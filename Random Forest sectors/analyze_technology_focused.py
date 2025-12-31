"""
Analysis Script: Technology Sector with Relevant Features
Focus on mining countries (batteries, rare earths) and tech hubs
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = '/Users/charlieormond/Desktop/charlie_projet/data/process/panel_monthly.csv'
RESULTS_DIR = '/Users/charlieormond/Desktop/charlie_projet/results/tech_analysis'

# Pays pertinents pour Technology
# Regroupés par catégorie pour analyse

# MÉTAUX BATTERIES & ÉLECTRONIQUE
BATTERY_MINING_COUNTRIES = [
    'COD',  # RD Congo - Cobalt (60% mondial)
    'CHL',  # Chili - Lithium (26% mondial)
    'ARG',  # Argentine - Lithium 
    'AUS',  # Australie - Lithium, Cobalt
    'ZAF',  # Afrique du Sud - Platine, Manganèse
    'ZMB',  # Zambie - Cobalt
    'PER',  # Pérou - Cuivre
    'IDN',  # Indonésie - Nickel (batteries)
    'PHL',  # Philippines - Nickel
]

# TERRES RARES
RARE_EARTH_COUNTRIES = [
    'CHN',  # Chine - 60% production terres rares
    'AUS',  # Australie - Terres rares
    'MYS',  # Malaisie - Traitement terres rares
    'IND',  # Inde - Thorium, terres rares
    'BRA',  # Brésil - Niobium
]

# HUBS TECH & SEMICONDUCTEURS
TECH_HUB_COUNTRIES = [
    'TWN',  # Taiwan - TSMC (semiconducteurs)
    'KOR',  # Corée - Samsung, SK Hynix
    'JPN',  # Japon - Sony, Toshiba
    'USA',  # USA - Silicon Valley
    'CHN',  # Chine - Shenzhen, assemblage
    'SGP',  # Singapore - Hub tech Asie
    'DEU',  # Allemagne - SAP, industrie 4.0
    'ISR',  # Israel - Cybersécurité, chips
]

# SUPPLY CHAIN CRITIQUE
SUPPLY_CHAIN_COUNTRIES = [
    'VNM',  # Vietnam - Assemblage électronique
    'THA',  # Thaïlande - Disques durs
    'MYS',  # Malaisie - Semiconducteurs
    'MEX',  # Mexique - Assemblage (USMCA)
]

# Combine all relevant countries (remove duplicates)
RELEVANT_COUNTRIES = list(set(
    BATTERY_MINING_COUNTRIES + 
    RARE_EARTH_COUNTRIES + 
    TECH_HUB_COUNTRIES + 
    SUPPLY_CHAIN_COUNTRIES
))

print(f"Total relevant countries: {len(RELEVANT_COUNTRIES)}")
print(f"Categories:")
print(f"  - Battery mining: {len(BATTERY_MINING_COUNTRIES)}")
print(f"  - Rare earths: {len(RARE_EARTH_COUNTRIES)}")
print(f"  - Tech hubs: {len(TECH_HUB_COUNTRIES)}")
print(f"  - Supply chain: {len(SUPPLY_CHAIN_COUNTRIES)}")

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def detect_regime(volatility_series, window=12):
    """Détecte régimes de volatilité"""
    rolling_vol = volatility_series.rolling(window).std()
    median_vol = rolling_vol.median()
    regime = pd.Series(index=volatility_series.index, dtype=str)
    regime[rolling_vol <= median_vol] = 'low_vol'
    regime[rolling_vol > median_vol] = 'high_vol'
    return regime

def calculate_metrics(y_true, y_pred, y_lag):
    """Calcule métriques complètes"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    direction_actual = np.sign(y_true - y_lag)
    direction_pred = np.sign(y_pred - y_lag)
    da = np.mean(direction_actual == direction_pred)
    
    threshold = y_true.std()
    large_moves = np.abs(y_true - y_lag) > threshold
    if large_moves.sum() > 0:
        hit_rate = np.mean(direction_actual[large_moves] == direction_pred[large_moves])
    else:
        hit_rate = np.nan
    
    returns = y_true.values - y_lag.values
    pred_returns = y_pred - y_lag.values
    signal = np.sign(pred_returns)
    strategy_returns = signal * returns
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-10) * np.sqrt(12)
    
    return {'MAE': mae, 'RMSE': rmse, 'DA': da, 'HitRate': hit_rate, 'Sharpe': sharpe}

def categorize_feature(feature_name, country_code):
    """Catégorise une feature selon le pays"""
    if country_code in BATTERY_MINING_COUNTRIES:
        return 'Battery Mining'
    elif country_code in RARE_EARTH_COUNTRIES:
        return 'Rare Earth'
    elif country_code in TECH_HUB_COUNTRIES:
        return 'Tech Hub'
    elif country_code in SUPPLY_CHAIN_COUNTRIES:
        return 'Supply Chain'
    else:
        return 'Other'

# ============================================================================
# ANALYSE PRINCIPALE
# ============================================================================

def analyze_technology_focused():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created directory: {RESULTS_DIR}")

    print(f"\nLoading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Pivot
    print("Pivoting data...")
    exclude_cols = ['Date', 'ISO', 'year', 'month']
    value_vars = [c for c in df.columns if c not in exclude_cols]
    df_pivot = df.pivot(index='Date', columns='ISO', values=value_vars)
    df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
    df_pivot = df_pivot.fillna(0)
    
    # Target
    target = 'RV_Technology_USA'
    
    # ========================================================================
    # EXPÉRIENCE 1: TOUTES LES FEATURES (BASELINE)
    # ========================================================================
    print(f"\n{'='*80}")
    print("EXPERIMENT 1: ALL FEATURES (Baseline)")
    print(f"{'='*80}")
    
    all_features = [c for c in df_pivot.columns if 'RV_' not in c and '_PAK' not in c]
    lag_col = f"{target}_lag1"
    
    df_exp1 = df_pivot.copy()
    df_exp1[lag_col] = df_exp1[target].shift(1)
    df_exp1 = df_exp1.dropna()
    
    # Exclude volatile periods
    exclude_mask = (
        ((df_exp1.index >= '2008-09-01') & (df_exp1.index <= '2009-06-01')) |
        ((df_exp1.index >= '2020-02-01') & (df_exp1.index <= '2020-06-01'))
    )
    df_exp1 = df_exp1[~exclude_mask]
    
    X_all = df_exp1[all_features + [lag_col]]
    y_all = df_exp1[target]
    
    split_idx = int(len(X_all) * 0.8)
    X_train_all, X_test_all = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_train_all, y_test_all = y_all.iloc[:split_idx], y_all.iloc[split_idx:]
    
    print(f"Training with {X_train_all.shape[1]} features...")
    model_all = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_all.fit(X_train_all, y_train_all)
    
    y_pred_all = model_all.predict(X_test_all)
    metrics_all = calculate_metrics(y_test_all, y_pred_all, X_test_all[lag_col])
    
    print(f"\nBaseline Results (All Features):")
    print(f"  MAE: {metrics_all['MAE']:.5f}")
    print(f"  DA: {metrics_all['DA']:.2%}")
    print(f"  Hit Rate: {metrics_all['HitRate']:.2%}")
    print(f"  Sharpe: {metrics_all['Sharpe']:.2f}")
    
    # ========================================================================
    # EXPÉRIENCE 2: SEULEMENT PAYS PERTINENTS
    # ========================================================================
    print(f"\n{'='*80}")
    print("EXPERIMENT 2: RELEVANT COUNTRIES ONLY")
    print(f"{'='*80}")
    
    # Filter features to only relevant countries
    relevant_features = []
    for feat in all_features:
        # Check if feature ends with any relevant country code
        for country in RELEVANT_COUNTRIES:
            if feat.endswith(f'_{country}'):
                relevant_features.append(feat)
                break
    
    print(f"Filtered to {len(relevant_features)} relevant features (from {len(all_features)})")
    
    df_exp2 = df_pivot.copy()
    df_exp2[lag_col] = df_exp2[target].shift(1)
    df_exp2 = df_exp2.dropna()
    df_exp2 = df_exp2[~exclude_mask]
    
    X_rel = df_exp2[relevant_features + [lag_col]]
    y_rel = df_exp2[target]
    
    split_idx = int(len(X_rel) * 0.8)
    X_train_rel, X_test_rel = X_rel.iloc[:split_idx], X_rel.iloc[split_idx:]
    y_train_rel, y_test_rel = y_rel.iloc[:split_idx], y_rel.iloc[split_idx:]
    
    print(f"Training with {X_train_rel.shape[1]} features...")
    model_rel = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_rel.fit(X_train_rel, y_train_rel)
    
    y_pred_rel = model_rel.predict(X_test_rel)
    metrics_rel = calculate_metrics(y_test_rel, y_pred_rel, X_test_rel[lag_col])
    
    print(f"\nRelevant Countries Results:")
    print(f"  MAE: {metrics_rel['MAE']:.5f} (baseline: {metrics_all['MAE']:.5f})")
    print(f"  DA: {metrics_rel['DA']:.2%} (baseline: {metrics_all['DA']:.2%})")
    print(f"  Hit Rate: {metrics_rel['HitRate']:.2%} (baseline: {metrics_all['HitRate']:.2%})")
    print(f"  Sharpe: {metrics_rel['Sharpe']:.2f} (baseline: {metrics_all['Sharpe']:.2f})")
    
    # ========================================================================
    # EXPÉRIENCE 3: SANS LES PAYS CHELOUS (TZA, KEN, GUF)
    # ========================================================================
    print(f"\n{'='*80}")
    print("EXPERIMENT 3: WITHOUT WEIRD COUNTRIES (TZA, KEN, GUF)")
    print(f"{'='*80}")
    
    EXCLUDE_WEIRD = ['TZA', 'KEN', 'GUF', 'BGR', 'AFG', 'ARM', 'DMA', 'LCA']
    
    clean_features = [f for f in all_features 
                     if not any(f.endswith(f'_{country}') for country in EXCLUDE_WEIRD)]
    
    print(f"Removed features from: {', '.join(EXCLUDE_WEIRD)}")
    print(f"Remaining: {len(clean_features)} features (from {len(all_features)})")
    
    df_exp3 = df_pivot.copy()
    df_exp3[lag_col] = df_exp3[target].shift(1)
    df_exp3 = df_exp3.dropna()
    df_exp3 = df_exp3[~exclude_mask]
    
    X_clean = df_exp3[clean_features + [lag_col]]
    y_clean = df_exp3[target]
    
    split_idx = int(len(X_clean) * 0.8)
    X_train_clean, X_test_clean = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
    y_train_clean, y_test_clean = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
    
    print(f"Training with {X_train_clean.shape[1]} features...")
    model_clean = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_clean.fit(X_train_clean, y_train_clean)
    
    y_pred_clean = model_clean.predict(X_test_clean)
    metrics_clean = calculate_metrics(y_test_clean, y_pred_clean, X_test_clean[lag_col])
    
    print(f"\nWithout Weird Countries Results:")
    print(f"  MAE: {metrics_clean['MAE']:.5f} (baseline: {metrics_all['MAE']:.5f})")
    print(f"  DA: {metrics_clean['DA']:.2%} (baseline: {metrics_all['DA']:.2%})")
    print(f"  Hit Rate: {metrics_clean['HitRate']:.2%} (baseline: {metrics_all['HitRate']:.2%})")
    print(f"  Sharpe: {metrics_clean['Sharpe']:.2f} (baseline: {metrics_all['Sharpe']:.2f})")
    
    # ========================================================================
    # ANALYSE DES FEATURES IMPORTANTES (Expérience 2 - Relevant Countries)
    # ========================================================================
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE ANALYSIS (Relevant Countries Model)")
    print(f"{'='*80}")
    
    importances = model_rel.feature_importances_
    feature_names = X_train_rel.columns
    
    # Create dataframe with categories
    importance_data = []
    for feat, imp in zip(feature_names, importances):
        # Extract country code
        parts = feat.split('_')
        country = parts[-1] if len(parts) > 1 else 'Unknown'
        category = categorize_feature(feat, country)
        
        importance_data.append({
            'Feature': feat,
            'Country': country,
            'Category': category,
            'Importance': imp
        })
    
    imp_df = pd.DataFrame(importance_data).sort_values('Importance', ascending=False)
    
    # Save full importance
    imp_df.to_csv(os.path.join(RESULTS_DIR, 'importance_relevant_countries.csv'), index=False)
    
    # Print top 30
    print("\nTop 30 Most Important Features:")
    print(imp_df.head(30).to_string(index=False))
    
    # Aggregate by category
    print(f"\n{'='*80}")
    print("IMPORTANCE BY CATEGORY")
    print(f"{'='*80}")
    
    category_importance = imp_df.groupby('Category')['Importance'].agg(['sum', 'mean', 'count'])
    category_importance = category_importance.sort_values('sum', ascending=False)
    category_importance['sum_pct'] = category_importance['sum'] / category_importance['sum'].sum() * 100
    
    print("\n" + category_importance.to_string())
    
    # Top countries by total importance
    print(f"\n{'='*80}")
    print("TOP 20 COUNTRIES BY TOTAL IMPORTANCE")
    print(f"{'='*80}")
    
    country_importance = imp_df.groupby('Country')['Importance'].agg(['sum', 'count'])
    country_importance = country_importance.sort_values('sum', ascending=False).head(20)
    country_importance['category'] = country_importance.index.map(
        lambda x: categorize_feature('', x)
    )
    
    print("\n" + country_importance.to_string())
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    # 1. Comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    experiments = ['Baseline\n(All Features)', 
                   'Relevant\nCountries', 
                   'Without\nWeird Countries']
    das = [metrics_all['DA'], metrics_rel['DA'], metrics_clean['DA']]
    colors = ['gray', 'steelblue', 'darkgreen']
    
    bars = ax.bar(experiments, das, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, da in zip(bars, das):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{da:.1%}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Directional Accuracy (%)', fontsize=12)
    ax.set_title('Technology Sector: Model Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random (50%)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=150)
    plt.close()
    print("✓ Saved model comparison chart")
    
    # 2. Category importance pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    explode = [0.05 if cat == 'Battery Mining' else 0 for cat in category_importance.index]
    colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    
    wedges, texts, autotexts = ax.pie(
        category_importance['sum_pct'], 
        labels=category_importance.index,
        autopct='%1.1f%%',
        startangle=90,
        explode=explode,
        colors=colors_pie,
        textprops={'fontsize': 11}
    )
    
    ax.set_title('Feature Importance by Category\n(Relevant Countries Model)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'category_importance.png'), dpi=150)
    plt.close()
    print("✓ Saved category importance chart")
    
    # 3. Time series with regime (Relevant Countries Model)
    regime = detect_regime(y_rel)
    df_exp2['regime'] = regime
    
    y_pred_train_rel = model_rel.predict(X_train_rel)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(y_train_rel.index, y_train_rel, label='Actual Train', 
            color='blue', alpha=0.3, linewidth=1)
    ax1.plot(y_train_rel.index, y_pred_train_rel, label='Predicted Train', 
            color='orange', alpha=0.5, linewidth=1)
    ax1.plot(y_test_rel.index, y_test_rel, label='Actual Test', 
            color='blue', alpha=0.8, linewidth=1.5)
    ax1.plot(y_test_rel.index, y_pred_rel, label='Predicted Test', 
            color='red', alpha=0.8, linestyle='--', linewidth=1.5)
    ax1.axvline(x=y_test_rel.index[0], color='black', linestyle=':', 
               linewidth=2, label='Train/Test Split')
    
    ax1.set_title(f"Technology - Relevant Countries Model\n" + 
                 f"Test MAE: {metrics_rel['MAE']:.5f} | DA: {metrics_rel['DA']:.2%} | " +
                 f"Sharpe: {metrics_rel['Sharpe']:.2f}",
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Realized Volatility', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Regime plot
    regime_colors = {'low_vol': 'green', 'high_vol': 'red'}
    for regime_type in ['low_vol', 'high_vol']:
        mask = df_exp2.loc[y_rel.index, 'regime'] == regime_type
        ax2.scatter(y_rel.index[mask], [0]*mask.sum(), 
                  c=regime_colors[regime_type], 
                  label=regime_type.replace('_', ' ').title(), 
                  alpha=0.6, s=15, marker='|')
    
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Regime', fontsize=11)
    ax2.set_yticks([])
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_ylim(-0.5, 0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'timeseries_relevant_countries.png'), dpi=150)
    plt.close()
    print("✓ Saved time series with regimes")
    
    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    summary_text = f"""
TECHNOLOGY SECTOR ANALYSIS - SUMMARY REPORT
{'='*80}

1. BASELINE (All Features: {len(all_features)} features)
   - DA: {metrics_all['DA']:.2%}
   - MAE: {metrics_all['MAE']:.5f}
   - Sharpe: {metrics_all['Sharpe']:.2f}

2. RELEVANT COUNTRIES ONLY ({len(relevant_features)} features)
   - DA: {metrics_rel['DA']:.2%} ({(metrics_rel['DA'] - metrics_all['DA'])*100:+.1f}pp)
   - MAE: {metrics_rel['MAE']:.5f} ({(metrics_rel['MAE'] - metrics_all['MAE']):.5f})
   - Sharpe: {metrics_rel['Sharpe']:.2f} ({metrics_rel['Sharpe'] - metrics_all['Sharpe']:+.2f})

3. WITHOUT WEIRD COUNTRIES ({len(clean_features)} features)
   - DA: {metrics_clean['DA']:.2%} ({(metrics_clean['DA'] - metrics_all['DA'])*100:+.1f}pp)
   - MAE: {metrics_clean['MAE']:.5f} ({(metrics_clean['MAE'] - metrics_all['MAE']):.5f})
   - Sharpe: {metrics_clean['Sharpe']:.2f} ({metrics_clean['Sharpe'] - metrics_all['Sharpe']:+.2f})

{'='*80}
CONCLUSION:

{'✓ BETTER' if metrics_rel['DA'] > metrics_all['DA'] else '✗ WORSE'}: Focusing on relevant countries {'IMPROVES' if metrics_rel['DA'] > metrics_all['DA'] else 'HURTS'} performance
{'✓ BETTER' if metrics_clean['DA'] > metrics_all['DA'] else '✗ WORSE'}: Removing weird countries {'IMPROVES' if metrics_clean['DA'] > metrics_all['DA'] else 'HURTS'} performance

Top Category: {category_importance.index[0]} ({category_importance.iloc[0]['sum_pct']:.1f}% importance)

Files saved to: {RESULTS_DIR}/
- model_comparison.png
- category_importance.png  
- timeseries_relevant_countries.png
- importance_relevant_countries.csv
"""
    
    print(summary_text)
    
    # Save report
    with open(os.path.join(RESULTS_DIR, 'ANALYSIS_REPORT.txt'), 'w') as f:
        f.write(summary_text)
    
    print(f"\n✓ Analysis complete! Results saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    analyze_technology_focused()
