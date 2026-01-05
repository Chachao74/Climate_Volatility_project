import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'process', 'panel_monthly.csv')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'final_analysis_enhanced')

# Top 3 Sectors for detailed plotting
TOP_SECTORS = ['RV_Utilities', 'RV_Technology', 'RV_Materials']

def detect_regime(volatility_series, window=12):
    """
    Détecte le régime de volatilité (low/high) pour la visualisation
    """
    rolling_vol = volatility_series.rolling(window).std()
    median_vol = rolling_vol.median()
    
    regime = pd.Series(index=volatility_series.index, dtype=str)
    regime[rolling_vol <= median_vol] = 'low_vol'
    regime[rolling_vol > median_vol] = 'high_vol'
    
    return regime

def calculate_advanced_metrics(y_true, y_pred, y_lag):
    """
    Calcule des métriques avancées incluant Sharpe-like ratio
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Directional Accuracy
    direction_actual = np.sign(y_true - y_lag)
    direction_pred = np.sign(y_pred - y_lag)
    da = np.mean(direction_actual == direction_pred)
    
    # Hit Rate (prédit correctement les grandes variations)
    threshold = y_true.std()
    large_moves = np.abs(y_true - y_lag) > threshold
    if large_moves.sum() > 0:
        hit_rate = np.mean(direction_actual[large_moves] == direction_pred[large_moves])
    else:
        hit_rate = np.nan
    
    # Sharpe-like metric
    returns = y_true.values - y_lag.values
    pred_returns = y_pred - y_lag.values
    signal = np.sign(pred_returns)
    strategy_returns = signal * returns
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-10) * np.sqrt(12)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'DA': da,
        'HitRate': hit_rate,
        'Sharpe': sharpe
    }

def analyze_all_sectors():
    # Create results directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created directory: {RESULTS_DIR}")

    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {DATA_PATH}")
        return

    # Preprocessing
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Pivot
    print("Pivoting data...")
    exclude_cols = ['Date', 'ISO', 'year', 'month']
    value_vars = [c for c in df.columns if c not in exclude_cols]
    df_pivot = df.pivot(index='Date', columns='ISO', values=value_vars)
    df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
    df_pivot = df_pivot.fillna(0)
    
    # Identify Features (exclude all RV columns)
    feature_cols = [c for c in df_pivot.columns if 'RV_' not in c]
    
    # EXCLUDE PAKISTAN
    feature_cols = [c for c in feature_cols if '_PAK' not in c]
    print(f"Using {len(feature_cols)} global features (Pakistan excluded).")
    
    # Identify ALL Targets
    target_cols = [c for c in df_pivot.columns if c.startswith('RV_') and c.endswith('_USA')]
    print(f"Found {len(target_cols)} USA targets.")
    
    results_summary = []

    for target in target_cols:
        target_name = target.replace('_USA', '')
        print(f"\n{'='*60}")
        print(f"Processing {target_name}...")
        print(f"{'='*60}")
        
        # Add Lagged Target
        lag_col = f"{target}_lag1"
        df_target = df_pivot.copy()
        df_target[lag_col] = df_target[target].shift(1)
        df_target = df_target.dropna()

        # Exclude Volatile Periods
        exclude_mask = (
            ((df_target.index >= '2008-09-01') & (df_target.index <= '2009-06-01')) |
            ((df_target.index >= '2020-02-01') & (df_target.index <= '2020-06-01'))
        )
        df_target = df_target[~exclude_mask]
        
        # Detect regime BEFORE split (for visualization)
        regime = detect_regime(df_target[target])
        df_target['regime'] = regime
        
        # Features
        current_feature_cols = feature_cols + [lag_col]
        X = df_target[current_feature_cols]
        y = df_target[target]

        # Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train
        print("Training Random Forest...")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        y_lag_test = X_test[lag_col]
        metrics = calculate_advanced_metrics(y_test, y_pred_test, y_lag_test)
        
        # Naive baseline
        naive_pred = X_test[lag_col]
        naive_mae = mean_absolute_error(y_test, naive_pred)
        
        print(f"\nTest Metrics:")
        print(f"MAE: {metrics['MAE']:.5f} (Naive: {naive_mae:.5f})")
        print(f"DA: {metrics['DA']:.2%}")
        print(f"Hit Rate: {metrics['HitRate']:.2%}")
        print(f"Sharpe: {metrics['Sharpe']:.2f}")
        
        results_summary.append({
            'Target': target_name, 
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'DA': metrics['DA'],
            'HitRate': metrics['HitRate'],
            'Sharpe': metrics['Sharpe'],
            'Naive_MAE': naive_mae
        })

        # Special processing for Top 3 Sectors
        if target_name in TOP_SECTORS:
            print(f"\nCreating detailed visualizations for {target_name}...")
            
            # 1. Feature Importance Plot
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_n = 20
            
            plt.figure(figsize=(12, 8))
            plt.title(f"Top {top_n} Features - {target_name}", fontsize=14, fontweight='bold')
            plt.barh(range(top_n), importances[indices[:top_n]], align="center", color='steelblue')
            plt.yticks(range(top_n), [X.columns[i] for i in indices[:top_n]], fontsize=9)
            plt.gca().invert_yaxis()
            plt.xlabel("Relative Importance", fontsize=11)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"{target_name}_importance.png"), dpi=150)
            plt.close()
            
            # Save importance CSV
            imp_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            imp_df.head(50).to_csv(os.path.join(RESULTS_DIR, f"{target_name}_importance.csv"), index=False)

            # 2. Full Time Series Plot WITH REGIME VISUALIZATION
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                           gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot principal - Time Series
            ax1.plot(y_train.index, y_train, label='Actual Train', 
                    color='blue', alpha=0.3, linewidth=1)
            ax1.plot(y_train.index, y_pred_train, label='Predicted Train', 
                    color='orange', alpha=0.5, linewidth=1)
            ax1.plot(y_test.index, y_test, label='Actual Test', 
                    color='blue', alpha=0.8, linewidth=1.5)
            ax1.plot(y_test.index, y_pred_test, label='Predicted Test', 
                    color='red', alpha=0.8, linestyle='--', linewidth=1.5)
            ax1.axvline(x=y_test.index[0], color='black', linestyle=':', 
                       linewidth=2, label='Train/Test Split')
            
            ax1.set_title(f"{target_name} - Full Time Series (Train + Test)\n" + 
                         f"Test MAE: {metrics['MAE']:.5f} | DA: {metrics['DA']:.2%} | Sharpe: {metrics['Sharpe']:.2f}",
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Realized Volatility', fontsize=11)
            ax1.legend(loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Plot des régimes (LE TRUC COOL !)
            regime_colors = {'low_vol': 'green', 'high_vol': 'red'}
            for regime_type in ['low_vol', 'high_vol']:
                mask = df_target.loc[y.index, 'regime'] == regime_type
                ax2.scatter(y.index[mask], [0]*mask.sum(), 
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
            plt.savefig(os.path.join(RESULTS_DIR, f"{target_name}_full_timeseries.png"), dpi=150)
            plt.close()
            
            print(f"✓ Saved visualizations for {target_name}")

    # Save Summary CSV
    summary_df = pd.DataFrame(results_summary)
    summary_df = summary_df.sort_values('DA', ascending=False)
    summary_path = os.path.join(RESULTS_DIR, 'summary_metrics.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*80}")
    print("SUMMARY - All Sectors by Directional Accuracy:")
    print(f"{'='*80}")
    print(summary_df[['Target', 'MAE', 'DA', 'HitRate', 'Sharpe']].to_string(index=False))
    print(f"\n{'='*80}")
    print(f"Top 3 Sectors (Detailed Analysis):")
    print(f"{'='*80}")
    top3 = summary_df[summary_df['Target'].isin(TOP_SECTORS)]
    print(top3.to_string(index=False))
    print(f"\n{'='*80}")
    print(f"Full summary saved to {summary_path}")
    print(f"Detailed visualizations saved for: {', '.join(TOP_SECTORS)}")
    print(f"{'='*80}")

if __name__ == "__main__":
    analyze_all_sectors()
