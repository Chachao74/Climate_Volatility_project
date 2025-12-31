"""
CLIMATE TRADING BOT - STRATÉGIE MULTI-SECTEUR (COMPLET)
========================================================
Génère des prédictions réelles et signaux de trading pour 7 secteurs.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from tensorflow.keras.models import load_model
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# CONFIGURATION
# ==============================================

MODELS_DIR = 'bot/models'
LIVE_DATA_PATH = 'bot/live_data.csv'

# ISOs for Lightweight LSTM
LIGHTWEIGHT_ISOS = ['MHL', 'GRL', 'KEN', 'MEX', 'USA', 'KAZ', 'BRA', 'CHN', 'DEU']

SECTORS = {
    'Financials': {'type': 'LSTM', 'model': 'lstm_financials.h5', 'scaler': 'scaler_financials.pkl', 'ticker': 'XLF', 'da': 0.72},
    'Consumer_Staples': {'type': 'LSTM', 'model': 'lstm_consumer_staples.h5', 'scaler': 'scaler_consumer_staples.pkl', 'ticker': 'XLP', 'da': 0.72},
    'SP500': {'type': 'LSTM', 'model': 'lstm_sp500.h5', 'scaler': 'scaler_sp500.pkl', 'ticker': 'SPY', 'da': 0.71},
    'Utilities': {'type': 'RF', 'model': 'rf_utilities.pkl', 'ticker': 'XLU', 'da': 0.75},
    'Materials': {'type': 'RF', 'model': 'rf_materials.pkl', 'ticker': 'XLB', 'da': 0.68},
    'Technology': {'type': 'RF', 'model': 'rf_technology.pkl', 'ticker': 'XLK', 'da': 0.70},
    'Agriculture': {'type': 'LSTM_GLOBAL', 'model': 'lstm_agriculture.h5', 'scaler': 'scaler_agriculture.pkl', 'ticker': 'DBA', 'da': 0.90}
}

# ==============================================
# FONCTIONS
# ==============================================

def load_live_data():
    if not os.path.exists(LIVE_DATA_PATH):
        raise FileNotFoundError(f"Live data not found. Run: python3 bot/live_weather_fetcher.py")
    df = pd.read_csv(LIVE_DATA_PATH, index_col=0, parse_dates=True)
    return df

def predict_lstm(model, scaler, input_seq, seq_len=52):
    """Helper LSTM prediction with scaler handling"""
    n_features = input_seq.shape[2]
    n_expected = scaler.n_features_in_
    n_targets = n_expected - n_features
    
    # Handle dimension mismatch
    if n_targets < 0:
        n_feats_needed = n_expected - 1
        input_seq = input_seq[:, :, :n_feats_needed]
        n_features = n_feats_needed
        n_targets = 1
        
    raw_input_2d = input_seq[0] 
    dummy_targets = np.zeros((seq_len, n_targets))
    full_input = np.hstack([raw_input_2d, dummy_targets])
    
    scaled_full = scaler.transform(full_input)
    X_pred = np.expand_dims(scaled_full, axis=0)
    
    pred_scaled = model.predict(X_pred, verbose=0)
    
    dummy_scaled_row = np.zeros((1, n_expected))
    dummy_scaled_row[0, -1] = pred_scaled[0, 0]
    pred_raw = scaler.inverse_transform(dummy_scaled_row)[0, -1]
    
    return pred_raw

def get_rf_prediction(sector_name, df_live):
    """RF Prediction"""
    model_path = f"{MODELS_DIR}/rf_{sector_name.lower()}.pkl"
    if not os.path.exists(model_path):
        return None
        
    rf = joblib.load(model_path)
    
    # Monthly aggregation (last 30 days)
    current_month = df_live.tail(30).mean().to_frame().T
    
    # Align features
    if hasattr(rf, 'feature_names_in_'):
        expected_cols = rf.feature_names_in_
        current_month = current_month.reindex(columns=expected_cols, fill_value=0)
    
    try:
        pred = rf.predict(current_month)
        return pred[0]
    except Exception as e:
        print(f"   ⚠️ RF {sector_name} error: {e}")
        return None

def get_lstm_prediction(sector_name, df_live, is_global=False):
    """LSTM Prediction"""
    config = SECTORS[sector_name]
    model_path = f"{MODELS_DIR}/{config['model']}"
    scaler_path = f"{MODELS_DIR}/{config['scaler']}"
    
    if not os.path.exists(model_path):
        return None
    
    model = load_model(model_path, custom_objects={"attention_block": None}, compile=False)
    scaler = joblib.load(scaler_path)
    
    # Filter columns
    if not is_global:
        cols_to_keep = []
        for iso in LIGHTWEIGHT_ISOS:
            cols_to_keep.extend([f't2m_{iso}', f'tp_{iso}'])
        available_cols = [c for c in cols_to_keep if c in df_live.columns]
        df_subset = df_live.reindex(columns=cols_to_keep, fill_value=0)
    else:
        # Global: use all columns
        df_subset = df_live.copy()
    
    # Resample to Weekly
    df_weekly = df_subset.resample('W-FRI').mean()
    
    # Pad to 52 weeks
    SEQ_LEN = 52
    if len(df_weekly) < SEQ_LEN:
        missing = SEQ_LEN - len(df_weekly)
        padding = pd.DataFrame([df_weekly.iloc[0]] * missing, columns=df_weekly.columns)
        df_weekly_padded = pd.concat([padding, df_weekly], ignore_index=True)
    else:
        df_weekly_padded = df_weekly.tail(SEQ_LEN)
        
    input_seq = np.expand_dims(df_weekly_padded.values, axis=0)
    
    try:
        pred = predict_lstm(model, scaler, input_seq)
        return pred
    except Exception as e:
        print(f"   ⚠️ LSTM {sector_name} error: {e}")
        return None

def generate_signal(sector_name, predicted_rv, model_type):
    """Generate trading signal"""
    config = SECTORS[sector_name]
    
    # Adaptive thresholds
    if sector_name == 'Agriculture':
        low_vol, high_vol = 0.015, 0.025
    elif sector_name in ['Utilities', 'Consumer_Staples']:
        low_vol, high_vol = 0.01, 0.02
    else:
        low_vol, high_vol = 0.015, 0.03
    
    if predicted_rv < low_vol:
        signal = "LONG"
        confidence = "HIGH" if config['da'] >= 0.7 else "MEDIUM"
        rationale = f"Faible volatilité ({predicted_rv:.4f}) → Environnement stable favorable"
    elif predicted_rv > high_vol:
        signal = "SHORT/HEDGE"
        confidence = "HIGH" if config['da'] >= 0.7 else "MEDIUM"
        rationale = f"Haute volatilité ({predicted_rv:.4f}) → Alerte risque, protection recommandée"
    else:
        signal = "NEUTRAL"
        confidence = "MEDIUM"
        rationale = f"Volatilité modérée ({predicted_rv:.4f}) → Maintenir position actuelle"
    
    return {
        'Secteur': sector_name,
        'Ticker': config['ticker'],
        'Modèle': model_type,
        'RV_Prédite': predicted_rv,
        'Signal': signal,
        'Confiance': confidence,
        'DA': config['da'],
        'Rationale': rationale
    }

def run_strategy():
    print("="*80)
    print("🤖 CLIMATE TRADING BOT - STRATÉGIE MULTI-SECTEUR")
    print(f"Généré: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load live data
    try:
        df_live = load_live_data()
        print(f"\n✅ Données live chargées: {len(df_live)} jours, {df_live.shape[1]} variables")
    except Exception as e:
        print(f"\n❌ Erreur chargement données: {e}")
        print("Lancez: python3 bot/live_weather_fetcher.py")
        return
    
    results = []
    
    # Process each sector
    for sector_name, config in SECTORS.items():
        print(f"\n📊 {sector_name} ({config['ticker']})...")
        
        try:
            if config['type'] == 'LSTM':
                predicted_rv = get_lstm_prediction(sector_name, df_live, is_global=False)
            elif config['type'] == 'LSTM_GLOBAL':
                predicted_rv = get_lstm_prediction(sector_name, df_live, is_global=True)
            else:  # RF
                predicted_rv = get_rf_prediction(sector_name, df_live)
            
            if predicted_rv is None:
                print(f"   ⚠️ Prédiction échouée")
                continue
            
            signal_data = generate_signal(sector_name, predicted_rv, config['type'])
            results.append(signal_data)
            
            print(f"   Modèle: {config['type']} (DA: {config['da']:.0%})")
            print(f"   RV Prédite: {predicted_rv:.4f}")
            print(f"   Signal: {signal_data['Signal']} ({signal_data['Confiance']})")
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    if not results:
        print("\n❌ Aucun signal généré")
        return
    
    # Save results
    df_results = pd.DataFrame(results)
    
    # CSV
    df_results.to_csv('bot/strategy_signals.csv', index=False)
    print(f"\n📊 Signaux sauvegardés: bot/strategy_signals.csv")
    
    # Markdown Report
    with open('bot/STRATEGY_REPORT.md', 'w', encoding='utf-8') as f:
        f.write("# 🌍 Climate Trading Bot - Rapport de Stratégie\n\n")
        f.write(f"**Généré**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Données**: {len(df_live)} jours de météo live ({df_live.index.min().strftime('%Y-%m-%d')} → {df_live.index.max().strftime('%Y-%m-%d')})\n\n")
        
        f.write("## 📊 Signaux de Trading\n\n")
        f.write("| Secteur | Ticker | Modèle | RV Prédite | Signal | Confiance | DA |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        for _, row in df_results.iterrows():
            f.write(f"| **{row['Secteur']}** | {row['Ticker']} | {row['Modèle']} | "
                   f"{row['RV_Prédite']:.4f} | **{row['Signal']}** | "
                   f"{row['Confiance']} | {row['DA']:.0%} |\n")
        
        f.write("\n## 💡 Analyse par Secteur\n\n")
        for _, row in df_results.iterrows():
            f.write(f"### {row['Secteur']} ({row['Ticker']})\n")
            f.write(f"- **Signal**: {row['Signal']}\n")
            f.write(f"- **Confiance**: {row['Confiance']} (DA historique: {row['DA']:.0%})\n")
            f.write(f"- **Analyse**: {row['Rationale']}\n\n")
        
        f.write("## 📈 Résumé des Positions Recommandées\n\n")
        
        long_sectors = df_results[df_results['Signal'] == 'LONG']['Secteur'].tolist()
        hedge_sectors = df_results[df_results['Signal'] == 'SHORT/HEDGE']['Secteur'].tolist()
        neutral_sectors = df_results[df_results['Signal'] == 'NEUTRAL']['Secteur'].tolist()
        
        if long_sectors:
            f.write(f"**🟢 LONG** ({len(long_sectors)}): {', '.join(long_sectors)}\n\n")
        if hedge_sectors:
            f.write(f"**🔴 SHORT/HEDGE** ({len(hedge_sectors)}): {', '.join(hedge_sectors)}\n\n")
        if neutral_sectors:
            f.write(f"**⚪ NEUTRAL** ({len(neutral_sectors)}): {', '.join(neutral_sectors)}\n\n")
        
        f.write("---\n\n")
        f.write("*Généré automatiquement par Climate Trading Bot - Basé sur données ERA5 + EM-DAT*\n")
    
    print(f"📄 Rapport sauvegardé: bot/STRATEGY_REPORT.md")
    
    # Display summary
    print(f"\n{'='*80}")
    print("📈 RÉSUMÉ DES SIGNAUX")
    print(f"{'='*80}")
    print(df_results[['Secteur', 'Ticker', 'Signal', 'RV_Prédite', 'Confiance']].to_string(index=False))
    print(f"\n✅ Stratégie complète générée!")
    
    return df_results

if __name__ == "__main__":
    run_strategy()
