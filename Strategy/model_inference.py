"""
CLIMATE TRADING BOT - INFERENCE ENGINE
======================================
Loads trained models (RF & LSTM) and live weather data to generate trading signals.

Inputs:
- `bot/live_data.csv`: Live weather data (t2m, tp).
- `bot/models/`: Trained models (.pkl, .h5) and scalers.

Outputs:
- Dictionary of signals for each sector/engine.
"""

import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# CONFIG
# CONFIG
MODELS_DIR = 'bot/models'
LIVE_DATA_PATH = 'bot/live_data.csv'

# Model Configs
RF_SECTORS = ['Utilities', 'Technology', 'Materials']
LSTM_SECTORS = ['Financials', 'Consumer_Staples', 'SP500']
AGRI_SECTOR = 'Agriculture'

# Key ISOs for "Lightweight" LSTMs (Financials, SP500, Staples)
# These models were trained on a subset of 9 locations.
LIGHTWEIGHT_ISOS = ['MHL', 'GRL', 'KEN', 'MEX', 'USA', 'KAZ', 'BRA', 'CHN', 'DEU']

def load_live_data():
    if not os.path.exists(LIVE_DATA_PATH):
        raise FileNotFoundError(f"Live data not found at {LIVE_DATA_PATH}. Run live_weather_fetcher.py first.")
    
    df = pd.read_csv(LIVE_DATA_PATH, index_col=0, parse_dates=True)
    return df

def get_rf_signals(df_live):
    signals = {}
    print("\n🌲 RF ENGINE (Alpha Generator)")
    
    # RF expects Monthly data (Global).
    # Resample to Monthly Mean (Last 30 days)
    current_month = df_live.tail(30).mean().to_frame().T
    
    # RF models were trained on panel_monthly.csv.
    # We assume live_data.csv has the same columns (or superset).
    # We might need to align columns if order differs.
    
    for sector in RF_SECTORS:
        model_path = f"{MODELS_DIR}/rf_{sector.lower()}.pkl"
        if not os.path.exists(model_path):
            print(f"   ⚠️ Model missing: {model_path}")
            continue
            
        rf = joblib.load(model_path)
        
        # Align Features
        # RF doesn't store feature names, but we know it uses all t2m/tp columns sorted?
        # Ideally we should have saved feature names.
        # For now, we pass the full global vector.
        # If shape mismatch, we catch it.
        
        try:
            # FIX: Align features with model expectations
            if hasattr(rf, 'feature_names_in_'):
                expected_cols = rf.feature_names_in_
                # Reindex to match model features (fills missing with 0, drops extra)
                # This handles missing 'affected_total' (disasters) by assuming 0 (no disaster)
                current_month = current_month.reindex(columns=expected_cols, fill_value=0)
            
            pred = rf.predict(current_month)
            vol_pred = pred[0]
            
            signal = "NEUTRAL"
            if vol_pred > 0.04:
                signal = "HIGH VOL (DEFENSIVE)"
            elif vol_pred < 0.02:
                signal = "LOW VOL (BULLISH)"
                
            signals[sector] = {
                "Pred_RV": vol_pred,
                "Signal": signal
            }
            print(f"   {sector}: {vol_pred:.4f} -> {signal}")
            
        except Exception as e:
            print(f"   ❌ Inference failed for {sector}: {e}")
            
    return signals

def predict_lstm(model, scaler, input_seq, seq_len=52):
    # Helper to predict with scaler dummy target handling
    n_features = input_seq.shape[2]
    n_expected = scaler.n_features_in_
    n_targets = n_expected - n_features
    
    # FIX: Handle dimension mismatch (e.g. extra ISOs in live data)
    if n_targets < 0:
        # We have MORE features than expected.
        # We must slice input_seq to match n_expected (minus 1 target usually)
        # Assuming 1 target
        n_feats_needed = n_expected - 1
        input_seq = input_seq[:, :, :n_feats_needed]
        n_features = n_feats_needed
        n_targets = 1
        # print(f"   ⚠️ Sliced input to {n_features} features")
        
    # Pad for scaler
    raw_input_2d = input_seq[0] 
    dummy_targets = np.zeros((seq_len, n_targets))
    full_input = np.hstack([raw_input_2d, dummy_targets])
    
    scaled_full = scaler.transform(full_input)
    X_pred = np.expand_dims(scaled_full, axis=0)
    
    # Predict
    pred_scaled = model.predict(X_pred, verbose=0)
    
    # Inverse Transform
    dummy_scaled_row = np.zeros((1, n_expected))
    dummy_scaled_row[0, -1] = pred_scaled[0, 0]
    pred_raw = scaler.inverse_transform(dummy_scaled_row)[0, -1]
    
    return pred_raw

def get_lstm_signals(df_live):
    signals = {}
    print("\n🧠 LSTM ENGINE (Risk Manager)")
    
    # Filter for Lightweight ISOs
    cols_to_keep = []
    for iso in LIGHTWEIGHT_ISOS:
        cols_to_keep.extend([f't2m_{iso}', f'tp_{iso}'])
        
    # Check if columns exist
    available_cols = [c for c in cols_to_keep if c in df_live.columns]
    if len(available_cols) != len(cols_to_keep):
        print(f"   ⚠️ Missing columns for Lightweight LSTM. Have {len(available_cols)}/{len(cols_to_keep)}")
        # We proceed with what we have, filling missing with 0
        
    df_subset = df_live.reindex(columns=cols_to_keep, fill_value=0)
    
    # Resample to Weekly
    df_weekly = df_subset.resample('W-FRI').mean()
    
    # Pad Sequence
    SEQ_LEN = 52
    if len(df_weekly) < SEQ_LEN:
        missing = SEQ_LEN - len(df_weekly)
        padding = pd.DataFrame([df_weekly.iloc[0]] * missing, columns=df_weekly.columns)
        df_weekly_padded = pd.concat([padding, df_weekly], ignore_index=True)
    else:
        df_weekly_padded = df_weekly.tail(SEQ_LEN)
        
    input_seq = np.expand_dims(df_weekly_padded.values, axis=0)
    
    for sector in LSTM_SECTORS:
        model_path = f"{MODELS_DIR}/lstm_{sector.lower()}.h5"
        scaler_path = f"{MODELS_DIR}/scaler_{sector.lower()}.pkl"
        
        if not os.path.exists(model_path): continue
        
        model = load_model(model_path, custom_objects={"attention_block": None}, compile=False)
        scaler = joblib.load(scaler_path)
        
        try:
            pred_raw = predict_lstm(model, scaler, input_seq)
            
            signal = "STABLE"
            if pred_raw > 0.025:
                signal = "CRISIS ALERT"
                
            signals[sector] = {"Pred_RV": pred_raw, "Signal": signal}
            print(f"   {sector}: {pred_raw:.4f} -> {signal}")
            
        except Exception as e:
            print(f"   ❌ Inference failed for {sector}: {e}")
            
    return signals

def get_agri_signals(df_live):
    print("\n🌾 AGRICULTURE ENGINE (Global)")
    
    # Agri uses ALL columns (Global)
    # Resample to Weekly
    df_weekly = df_live.resample('W-FRI').mean()
    
    # Pad Sequence
    SEQ_LEN = 52
    if len(df_weekly) < SEQ_LEN:
        missing = SEQ_LEN - len(df_weekly)
        padding = pd.DataFrame([df_weekly.iloc[0]] * missing, columns=df_weekly.columns)
        df_weekly_padded = pd.concat([padding, df_weekly], ignore_index=True)
    else:
        df_weekly_padded = df_weekly.tail(SEQ_LEN)
        
    input_seq = np.expand_dims(df_weekly_padded.values, axis=0)
    
    model_path = f"{MODELS_DIR}/lstm_agriculture.h5"
    scaler_path = f"{MODELS_DIR}/scaler_agriculture.pkl" # Wait, did I save scaler?
    # In agri.py: joblib.dump(scaler, f'{MODELS_DIR}/scaler_agriculture.pkl') -> Yes.
    
    if not os.path.exists(model_path):
        print("   ⚠️ Agri model missing")
        return {}
        
    model = load_model(model_path, custom_objects={"attention_block": None}, compile=False)
    scaler = joblib.load(scaler_path)
    
    try:
        pred_raw = predict_lstm(model, scaler, input_seq)
        
        signal = "NEUTRAL"
        if pred_raw > 0.02: # Tune threshold
            signal = "HIGH VOL (SHORT)"
        else:
            signal = "LOW VOL (LONG)"
            
        print(f"   Agriculture: {pred_raw:.4f} -> {signal}")
        return {"Agriculture": {"Pred_RV": pred_raw, "Signal": signal}}
        
    except Exception as e:
        print(f"   ❌ Agri Inference failed: {e}")
        return {}

def run_inference():
    print(f"{'='*60}")
    print("🤖 CLIMATE TRADING BOT - INFERENCE")
    print(f"{'='*60}")
    
    df_live = load_live_data()
    print(f"   Loaded Live Data: {len(df_live)} rows, {df_live.shape[1]} cols")
    
    signals_rf = get_rf_signals(df_live)
    signals_lstm = get_lstm_signals(df_live)
    signals_agri = get_agri_signals(df_live)
    
    all_signals = {**signals_rf, **signals_lstm, **signals_agri}
    
    # Save Report
    with open('bot/CLIMATE_TRADING_REPORT.md', 'w') as f:
        f.write("# 🌍 Climate Trading Bot Report\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        f.write("## 📊 Signals\n")
        f.write("| Sector | Engine | Predicted RV | Signal |\n")
        f.write("|---|---|---|---|\n")
        for sector, data in all_signals.items():
            engine = "LSTM" if sector in LSTM_SECTORS or sector == 'Agriculture' else "RF"
            f.write(f"| {sector} | {engine} | {data['Pred_RV']:.4f} | {data['Signal']} |\n")
            
    print(f"\n✅ Report saved to bot/CLIMATE_TRADING_REPORT.md")
    return all_signals

if __name__ == "__main__":
    run_inference()
