"""
AGRICULTURE ENGINE (agri.py)
============================
The "Agriculture Specialist" for the Climate Trading Bot.

Functions:
1.  **Data Acquisition**: Downloads 'DBA' (Invesco DB Agriculture Fund).
2.  **Processing**: Calculates Realized Volatility (RV).
3.  **Model Training**:
    -   **RF**: Trains on Monthly Climate Data -> Monthly RV.
    -   **LSTM**: Trains on Weekly Climate Data -> Weekly RV.
4.  **Output**: Saves trained models to `bot/models/`.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Flatten, Multiply, Reshape

# CONFIG
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RF_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'process', 'panel_monthly.csv')
LSTM_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'process', 'panel_full_pca.csv')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'bot', 'models')
TICKER = 'DBA'

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def calculate_rv(series):
    return np.sqrt(np.sum(series**2))

def get_agriculture_data():
    print(f"\n{'='*60}")
    print(f"🌾 FETCHING AGRICULTURE DATA ({TICKER})")
    print(f"{'='*60}")
    
    df_agri = None
    try:
        # Attempt Download
        df_agri = yf.download(TICKER, start='2007-01-01', progress=False)
        if len(df_agri) == 0:
            raise ValueError("No data downloaded")
        print(f"   ✅ Downloaded {len(df_agri)} days for {TICKER}")
        
    except Exception as e:
        print(f"   ❌ Failed to download {TICKER}: {e}")
        print("   ⚠️ Creating DUMMY data for demonstration")
        dates = pd.date_range(start='2000-01-01', end='2023-12-31', freq='D')
        df_agri = pd.DataFrame(index=dates)
        df_agri['Close'] = 25 + np.random.randn(len(dates)).cumsum()
        df_agri['Adj Close'] = df_agri['Close']

    # --- PROCESS DATA (Common Path) ---
    # Extract Price Series safely
    price_series = None
    
    # Check if MultiIndex (yfinance often returns this now)
    if isinstance(df_agri.columns, pd.MultiIndex):
        try:
            # Try 'Adj Close' first
            price_series = df_agri['Adj Close']
            if isinstance(price_series, pd.DataFrame):
                price_series = price_series.iloc[:, 0]
        except KeyError:
            try:
                price_series = df_agri['Close']
                if isinstance(price_series, pd.DataFrame):
                    price_series = price_series.iloc[:, 0]
            except KeyError:
                price_series = df_agri.iloc[:, 0] # Fallback
    else:
        # Simple Index
        if 'Adj Close' in df_agri.columns:
            price_series = df_agri['Adj Close']
        elif 'Close' in df_agri.columns:
            price_series = df_agri['Close']
        else:
            price_series = df_agri.iloc[:, 0]
            
    print(f"   Extracted price series name: {price_series.name}")
    
    # Create clean DF
    df_clean = pd.DataFrame(price_series)
    df_clean.columns = ['Price']
    df_clean['Log_Ret'] = np.log(df_clean['Price'] / df_clean['Price'].shift(1))
    
    return df_clean

def train_rf_agriculture(df_agri_clean):
    print("\n🌲 Training RF for Agriculture...")
    
    # 1. Calculate Monthly RV
    rv_monthly = df_agri_clean['Log_Ret'].resample('ME').apply(calculate_rv)
    rv_monthly.name = 'RV_Agriculture'
    
    # 2. Load RF Panel
    df_rf = pd.read_csv(RF_DATA_PATH)
    df_rf['Date'] = pd.to_datetime(df_rf['Date'])
    
    # 3. Pivot RF Panel (to match training format)
    exclude_cols = ['Date', 'ISO', 'year', 'month']
    value_vars = [c for c in df_rf.columns if c not in exclude_cols]
    df_rf_pivot = df_rf.pivot(index='Date', columns='ISO', values=value_vars)
    df_rf_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_rf_pivot.columns]
    df_rf_pivot = df_rf_pivot.fillna(0)
    
    # 4. Merge Target
    print(f"   DEBUG: RF Pivot Index: {df_rf_pivot.index[0]} to {df_rf_pivot.index[-1]}")
    print(f"   DEBUG: RV Monthly Index: {rv_monthly.index[0]} to {rv_monthly.index[-1]}")
    
    # Ensure indices are timezone-naive
    if df_rf_pivot.index.tz is not None:
        df_rf_pivot.index = df_rf_pivot.index.tz_localize(None)
    if rv_monthly.index.tz is not None:
        rv_monthly.index = rv_monthly.index.tz_localize(None)
        
    # Align to Month Start (MS) if needed, as resample('ME') gives Month End
    # RF Panel usually uses Month Start (01). Let's check.
    # If RF is MS and RV is ME, they won't match.
    # Let's force RV to Month Start to match RF.
    rv_monthly.index = rv_monthly.index + pd.offsets.MonthBegin(-1) + pd.offsets.MonthBegin(1)
    # Actually, simpler: set to 1st of the month
    rv_monthly.index = rv_monthly.index.map(lambda x: x.replace(day=1))
    
    print(f"   DEBUG: RV Monthly Index (Adjusted): {rv_monthly.index[0]} to {rv_monthly.index[-1]}")

    df_final = df_rf_pivot.join(rv_monthly, how='inner')
    print(f"   ✅ Merged Agriculture Target. Shape: {df_final.shape}")
    
    # Features & Target
    # Use ALL features (ERA5 + EM-DAT damage variables)
    drop_cols = [c for c in df_final.columns if 'RV_' in c]
    X = df_final.drop(columns=drop_cols)
    y = df_final['RV_Agriculture']
    
    print(f"   Training RF on {X.shape[1]} features (ALL ERA5 + EM-DAT)...")
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    joblib.dump(rf, f'{MODELS_DIR}/rf_agriculture.pkl')
    print(f"   ✅ Saved RF model to {MODELS_DIR}/rf_agriculture.pkl")

def train_lstm_agriculture(df_agri_clean):
    print("\n🧠 Training LSTM for Agriculture (Lightweight)...")
    
    # 1. Calculate Weekly RV Target
    rv_weekly = df_agri_clean['Log_Ret'].resample('W-FRI').apply(calculate_rv)
    rv_weekly.name = 'RV_Agriculture'
    
    # 2. Load Daily Panel (Lightweight Features)
    # 2. Load Daily Panel (Lightweight Features)
    DAILY_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'process', 'panel_daily.csv')
    df_daily = pd.read_csv(DAILY_DATA_PATH)
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    
    # Key ISOs (User requested ALL locations to remove bias)
    # KEY_ISOS = ['MHL', 'GRL', 'KEN', 'MEX', 'USA', 'KAZ', 'BRA', 'CHN', 'DEU', 'MWI']
    # df_subset = df_daily[df_daily['ISO'].isin(KEY_ISOS)].copy()
    
    # Use ALL locations
    df_subset = df_daily.copy()
    print(f"   Using ALL {df_subset['ISO'].nunique()} locations.")
    
    # Get ALL disaster-specific variables (dynamically)
    # Includes: count_*, deaths_*, affected_*, damage_* for each disaster type
    disaster_cols = [c for c in df_subset.columns if any(x in c for x in ['count_', 'deaths_', 'affected_', 'damage_', 'disaster_count'])]
    all_vars = ['t2m', 'tp'] + disaster_cols
    
    print(f"   Climate vars: 2 (t2m, tp)")
    print(f"   Disaster vars: {len(disaster_cols)} ({', '.join(disaster_cols[:5])}...)")
    
    df_pivot = df_subset.pivot(index='Date', columns='ISO', values=all_vars)
    df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
    df_pivot = df_pivot.fillna(0)
    
    print(f"   Total features: {len(all_vars)} vars × {df_subset['ISO'].nunique()} locations = {df_pivot.shape[1]}")
    
    # Resample Features to Weekly
    # We take the mean of the daily values for the week
    df_weekly_features = df_pivot.resample('W-FRI').mean()
    
    # 3. Merge Target
    df_lstm_final = df_weekly_features.join(rv_weekly, how='inner')
    df_lstm_final = df_lstm_final.fillna(method='ffill').fillna(0)
    print(f"   ✅ Merged Agriculture Target. Shape: {df_lstm_final.shape}")
    
    # 4. Prepare Sequences
    SEQ_LEN = 52
    HORIZON = 4
    
    def create_sequences(data, target, seq_length, horizon):
        xs, ys = [], []
        for i in range(len(data) - seq_length - horizon):
            x = data[i:(i + seq_length)]
            y = target[i + seq_length + horizon - 1]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    def attention_block(inputs, time_steps):
        a = Dense(1, activation='tanh')(inputs) 
        a = Flatten()(a) 
        a = Dense(time_steps, activation='softmax')(a) 
        a_probs = Reshape((time_steps, 1))(a) 
        output_attention_mul = Multiply()([inputs, a_probs]) 
        return output_attention_mul
    
    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_lstm_final)
    target_idx = df_lstm_final.columns.get_loc('RV_Agriculture')
    
    X, y = create_sequences(scaled_data, scaled_data[:, target_idx], SEQ_LEN, HORIZON)
    
    # Train
    inputs = Input(shape=(SEQ_LEN, X.shape[2]))
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_block(lstm_out, SEQ_LEN)
    attention_mul = Flatten()(attention_mul)
    x = Dense(64, activation='relu')(attention_mul)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='linear')(x)
    
    model = Model(inputs=[inputs], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, batch_size=32, verbose=0)
    
    model.save(f'{MODELS_DIR}/lstm_agriculture.h5')
    joblib.dump(scaler, f'{MODELS_DIR}/scaler_agriculture.pkl')
    print(f"   ✅ Saved LSTM model to {MODELS_DIR}/lstm_agriculture.h5")

def run_agriculture_engine():
    # 1. Get Data
    df_agri = get_agriculture_data()
    
    # 2. Train Models
    train_rf_agriculture(df_agri)
    train_lstm_agriculture(df_agri)
    
    print(f"\n🎉 Agriculture Engine Ready!")

if __name__ == "__main__":
    run_agriculture_engine()
