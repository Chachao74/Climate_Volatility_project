"""
AGRICULTURE BASELINE (NO CLIMATE)
=================================
Strict baseline benchmark using ONLY lagged volatility.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Flatten, Multiply, Reshape
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

TICKER = 'DBA'

def calculate_rv(series):
    return np.sqrt(np.sum(series**2))

def attention_block(inputs, time_steps):
    a = Dense(1, activation='tanh')(inputs) 
    a = Flatten()(a) 
    a = Dense(time_steps, activation='softmax')(a) 
    a_probs = Reshape((time_steps, 1))(a) 
    output_attention_mul = Multiply()([inputs, a_probs]) 
    return output_attention_mul

def run_baseline():
    print(f"\n{'='*60}")
    print(f"🌾 RUNNING AGRICULTURE BASELINE (NO CLIMATE)")
    print(f"{'='*60}")
    
    # 1. Get Target Data (DBA)
    # -----------------------
    print("   Fetching DBA data...")
    df_agri = yf.download(TICKER, start='2007-01-01', progress=False)
    
    # Extract Price Series safely
    price_series = None
    if isinstance(df_agri.columns, pd.MultiIndex):
        try:
            price_series = df_agri['Adj Close'].iloc[:, 0]
        except:
            price_series = df_agri['Close'].iloc[:, 0]
    else:
        price_series = df_agri['Adj Close'] if 'Adj Close' in df_agri.columns else df_agri['Close']
    
    df_clean = pd.DataFrame(price_series)
    df_clean.columns = ['Price']
    df_clean['Log_Ret'] = np.log(df_clean['Price'] / df_clean['Price'].shift(1))
    
    # Calculate Weekly RV Target
    rv_weekly = df_clean['Log_Ret'].resample('W-FRI').apply(calculate_rv)
    rv_weekly.name = 'RV_Agriculture'
    
    # 2. Prepare Features (ONLY RV)
    # ----------------------------
    # We use the target itself as the only feature (Autoregressive)
    df_features = pd.DataFrame(rv_weekly)
    
    # Fill NaNs
    df_features = df_features.fillna(method='ffill').fillna(0)
    
    print(f"   Features shape: {df_features.shape} (Only RV)")
    
    # 3. Prepare Sequences
    # --------------------
    SEQ_LEN = 52
    HORIZON = 4
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_features)
    
    def create_sequences(data, seq_length, horizon):
        xs, ys = [], []
        for i in range(len(data) - seq_length - horizon):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length + horizon - 1, 0] # Target is RV (column 0)
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    X, y = create_sequences(scaled_data, SEQ_LEN, HORIZON)
    
    # 4. Train/Test Split (80/20)
    # --------------------------
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples:  {len(X_test)}")
    
    # 5. Build Model (Same Architecture as Best Run)
    # ---------------------------------------------
    # LSTM 64 -> Attention -> Dense 64
    inputs = Input(shape=(SEQ_LEN, X.shape[2]))
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_block(lstm_out, SEQ_LEN)
    attention_mul = Flatten()(attention_mul)
    x = Dense(64, activation='relu')(attention_mul)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='linear')(x)
    
    model = Model(inputs=[inputs], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    print("   Training Baseline Model...")
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0, validation_data=(X_test, y_test))
    
    # 6. Evaluate
    # -----------
    y_pred = model.predict(X_test)
    
    # Inverse transform
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    
    # Directional Accuracy
    # We need previous values to calculate direction
    # y_test corresponds to t+horizon. We need t+horizon-1 (or t) to compare change?
    # Usually DA is: sign(pred_t - actual_t-1) == sign(actual_t - actual_t-1)
    # Here we have sequences.
    
    # Reconstruct the "previous" values for the test set
    # The target y_test[i] corresponds to data[split_idx + i + seq_len + horizon - 1]
    # The previous value is data[split_idx + i + seq_len + horizon - 2]
    
    # Let's get the raw data indices for test set targets
    test_indices = range(split_idx, len(X))
    prev_values = []
    actual_values = []
    pred_values = []
    
    for i, idx in enumerate(test_indices):
        # Index in original scaled_data
        target_idx = idx + SEQ_LEN + HORIZON - 1
        prev_idx = target_idx - 1 # Previous week
        
        prev_val = scaled_data[prev_idx, 0]
        curr_val = scaled_data[target_idx, 0]
        pred_val = y_pred[i, 0]
        
        prev_values.append(prev_val)
        actual_values.append(curr_val)
        pred_values.append(pred_val)
        
    prev_values = np.array(prev_values)
    actual_values = np.array(actual_values)
    pred_values = np.array(pred_values)
    
    actual_dir = np.sign(actual_values - prev_values)
    pred_dir = np.sign(pred_values - prev_values)
    
    da = np.mean(actual_dir == pred_dir) * 100
    
    print(f"\n{'='*60}")
    print(f"📊 BASELINE RESULTS (NO CLIMATE)")
    print(f"{'='*60}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   DA:   {da:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_baseline()
