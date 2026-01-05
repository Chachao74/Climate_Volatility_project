"""
LSTM COMPLETE ALL-IN-ONE
========================
This script performs the entire LSTM pipeline in one go:
1. Data Loading (Full Panel: Weather + Disasters)
2. Training (Weekly + Attention) for ALL sectors
3. Evaluation (MAE, RMSE, DA, Hit Rate, Sharpe)
4. Regime Analysis (High vs Low Volatility)
5. Feature Importance (Permutation on PCA -> Mapped to Original)
6. Plotting (Time Series, Regimes, Importance)

Output: All results saved to 'results/LSTM_Final_Complete'
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Flatten, Multiply, Reshape
from tensorflow.keras.callbacks import EarlyStopping

# ======================================================
# CONFIGURATION
# ======================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'process', 'panel_full_pca.csv')
PCA_MODEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'process', 'pca_full_model.pkl')
PCA_FEATURES_PATH = os.path.join(PROJECT_ROOT, 'data', 'process', 'pca_full_features.pkl')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'LSTM_Final_Complete')

SEQUENCE_LENGTH = 52 # Weeks
FORECAST_HORIZON = 4 # Weeks

# Top sectors for detailed Feature Importance analysis
TOP_SECTORS_FOR_IMPORTANCE = ['RV_Utilities', 'RV_Healthcare', 'RV_Consumer_Discretionary']

# ======================================================
# UTILS
# ======================================================
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

def detect_regime(volatility_series, window=12):
    rolling_std = volatility_series.rolling(window).std()
    median_vol = rolling_std.median()
    regime = pd.Series(index=volatility_series.index, dtype='object')
    regime[rolling_std <= median_vol] = 'low_vol'
    regime[rolling_std > median_vol] = 'high_vol'
    return regime

def calculate_metrics(y_true, y_pred, y_lag):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    direction_actual = np.sign(y_true - y_lag)
    direction_pred = np.sign(y_pred - y_lag)
    da = np.mean(direction_actual == direction_pred)
    
    threshold = np.std(y_true)
    large_moves = np.abs(y_true - y_lag) > threshold
    if large_moves.sum() > 0:
        hit_rate = np.mean(direction_actual[large_moves] == direction_pred[large_moves])
    else:
        hit_rate = np.nan
        
    returns = y_true - y_lag
    pred_returns = y_pred - y_lag
    signal = np.sign(pred_returns)
    strategy_returns = signal * returns
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-10) * np.sqrt(12)
    
    return {'MAE': mae, 'RMSE': rmse, 'DA': da, 'HitRate': hit_rate, 'Sharpe': sharpe}

# ======================================================
# MAIN PIPELINE
# ======================================================
def run_pipeline():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created directory: {RESULTS_DIR}")

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Load PCA for importance mapping
    pca = joblib.load(PCA_MODEL_PATH)
    feature_names = joblib.load(PCA_FEATURES_PATH)

    # Preprocessing (Weekly)
    print("Resampling to Weekly (W-FRI)...")
    df_pivot = df.set_index('Date')
    weekly_rules = {}
    for col in df_pivot.columns:
        if col.startswith('RV_') or col.startswith('Level_'):
            weekly_rules[col] = 'mean'
        else:
            weekly_rules[col] = 'mean'
            
    df_model = df_pivot.resample('W-FRI').agg(weekly_rules)
    df_model = df_model.fillna(method='ffill').fillna(0)
    
    target_cols = [c for c in df_model.columns if c.startswith('RV_')]
    results_summary = []
    
    for TARGET_SECTOR in target_cols:
        print(f"\n{'='*60}")
        print(f"Processing {TARGET_SECTOR}...")
        print(f"{'='*60}")
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_model)
        target_idx = df_model.columns.get_loc(TARGET_SECTOR)
        
        X, y = create_sequences(scaled_data, scaled_data[:, target_idx], SEQUENCE_LENGTH, FORECAST_HORIZON)
        
        # Dates
        dates = df_model.index[SEQUENCE_LENGTH + FORECAST_HORIZON - 1:]
        dates = dates[:len(y)] # Ensure match
        
        # Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        dates_test = dates[split_idx:]
        dates_train = dates[:split_idx]
        
        # Train Model
        inputs = Input(shape=(SEQUENCE_LENGTH, X_train.shape[2]))
        lstm_out = LSTM(64, return_sequences=True)(inputs)
        lstm_out = Dropout(0.3)(lstm_out)
        attention_mul = attention_block(lstm_out, SEQUENCE_LENGTH)
        attention_mul = Flatten()(attention_mul)
        x = Dense(64, activation='relu')(attention_mul)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='linear')(x)
        
        model = Model(inputs=[inputs], outputs=output)
        model.compile(optimizer='adam', loss='mse')
        
        model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0, 
                  callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
        
        # Predict
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred_train_scaled = model.predict(X_train, verbose=0)
        
        # Inverse Transform
        def inverse(y_scaled):
            dummy = np.zeros((len(y_scaled), df_model.shape[1]))
            dummy[:, target_idx] = y_scaled.flatten()
            return scaler.inverse_transform(dummy)[:, target_idx]
            
        y_test_actual = inverse(y_test)
        y_pred_actual = inverse(y_pred_scaled)
        y_train_actual = inverse(y_train)
        y_pred_train_actual = inverse(y_pred_train_scaled)
        
        # Baseline (t) for DA
        prev_val_scaled = X_test[:, -1, target_idx]
        prev_val_actual = inverse(prev_val_scaled)
        
        # Metrics
        metrics = calculate_metrics(y_test_actual, y_pred_actual, prev_val_actual)
        
        print(f"MAE: {metrics['MAE']:.5f}")
        print(f"DA: {metrics['DA']:.2%}")
        print(f"Hit Rate: {metrics['HitRate']:.2%}")
        print(f"Sharpe: {metrics['Sharpe']:.2f}")
        
        results_summary.append({
            'Target': TARGET_SECTOR,
            **metrics
        })
        
        # -------------------------------------------------------
        # PLOTTING & REGIME ANALYSIS
        # -------------------------------------------------------
        y_test_series = pd.Series(y_test_actual, index=dates_test)
        regime = detect_regime(y_test_series, window=6)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(dates_train, y_train_actual, label='Actual Train', color='blue', alpha=0.3)
        ax1.plot(dates_train, y_pred_train_actual, label='Predicted Train', color='orange', alpha=0.5)
        ax1.plot(dates_test, y_test_actual, label='Actual Test', color='blue', alpha=0.8, linewidth=1.5)
        ax1.plot(dates_test, y_pred_actual, label='Predicted Test', color='red', alpha=0.8, linestyle='--', linewidth=1.5)
        ax1.axvline(x=dates_test[0], color='black', linestyle=':', label='Split')
        
        ax1.set_title(f"LSTM {TARGET_SECTOR}\nDA: {metrics['DA']:.2%} | HitRate: {metrics['HitRate']:.2%} | Sharpe: {metrics['Sharpe']:.2f}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        regime_colors = {'low_vol': 'green', 'high_vol': 'red'}
        for r_type in ['low_vol', 'high_vol']:
            mask = regime == r_type
            if mask.sum() > 0:
                ax2.scatter(dates_test[mask], [0]*mask.sum(), c=regime_colors[r_type], label=r_type, alpha=0.6, marker='|', s=50)
        
        ax2.set_yticks([])
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.set_title('Volatility Regimes (Test Set)')
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/{TARGET_SECTOR}_timeseries.png')
        plt.close()
        
        # -------------------------------------------------------
        # FEATURE IMPORTANCE (Only for Top Sectors)
        # -------------------------------------------------------
        if TARGET_SECTOR in TOP_SECTORS_FOR_IMPORTANCE:
            print(f"Calculating Feature Importance for {TARGET_SECTOR}...")
            
            # Permutation Importance on PCA Components
            pc_importance = np.zeros(X_test.shape[2])
            pc_cols = [c for c in df_model.columns if c.startswith('PC_')]
            pc_indices = [df_model.columns.get_loc(c) for c in pc_cols]
            
            base_mae = metrics['MAE']
            
            for i, col_idx in enumerate(pc_indices):
                original_col = X_test[:, :, col_idx].copy()
                np.random.shuffle(X_test[:, :, col_idx])
                
                y_pred_perm_scaled = model.predict(X_test, verbose=0)
                y_pred_perm = inverse(y_pred_perm_scaled)
                mae_perm = mean_absolute_error(y_test_actual, y_pred_perm)
                
                pc_importance[col_idx] = mae_perm - base_mae
                X_test[:, :, col_idx] = original_col
            
            # Map to Original Features
            pc_importance = np.maximum(pc_importance, 0)
            if pc_importance.sum() > 0:
                pc_importance = pc_importance / pc_importance.sum()
                
            loadings = np.abs(pca.components_)
            pc_imp_vector = pc_importance[pc_indices]
            
            if len(pc_imp_vector) != loadings.shape[0]:
                min_len = min(len(pc_imp_vector), loadings.shape[0])
                pc_imp_vector = pc_imp_vector[:min_len]
                loadings = loadings[:min_len, :]
                
            feature_importance = loadings.T @ pc_imp_vector
            
            df_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            df_imp.to_csv(f'{RESULTS_DIR}/{TARGET_SECTOR}_importance.csv', index=False)
            
            # Plot Top 20
            top_20 = df_imp.head(20)
            plt.figure(figsize=(10, 8))
            plt.barh(top_20['Feature'][::-1], top_20['Importance'][::-1])
            plt.title(f'Feature Importance: {TARGET_SECTOR}')
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/{TARGET_SECTOR}_importance.png')
            plt.close()
            print(f"Saved importance plot for {TARGET_SECTOR}")

    # Save Final Summary
    df_res = pd.DataFrame(results_summary).sort_values('DA', ascending=False)
    df_res.to_csv(f'{RESULTS_DIR}/summary_metrics.csv', index=False)
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY (All Sectors)")
    print(f"{'='*80}")
    print(df_res.to_string(index=False))
    print(f"\nAll results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    run_pipeline()
