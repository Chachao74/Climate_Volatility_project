"""
AGRICULTURE LSTM - EXACT LSTM FORMAT
=====================================
Matches the exact plotting style from lstm_complete_all_in_one.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'results/Agriculture'

def calculate_rv(series):
    return np.sqrt(np.sum(series**2))

def get_agri_data():
    df = yf.download('DBA', start='2007-01-01', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    price = df['Adj Close'].squeeze() if 'Adj Close' in df.columns else df['Close'].squeeze()
    df_clean = pd.DataFrame({'Price': price})
    df_clean['Log_Ret'] = np.log(df_clean['Price'] / df_clean['Price'].shift(1))
    return df_clean

def detect_regime(volatility_series, window=6):
    rolling_std = volatility_series.rolling(window).std()
    median_vol = rolling_std.median()
    regime = pd.Series(index=volatility_series.index, dtype='object')
    regime[rolling_std <= median_vol] = 'low_vol'
    regime[rolling_std > median_vol] = 'high_vol'
    return regime

print("="*80)
print("🌾 AGRICULTURE LSTM - EXACT FORMAT")
print("="*80)

# Load model
model = load_model('bot/models/lstm_agriculture.h5', custom_objects={"attention_block": None}, compile=False)
scaler = joblib.load('bot/models/scaler_agriculture.pkl')

df_agri = get_agri_data()
rv_weekly = df_agri['Log_Ret'].resample('W-FRI').apply(calculate_rv)
rv_weekly.name = 'RV_Agriculture'

# Load disaster-specific panel
df_daily = pd.read_csv('data/process/panel_daily_disaster_specific.csv')
df_daily['Date'] = pd.to_datetime(df_daily['Date'])

disaster_cols = [c for c in df_daily.columns if any(x in c for x in ['count_', 'deaths_', 'affected_', 'damage_', 'disaster_count'])]
all_vars = ['t2m', 'tp'] + disaster_cols

df_subset = df_daily.copy()
df_pivot = df_subset.pivot(index='Date', columns='ISO', values=all_vars)
df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
df_pivot = df_pivot.fillna(0)

df_weekly = df_pivot.resample('W-FRI').mean()
df_final = df_weekly.join(rv_weekly, how='inner')
df_final = df_final.ffill().fillna(0)

# Sequences
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

scaled_data = scaler.fit_transform(df_final)
target_idx = df_final.columns.get_loc('RV_Agriculture')

X, y = create_sequences(scaled_data, scaled_data[:, target_idx], SEQ_LEN, HORIZON)

# Dates
dates = df_final.index[SEQ_LEN + HORIZON - 1:]
dates = dates[:len(y)]

# 80/20 Split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_train = dates[:split_idx]
dates_test = dates[split_idx:]

# Predict
y_pred_train_scaled = model.predict(X_train, verbose=0)
y_pred_test_scaled = model.predict(X_test, verbose=0)

# Inverse transform
def inverse(y_scaled):
    dummy = np.zeros((len(y_scaled), scaled_data.shape[1]))
    dummy[:, target_idx] = y_scaled.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]

y_train_actual = inverse(y_train)
y_pred_train_actual = inverse(y_pred_train_scaled)
y_test_actual = inverse(y_test)
y_pred_actual = inverse(y_pred_test_scaled)

# Metrics
mae_test = mean_absolute_error(y_test_actual, y_pred_actual)
rmse_test = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
y_diff = np.diff(y_test_actual)
pred_diff = np.diff(y_pred_actual)
da_test = np.mean(np.sign(y_diff) == np.sign(pred_diff))

print(f"Test MAE: {mae_test:.5f}")
print(f"Test DA: {da_test:.2%}")

# PLOT - EXACT LSTM FORMAT
y_test_series = pd.Series(y_test_actual, index=dates_test)
regime = detect_regime(y_test_series, window=6)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

# Top: Time Series (EXACT copy from lines 203-211)
ax1.plot(dates_train, y_train_actual, label='Actual Train', color='blue', alpha=0.3)
ax1.plot(dates_train, y_pred_train_actual, label='Predicted Train', color='orange', alpha=0.5)
ax1.plot(dates_test, y_test_actual, label='Actual Test', color='blue', alpha=0.8, linewidth=1.5)
ax1.plot(dates_test, y_pred_actual, label='Predicted Test', color='red', alpha=0.8, linestyle='--', linewidth=1.5)
ax1.axvline(x=dates_test[0], color='black', linestyle=':', label='Split')

ax1.set_title(f"LSTM Agriculture\nDA: {da_test:.2%} | MAE: {mae_test:.5f}")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom: Regime Classification (EXACT copy from lines 213-222)
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
plt.savefig(f'{OUTPUT_DIR}/plots/LSTM_full_train_test.png', dpi=300)
print(f"✅ Saved: LSTM_full_train_test.png")
plt.close()

# TEST ZOOM
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(dates_test, y_test_actual, label='Actual Test', color='blue', alpha=0.8, linewidth=1.5)
ax.plot(dates_test, y_pred_actual, label='Predicted Test', color='red', alpha=0.8, linestyle='--', linewidth=1.5)

ax.set_title(f'Agriculture LSTM: Test Set Detail (DA={da_test:.2%}, MAE={mae_test:.5f})', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=13)
ax.set_ylabel('Realized Volatility', fontsize=13)
ax.legend(fontsize=12, loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/LSTM_test_zoom.png', dpi=300)
print(f"✅ Saved: LSTM_test_zoom.png")
plt.close()

print(f"\n{'='*80}")
print(f"✅ Agriculture plots updated to EXACT LSTM format")
print(f"{'='*80}")
