"""
AGRICULTURE - STRATÉGIE FINALE (ULTRA AGRESSIVE)
================================================
Génération de la figure finale pour le rapport
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os

# Charger prédictions
pred_path = 'results/LSTM AGRICULTURE/data/LSTM_predictions.csv'
predictions = pd.read_csv(pred_path)
predictions.columns = ['date', 'actual_rv', 'predicted_rv', 'regime']
predictions['date'] = pd.to_datetime(predictions['date'])
predictions = predictions.set_index('date')

# Charger prix DBA
dba_data = yf.download('DBA', start='2015-01-01', end='2024-12-31', progress=False) # Focus 2015-2024 pour le rapport
dba_prices = dba_data['Close'] if len(dba_data.columns) == 1 else dba_data['Close'].iloc[:, 0]
dba_weekly = dba_prices.resample('W-FRI').last()

# Merger
data = predictions.join(dba_weekly.rename('price'), how='inner')
data = data.dropna()
data['returns'] = data['price'].pct_change()
data = data.dropna()

# === STRATÉGIES ===

# 0. BUY & HOLD
bh_equity = np.cumprod(1 + data['returns'].values)

# 7. ULTRA AGGRESSIVE (200% / -100%)
ultra_list = [1.0]
for i in range(1, len(data)):
    ret = data['returns'].iloc[i]
    
    if i >= 2:
        prev_vol = data['predicted_rv'].iloc[i-1]
        curr_vol = data['predicted_rv'].iloc[i]
        median_vol = data['predicted_rv'].median()
        
        # Breakout: vol était haute, devient basse (Rebond)
        if prev_vol > median_vol and curr_vol < median_vol:
            bot_ret = ret * 2.0 - 0.0005  # Boost 200%
        
        # Breakdown: vol était basse, devient haute (Crash potentiel)
        elif prev_vol < median_vol and curr_vol > median_vol:
            # Short 100%
            bot_ret = (-1 * ret * 1.0) - 0.0005 
        
        else:
            bot_ret = ret - 0.0005
    else:
        bot_ret = ret - 0.0005
    
    ultra_list.append(ultra_list[-1] * (1 + bot_ret))

ultra_eq = np.array(ultra_list)

# === PLOT FINAL ===
fig, ax = plt.subplots(figsize=(10, 5)) # Format compact pour le rapport
dates = data.index

# Plot curves
ax.plot(dates, ultra_eq, label='Investment Strategy', color='#8E44AD', linewidth=2.5)
ax.plot(dates, bh_equity, label='Buy & Hold (Benchmark)', color='gray', linestyle='--', linewidth=2, alpha=0.8)

# Fill between
ax.fill_between(dates, ultra_eq, bh_equity, where=(ultra_eq >= bh_equity), color='#8E44AD', alpha=0.1)
ax.fill_between(dates, ultra_eq, bh_equity, where=(ultra_eq < bh_equity), color='red', alpha=0.1)

# Style
ax.set_title('Agriculture Strategy vs Benchmark (2015-2024)', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Return', fontsize=10)
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)
ax.axhline(1.0, color='black', linestyle=':', alpha=0.5)

# Add stats box
final_ret = (ultra_eq[-1] - 1) * 100
bh_ret = (bh_equity[-1] - 1) * 100
stats_text = f"Strategy: {final_ret:+.1f}%\nBenchmark: {bh_ret:+.1f}%"
ax.text(0.02, 0.05, stats_text, transform=ax.transAxes, fontsize=9, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), verticalalignment='bottom')

plt.tight_layout()
plt.savefig('VF/Report_Final/fig_agriculture_backtest_clean.png', dpi=300)
print(f"✅ Saved: VF/Report_Final/fig_agriculture_backtest_clean.png")
