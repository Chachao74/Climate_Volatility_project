 """
AGRICULTURE - STRATÉGIES AMÉLIORÉES (TEST ULTRA AGRESSIF)
=========================================================
Test de la stratégie Ultra Agressive (200% Long / 100% Short)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os

print("="*80)
print("200% long -100 % short- AGRICULTURE LSTM (TEST ")
print("="*80)

# Charger prédictions
pred_path = 'results/LSTM AGRICULTURE/data/LSTM_predictions.csv'
if not os.path.exists(pred_path):
    print(f"Fichier non trouvé: {pred_path}")
    exit()

predictions = pd.read_csv(pred_path)
predictions.columns = ['date', 'actual_rv', 'predicted_rv', 'regime']
predictions['date'] = pd.to_datetime(predictions['date'])
predictions = predictions.set_index('date')

# Charger prix DBA
dba_data = yf.download('DBA', start='2010-01-01', end='2024-12-31', progress=False)
dba_prices = dba_data['Close'] if len(dba_data.columns) == 1 else dba_data['Close'].iloc[:, 0]
dba_weekly = dba_prices.resample('W-FRI').last()

# Merger
data = predictions.join(dba_weekly.rename('price'), how='inner')
data = data.dropna()
data['returns'] = data['price'].pct_change()
data = data.dropna()

print(f"{len(data)} semaines de données")

# === STRATÉGIES ===

# 0. BUY & HOLD
bh_equity = np.cumprod(1 + data['returns'].values)

# 4. VOLATILITY BREAKOUT (ORIGINAL 130/0)
breakout_list = [1.0]
for i in range(1, len(data)):
    ret = data['returns'].iloc[i]
    if i >= 2:
        prev_vol = data['predicted_rv'].iloc[i-1]
        curr_vol = data['predicted_rv'].iloc[i]
        median_vol = data['predicted_rv'].median()
        if prev_vol > median_vol and curr_vol < median_vol:
            bot_ret = ret * 1.3 - 0.0005
        elif prev_vol < median_vol and curr_vol > median_vol:
            bot_ret = 0
        else:
            bot_ret = ret - 0.0005
    else:
        bot_ret = ret - 0.0005
    breakout_list.append(breakout_list[-1] * (1 + bot_ret))
breakout_eq = np.array(breakout_list)

# 6. AGGRESSIVE BREAKOUT (200/-100)
agg_list = [1.0]
for i in range(1, len(data)):
    ret = data['returns'].iloc[i]
    if i >= 2:
        prev_vol = data['predicted_rv'].iloc[i-1]
        curr_vol = data['predicted_rv'].iloc[i]
        median_vol = data['predicted_rv'].median()
        if prev_vol > median_vol and curr_vol < median_vol:
            bot_ret = ret * 2.0 - 0.0005
        elif prev_vol < median_vol and curr_vol > median_vol:
            bot_ret = (-1 * ret * 1.0) - 0.0005
        else:
            bot_ret = ret - 0.0005
    else:
        bot_ret = ret - 0.0005
    agg_list.append(agg_list[-1] * (1 + bot_ret))
agg_eq = np.array(agg_list)

# 7. ULTRA AGGRESSIVE (200/-100)
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

# === STATS ===
    'Buy & Hold': bh_equity,
    'Original (130/0)': breakout_eq,
    'Strategy (200/-100)': agg_eq

print("\n📊 RÉSULTATS COMPARATIFS:")
print("-" * 80)
for name, eq in strategies.items():
    final_ret = (eq[-1] - 1) * 100
    
    # Max DD
    dd = ((eq / np.maximum.accumulate(eq)) - 1).min() * 100
    
    # Sharpe (approx)
    daily_rets = pd.Series(eq).pct_change().dropna()
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(52) if daily_rets.std() > 0 else 0
    
    print(f"{name:30s}: Return {final_ret:+7.1f}% | MaxDD {dd:5.1f}% | Sharpe {sharpe:.2f}")

# === PLOT ===
print("\n📊 Création graphique...")
fig, ax = plt.subplots(figsize=(12, 6))
dates = data.index

colors = ['gray', '#F39C12', '#27AE60', '#8E44AD']
styles = ['--', '-', '-', '-']
widths = [2, 2, 2.5, 3]

for (name, eq), color, style, width in zip(strategies.items(), colors, styles, widths):
    ax.plot(dates, eq, label=name, color=color, linestyle=style, linewidth=width, alpha=0.9)

ax.set_title('Agriculture Strategy: Comparison of Leverage Levels', fontsize=14, fontweight='bold')
ax.set_ylabel('Cumulative Return', fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.axhline(1.0, color='black', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('VF/agriculture_ultra_test.png', dpi=300)
print(f"✅ Saved: VF/agriculture_ultra_test.png")
