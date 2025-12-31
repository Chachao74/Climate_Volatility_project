"""
ANALYSE: Quand Vol Breakout surperforme vs B&H?
================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

print("📊 Analyse Rolling Performance")

# Charger données
predictions = pd.read_csv('results/LSTM AGRICULTURE/data/LSTM_predictions.csv')
predictions.columns = ['date', 'actual_rv', 'predicted_rv', 'regime']
predictions['date'] = pd.to_datetime(predictions['date'])
predictions = predictions.set_index('date')

dba_data = yf.download('DBA', start='2010-01-01', end='2024-12-31', progress=False)
dba_prices = dba_data['Close'] if len(dba_data.columns) == 1 else dba_data['Close'].iloc[:, 0]
dba_weekly = dba_prices.resample('W-FRI').last()

data = predictions.join(dba_weekly.rename('price'), how='inner')
data = data.dropna()
data['returns'] = data['price'].pct_change()
data = data.dropna()

print(f"✅ {len(data)} semaines totales")

# Vol Breakout
median_vol = data['predicted_rv'].median()
bot_returns = []

for i in range(len(data)):
    ret = data['returns'].iloc[i]
    
    if i >= 2:
        prev_vol = data['predicted_rv'].iloc[i-1]
        curr_vol = data['predicted_rv'].iloc[i]
        
        if prev_vol > median_vol and curr_vol < median_vol:
            bot_ret = ret * 1.3 - 0.0005
        elif prev_vol < median_vol and curr_vol > median_vol:
            bot_ret = 0
        else:
            bot_ret = ret - 0.0005
    else:
        bot_ret = ret - 0.0005
    
    bot_returns.append(bot_ret)

bot_returns = np.array(bot_returns)
bh_returns = data['returns'].values

# Cumulative
bot_equity = np.cumprod(1 + bot_returns)
bh_equity = np.cumprod(1 + bh_returns)

# ANALYSE PAR PÉRIODE
periods = [
    ('2010-2015', '2010', '2015'),
    ('2015-2020', '2015', '2020'),
    ('2020-2024', '2020', '2024'),
    ('Full', '2010', '2024')
]

print("\n" + "="*70)
print("PERFORMANCE PAR PÉRIODE")
print("="*70)

results = []
for name, start, end in periods:
    mask = (data.index >= start) & (data.index <= end)
    if mask.sum() == 0:
        continue
    
    bot_period = np.cumprod(1 + bot_returns[mask])
    bh_period = np.cumprod(1 + bh_returns[mask])
    
    bot_ret = (bot_period[-1] - 1) * 100
    bh_ret = (bh_period[-1] - 1) * 100
    outperf = bot_ret - bh_ret
    
    results.append({
        'Period': name,
        'Bot': bot_ret,
        'B&H': bh_ret,
        'Outperf': outperf
    })
    
    symbol = "✅" if outperf > 0 else "❌"
    print(f"{name:12s}: Bot {bot_ret:+6.1f}% | B&H {bh_ret:+6.1f}% | Δ {outperf:+6.1f}% {symbol}")

# GRAPHIQUE
fig, axes = plt.subplots(3, 1, figsize=(18, 14), gridspec_kw={'height_ratios': [3, 1, 1]})

dates = data.index

# 1. Equity curves
ax1 = axes[0]
ax1.plot(dates, bot_equity, label='Vol Breakout', linewidth=4, color='#27AE60', zorder=3)
ax1.plot(dates, bh_equity, label='Buy & Hold', linewidth=3, linestyle='--', color='#E67E22', zorder=2)

# Highlight périodes
colors_periods = ['lightblue', 'lightgreen', 'lightyellow']
for (name, start, end), color in zip(periods[:-1], colors_periods):
    ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.1, color=color, zorder=1)
    mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
    ax1.text(mid, ax1.get_ylim()[1] * 0.95, name, ha='center', fontsize=10, fontweight='bold')

ax1.set_title('Vol Breakout vs Buy & Hold: Full Period Analysis', fontsize=16, fontweight='bold')
ax1.set_ylabel('Cumulative Return', fontsize=14)
ax1.legend(fontsize=13, loc='upper left')
ax1.grid(alpha=0.3)
ax1.axhline(1.0, color='black', linestyle=':', alpha=0.5)

# 2. Outperformance rolling
ax2 = axes[1]
window = 52  # 1 an
rolling_outperf = []
rolling_dates = []

for i in range(window, len(bot_equity)):
    bot_period_ret = (bot_equity[i] / bot_equity[i-window] - 1) * 100
    bh_period_ret = (bh_equity[i] / bh_equity[i-window] - 1) * 100
    rolling_outperf.append(bot_period_ret - bh_period_ret)
    rolling_dates.append(dates[i])

ax2.plot(rolling_dates, rolling_outperf, linewidth=2, color='purple')
ax2.fill_between(rolling_dates, 0, rolling_outperf, 
                  where=np.array(rolling_outperf) >= 0, color='green', alpha=0.3)
ax2.fill_between(rolling_dates, 0, rolling_outperf, 
                  where=np.array(rolling_outperf) < 0, color='red', alpha=0.3)
ax2.axhline(0, color='black', linestyle='-', linewidth=1)
ax2.set_ylabel('1Y Rolling\nOutperformance (%)', fontsize=12)
ax2.set_title('Rolling 1-Year Outperformance (Bot vs B&H)', fontsize=12)
ax2.grid(alpha=0.3)

# 3. Volatility
ax3 = axes[2]
ax3.plot(dates, data['predicted_rv'], linewidth=1.5, color='red', alpha=0.7)
ax3.fill_between(dates, 0, data['predicted_rv'], color='red', alpha=0.2)
ax3.axhline(median_vol, color='black', linestyle='--', linewidth=2, label='Median')
ax3.set_xlabel('Date', fontsize=14)
ax3.set_ylabel('Predicted Vol', fontsize=12)
ax3.set_title('LSTM Predicted Volatility', fontsize=12)
ax3.legend()
ax3.grid(alpha=0.3)

for ax in axes:
    ax.tick_params(axis='x', rotation=35)

plt.tight_layout()
plt.savefig('VF/agriculture_rolling_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: VF/agriculture_rolling_analysis.png")

print("\n" + "="*70)
print("💡 CONCLUSION")
print("="*70)
print("""
Vol Breakout NE surperforme PAS toujours!

✅ GAGNE sur périodes avec:
   - Hautes volatilités suivies de corrections
   - Bear markets évités (cash)
   
❌ PERD sur périodes avec:
   - Bull runs volatils (2020-2022)
   - Hausse continue malgré vol

La stratégie est DÉFENSIVE, pas offensive!
""")
