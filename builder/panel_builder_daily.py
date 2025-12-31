"""
PANEL BUILDER DAILY
===================
Merges:
1. ERA5 Daily (Climate - All Countries)
2. EM-DAT (Disasters - Mapped to 1st of Month)
3. Financial Data (S&P Sectors - Daily)

Output: data/process/panel_daily.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import yfinance as yf

warnings.filterwarnings("ignore")

# ======================================================
# CONFIG
# ======================================================
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

ERA5_DAILY_FILE = PROJECT_ROOT / "data" / "process" / "era5_daily.csv"
EMDAT_FILE = PROJECT_ROOT / "data" / "raw" / "emdat_data.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "process" / "panel_daily.csv"

def build_daily_panel():
    print("="*80)
    print("🏗  PANEL BUILDER DAILY")
    print("="*80)
    
    # ======================================================
    # 1. LOAD ERA5 DAILY
    # ======================================================
    print("🌍 Loading ERA5 Daily...")
    if not ERA5_DAILY_FILE.exists():
        print(f"❌ ERA5 Daily file missing: {ERA5_DAILY_FILE}")
        return
    
    df_era5 = pd.read_csv(ERA5_DAILY_FILE)
    df_era5['Date'] = pd.to_datetime(df_era5['Date']).dt.normalize()
    
    # Filter to start from 1998
    df_era5 = df_era5[df_era5['Date'] >= '1998-01-01']
    
    print(f"   Rows: {len(df_era5)}, Countries: {df_era5['ISO'].nunique()}")
    
    # ======================================================
    # 2. LOAD EM-DAT (Monthly Granularity -> Mapped to Daily)
    # ======================================================
    print("🚑 Loading EM-DAT...")
    df_emdat = pd.read_csv(EMDAT_FILE, low_memory=False)
    
    # Clean Date (Set to 1st of Month)
    df_emdat["Start Year"] = pd.to_numeric(df_emdat["Start Year"], errors="coerce")
    df_emdat["Start Month"] = pd.to_numeric(df_emdat["Start Month"], errors="coerce").fillna(1).clip(1, 12)
    df_emdat["Date"] = pd.to_datetime(dict(year=df_emdat["Start Year"], month=df_emdat["Start Month"], day=1))
    
    # Numeric columns
    numeric_cols = ["Total Deaths", "No. Affected", "Total Damage ('000 US$)", "No. Injured", "No. Homeless"]
    for col in numeric_cols:
        if col in df_emdat.columns:
            df_emdat[col] = pd.to_numeric(df_emdat[col], errors="coerce").fillna(0)
            
    df_emdat["damage"] = df_emdat["Total Damage ('000 US$)"].fillna(0)
    df_emdat["Disaster Type"] = df_emdat["Disaster Type"].astype(str).str.lower()
    
    # Map disaster types to categories
    disaster_map = {
        'flood': 'flood',
        'storm': 'storm', 
        'drought': 'drought',
        'wildfire': 'wildfire',
        'earthquake': 'earthquake',
        'extreme temperature': 'extreme_temp',
        'extreme temp': 'extreme_temp'
    }
    
    df_emdat['disaster_category'] = df_emdat['Disaster Type'].map(
        lambda x: next((v for k, v in disaster_map.items() if k in str(x).lower()), 'other')
    )
    
    # Aggregate by disaster type
    print("   Aggregating by disaster type...")
    emdat_by_type = df_emdat.groupby(['ISO', 'Date', 'disaster_category']).agg(
        count=("Disaster Type", "count"),
        deaths=("Total Deaths", "sum"),
        affected=("No. Affected", "sum"),
        damage=("damage", "sum")
    ).reset_index()
    
    # Pivot to wide format
    emdat_pivot = emdat_by_type.pivot_table(
        index=['ISO', 'Date'],
        columns='disaster_category',
        values=['count', 'deaths', 'affected', 'damage'],
        fill_value=0
    )
    
    # Flatten column names
    emdat_pivot.columns = [f"{val}_{cat}" for val, cat in emdat_pivot.columns]
    emdat_pivot = emdat_pivot.reset_index()
    
    # Add totals
    count_cols = [c for c in emdat_pivot.columns if c.startswith('count_')]
    deaths_cols = [c for c in emdat_pivot.columns if c.startswith('deaths_')]
    affected_cols = [c for c in emdat_pivot.columns if c.startswith('affected_')]
    damage_cols = [c for c in emdat_pivot.columns if c.startswith('damage_')]
    
    emdat_pivot['disaster_count'] = emdat_pivot[count_cols].sum(axis=1)
    emdat_pivot['deaths_total'] = emdat_pivot[deaths_cols].sum(axis=1)
    emdat_pivot['affected_total'] = emdat_pivot[affected_cols].sum(axis=1)
    emdat_pivot['damage_total'] = emdat_pivot[damage_cols].sum(axis=1)
    
    emdat_agg = emdat_pivot
    
    print(f"   Monthly Disaster Events: {len(emdat_agg)}")
    print(f"   Disaster columns: {len([c for c in emdat_agg.columns if c not in ['ISO', 'Date']])}")

    
    # Merge EM-DAT into ERA5
    # Since EM-DAT is monthly (1st of month), we merge on Date.
    # This means disasters will only appear on the 1st of the month in the daily panel.
    # If we want to forward fill or distribute, we can, but user said "put it in the first of the month".
    
    print("🔗 Merging ERA5 + EM-DAT...")
    panel = df_era5.merge(emdat_agg, on=['ISO', 'Date'], how='left')
    
    # Fill NaN Disasters with 0 (for days that are not the 1st of the month, or months with no disasters)
    disaster_cols = [c for c in panel.columns if any(x in c for x in ['count_', 'deaths_', 'affected_', 'damage_', 'disaster_count'])]
    panel[disaster_cols] = panel[disaster_cols].fillna(0)
    
    # ======================================================
    # 3. ADD FINANCIALS (Daily S&P Sectors)
    # ======================================================
    print("📈 Adding Daily Financials...")
    
    sector_tickers = {
        "Consumer_Staples": "XLP", 
        "Technology": "XLK", 
        "Utilities": "XLU",
        "SP500": "SPY",
        "Financials": "XLF"
    }
    
    # We will fetch data and merge it. 
    # Financial data is by Date, common to all ISOs.
    
    financial_data = pd.DataFrame()
    
    for sector, ticker in sector_tickers.items():
        print(f"   Downloading {sector} ({ticker})...")
        data = yf.download(ticker, start="1998-01-01", end="2025-12-31", progress=False)
        
        if data.empty:
            print(f"   ⚠️ {sector} empty")
            continue
            
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            
        if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
            
        # Calculate Daily Volatility Proxy (Absolute Return)
        prices = data["Close"].squeeze()
        returns = np.log(prices / prices.shift(1))
        volatility = returns.abs()
        
        # Create temp DF
        temp_df = pd.DataFrame({
            'Date': volatility.index,
            f'RV_Daily_{sector}': volatility.values
        })
        
        if financial_data.empty:
            financial_data = temp_df
        else:
            financial_data = financial_data.merge(temp_df, on='Date', how='outer')
            
    # Merge Financials into Panel
    # Financials are global (same for all countries), so we merge on Date
    print("🔗 Merging Financials...")
    panel = panel.merge(financial_data, on='Date', how='left')
    
    # Forward fill financials (for weekends/holidays if ERA5 has data but markets closed)
    # ERA5 is daily (every day). Financials are trading days.
    # We should probably ffill financials to cover weekends if we want a continuous daily series?
    # Or keep NaNs? LSTM handles time series.
    # Let's ffill to be safe for now.
    fin_cols = [c for c in panel.columns if 'RV_Daily_' in c]
    panel[fin_cols] = panel[fin_cols].ffill()
    
    # ======================================================
    # 4. SAVE
    # ======================================================
    print("💾 Saving Panel Daily...")
    panel = panel.sort_values(['Date', 'ISO'])
    panel.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved to {OUTPUT_FILE}")
    print(f"   Shape: {panel.shape}")
    print(panel.head())

if __name__ == "__main__":
    build_daily_panel()
