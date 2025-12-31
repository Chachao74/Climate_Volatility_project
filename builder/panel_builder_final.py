"""
PANEL BUILDER FINAL
===================
Merges:
1. EM-DAT (Disasters by type)
2. ERA5 Final (Climate - All Countries)
3. Financial Data (S&P, MSCI)
4. ENSO & CPU Indices
5. Adds Lags & Rolling Features

Output: FINAL_DELIVERY/Data_Pipeline/panel_final.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ======================================================
# CONFIG
# ======================================================
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

EMDAT_FILE = PROJECT_ROOT / "data" / "raw" / "emdat_data.csv"
ERA5_FILE = PROJECT_ROOT / "data" / "process" / "era5_monthly.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "process" / "panel_final.csv"

# Financials
MSCI_EU = PROJECT_ROOT / "data" / "raw" / "Index" / "msci_europe.csv"
MSCI_EM = PROJECT_ROOT / "data" / "raw" / "Index" / "msci_em.csv"

# Indices
ENSO_FILE = PROJECT_ROOT / "data" / "raw" / "nino34.long.anom.csv"
CPU_FILE = PROJECT_ROOT / "data" / "raw" / "cpu_index.csv"

def build_panel():
    print("="*80)
    print("🏗  PANEL BUILDER FINAL (ENHANCED)")
    print("="*80)
    
    # ======================================================
    # 1. LOAD ERA5
    # ======================================================
    print("🌍 Loading ERA5...")
    if not ERA5_FILE.exists():
        print("❌ ERA5 file missing. Run era5_builder_final.py first.")
        return
    
    df_era5 = pd.read_csv(ERA5_FILE)
    df_era5['Date'] = pd.to_datetime(df_era5['Date'])
    
    # Keep both 00:00 and 06:00, then merge to get complete data
    # (Some variables like tp are only at 06:00)
    df_00 = df_era5[df_era5['Date'].dt.hour == 0].copy()
    df_06 = df_era5[df_era5['Date'].dt.hour == 6].copy()
    
    # Normalize dates to remove hour component
    df_00['Date'] = df_00['Date'].dt.normalize()
    df_06['Date'] = df_06['Date'].dt.normalize()
    
    # Merge, preferring 00:00 values but filling with 06:00 where needed
    df_era5 = df_00.merge(df_06, on=['Date', 'ISO'], how='outer', suffixes=('', '_06'))
    
    # For each variable, use 00:00 value if available, otherwise use 06:00
    for col in ['t2m', 'tp', 'u10', 'v10', 'tcc', 'd2m', 'msl', 'wind_speed', 'humidity']:
        if f'{col}_06' in df_era5.columns:
            df_era5[col] = df_era5[col].fillna(df_era5[f'{col}_06'])
            df_era5 = df_era5.drop(columns=[f'{col}_06'])
    
    # DON'T filter yet - we need data from Jan 1998 to calculate lags for Dec 1998
    
    print(f"   Rows: {len(df_era5)}, Countries: {df_era5['ISO'].nunique()}")
    
    # ======================================================
    # 2. LOAD EM-DAT (WITH DISASTER TYPES)
    # ======================================================
    print("🚑 Loading EM-DAT...")
    df_emdat = pd.read_csv(EMDAT_FILE, low_memory=False)
    
    # Clean Date
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
    
    # Damage by type
    df_emdat["damage_flood"] = np.where(df_emdat["Disaster Type"].str.contains("flood"), df_emdat["damage"], 0)
    df_emdat["damage_storm"] = np.where(df_emdat["Disaster Type"].str.contains("storm"), df_emdat["damage"], 0)
    df_emdat["damage_drought"] = np.where(df_emdat["Disaster Type"].str.contains("drought"), df_emdat["damage"], 0)
    df_emdat["damage_wildfire"] = np.where(df_emdat["Disaster Type"].str.contains("wildfire|fire"), df_emdat["damage"], 0)
    df_emdat["damage_earthquake"] = np.where(df_emdat["Disaster Type"].str.contains("earthquake"), df_emdat["damage"], 0)
    df_emdat["damage_extreme_temp"] = np.where(df_emdat["Disaster Type"].str.contains("extreme temperature"), df_emdat["damage"], 0)
    
    # Aggregate Monthly
    emdat_agg = df_emdat.groupby(['ISO', 'Date']).agg(
        flood_count=("Disaster Type", lambda x: x.str.contains("flood").sum()),
        storm_count=("Disaster Type", lambda x: x.str.contains("storm").sum()),
        drought_count=("Disaster Type", lambda x: x.str.contains("drought").sum()),
        wildfire_count=("Disaster Type", lambda x: x.str.contains("wildfire|fire").sum()),
        earthquake_count=("Disaster Type", lambda x: x.str.contains("earthquake").sum()),
        extreme_temp_count=("Disaster Type", lambda x: x.str.contains("extreme temperature").sum()),
        disaster_count=("Disaster Type", "count"),
        deaths_total=("Total Deaths", "sum"),
        injured_total=("No. Injured", "sum"),
        homeless_total=("No. Homeless", "sum"),
        affected_total=("No. Affected", "sum"),
        damage_total=("damage", "sum"),
        damage_flood=("damage_flood", "sum"),
        damage_storm=("damage_storm", "sum"),
        damage_drought=("damage_drought", "sum"),
        damage_wildfire=("damage_wildfire", "sum"),
        damage_earthquake=("damage_earthquake", "sum"),
        damage_extreme_temp=("damage_extreme_temp", "sum"),
    ).reset_index()
    
    # Calculate damage_other
    damage_subtypes = ['damage_flood', 'damage_storm', 'damage_drought', 'damage_wildfire', 'damage_earthquake', 'damage_extreme_temp']
    emdat_agg['damage_other'] = emdat_agg['damage_total'] - emdat_agg[damage_subtypes].sum(axis=1)
    emdat_agg['damage_other'] = emdat_agg['damage_other'].clip(lower=0) # Ensure no negative values due to float precision
    
    print(f"   Rows: {len(emdat_agg)}")
    
    # ======================================================
    # 3. MERGE ERA5 + EM-DAT
    # ======================================================
    print("🔗 Merging ERA5 + EM-DAT...")
    panel = df_era5.merge(emdat_agg, on=['ISO', 'Date'], how='left')
    
    # Fill NaN Disasters with 0
    disaster_cols = [c for c in panel.columns if 'count' in c or 'damage' in c or 'deaths' in c or 'affected' in c or 'injured' in c or 'homeless' in c]
    panel[disaster_cols] = panel[disaster_cols].fillna(0)
    
    # ======================================================
    # 4. ADD FINANCIALS (S&P SECTORS, S&P500, VIX, AGRICULTURE)
    # ======================================================
    print("📈 Adding Financials (Sectors, S&P500, VIX, Agriculture)...")
    
    # Define all tickers
    sector_tickers = {
        "Consumer_Discretionary": "XLY", 
        "Consumer_Staples": "XLP", 
        "Energy": "XLE",
        "Financials": "XLF", 
        "Healthcare": "XLV", 
        "Industrials": "XLI",
        "Materials": "XLB", 
        "Technology": "XLK", 
        "Utilities": "XLU",
        "SP500": "^GSPC", 
        "VIX": "^VIX"
    }
    
    try:
        import yfinance as yf
        sector_panels = []
        
        for sector, ticker in sector_tickers.items():
            try:
                print(f"   Downloading {sector} ({ticker})...")
                data = yf.download(ticker, start="1998-01-01", end="2025-12-31", progress=False)
                if data.empty: 
                    print(f"   ⚠️ {sector} empty, skipped")
                    continue
                
                # Handle MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex): 
                    data.columns = data.columns.droplevel(1)
                
                # Remove timezone
                if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                
                # Calculate RV
                prices = data["Close"]
                
                # Handle scalar vs Series
                if isinstance(prices, (int, float, np.number)):
                    print(f"   ⚠️ {sector} returned scalar, skipped")
                    continue
                    
                prices = prices.squeeze()
                returns = np.log(prices / prices.shift(1))
                rv_monthly = (returns ** 2).resample("MS").sum()
                rv_monthly.index = rv_monthly.index.to_period("M").to_timestamp()
                
                sector_panels.append(pd.DataFrame({
                    "Date": rv_monthly.index, 
                    f"RV_{sector}": rv_monthly.values
                }))
                print(f"   ✓ {sector} added")
                
            except Exception as e:
                print(f"   ⚠️ {sector} failed: {e}")
        
        # Merge all sector panels
        if sector_panels:
            df_sectors = sector_panels[0]
            for df in sector_panels[1:]: 
                df_sectors = df_sectors.merge(df, on="Date", how="outer")
            
            # Merge with main panel
            panel = panel.merge(df_sectors, on='Date', how='left')
            
            # Forward fill financial data
            rv_cols = [c for c in panel.columns if c.startswith('RV_')]
            panel[rv_cols] = panel[rv_cols].ffill()
            
            print(f"   ✓ Added {len(sector_panels)} financial series")
        else:
            print("   ⚠️ No financial data downloaded")
            
    except ImportError:
        print("   ⚠️ yfinance not found, skipping sector download")
    
    # Add MSCI indices
    if MSCI_EU.exists():
        eu = pd.read_csv(MSCI_EU)
        eu['Date'] = pd.to_datetime(eu['Date'])
        eu['ret'] = np.log(eu['MSCI_EUROPE'] / eu['MSCI_EUROPE'].shift(1))
        eu_vol = eu.set_index('Date')['ret'].pow(2).resample('MS').sum().reset_index(name='RV_MSCI_EU')
        panel = panel.merge(eu_vol, on='Date', how='left')
        panel['RV_MSCI_EU'] = panel['RV_MSCI_EU'].ffill()
        
    if MSCI_EM.exists():
        em = pd.read_csv(MSCI_EM)
        em['Date'] = pd.to_datetime(em['Date'])
        em['ret'] = np.log(em['MSCI_EM'] / em['MSCI_EM'].shift(1))
        em_vol = em.set_index('Date')['ret'].pow(2).resample('MS').sum().reset_index(name='RV_MSCI_EM')
        panel = panel.merge(em_vol, on='Date', how='left')
        panel['RV_MSCI_EM'] = panel['RV_MSCI_EM'].ffill()
    
    # ======================================================
    # 5. ADD ENSO & CPU INDICES
    # ======================================================
    print("🌊 Adding ENSO & CPU Indices...")
    try:
        if ENSO_FILE.exists():
            enso = pd.read_csv(ENSO_FILE, skiprows=1, names=['Date', 'nino34']) # Renamed to nino34
            enso['Date'] = pd.to_datetime(enso['Date'])
            enso = enso[['Date', 'nino34']]
            panel = panel.merge(enso, on='Date', how='left')
            panel['nino34'] = panel['nino34'].ffill()
            print("   ✓ ENSO (nino34) added")
    except Exception as e:
        print(f"   ⚠️ ENSO skipped: {e}")
    
    try:
        if CPU_FILE.exists():
            cpu = pd.read_csv(CPU_FILE)
            # Check if Date column exists, otherwise skip
            if 'Date' in cpu.columns:
                cpu['Date'] = pd.to_datetime(cpu['Date'])
                cpu = cpu.rename(columns={'CPU': 'cpu'}) # Renamed to cpu
                cpu = cpu[['Date', 'cpu']]
                panel = panel.merge(cpu, on='Date', how='left')
                panel['cpu'] = panel['cpu'].ffill()
                print("   ✓ CPU added")
            else:
                print("   ⚠️ CPU file missing Date column, skipped")
    except Exception as e:
        print(f"   ⚠️ CPU skipped: {e}")
    
    # ======================================================
    # 6. ADD LAGS & ROLLING FEATURES
    # ======================================================
    print("🔄 Adding Lags & Rolling Features...")
    panel = panel.sort_values(['ISO', 'Date'])
    
    # 6.1 Climate Lags & Rolls
    climate_vars = ['t2m', 'tp', 'wind_speed', 'humidity', 'msl']
    # Renaming map to match monthly panel if needed, but monthly uses t2m, tp etc.
    # Monthly panel has temp_c, precip_m. Let's rename t2m -> temp_c, tp -> precip_m for consistency?
    # The user asked to add variables missing from monthly. Monthly has temp_c, precip_m.
    # Let's keep ERA5 names but maybe alias them or just ensure we have the derived ones.
    # Actually, looking at monthly panel, it has temp_c, precip_m.
    # I will rename them to match monthly panel for consistency.
    
    panel = panel.rename(columns={'t2m': 'temp_c', 'tp': 'precip_m'})
    climate_vars = ['temp_c', 'precip_m', 'wind_speed', 'humidity', 'msl']
    
    for var in climate_vars:
        if var in panel.columns:
            # Lag 1, 3, 6, 12 months
            panel[f'{var}_lag1m'] = panel.groupby('ISO')[var].shift(1)
            panel[f'{var}_lag3m'] = panel.groupby('ISO')[var].shift(3)
            panel[f'{var}_lag6m'] = panel.groupby('ISO')[var].shift(6)
            panel[f'{var}_lag12m'] = panel.groupby('ISO')[var].shift(12)
            
            # Rolling 3, 6, 12 months
            panel[f'{var}_roll3m'] = panel.groupby('ISO')[var].transform(lambda x: x.rolling(3, min_periods=1).mean())
            panel[f'{var}_roll6m'] = panel.groupby('ISO')[var].transform(lambda x: x.rolling(6, min_periods=1).mean())
            panel[f'{var}_roll12m'] = panel.groupby('ISO')[var].transform(lambda x: x.rolling(12, min_periods=1).mean())

    # 6.2 Climate Volatility & Spikes
    # Volatility = Rolling Std Dev (3m)
    if 'temp_c' in panel.columns:
        panel['temp_vol_3m'] = panel.groupby('ISO')['temp_c'].transform(lambda x: x.rolling(3).std())
    if 'precip_m' in panel.columns:
        panel['precip_vol_3m'] = panel.groupby('ISO')['precip_m'].transform(lambda x: x.rolling(3).std())
        
    # Spikes = (Val - Mean_12m) / Std_12m (Z-Score)
    for var, name in [('temp_c', 'temp_spike'), ('precip_m', 'precip_spike')]:
        if var in panel.columns:
            roll_mean = panel.groupby('ISO')[var].transform(lambda x: x.rolling(12, min_periods=1).mean())
            roll_std = panel.groupby('ISO')[var].transform(lambda x: x.rolling(12, min_periods=1).std())
            panel[name] = (panel[var] - roll_mean) / (roll_std + 1e-6) # Avoid div by zero

    # 6.3 Disaster Lags & Rolls
    # Variables to lag/roll: counts + damage + deaths + affected
    disaster_targets = [
        'flood_count', 'storm_count', 'drought_count', 'wildfire_count', 'earthquake_count', 'extreme_temp_count', 'disaster_count',
        'damage_total', 'deaths_total', 'affected_total'
    ]
    
    for var in disaster_targets:
        if var in panel.columns:
            # Lags: 1, 3, 6, 12
            panel[f'{var}_lag1m'] = panel.groupby('ISO')[var].shift(1)
            panel[f'{var}_lag3m'] = panel.groupby('ISO')[var].shift(3)
            panel[f'{var}_lag6m'] = panel.groupby('ISO')[var].shift(6)
            panel[f'{var}_lag12m'] = panel.groupby('ISO')[var].shift(12)
            
            # Rolling Sums: 3, 6, 12 (Disasters are events, so sum makes sense for "recent history")
            # Note: Monthly panel uses 'roll' suffix for these.
            panel[f'{var}_roll3m'] = panel.groupby('ISO')[var].transform(lambda x: x.rolling(3, min_periods=1).sum())
            panel[f'{var}_roll6m'] = panel.groupby('ISO')[var].transform(lambda x: x.rolling(6, min_periods=1).sum())
            panel[f'{var}_roll12m'] = panel.groupby('ISO')[var].transform(lambda x: x.rolling(12, min_periods=1).sum())

    # ======================================================
    # 7. FILTER TO DECEMBER 1998 (AFTER LAGS CALCULATED)
    # ======================================================
    print("📅 Filtering to December 1998...")
    panel = panel[panel['Date'] >= '1998-12-01'].copy()
    print(f"   Final shape after filtering: {panel.shape}")
    
    # ======================================================
    # 8. SAVE
    # ======================================================
    panel = panel.sort_values(['Date', 'ISO'])
    panel.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved to {OUTPUT_FILE}")
    print(f"   Shape: {panel.shape}")
    print(f"   Columns: {len(panel.columns)}")
    print(panel.head())

if __name__ == "__main__":
    build_panel()
