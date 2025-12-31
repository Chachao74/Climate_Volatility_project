"""
BUILD FULL LSTM PANEL (ERA5 + EM-DAT + PCA)
===========================================
Combines Daily Weather (ERA5) and Disaster Data (EM-DAT) into a single PCA-compressed panel.

Process:
1. Load ERA5 Daily Weather (t2m, tp).
2. Load EM-DAT Disasters (Deaths, Affected, Damage).
   - Parse Dates (Handle missing days/months).
   - Pivot to Daily format (Country x Disaster_Variable).
3. Merge Weather + Disasters.
4. Apply PCA (Keep 95% variance).
5. Add Financial Targets.
6. Save to 'data/process/panel_full_pca.csv'.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import yfinance as yf
import joblib
import warnings

warnings.filterwarnings("ignore")

# ======================================================
# CONFIG
# ======================================================
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

ERA5_FILE = PROJECT_ROOT / "data" / "process" / "era5_daily.csv"
EMDAT_FILE = PROJECT_ROOT / "data" / "raw" / "emdat_data.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "process" / "panel_full_pca.csv"
PCA_MODEL_PATH = PROJECT_ROOT / "data" / "process" / "pca_full_model.pkl"
PCA_FEATURES_PATH = PROJECT_ROOT / "data" / "process" / "pca_full_features.pkl"

def build_full_panel():
    print("=" * 60)
    print("🌍 BUILDING FULL LSTM PANEL (WEATHER + DISASTERS)")
    print("=" * 60)

    # ==========================================
    # 1. PROCESS WEATHER DATA (ERA5)
    # ==========================================
    print("\n1️⃣ Processing Weather Data (ERA5)...")
    df_weather = pd.read_csv(ERA5_FILE)
    df_weather['Date'] = pd.to_datetime(df_weather['Date']).dt.normalize()
    df_weather = df_weather[df_weather['Date'] >= '1998-01-01']
    
    # Exclude USA from features (to avoid leakage if predicting USA, or keep global context)
    # User instruction: "global to USA". Usually implies using ROW to predict USA.
    # But let's keep USA in features for now as it contains local info, unless strictly requested otherwise.
    # Previous script excluded USA. Let's stick to that for consistency? 
    # Actually, user asked for "panel with era 5 daily and emdat". 
    # Let's exclude USA from *features* to be safe and consistent with previous logic.
    df_weather = df_weather[df_weather['ISO'] != 'USA']

    print("   Pivoting Weather...")
    weather_pivot = df_weather.pivot(index='Date', columns='ISO', values=['t2m', 'tp'])
    weather_pivot.columns = [f'{col[0]}_{col[1]}' for col in weather_pivot.columns]
    weather_pivot = weather_pivot.fillna(method='ffill').fillna(0)
    
    print(f"   Weather Shape: {weather_pivot.shape}")

    # ==========================================
    # 2. PROCESS DISASTER DATA (EM-DAT)
    # ==========================================
    print("\n2️⃣ Processing Disaster Data (EM-DAT)...")
    df_emdat = pd.read_csv(EMDAT_FILE)
    
    # Filter columns
    cols_to_keep = ['ISO', 'Start Year', 'Start Month', 'Start Day', 
                    'Total Deaths', 'Total Affected', "Total Damage ('000 US$)"]
    df_emdat = df_emdat[cols_to_keep]
    
    # Clean ISO
    df_emdat = df_emdat.dropna(subset=['ISO'])
    df_emdat = df_emdat[df_emdat['ISO'] != 'USA'] # Exclude USA features
    
    # Construct Date
    # Fill missing Day with 1, missing Month with 1
    df_emdat['Start Month'] = df_emdat['Start Month'].fillna(1).astype(int)
    df_emdat['Start Day'] = df_emdat['Start Day'].fillna(1).astype(int)
    
    # Create Date column
    def create_date(row):
        try:
            return pd.Timestamp(year=int(row['Start Year']), month=int(row['Start Month']), day=int(row['Start Day']))
        except:
            return pd.NaT
            
    df_emdat['Date'] = df_emdat.apply(create_date, axis=1)
    df_emdat = df_emdat.dropna(subset=['Date'])
    df_emdat = df_emdat[df_emdat['Date'] >= '1998-01-01']
    
    # Aggregate by Date and ISO (summing damages if multiple events)
    # Rename columns for clarity
    df_emdat = df_emdat.rename(columns={
        'Total Deaths': 'deaths',
        'Total Affected': 'affected',
        "Total Damage ('000 US$)": 'damage'
    })
    
    emdat_agg = df_emdat.groupby(['Date', 'ISO'])[['deaths', 'affected', 'damage']].sum().reset_index()
    
    print("   Pivoting Disasters...")
    emdat_pivot = emdat_agg.pivot(index='Date', columns='ISO', values=['deaths', 'affected', 'damage'])
    emdat_pivot.columns = [f'{col[0]}_{col[1]}' for col in emdat_pivot.columns]
    
    # Fill NaNs with 0 (No disaster = 0 damage)
    emdat_pivot = emdat_pivot.fillna(0)
    
    print(f"   Disaster Shape: {emdat_pivot.shape}")

    # ==========================================
    # 3. MERGE & ALIGN
    # ==========================================
    print("\n3️⃣ Merging Weather & Disasters...")
    # Use Weather index as master (daily continuity)
    df_features = weather_pivot.join(emdat_pivot, how='left')
    
    # Fill missing disaster days with 0
    df_features = df_features.fillna(0)
    
    print(f"   Combined Features: {df_features.shape[1]}")
    
    # ==========================================
    # 4. PCA COMPRESSION
    # ==========================================
    print("\n4️⃣ Applying PCA Compression...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    n_components = X_pca.shape[1]
    print(f"   ✅ Compressed to {n_components} components (95% variance)")
    
    # Save PCA Model
    joblib.dump(pca, PCA_MODEL_PATH)
    joblib.dump(list(df_features.columns), PCA_FEATURES_PATH)
    print(f"   💾 Saved PCA model to {PCA_MODEL_PATH}")
    
    # Create PCA DataFrame
    pca_cols = [f'PC_{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df_features.index)

    # ==========================================
    # 5. ADD TARGETS
    # ==========================================
    print("\n5️⃣ Adding Financial Targets...")
    sector_tickers = {
        "Consumer_Staples": "XLP", "Technology": "XLK", "Utilities": "XLU",
        "Materials": "XLB", "Energy": "XLE", "Financials": "XLF",
        "Healthcare": "XLV", "Industrials": "XLI", "Consumer_Discretionary": "XLY",
        "SP500": "^GSPC"
    }
    
    financial_data = pd.DataFrame()
    for sector, ticker in sector_tickers.items():
        try:
            data = yf.download(ticker, start="1998-01-01", end="2025-12-31", progress=False)
            if data.empty: continue
            
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)
            if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
                
            prices = data["Close"].squeeze()
            returns = np.log(prices / prices.shift(1))
            volatility = returns.abs()
            
            temp = pd.DataFrame({f'RV_{sector}': volatility})
            if financial_data.empty: financial_data = temp
            else: financial_data = financial_data.join(temp, how='outer')
        except Exception as e:
            print(f"   ⚠️ Failed {sector}: {e}")

    # Merge PCA + Financials
    df_final = df_pca.join(financial_data, how='left')
    
    # Forward fill financials (weekend gaps etc)
    fin_cols = [c for c in df_final.columns if 'RV_' in c]
    df_final[fin_cols] = df_final[fin_cols].ffill()
    df_final = df_final.dropna(subset=fin_cols, how='all')
    
    # ==========================================
    # 6. SAVE
    # ==========================================
    print("\n6️⃣ Saving Final Panel...")
    df_final.to_csv(OUTPUT_FILE)
    print(f"   ✅ Saved to: {OUTPUT_FILE}")
    print(f"   Final Shape: {df_final.shape}")

if __name__ == "__main__":
    build_full_panel()
