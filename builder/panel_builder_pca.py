"""
BUILD LSTM DATA WITH PCA (ADAPTED)
==================================
Adapted from test/lstm_builder.py to work with current project structure.

Process:
1. Load ERA5 daily weather data (data/process/era5_daily.csv)
2. Pivot to wide format (Date x [Country_Variable])
3. Apply PCA to compress features (Keep 95% variance)
4. Add Financial Targets (S&P Sectors)
5. Save compressed dataset

Output: data/process/panel_daily_pca.csv
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ======================================================
# CONFIG
# ======================================================
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

ERA5_DAILY_FILE = PROJECT_ROOT / "data" / "process" / "era5_daily.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "process" / "panel_daily_pca.csv"

def build_lstm_pca():
    print("=" * 60)
    print("🧠 BUILDING LSTM DATA WITH PCA COMPRESSION (ADAPTED)")
    print("=" * 60)
    
    # ================================
    # 1. LOAD RAW DATA
    # ================================
    print("\n1️⃣ Loading Daily Weather Data...")
    if not ERA5_DAILY_FILE.exists():
        print(f"❌ File not found: {ERA5_DAILY_FILE}")
        return
        
    df_daily = pd.read_csv(ERA5_DAILY_FILE)
    df_daily['Date'] = pd.to_datetime(df_daily['Date']).dt.normalize()
    
    # Filter to start from 1998
    df_daily = df_daily[df_daily['Date'] >= '1998-01-01']
    
    print(f"   Loaded: {len(df_daily)} rows, {df_daily['ISO'].nunique()} countries")
    print(f"   Date range: {df_daily['Date'].min()} to {df_daily['Date'].max()}")
    
    # ================================
    # 2. PIVOT TO WIDE FORMAT
    # ================================
    print("\n2️⃣ Pivoting to Wide Format (Date x Features)...")
    
    # We want to pivot all variables (t2m, tp) for all countries
    # The user's script separated USA, but let's keep it simple and include all.
    # If we want to exclude USA from features (to avoid leakage if predicting USA), we can.
    # The user's script: df_global = df_daily[df_daily['ISO'] != 'USA']
    # Let's follow that logic - use global climate to predict USA.
    
    df_global = df_daily[df_daily['ISO'] != 'USA'].copy()
    
    # Pivot
    print("   ⏳ Pivoting (this takes time)...")
    # Pivot table with multiple values
    daily_pivot = df_global.pivot(index='Date', columns='ISO', values=['t2m', 'tp'])
    
    # Flatten columns: Var_ISO
    daily_pivot.columns = [f'{col[0]}_{col[1]}' for col in daily_pivot.columns]
    daily_pivot = daily_pivot.reset_index()
    
    print(f"   Pivoted: {len(daily_pivot)} dates, {len(daily_pivot.columns)-1} features")
    
    # ================================
    # 3. APPLY PCA COMPRESSION
    # ================================
    print("\n3️⃣ Applying PCA Compression...")
    
    # Separate Date column
    dates = daily_pivot['Date']
    X = daily_pivot.drop('Date', axis=1)
    
    # Fill missing values (ffill then 0)
    X = X.fillna(method='ffill').fillna(0)
    
    print(f"   Original features: {X.shape[1]}")
    
    # Standardize (required for PCA)
    print("   Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA: Keep 95% variance
    print("   Running PCA (keep 95% variance)...")
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    n_components = X_pca.shape[1]
    variance_explained = pca.explained_variance_ratio_.sum()
    
    print(f"   ✅ Compressed to {n_components} components")
    print(f"   📊 Variance retained: {variance_explained:.2%}")
    
    # Save PCA Model & Feature Names for later analysis
    import joblib
    pca_model_path = PROJECT_ROOT / "data" / "process" / "pca_model.pkl"
    feature_names_path = PROJECT_ROOT / "data" / "process" / "pca_features.pkl"
    
    joblib.dump(pca, pca_model_path)
    joblib.dump(list(X.columns), feature_names_path)
    print(f"   💾 Saved PCA model to {pca_model_path}")
    
    # Create PCA DataFrame
    pca_cols = [f'PC_{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)
    df_pca['Date'] = dates.values
    
    # ================================
    # 4. ADD TARGETS (Financials)
    # ================================
    print("\n4️⃣ Adding Target Variables (RV)...")
    
    sector_tickers = {
        "Consumer_Staples": "XLP", 
        "Technology": "XLK", 
        "Utilities": "XLU",
        "Materials": "XLB",
        "Energy": "XLE",
        "Financials": "XLF",
        "Healthcare": "XLV",
        "Industrials": "XLI",
        "Consumer_Discretionary": "XLY",
        "SP500": "^GSPC"
    }
    
    financial_data = pd.DataFrame()
    
    for sector, ticker in sector_tickers.items():
        try:
            # print(f"   Downloading {sector} ({ticker})...")
            data = yf.download(ticker, start="1998-01-01", end="2025-12-31", progress=False)
            
            if data.empty: continue
                
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
                
            # Calculate Daily Volatility Proxy (Absolute Return)
            prices = data["Close"].squeeze()
            returns = np.log(prices / prices.shift(1))
            volatility = returns.abs()
            
            temp_df = pd.DataFrame({
                'Date': volatility.index,
                f'RV_{sector}': volatility.values
            })
            
            if financial_data.empty:
                financial_data = temp_df
            else:
                financial_data = financial_data.merge(temp_df, on='Date', how='outer')
        except Exception as e:
            print(f"   ⚠️ Failed {sector}: {e}")

    # Merge PCA + Financials
    if not financial_data.empty:
        df_lstm = pd.merge(df_pca, financial_data, on='Date', how='left')
        
        # Forward fill financials
        fin_cols = [c for c in df_lstm.columns if 'RV_' in c]
        df_lstm[fin_cols] = df_lstm[fin_cols].ffill()
        
        # Drop rows where financials are still NaN (start of series)
        df_lstm = df_lstm.dropna(subset=fin_cols, how='all')
        
        print(f"   Merged RV targets. Final shape: {df_lstm.shape}")
    else:
        df_lstm = df_pca
        print("   ⚠️ No financial data added.")

    # ================================
    # 5. SAVE
    # ================================
    print("\n5️⃣ Saving Compressed Dataset...")
    
    df_lstm.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ LSTM PCA DATA CREATED!")
    print(f"   Saved: {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    build_lstm_pca()
