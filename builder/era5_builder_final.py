"""
ERA5 BUILDER FINAL
==================
Downloads and processes ERA5 climate data for ALL countries.

Strategy:
1. Iterate by VARIABLE.
2. Download FULL PERIOD (1998-2025) for that variable (Global, Monthly, 00:00).
3. Extract spatial mean for all 203 countries.
4. Delete raw NetCDF to save space.
5. Merge all variables into final CSV.

Variables:
- t2m (Temp)
- tp (Precip)
- u10, v10 (Wind) -> wind_speed
- tcc (Cloud Cover)
- d2m (Dewpoint) -> humidity
- msl (Pressure)

Output: FINAL_DELIVERY/Data_Pipeline/era5_final_monthly.csv
"""

import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')

# ======================================================
# CONFIG
# ======================================================
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
RAW_DIR = DATA_DIR / "raw_era5"
OUTPUT_FILE = THIS_DIR / "era5_final_monthly.csv"
BBOX_FILE = THIS_DIR / "country_bboxes.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1998
END_YEAR = 2025 

ERA5_VARS = {
    '2m_temperature': 't2m',
    'total_precipitation': 'tp',
    '10m_u_component_of_wind': 'u10',
    '10m_v_component_of_wind': 'v10',
    'total_cloud_cover': 'tcc',
    '2m_dewpoint_temperature': 'd2m',
    'mean_sea_level_pressure': 'msl'
}

# ======================================================
# MAIN PROCESS
# ======================================================
def process_era5():
    print("="*80)
    print("🌍 ERA5 BUILDER (By Variable, Full History)")
    print("="*80)
    
    # Load BBoxes
    if not BBOX_FILE.exists():
        raise FileNotFoundError("Run generate_bboxes.py first!")
    bboxes = pd.read_csv(BBOX_FILE)
    print(f"Loaded {len(bboxes)} countries.")
    
    c = cdsapi.Client()
    
    # Dictionary to store results: {ISO: {Date: {Var: Val}}}
    # Or better: List of DataFrames, one per variable, then merge.
    variable_dfs = []
    
    for var_long, var_short in ERA5_VARS.items():
        print(f"\n🔹 Processing Variable: {var_long} ({var_short})")
        
        outfile = RAW_DIR / f"{var_short}_full.nc"
        
        # 1. DOWNLOAD
        if not outfile.exists():
            print(f"   ⬇️  Downloading 1998-{END_YEAR}...")
            try:
                c.retrieve(
                    'reanalysis-era5-single-levels-monthly-means',
                    {
                        'product_type': 'monthly_averaged_reanalysis',
                        'variable': var_long,
                        'year': [str(y) for y in range(START_YEAR, END_YEAR + 1)],
                        'month': [f"{m:02d}" for m in range(1, 13)],
                        'time': '00:00', # STRICTLY 00:00
                        'format': 'netcdf',
                        'area': [90, -180, -90, 180], # Global
                    },
                    str(outfile)
                )
            except Exception as e:
                print(f"   ❌ Download Error: {e}")
                continue
        else:
            print("   ✓ File already exists")
            
        # 2. EXTRACT
        print("   ⚙️  Extracting country data...")
        try:
            ds = xr.open_dataset(outfile)
            
            # Handle time dimension naming (ERA5 monthly uses 'valid_time')
            if 'valid_time' in ds.dims:
                ds = ds.rename({'valid_time': 'time'})
            
            # Rename to standard
            ds = ds.rename({list(ds.data_vars)[0]: var_short})
            
            # Pre-calc conversions if needed (to save memory/cpu later)
            if var_short == 't2m': ds['t2m'] = ds['t2m'] - 273.15
            if var_short == 'd2m': ds['d2m'] = ds['d2m'] - 273.15
            if var_short == 'msl': ds['msl'] = ds['msl'] / 100
            
            extracted_rows = []
            
            for _, row in bboxes.iterrows():
                iso = row['ISO']
                min_lon, max_lon = row['min_lon'], row['max_lon']
                min_lat, max_lat = row['min_lat'], row['max_lat']
                
                try:
                    # Crop
                    ds_crop = ds.sel(
                        longitude=slice(min_lon, max_lon),
                        latitude=slice(max_lat, min_lat)
                    )
                    
                    # Handle points
                    if ds_crop.sizes['longitude'] == 0 or ds_crop.sizes['latitude'] == 0:
                        ds_crop = ds.sel(
                            longitude=slice(min_lon, max_lon),
                            latitude=slice(max_lat, min_lat),
                            method='nearest'
                        )
                        
                    # Mean
                    mean_vals = ds_crop.mean(dim=['latitude', 'longitude'])
                    
                    # To DF
                    df = mean_vals.to_dataframe().reset_index()
                    df['ISO'] = iso
                    
                    # Keep only relevant columns (drop 'number', 'expver' if present)
                    cols_to_keep = ['time', 'ISO', var_short]
                    df = df[[c for c in cols_to_keep if c in df.columns]]
                    
                    extracted_rows.append(df)
                    
                except Exception as e:
                    pass
            
            # Combine all countries for this variable
            if extracted_rows:
                df_var = pd.concat(extracted_rows)
                df_var = df_var.rename(columns={'time': 'Date'})
                variable_dfs.append(df_var)
                print(f"   ✓ Extracted {len(extracted_rows)} countries")
                
            ds.close()
            
            # 3. DELETE RAW (DISABLED BY USER REQUEST)
            # print("   🧹 Deleting raw file...")
            # os.remove(outfile)
            
        except Exception as e:
            print(f"   ❌ Processing Error: {e}")

    # ======================================================
    # MERGE & DERIVED VARS
    # ======================================================
    print("\n💾 Merging Variables...")
    
    if not variable_dfs:
        print("❌ No data processed.")
        return

    # Start with the first variable
    final_df = variable_dfs[0]
    
    # Merge others
    for df in variable_dfs[1:]:
        final_df = final_df.merge(df, on=['ISO', 'Date'], how='outer')
        
    # Derived Variables
    print("🧮 Calculating Derived Variables...")
    
    # Wind Speed
    if 'u10' in final_df.columns and 'v10' in final_df.columns:
        final_df['wind_speed'] = np.sqrt(final_df['u10']**2 + final_df['v10']**2)
        
    # Humidity
    if 't2m' in final_df.columns and 'd2m' in final_df.columns:
        # T and Td are already in Celsius from extraction step
        T = final_df['t2m']
        Td = final_df['d2m']
        # Magnus formula requires T in Celsius
        final_df['humidity'] = 100 * np.exp((17.625 * Td) / (243.04 + Td)) / np.exp((17.625 * T) / (243.04 + T))

    # Sort and Save
    final_df = final_df.sort_values(['ISO', 'Date'])
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ Saved to {OUTPUT_FILE}")
    print(f"   Shape: {final_df.shape}")
    print(final_df.head())

if __name__ == "__main__":
    process_era5()
