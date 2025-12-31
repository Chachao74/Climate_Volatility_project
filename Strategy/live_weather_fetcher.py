"""
LIVE WEATHER FETCHER (GLOBAL)
=============================
Connects to Open-Meteo API to get real-time weather data for ALL 187 countries.
Uses batched requests to handle API limits.

Inputs:
- `data/process/country_coords.csv`: Centroids (ISO, lat, lon).
- `data/process/panel_daily.csv`: Source of target ISO list (cached to `bot/iso_list.json`).

Outputs:
- `bot/live_data.csv`: Daily T2M and TP for all locations (last 90 days).
"""

import requests
import pandas as pd
import numpy as np
import os
import json
import time

# CONFIG
OUTPUT_FILE = 'bot/live_data.csv'
COORDS_FILE = 'data/process/country_coords.csv'
PANEL_FILE = 'data/process/panel_daily.csv'
ISO_CACHE_FILE = 'bot/iso_list.json'
BATCH_SIZE = 50
ISO_CODES_FILE = 'data/process/iso_codes.csv'

def get_target_isos():
    # Check cache
    if os.path.exists(ISO_CACHE_FILE):
        with open(ISO_CACHE_FILE, 'r') as f:
            return json.load(f)
            
    print("   ⏳ Reading panel file to extract ISO list (First Run)...")
    if not os.path.exists(PANEL_FILE):
        raise FileNotFoundError(f"Panel file not found: {PANEL_FILE}")
        
    # Read only ISO column
    df = pd.read_csv(PANEL_FILE, usecols=['ISO'])
    isos = df['ISO'].unique().tolist()
    
    # Save cache
    with open(ISO_CACHE_FILE, 'w') as f:
        json.dump(isos, f)
        
    print(f"   ✅ Cached {len(isos)} ISOs to {ISO_CACHE_FILE}")
    return isos

def load_coordinates(target_isos):
    if not os.path.exists(COORDS_FILE):
        raise FileNotFoundError(f"Coords file not found: {COORDS_FILE}")
    if not os.path.exists(ISO_CODES_FILE):
        raise FileNotFoundError(f"ISO codes file not found: {ISO_CODES_FILE}")
        
    # 1. Load Coords (Alpha-2)
    df_coords = pd.read_csv(COORDS_FILE)
    # Ensure ISO column is named 'alpha-2' for merge
    if 'ISO' in df_coords.columns:
        df_coords.rename(columns={'ISO': 'alpha-2'}, inplace=True)
        
    # 2. Load ISO Codes (Alpha-2 -> Alpha-3)
    df_iso = pd.read_csv(ISO_CODES_FILE)
    # Columns: name,alpha-2,alpha-3,...
    
    # 3. Merge
    df_merged = pd.merge(df_coords, df_iso[['alpha-2', 'alpha-3']], on='alpha-2', how='inner')
    
    # 4. Filter for Target ISOs (Alpha-3)
    df_merged.rename(columns={'alpha-3': 'ISO'}, inplace=True)
    df_filtered = df_merged[df_merged['ISO'].isin(target_isos)].copy()
    
    # Check for missing
    found = set(df_filtered['ISO'])
    missing = set(target_isos) - found
    if missing:
        print(f"   ⚠️ Missing coordinates for {len(missing)} ISOs: {list(missing)[:5]}...")
        
    return df_filtered

def fetch_weather_batch(lats, lons, isos):
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
    
    params = {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons)),
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "UTC"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"   ❌ Batch Request failed: {e}")
        return None

def fetch_weather():
    print(f"\n{'='*60}")
    print("🌍 FETCHING LIVE WEATHER DATA (GLOBAL - 187 LOCATIONS)")
    print(f"{'='*60}")
    
    # 1. Get Target ISOs
    target_isos = get_target_isos()
    
    # 2. Get Coordinates
    df_coords = load_coordinates(target_isos)
    print(f"   Loaded coordinates for {len(df_coords)} locations.")
    
    # 3. Batch Fetch
    all_dfs = []
    
    # Convert to list of dicts for iteration
    locations = df_coords.to_dict('records')
    
    total_batches = (len(locations) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(locations), BATCH_SIZE):
        batch = locations[i:i+BATCH_SIZE]
        print(f"   Fetching batch {i//BATCH_SIZE + 1}/{total_batches} ({len(batch)} locs)...")
        
        lats = [loc['latitude'] for loc in batch]
        lons = [loc['longitude'] for loc in batch]
        batch_isos = [loc['ISO'] for loc in batch]
        
        data = fetch_weather_batch(lats, lons, batch_isos)
        
        if not data:
            continue
            
        # Process Batch
        if isinstance(data, dict): data = [data] # Handle single result case
        
        for j, res in enumerate(data):
            iso = batch_isos[j]
            daily = res.get('daily', {})
            
            dates = daily.get('time', [])
            t2m = daily.get('temperature_2m_mean', [])
            tp = daily.get('precipitation_sum', [])
            
            if not dates:
                print(f"     ⚠️ No data for {iso}")
                continue
                
            df_loc = pd.DataFrame({
                'Date': pd.to_datetime(dates),
                f't2m_{iso}': t2m,
                f'tp_{iso}': tp
            })
            df_loc.set_index('Date', inplace=True)
            all_dfs.append(df_loc)
            
        # Sleep to be nice to API
        time.sleep(1)
        
    # 4. Merge
    print("   Merging data...")
    if not all_dfs:
        print("❌ No data fetched.")
        return
        
    df_final = pd.concat(all_dfs, axis=1)
    df_final = df_final.fillna(method='ffill').fillna(0)
    
    # 5. Save
    df_final.to_csv(OUTPUT_FILE)
    print(f"   ✅ Saved live weather data to {OUTPUT_FILE}")
    print(f"   Shape: {df_final.shape}")
    
    return df_final

if __name__ == "__main__":
    fetch_weather()
