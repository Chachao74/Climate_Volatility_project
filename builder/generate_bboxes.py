"""
GENERATE COUNTRY BOUNDING BOXES
===============================
Uses geopandas and naturalearth_lowres to generate a CSV
containing the bounding box (min_lon, min_lat, max_lon, max_lat)
for ALL countries available.

Output: FINAL_DELIVERY/Data_Pipeline/country_bboxes.csv
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path

# Config
THIS_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = THIS_DIR / "country_bboxes.csv"

def generate_bboxes():
    print("🌍 Loading Natural Earth Data...")
    # Load world map
    # Load world map from URL (since local dataset is deprecated)
    url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    
    print(f"Columns: {world.columns}")
    # Filter out Antarctica
    if 'NAME' in world.columns:
        world = world[world.NAME != "Antarctica"]
        name_col = 'NAME'
    elif 'ADMIN' in world.columns:
        world = world[world.ADMIN != "Antarctica"]
        name_col = 'ADMIN'
    else:
        name_col = world.columns[0] # Fallback
    
    iso_col = 'ISO_A3' if 'ISO_A3' in world.columns else 'ADM0_A3'
    
    print(f"   → Found {len(world)} countries.")
    
    bboxes = []
    
    for _, row in world.iterrows():
        iso = row[iso_col]
        name = row[name_col]
        if iso == "-99": continue # Skip invalid ISOs
        
        bounds = row.geometry.bounds # (minx, miny, maxx, maxy)
        
        bboxes.append({
            "ISO": iso,
            "Name": name,
            "min_lon": bounds[0],
            "min_lat": bounds[1],
            "max_lon": bounds[2],
            "max_lat": bounds[3]
        })
        
    # Add Missing Countries (Small Islands / Territories often missing in low-res)
    missing_countries = {
        "AIA": {"Name": "Anguilla", "bbox": [-63.16, 18.16, -62.96, 18.28]},
        "ATG": {"Name": "Antigua and Barbuda", "bbox": [-61.9, 16.9, -61.6, 17.7]},
        "ABW": {"Name": "Aruba", "bbox": [-70.08, 12.4, -69.86, 12.63]},
        "BHR": {"Name": "Bahrain", "bbox": [50.3, 25.5, 50.8, 26.3]},
        "BRB": {"Name": "Barbados", "bbox": [-59.7, 13.0, -59.4, 13.4]},
        "BMU": {"Name": "Bermuda", "bbox": [-64.9, 32.2, -64.6, 32.4]},
        "CYM": {"Name": "Cayman Islands", "bbox": [-81.45, 19.25, -79.7, 19.75]},
        "COM": {"Name": "Comoros", "bbox": [43.0, -12.5, 44.6, -11.3]},
        "CPV": {"Name": "Cabo Verde", "bbox": [-25.4, 14.8, -22.6, 17.2]},
        "DMA": {"Name": "Dominica", "bbox": [-61.55, 15.1, -61.2, 15.7]},
        "FSM": {"Name": "Micronesia", "bbox": [137.0, 3.0, 164.0, 10.0]},
        "GRD": {"Name": "Grenada", "bbox": [-61.8, 11.9, -61.5, 12.3]},
        "HKG": {"Name": "Hong Kong", "bbox": [113.8, 22.1, 114.4, 22.6]},
        "KIR": {"Name": "Kiribati", "bbox": [-176.0, -12.0, -150.0, 5.0]}, # Spread out
        "KNA": {"Name": "Saint Kitts and Nevis", "bbox": [-62.9, 17.0, -62.5, 17.5]},
        "LCA": {"Name": "Saint Lucia", "bbox": [-61.1, 13.7, -60.8, 14.1]},
        "MAC": {"Name": "Macao", "bbox": [113.5, 22.1, 113.6, 22.2]},
        "MDV": {"Name": "Maldives", "bbox": [72.5, -1.0, 74.0, 8.0]},
        "MLT": {"Name": "Malta", "bbox": [14.1, 35.7, 14.6, 36.1]},
        "MUS": {"Name": "Mauritius", "bbox": [57.3, -20.6, 57.8, -19.9]},
        "MHL": {"Name": "Marshall Islands", "bbox": [160.0, 4.0, 172.0, 15.0]},
        "PLW": {"Name": "Palau", "bbox": [134.0, 6.5, 135.0, 8.5]},
        "PYF": {"Name": "French Polynesia", "bbox": [-155.0, -28.0, -134.0, -8.0]},
        "SGP": {"Name": "Singapore", "bbox": [103.6, 1.2, 104.1, 1.5]},
        "STP": {"Name": "Sao Tome and Principe", "bbox": [6.4, 0.0, 7.5, 1.7]},
        "SYC": {"Name": "Seychelles", "bbox": [55.0, -10.0, 56.0, -4.0]},
        "TON": {"Name": "Tonga", "bbox": [-176.0, -22.0, -173.0, -15.0]},
        "TUV": {"Name": "Tuvalu", "bbox": [176.0, -11.0, 180.0, -5.0]},
        "VCT": {"Name": "Saint Vincent and the Grenadines", "bbox": [-61.3, 12.5, -61.0, 13.5]},
        "WSM": {"Name": "Samoa", "bbox": [-173.0, -14.5, -171.0, -13.0]},
        # Add FRA/NOR explicitly if missing due to overseas territories confusion
        "FRA": {"Name": "France", "bbox": [-5.5, 41.0, 10.0, 51.5]},
        "NOR": {"Name": "Norway", "bbox": [4.0, 57.0, 32.0, 72.0]},
    }
    
    existing_isos = set(b["ISO"] for b in bboxes)
    
    for iso, data in missing_countries.items():
        if iso not in existing_isos:
            bboxes.append({
                "ISO": iso,
                "Name": data["Name"],
                "min_lon": data["bbox"][0],
                "min_lat": data["bbox"][1],
                "max_lon": data["bbox"][2],
                "max_lat": data["bbox"][3]
            })
            print(f"   + Added {iso} ({data['Name']}) manually.")
            
    df = pd.DataFrame(bboxes)
    df = df.sort_values("ISO").drop_duplicates(subset="ISO", keep="first")
    
    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(df)} countries to {OUTPUT_FILE}")
    print(df.head())

if __name__ == "__main__":
    generate_bboxes()
