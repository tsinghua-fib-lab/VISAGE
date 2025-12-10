import json
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os

# ==========================================
#  CONFIGURATION & PATHS
# ==========================================

# List of city display names 
Cities = [
    'Boston', 'Chicago', 'Dallas', 'Detroit', 'Los Angeles', 'Miami', 'New York', 'Philadelphia',
    'San Francisco', 'Seattle', 'Washington', 'Albuquerque', 'Austin', 'Baltimore', 'Charlotte',
    'Columbus', 'Denver', 'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas',
    'Memphis', 'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland', 'San Antonio',
    'San Diego', 'San Jose', 'Tucson'
]

# Path to the US Census Tract Shapefile
SHAPEFILE_PATH = "/data3/maruolong/VISAGE/data/raw/census/cb_2019_us_tract_500k/cb_2019_us_tract_500k.shp"

# Base paths for data
SEGREGATION_BASE_PATH = "/data3/maruolong/VISAGE/data/raw/mobility/visit_data"
SEGMENTATION_BASE_PATH = "/data3/maruolong/VISAGE/data/processed/baseline/seg/segment_results"

# ==========================================
#  MAIN PROCESSING
# ==========================================

def plot_segregation_maps():
    print(">>> Loading US Tract Shapefile...")
    # Load Census Tract Shapefile
    tracts = gpd.read_file(SHAPEFILE_PATH)
    
    # Ensure GEOID is a string and padded to 11 digits
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
    
    # Set plot font globally (optional)
    plt.rcParams['font.family'] = 'Times New Roman'

    # Iterate through each city
    for city in Cities:
        print(f"Processing city: {city}...")

        # 1. Load Segregation Data
        # ---------------------------------------------------------
        segregation_file = f"{SEGREGATION_BASE_PATH}/{city}_2019/{city}_2019_tract_segregation.jsonl"
        segregation_data = {}
        
        if not os.path.exists(segregation_file):
            print(f"  [!] Warning: Segregation file not found for {city}. Skipping.")
            continue

        with open(segregation_file, "r") as f:
            for line in f:
                data = json.loads(line)
                segregation_data[data["tract_id"]] = data["segregation"]

        # 2. Load Segmentation/Street View Data (Used here to filter valid tracts)
        # ---------------------------------------------------------
        segmentation_file = f"{SEGMENTATION_BASE_PATH}/{city}_segmentation/tract_averages.jsonl"
        segmentation_list = []
        
        if not os.path.exists(segmentation_file):
            print(f"  [!] Warning: Segmentation file not found for {city}. Skipping.")
            continue

        with open(segmentation_file, "r") as f:
            for line in f:
                segmentation_list.append(json.loads(line))

        # 3. Merge Data
        # ---------------------------------------------------------
        df2 = pd.DataFrame(segmentation_list)
        # Map segregation scores to the DataFrame
        df2["segregation"] = df2["tract_id"].map(segregation_data)
        # Drop tracts that don't have segregation scores
        df2 = df2.dropna()

        # 4. Prepare GeoDataFrame
        # ---------------------------------------------------------
        # Filter the national shapefile to only include tracts present in our current city's data
        city_tracts = tracts[tracts["GEOID"].isin(df2["tract_id"])].copy()

        # Map the segregation score to the GeoDataFrame
        city_tracts["segregation"] = city_tracts["GEOID"].map(df2.set_index("tract_id")["segregation"])

        # 5. Plotting
        # ---------------------------------------------------------
        # Initialize figure
        fig, ax = plt.subplots(figsize=(12, 8), dpi=600)
        
        # Plot tracts colored by segregation score
        city_tracts.plot(
            column="segregation", 
            cmap="YlGnBu",          # Color map
            legend=True,            # Show legend bar
            edgecolor="black",      # Tract border color
            linewidth=0.1,          # Tract border width (thinner is often better for whole city maps)
            ax=ax
        )

        # Set title
        ax.set_title(f"Segregation Distribution in {city}", fontsize=16)
        
        # Remove axis ticks for a cleaner map look
        ax.set_axis_off()

        # Define output path
        output_dir = f"{SEGREGATION_BASE_PATH}/{city}_2019"
        output_path = f"{output_dir}/{city}_segregation_distribution_map.png"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save figure
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()

        print(f"  ✅ Map saved to: {output_path}")

    print("\n>>> All cities processed successfully!")

if __name__ == "__main__":
    plot_segregation_maps()