import os
import json
import shutil
import math
import geopandas as gpd
import pandas as pd
from heapq import nsmallest

# ==========================================
#  USER CONFIGURATION SECTION
# ==========================================
# Please update these paths to match your specific environment.

# --- INPUTS ---
# 1. Path to the US Census Tract Shapefile (.shp)
SHAPEFILE_PATH = "/data3/maruolong/VISAGE/data/raw/census/cb_2019_us_tract_500k/cb_2019_us_tract_500k.shp"

# 2. Path to the JSON file containing the list of target tract IDs
#    (The script will filter the shapefile to only include tracts found in this list)
TARGET_LIST_JSONL = "/data3/maruolong/VISAGE/data/raw/census/Urbanarea_tracts.json"

# 3. Directory containing the source satellite images (tiles)
#    (The script expects filenames in the format "y_x.jpg" or similar) (replace the path to your actual source)
SOURCE_IMAGE_DIR = "/data4/zhangxin/llm/data/esri/17"

# --- OUTPUTS ---
# 4. Path to save the intermediate mapping file (Tract ID -> Center Tile Coordinate)
MAPPING_OUTPUT_FILE = "/data3/maruolong/VISAGE/data/raw/census/seg_tract_tile_mapping.jsonl"

# 5. Directory where the selected images will be copied to
#    (Images will be organized into subfolders by tract_id: /.../merged/{tract_id}/...)
DESTINATION_IMAGE_DIR = "/data3/maruolong/VISAGE/data/raw/imagery/rs_image_new/merged"

# --- PARAMETERS ---
# Zoom level used to calculate the 'center' tile of a census tract
TRACT_ZOOM_LEVEL = 15

# Zoom level of the actual satellite images in SOURCE_IMAGE_DIR (used for distance calc)
IMAGE_ZOOM_LEVEL = 17

# ==========================================
#  HELPER FUNCTIONS: GEOSPATIAL & TILES
# ==========================================

def lat_lon_to_tile(lat, lon, zoom):
    """
    Convert latitude and longitude to tile coordinates (y, x) at a specific zoom level.
    Returns (y, x) integers.
    """
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )
    return y_tile, x_tile

def tile_to_lat_lon(y_tile, x_tile, zoom):
    """
    Convert tile coordinates (y, x) back to latitude and longitude.
    """
    n = 2.0 ** zoom
    lon = x_tile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_tile / n)))
    lat = math.degrees(lat_rad)
    return lat, lon

def get_tile_for_tract(tract_geometry, zoom_level):
    """
    Calculate the tile coordinate for the centroid of a tract geometry.
    Returns string format "y_x".
    """
    centroid = tract_geometry.centroid
    lat = centroid.y
    lon = centroid.x
    tile_y, tile_x = lat_lon_to_tile(lat, lon, zoom_level)
    return f"{tile_y}_{tile_x}"

def find_nearest_tiles(target_lat, target_lon, available_tiles, image_zoom, top_n=27):
    """
    Find the nearest 'top_n' tiles from the available set based on Euclidean distance 
    in lat/lon space.
    """
    distances = []
    
    for tile in available_tiles:
        try:
            y, x = map(int, tile.split("_"))
            # Convert the image tile coordinate back to lat/lon to measure distance
            tile_lat, tile_lon = tile_to_lat_lon(y, x, zoom=image_zoom)
            
            # Euclidean distance approximation
            distance = math.sqrt((tile_lat - target_lat) ** 2 + (tile_lon - target_lon) ** 2)
            distances.append((distance, tile))
        except ValueError:
            continue 

    # Return the closest 'top_n' tiles
    return [tile for _, tile in nsmallest(top_n, distances)]

# ==========================================
#  STEP 1: GENERATE TRACT MAPPING
# ==========================================

def generate_tract_mapping():
    print("\n>>> [Step 1/2] Loading Shapefile and Generating Mapping...")
    
    if not os.path.exists(SHAPEFILE_PATH):
        print(f"Error: Shapefile not found at {SHAPEFILE_PATH}")
        return None

    # 1. Load Shapefile
    gdf = gpd.read_file(SHAPEFILE_PATH)
    if "GEOID" not in gdf.columns:
        print("Error: 'GEOID' column not found in shapefile.")
        return None
    
    # Standardize GEOID format (11 digits)
    gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(11)
    print(f"    Loaded {len(gdf)} tracts from shapefile.")

    # 2. Calculate Center Tile for every tract
    print(f"    Calculating center tiles (Zoom: {TRACT_ZOOM_LEVEL})...")
    gdf["tile"] = gdf.apply(lambda row: get_tile_for_tract(row.geometry, TRACT_ZOOM_LEVEL), axis=1)

    # 3. Load Target Tract List
    if not os.path.exists(TARGET_LIST_JSONL):
        print(f"Error: Target list file not found at {TARGET_LIST_JSONL}")
        return None

    target_tract_ids = set()
    print("    Loading target tract IDs...")
    with open(TARGET_LIST_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "tract_id" in data:
                    t_id = str(data["tract_id"]).zfill(11)
                    target_tract_ids.add(t_id)
            except json.JSONDecodeError:
                continue
    
    print(f"    Loaded {len(target_tract_ids)} unique target IDs.")

    # 4. Filter and Export
    filtered_data = gdf[gdf["GEOID"].isin(target_tract_ids)]
    result_mapping = dict(zip(filtered_data["GEOID"], filtered_data["tile"]))
    
    # Check for missing targets
    missing_ids = target_tract_ids - set(result_mapping.keys())
    if missing_ids:
        print(f"    Warning: {len(missing_ids)} IDs from target list not found in shapefile.")

    # Save to intermediate file
    os.makedirs(os.path.dirname(MAPPING_OUTPUT_FILE), exist_ok=True)
    with open(MAPPING_OUTPUT_FILE, "w", encoding="utf-8") as f:
        for tract_id, tile in result_mapping.items():
            record = {"tract_id": tract_id, "tile": tile}
            json.dump(record, f)
            f.write("\n")

    print(f"    Mapping saved to: {MAPPING_OUTPUT_FILE}")
    return result_mapping

# ==========================================
#  STEP 2: EXTRACT AND COPY IMAGES
# ==========================================

def extract_images(tract_mapping):
    print("\n>>> [Step 2/2] Extracting and Copying Satellite Images...")

    if not tract_mapping:
        print("Error: No mapping data available. Skipping image extraction.")
        return

    # 1. Scan Source Directory
    print(f"    Scanning source directory: {SOURCE_IMAGE_DIR}")
    if not os.path.exists(SOURCE_IMAGE_DIR):
        print(f"Error: Source directory does not exist.")
        return

    available_tiles = set()
    for file in os.listdir(SOURCE_IMAGE_DIR):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            available_tiles.add(os.path.splitext(file)[0]) # Store without extension

    print(f"    Found {len(available_tiles)} available image tiles.")

    # 2. Process Each Tract
    os.makedirs(DESTINATION_IMAGE_DIR, exist_ok=True)
    not_found_tracts = []
    
    print(f"    Processing {len(tract_mapping)} tracts...")
    
    for tract_id, center_tile_str in tract_mapping.items():
        found_for_this_tract = False
        
        # Parse center tile (Zoom 15) and convert to real-world lat/lon
        try:
            ct_y, ct_x = map(int, center_tile_str.split("_"))
            center_lat, center_lon = tile_to_lat_lon(ct_y, ct_x, zoom=TRACT_ZOOM_LEVEL)
        except ValueError:
            print(f"    Skipping invalid tile format for tract {tract_id}: {center_tile_str}")
            continue

        # Find nearest tiles in the source folder (Zoom 17)
        # We look for 27 candidates to allow for our sparse sampling pattern
        nearest_tiles = find_nearest_tiles(
            center_lat, center_lon, 
            available_tiles, 
            image_zoom=IMAGE_ZOOM_LEVEL, 
            top_n=27
        )

        # Strategy: Pick specific indices to get a diverse spread of images
        selected_indices = [0, 7, 13, 15, 18, 20, 26]
        
        # Filter valid indices
        selected_tiles = [nearest_tiles[i] for i in selected_indices if i < len(nearest_tiles)]

        # If not enough tiles were found via indices, fill up with the closest remaining ones
        if len(selected_tiles) < len(selected_indices):
            selected_tiles.extend(nearest_tiles[len(selected_tiles):len(selected_indices)])

        # Cap at 8 images max (adjustable)
        selected_tiles = selected_tiles[:8]

        # Prepare destination folder
        tract_dest_dir = os.path.join(DESTINATION_IMAGE_DIR, str(tract_id))
        os.makedirs(tract_dest_dir, exist_ok=True)

        # Copy images
        for tile_name in selected_tiles:
            for ext in [".png", ".jpg", ".jpeg"]:
                src_path = os.path.join(SOURCE_IMAGE_DIR, tile_name + ext)
                dst_path = os.path.join(tract_dest_dir, tile_name + ext)
                
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    found_for_this_tract = True
                    break # Found the extension, move to next tile
        
        if not found_for_this_tract:
            not_found_tracts.append(tract_id)

    # 3. Summary
    print("\n>>> Processing Complete.")
    if not_found_tracts:
        print(f"    Warning: {len(not_found_tracts)} tracts had no matching images found.")
        # print(f"    Missing IDs: {not_found_tracts}")
    else:
        print("    Success: All tracts have associated images.")
    
    print(f"    Images output to: {DESTINATION_IMAGE_DIR}")

# ==========================================
#  MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Run Step 1: Mapping
    mapping_data = generate_tract_mapping()
    
    # Run Step 2: Extraction (only if mapping succeeded)
    if mapping_data:
        extract_images(mapping_data)