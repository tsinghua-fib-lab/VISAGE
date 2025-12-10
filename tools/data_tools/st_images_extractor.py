import os
import shutil
import json
import time

# ==========================================
#  USER CONFIGURATION SECTION
# ==========================================

# 1. List of Cities to Process
CITIES = [
    'Boston', 'Chicago', 'Dallas', 'Detroit', 'Los Angeles', 'Miami', 'New York', 'Philadelphia',
    'San Francisco', 'Seattle', 'Washington', 'Albuquerque', 'Austin', 'Baltimore', 'Charlotte',
    'Columbus', 'Denver', 'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas',
    'Memphis', 'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland', 'San Antonio',
    'San Diego', 'San Jose', 'Tucson'
]

year = '2019'

# 2. Source Directories containing the Street View Images
#    IMPORTANT: Update these paths to point to your actual image datasets.
#    The script will search these directories for folders matching the tract IDs.
SOURCE_DIRS = [
    # --- Example Paths (Please replace/add your own) ---
    "/data2/ouyangtianjian/NEW_StreetView_Images_US_100000_CUT_merged/US_StreetView_10000_to_12500_CUT",
    "/data2/ouyangtianjian/NEW_StreetView_Images_US_100000_CUT_merged/US_StreetView_12500_to_15000_CUT",
    "/data2/ouyangtianjian/NEW_StreetView_Images_US_100000_CUT_merged/US_StreetView_15000_to_17500_CUT",
    # ... Add all your source paths here ...
    "/data2/ouyangtianjian/US_StreetView_Others_40000_to_50000_CUT_512"
]

# 3. Base Paths for Input and Output
#    Input Pattern: .../{City}_2019/{City}_2019_tract_segregation.jsonl
#    Output Pattern: .../{City}/images/{tract_id}/...
INPUT_BASE_DIR = "/data3/maruolong/VISAGE/data/raw/mobility/visit_data"
OUTPUT_BASE_DIR = "/data3/maruolong/VISAGE/data/raw/imagery/street_view_images_extracted"

# Valid image extensions to copy
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# ==========================================
#  MAIN PROCESSING LOGIC
# ==========================================

def process_cities():
    print(f"Starting processing for {len(CITIES)} cities...")

    for city in CITIES:
        print(f"\n========================================")
        print(f" Processing City: {city}")
        print(f"========================================")
        
        # 1. Construct dynamic file paths for the current city
        #    Pattern matches: Chicago_2019/Chicago_2019_tract_segregation.jsonl
        jsonl_file = os.path.join(INPUT_BASE_DIR, f"{city}_{year}", f"{city}_{year}_tract_segregation.jsonl")
        
        #    Pattern matches: Baseline/Chicago/images
        target_root_dir = os.path.join(OUTPUT_BASE_DIR, city, "images")
        os.makedirs(target_root_dir, exist_ok=True)

        # 2. Check if input file exists
        if not os.path.exists(jsonl_file):
            print(f"  [!] Input file not found: {jsonl_file}")
            print(f"  Skipping {city}...")
            continue

        # 3. Load Tract IDs from JSONL
        #    Using a set for O(1) lookups is much faster than a list
        target_tracts = set()
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if "tract_id" in data:
                            target_tracts.add(str(data["tract_id"]))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"  Error reading JSONL: {e}")
            continue

        print(f"  Loaded {len(target_tracts)} target tracts for {city}.")

        # 4. Prepare Target Directory
        if not os.path.exists(target_root_dir):
            os.makedirs(target_root_dir, exist_ok=True)

        # 5. Search and Copy Images
        #    Optimization: Instead of looping tracts -> sources (which is slow),
        #    we loop sources -> folders and check if the folder is in our target list.
        copied_count = 0
        
        for source_dir in SOURCE_DIRS:
            if not os.path.exists(source_dir):
                # print(f"  [Warn] Source dir not found: {source_dir}") # Optional logging
                continue

            # Walk through the source directory
            for root, dirs, files in os.walk(source_dir):
                folder_name = os.path.basename(root)

                # Check if this folder corresponds to one of our target tracts
                if folder_name in target_tracts:
                    
                    # Create specific destination folder: target_root_dir/{tract_id}
                    dest_dir = os.path.join(target_root_dir, folder_name)
                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir, exist_ok=True)

                    # Copy valid images
                    for file in files:
                        _, ext = os.path.splitext(file)
                        if ext.lower() in VALID_EXTENSIONS:
                            src_file = os.path.join(root, file)
                            dst_file = os.path.join(dest_dir, file)
                            
                            # Only copy if it doesn't exist to save time
                            if not os.path.exists(dst_file):
                                shutil.copy2(src_file, dst_file)
                                # print(f"    Copied: {file} -> {dest_dir}") # Verbose logging
                                copied_count += 1
        
        print(f"  Done. Copied {copied_count} images for {city}.")
        print(f"  Output directory: {target_root_dir}")

    print("\n>>> All cities processed successfully.")

if __name__ == "__main__":
    process_cities()