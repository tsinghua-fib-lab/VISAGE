#!/usr/bin/env python3
"""
Script: Data Directory Setup
Description: Creates the required folder structure for VISAGE data storage.
Usage: python VISAGE/scripts/setup/download_data.py
"""

import os
import argparse

# Default path matching your server configuration
DEFAULT_BASE_DIR = "/data3/maruolong/VISAGE/data"

def create_directory_structure(base_dir):
    """Creates the necessary subdirectories for raw and processed data."""
    print(f"🚀 Initializing directory structure at: {base_dir}")
    
    # List of required subdirectories
    subdirs = [
        "raw/imagery",              # Stores city street view images
        "raw/mobility/visit_data",  # Stores mobility/segregation data
        "raw/census",               # Stores census tract shapefiles, mappings and other information
        "processed/literature",     # Stores literature analysis outputs
        "processed/training_data",  # Stores training datasets
        "processed/cue_frequencies" # Stores cue detection results
    ]
    
    for sub in subdirs:
        path = os.path.join(base_dir, sub)
        os.makedirs(path, exist_ok=True)
        print(f"   ✅ Created: {path}")

def print_instructions():
    """Prints instructions for manual data placement."""
    print("\n📦 Data Setup Instructions:")
    print("   Since the dataset involves large satellite/street-view imagery,")
    print("   please manually move your dataset to the directories created above:")
    print("   ---------------------------------------------------------------")
    print("   1. City Images -> data/raw/imagery/{CityName}/images") # the inner folder should use the tract ID as the folder name, for example: data/raw/imagery/Chicago/images/17031010100/***.png
    print("   2. RS Images   -> data/raw/imagery/rs_image_new/merged") # the inner folder should use the tract ID as the folder name and contain all the rs images like in the tract: data/raw/imagery/rs_image_new/merged/10_391_395.png
    print("   3. JSONL Data  -> data/raw/mobility/visit_data/") # the inner folder should be named as {CityName}_{year}, with .csv.gz files inside, for example: data/raw/mobility/visit_data/Chicago_2019/Chicago_2019_***.csv.gz
    print("   4. Census Data -> data/raw/census/") # Place your census shapefile, tile mapping files, and the tract income data(.xlsx) here. You can download them from the links in the README.md (through ACS website)
    print("   ---------------------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup Data Directories")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR, 
                        help="Root directory for data storage")
    args = parser.parse_args()
    
    create_directory_structure(args.base_dir)
    print_instructions()