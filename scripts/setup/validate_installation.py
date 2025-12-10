#!/usr/bin/env python3
"""
Script: Installation Validator
Description: Checks if all modules are importable and if data paths exist.
Usage: python scripts/setup/validate_installation.py
"""

import os
import sys
import importlib

# Add project root to python path to find modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

# Add external experiments folder if it exists outside the project
sys.path.append("/data3/maruolong/VISAGE/experiments") 
sys.path.append("/data3/maruolong/VISAGE/workflows")

def check_modules():
    """Verifies that core modules can be imported."""
    print("🔍 Checking System Modules...")
    
    required_modules = [
        "closed_loop_workflow",
        "experiment_tracker",
        "numpy",
        "pandas",
        "sklearn",
        "torch"
    ]
    
    all_pass = True
    for mod in required_modules:
        try:
            importlib.import_module(mod)
            print(f"   ✅ Module found: '{mod}'")
        except ImportError as e:
            print(f"   ❌ Module MISSING: '{mod}'")
            print(f"      Error details: {e}")
            all_pass = False
            
    # Check for predictor.py specifically (it might be optional)
    try:
        importlib.import_module("predictor")
        print(f"   ✅ Module found: 'predictor'")
    except ImportError:
        print(f"   ⚠️ Module 'predictor' not found (Regression Viz might fail)")
        
    return all_pass

def check_paths():
    """Verifies that key data directories exist."""
    print("\n🔍 Checking Data Paths...")
    
    # Change this if your data is located elsewhere
    base_data = "/data3/maruolong/VISAGE/data"
    
    if os.path.exists(base_data):
        print(f"   ✅ Data Root found: {base_data}")
    else:
        print(f"   ⚠️ Data Root NOT found: {base_data}")
        print("      (Please ensure your storage is mounted or run download_data.py)")

if __name__ == "__main__":
    print(f"📂 Project Root detected as: {PROJECT_ROOT}")
    
    if check_modules():
        check_paths()
        print("\n✨ System check passed! You are ready to run the pipeline.")
    else:
        print("\n❌ System check FAILED. Please fix the missing modules above.")