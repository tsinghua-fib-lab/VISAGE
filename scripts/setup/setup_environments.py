#!/usr/bin/env python3
"""
Script: Environment Setup
Description: Installs Python dependencies and configures API keys (.env).
             Includes setup for LLM and Academic Literature APIs.
Usage: python VISAGE/scripts/setup/setup_environments.py
"""

import subprocess
import sys
import os

def install_requirements():
    """Installs dependencies listed in requirements.txt."""
    print("🔧 Installing dependencies from requirements.txt...")
    
    # Locate requirements.txt relative to this script
    req_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "requirements.txt")
    
    if not os.path.exists(req_path):
        print(f"❌ Error: requirements.txt not found at {req_path}")
        return

    try:
        # Run pip install
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
        print("✅ Dependencies installed successfully.")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies. Please check your internet connection.")

def setup_api_keys():
    """Interactively configures API keys and saves them to a .env file."""
    print("\n🔑 API Key Configuration")
    print("   (Leave blank and press Enter to skip any specific key)")
    
    # --- 1. LLM Configuration ---
    print("\n   [1/2] LLM Model Configuration (for Text Generation)")
    llm_api_key = input("   Enter your OpenAI/LLM API Key: ").strip()
    llm_base_url = input("   Enter your LLM Base URL (optional): ").strip()
    
    # --- 2. Literature Configuration ---
    print("\n   [2/2] Literature Agent Configuration (for Paper Retrieval)")
    # Usually SerpApi is used for Google Scholar scraping
    scholar_key = input("   Enter your Google Scholar/SerpApi Key: ").strip()
    ieee_key = input("   Enter your IEEE Xplore API Key: ").strip()
    elsevier_key = input("   Enter your Elsevier/Scopus API Key (optional): ").strip()
    # You can add more literature sources as needed
    
    # Check if any key was provided
    if any([llm_api_key, llm_base_url, scholar_key, ieee_key, elsevier_key]):
        env_file = ".env"
        with open(env_file, "w") as f:
            # LLM Keys
            if llm_api_key: f.write(f"VISAGE_API_KEY={llm_api_key}\n")
            if llm_base_url: f.write(f"VISAGE_BASE_URL={llm_base_url}\n")
            
            # Literature Keys
            if scholar_key: f.write(f"SERPAPI_KEY={scholar_key}\n") # Common naming for Scholar
            if ieee_key: f.write(f"IEEE_API_KEY={ieee_key}\n")
            if elsevier_key: f.write(f"ELSEVIER_API_KEY={elsevier_key}\n")
            
        print(f"\n✅ Credentials saved to {env_file}")
        print("   The Literature Agent will now be able to fetch academic papers.")
    else:
        print("\n   Skipped API configuration.")

if __name__ == "__main__":
    install_requirements()
    setup_api_keys()