import pandas as pd
import os

# ==========================================
#  USER CONFIGURATION
# ==========================================
# Input CSV file path (Please replace with your actual file path)
# Update 'BASE_DIR' to your root directory.
BASE_DIR = '/data3/maruolong/VISAGE/data'

INPUT_CSV_FILE = 'ACSST5Y2019.S1901-Data.csv'
INPUT_CSV_FILE = os.path.join(BASE_DIR, INPUT_CSV_FILE)

# Output Excel file path
OUTPUT_XLSX_FILE = 'Median_Income.xlsx'
OUTPUT_XLSX_FILE = os.path.join(BASE_DIR, OUTPUT_XLSX_FILE)

# ==========================================
#  MAIN LOGIC
# ==========================================

def process_income_data():
    print(f"Reading CSV file: {INPUT_CSV_FILE}...")
    
    if not os.path.exists(INPUT_CSV_FILE):
        print(f"Error: Input file not found at {INPUT_CSV_FILE}")
        return

    # Read CSV file
    # Usually, Census data has variable codes in the first row (header=0) 
    # and variable descriptions in the second row.
    # Pandas automatically uses the first row as column names.
    df = pd.read_csv(INPUT_CSV_FILE, dtype=str) # Read as string to preserve leading zeros

    # Check if necessary columns exist
    # 'GEO_ID' is the geographic identifier
    # 'S1901_C01_012E' is typically Median Household Income (Estimate)
    required_cols = ['GEO_ID', 'S1901_C01_012E']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in CSV.")
            return

    # Data Preprocessing
    # Requirement: Start reading data from the third row.
    # In Pandas:
    # Row 0 (index 0) is the second row of the CSV (usually descriptive text like "Estimate!!Median...")
    # Row 1 (index 1) is the third row of the CSV (start of actual data)
    
    # We extract index 1 and all subsequent rows
    df_data = df.iloc[1:].copy()

    # 1. Process Tract column: Extract the last 11 digits of GEO_ID
    # Example: "1400000US17031010100" -> "17031010100"
    df_data['tract'] = df_data['GEO_ID'].astype(str).str[-11:]

    # 2. Process Income column
    df_data['Income'] = df_data['S1901_C01_012E']

    # 3. Keep only the two required columns
    final_df = df_data[['tract', 'Income']]

    # Save as Excel file
    print(f"Saving to {OUTPUT_XLSX_FILE}...")
    final_df.to_excel(OUTPUT_XLSX_FILE, index=False)
    
    print("Done! Processing complete.")
    print(f"Total rows processed: {len(final_df)}")
    print(final_df.head()) # Print preview of the first few rows

if __name__ == "__main__":
    process_income_data()