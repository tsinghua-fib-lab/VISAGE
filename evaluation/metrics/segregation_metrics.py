import pandas as pd
import json
import os
import sys

# ==========================================
#  USER CONFIGURATION SECTION
#  Please modify the paths and settings below
# ==========================================

# 1. Define the list of cities to process
CITIES = [
    'Boston', 'Chicago', 'Dallas', 'Detroit', 'Los Angeles', 'Miami', 'New York', 'Philadelphia',
    'San Francisco', 'Seattle', 'Washington', 'Albuquerque', 'Austin', 'Baltimore', 'Charlotte',
    'Columbus', 'Denver', 'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas',
    'Memphis', 'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland', 'San Antonio',
    'San Diego', 'San Jose', 'Tucson'
]

# 2. Define the Year (Optional/Configurable)
YEAR = '2019'

# 3. Define the list of Weeks to process
# Note: Ensure these date strings match your file naming convention.
WEEKS = [
    '2019-08-09', '2019-08-16', '2019-08-23', '2019-08-30', 
    '2019-09-06', '2019-09-13', '2019-09-20', '2019-09-27', 
    '2019-10-04', '2019-10-11', '2019-10-18', '2019-10-25'
]

# 4. Define Base Paths
# Update 'BASE_DIR' to your root directory.
BASE_DIR = '/data3/maruolong/VISAGE/data/raw'

# Path to the Median Income Excel file
INCOME_FILE_PATH = os.path.join(BASE_DIR, "census", "Median_Income.xlsx")

# Input/Output directories
# The script assumes files are at: {VISIT_DATA_DIR}/{City}_{Year}/{City}_{Week}.csv.gz
VISIT_DATA_DIR = os.path.join(BASE_DIR, 'mobility', 'visit_data')


# ==========================================
#  HELPER FUNCTIONS
# ==========================================

def process_income_value(income):
    """
    Cleans and converts income strings to numerical values.
    """
    if str(income) == "-":
        return None
    elif str(income) == "250,000+":
        return 300000
    elif str(income) == "2,500-":
        return 2000
    try:
        # Remove commas if present and convert
        return int(str(income).replace(',', ''))
    except ValueError:
        return None

def convert_int64(obj):
    """
    Recursively converts pandas/numpy int64/float64 types to native Python types
    to ensure JSON serialization works.
    """
    if isinstance(obj, pd.Series):
        return obj.apply(convert_int64)
    elif isinstance(obj, dict):
        return {key: convert_int64(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64(item) for item in obj]
    
    # Handle numpy types
    try:
        import numpy as np
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
    except ImportError:
        pass
        
    return obj

def calculate_segregation_index(income_counts, total_visits):
    """
    Calculates the segregation index S based on income proportions.
    Formula: S = (2/3) * sum(|p_i - 0.25|)
    """
    if total_visits == 0:
        return 0.0, {}
    
    # Calculate proportion p_i for each income level
    p_values = {k: v / total_visits for k, v in income_counts.items()}
    
    # Calculate S
    # Assuming 4 income levels (quartiles), expected proportion is 0.25
    s_value = (2 / 3) * sum(abs(p - 0.25) for p in p_values.values())
    
    return round(s_value, 2), p_values

# ==========================================
#  MAIN PROCESSOR
# ==========================================

def run_processing():
    print(">>> Loading Median Income Data...")
    if not os.path.exists(INCOME_FILE_PATH):
        print(f"Error: Income file not found at {INCOME_FILE_PATH}")
        return

    # Load Income Data once
    tract_income_df = pd.read_excel(INCOME_FILE_PATH)
    # Create a dictionary mapping Tract ID (padded to 11 digits) to Income
    tract_income_dict = dict(zip(
        tract_income_df.tract.astype(str).apply(lambda x: x.zfill(11)), 
        tract_income_df.Income
    ))
    print(">>> Income Data Loaded.\n")

    # Loop through each city
    for city in CITIES:
        print(f"==================================================")
        print(f" PROCESSING CITY: {city} | YEAR: {YEAR}")
        print(f"==================================================")

        # Define file paths specific to this city
        city_dir = os.path.join(VISIT_DATA_DIR, f"{city}_{YEAR}")
        
        # Output filenames
        file_step1_visits = os.path.join(city_dir, f"{city}_{YEAR}_tract_visits.jsonl")
        file_step2_levels = os.path.join(city_dir, f"{city}_{YEAR}_tract_income_levels_by_visiting.jsonl")
        file_step3_segregation = os.path.join(city_dir, f"{city}_{YEAR}_tract_segregation.jsonl")
        file_step3_distribution = os.path.join(city_dir, f"{city}_{YEAR}_tract_income_distribution.jsonl")

        # Check if city directory exists
        if not os.path.exists(city_dir):
            print(f"Warning: Directory not found for {city}. Skipping...")
            continue

        # ---------------------------------------------------------
        # STEP 1: Extract and Aggregate Visit Data
        # ---------------------------------------------------------
        print(f"--> Step 1: Extracting visit data from SafeGraph CSVs...")
        tract_visits = {}

        for week in WEEKS:
            csv_path = os.path.join(city_dir, f"{city}_{week}.csv.gz")
            if not os.path.exists(csv_path):
                print(f"    Warning: File not found {csv_path}, skipping week.")
                continue
            
            # print(f"    Processing week: {week}") # Uncomment for detailed logs
            try:
                df = pd.read_csv(csv_path)

                # Filter data
                df = df[df.parent_placekey.isna()]
                df = df[~df.visitor_home_aggregation.isna()]
                df = df[df.visitor_home_aggregation != '{}']

                # Calculate normalization factor
                df['norm'] = df.normalized_visits_by_state_scaling / df.raw_visitor_counts

                # Process POIs
                for placekey, item, norm_val, poi_cbg in zip(df.placekey, df.visitor_home_aggregation, df.norm, df.poi_cbg):
                    vdict = json.loads(item)
                    # Extract Tract ID (first 11 digits)
                    tract_id = str(poi_cbg).strip("'")[:11]

                    for cbg_id in vdict:
                        if tract_id not in tract_visits:
                            tract_visits[tract_id] = {}
                        if cbg_id not in tract_visits[tract_id]:
                            tract_visits[tract_id][cbg_id] = 0
                        tract_visits[tract_id][cbg_id] += vdict[cbg_id] * norm_val
            except Exception as e:
                print(f"    Error processing {week}: {e}")

        # Merge with Income Data
        step1_data = []
        for tract_id, cbg_data in tract_visits.items():
            # Get income for the POI's tract
            tract_income = tract_income_dict.get(tract_id, None)
            if tract_income is not None:
                tract_income = process_income_value(tract_income)
            
            if tract_income is None:
                continue

            total_visits = sum(cbg_data.values())
            visiting_tracts = []

            # Process visitors' tracts
            for visiting_tract_id in cbg_data.keys():
                visitor_income = tract_income_dict.get(visiting_tract_id, None)
                if visitor_income is not None:
                    visitor_income = process_income_value(visitor_income)
                
                if visitor_income is None:
                    continue

                visiting_tracts.append({
                    'tract_id': visiting_tract_id,
                    'income': visitor_income,
                    'visits': int(cbg_data[visiting_tract_id])
                })

            result = {
                'tract_id': tract_id,
                'visiting_tracts': visiting_tracts,
                'income': tract_income,
                'total_visits': total_visits
            }
            step1_data.append(result)

        # Save Step 1 Output
        with open(file_step1_visits, 'w') as f:
            for data in step1_data:
                json.dump(data, f)
                f.write('\n')
        print(f"    Saved raw visits to: {os.path.basename(file_step1_visits)}")

        # ---------------------------------------------------------
        # STEP 2: Calculate Income Levels (Quartiles)
        # ---------------------------------------------------------
        print(f"--> Step 2: Calculating income levels (Quartiles)...")
        
        all_incomes = []
        tract_income_mapping = {} # Map tract_id -> list of incomes (usually just one, but logic follows original)
        visited_ids = set()

        # Gather incomes from the data generated in Step 1
        for data in step1_data:
            visiting_tracts_list = data['visiting_tracts']
            for v_tract in visiting_tracts_list:
                t_id = v_tract['tract_id']
                inc = v_tract['income']
                
                if t_id not in visited_ids:
                    all_incomes.append(inc)
                    visited_ids.add(t_id)
                    if t_id not in tract_income_mapping:
                        tract_income_mapping[t_id] = []
                    tract_income_mapping[t_id].append(inc)

        # Calculate Quartiles using qcut
        income_series = pd.Series(all_incomes)
        if len(income_series.unique()) > 1:
            try:
                bins = pd.qcut(income_series, 4, labels=[1, 2, 3, 4], duplicates='drop')
                bin_labels = bins.cat.codes + 1
            except IndexError:
                bin_labels = pd.Series([1] * len(income_series))
        else:
            bin_labels = pd.Series([1] * len(income_series))
        
        income_to_level = dict(zip(income_series, bin_labels))
        
        # Map tract IDs to calculated Levels
        tract_level_dict = {}
        for t_id, inc_list in tract_income_mapping.items():
            # Use the first income value to determine level
            tract_level_dict[t_id] = income_to_level.get(inc_list[0], 1)

        # Update data with levels
        step2_data = []
        for data in step1_data:
            visiting_tracts_list = data['visiting_tracts']
            for v_tract in visiting_tracts_list:
                t_id = v_tract['tract_id']
                lvl = tract_level_dict.get(t_id, None)
                if lvl is not None:
                    v_tract['income'] = lvl # Replace raw income with level
            
            # Clean int64 types for JSON serialization
            step2_data.append(convert_int64(data))

        # Save Step 2 Output
        with open(file_step2_levels, 'w') as f:
            for data in step2_data:
                json.dump(data, f)
                f.write('\n')
        print(f"    Saved income levels to: {os.path.basename(file_step2_levels)}")

        # ---------------------------------------------------------
        # STEP 3: Calculate Segregation Indicators
        # ---------------------------------------------------------
        print(f"--> Step 3: Calculating Segregation Index & Distribution...")
        
        segregation_output = []
        distribution_output = []

        # Process the data from Step 2
        for data in step2_data:
            tract_id = data['tract_id']
            visiting_tracts = data['visiting_tracts']

            # Count visits by income level
            income_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            total_visits_calc = 0

            for vt in visiting_tracts:
                lvl = vt['income']
                v_count = vt['visits']
                if lvl in income_counts:
                    income_counts[lvl] += v_count
                    total_visits_calc += v_count
            
            # Calculate metrics
            segregation_score, p_values = calculate_segregation_index(income_counts, total_visits_calc)

            segregation_output.append({
                'tract_id': tract_id,
                'segregation': segregation_score
            })

            # Format distribution list [level1%, level2%, level3%, level4%]
            dist_list = [round(p_values.get(i, 0.0), 2) for i in range(1, 5)]
            distribution_output.append({
                'tract_id': tract_id,
                'income_distribution': dist_list
            })

        # Save Step 3 Outputs
        with open(file_step3_segregation, 'w') as f:
            for item in segregation_output:
                json.dump(item, f)
                f.write('\n')
        
        with open(file_step3_distribution, 'w') as f:
            for item in distribution_output:
                json.dump(item, f)
                f.write('\n')

        print(f"    Saved segregation data to: {os.path.basename(file_step3_segregation)}")
        print(f"    Saved distribution data to: {os.path.basename(file_step3_distribution)}")
        print(f"✅ DONE: {city} processing complete.\n")

    print(">>> All cities processed successfully.")

if __name__ == "__main__":
    run_processing()