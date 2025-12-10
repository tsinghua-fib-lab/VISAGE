import json
import os

# ==========================================
#  USER CONFIGURATION
# ==========================================

CITIES = [
    'Boston', 'Chicago', 'Dallas', 'Detroit', 'Los Angeles', 'Miami', 'New York', 'Philadelphia',
    'San Francisco', 'Seattle', 'Washington', 'Albuquerque', 'Austin', 'Baltimore', 'Charlotte',
    'Columbus', 'Denver', 'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas',
    'Memphis', 'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland', 'San Antonio',
    'San Diego', 'San Jose', 'Tucson'
]

YEAR = '2019'

# Root directory path
BASE_DIR = '/data3/maruolong/VISAGE/data/raw'
VISIT_DATA_DIR = os.path.join(BASE_DIR, 'mobility', 'visit_data')

# ==========================================
#  CALCULATION FUNCTION
# ==========================================

def calculate_segregation_index(income_counts, total_visits):
    """
    Calculates the segregation index S based on income proportions.
    Formula: S = (2/3) * sum(|p_i - 0.25|)
    
    Args:
        income_counts (dict): Dictionary of counts per income level {1: count, 2: count...}
        total_visits (int): Total number of visits
    Returns:
        tuple: (Segregation Score, Dictionary of proportions)
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
#  MAIN PROCESSING LOOP
# ==========================================

def run_segregation_calculation():
    print(">>> Starting Segregation Calculation Process...\n")

    for city in CITIES:
        print(f"Processing City: {city} ({YEAR})...")
        
        # Define directory and file paths
        city_dir = os.path.join(VISIT_DATA_DIR, f"{city}_{YEAR}")
        
        # Input file (generated from the previous step)
        input_file = os.path.join(city_dir, f"{city}_{YEAR}_tract_income_levels_by_visiting.jsonl")
        
        # Output files
        output_segregation_file = os.path.join(city_dir, f"{city}_{YEAR}_tract_segregation.jsonl")
        output_distribution_file = os.path.join(city_dir, f"{city}_{YEAR}_tract_income_distribution.jsonl")

        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"  [!] Input file not found: {input_file}. Skipping...")
            continue

        segregation_data = []
        income_distribution_data = []

        # Read and process the input JSONL file
        with open(input_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                
                tract_id = data['tract_id']
                visiting_tracts = data['visiting_tracts']
                
                # Initialize counters for 4 income levels
                income_counts = {1: 0, 2: 0, 3: 0, 4: 0}
                total_visits = 0
                
                # Aggregate visits by income level
                for visiting_tract in visiting_tracts:
                    income_level = visiting_tract['income']
                    visits = visiting_tract['visits']
                    
                    # Ensure income level is valid (1-4)
                    if income_level in income_counts:
                        income_counts[income_level] += visits
                        total_visits += visits
                
                # Calculate metrics
                segregation, p_values = calculate_segregation_index(income_counts, total_visits)

                # Append segregation result
                segregation_data.append({
                    'tract_id': tract_id,
                    'segregation': segregation
                })

                # Append distribution result (format list: [Level1, Level2, Level3, Level4])
                distribution_list = [round(p_values.get(i, 0.0), 2) for i in range(1, 5)]
                income_distribution_data.append({
                    'tract_id': tract_id,
                    'income_distribution': distribution_list
                })

        # Save Segregation Output
        with open(output_segregation_file, 'w') as f:
            for item in segregation_data:
                json.dump(item, f)
                f.write('\n')

        # Save Income Distribution Output
        with open(output_distribution_file, 'w') as f:
            for item in income_distribution_data:
                json.dump(item, f)
                f.write('\n')

        print(f"  ✅ Saved: {os.path.basename(output_segregation_file)}")
        print(f"  ✅ Saved: {os.path.basename(output_distribution_file)}")

    print("\n>>> All cities processed.")

if __name__ == "__main__":
    run_segregation_calculation()