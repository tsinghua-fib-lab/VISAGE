import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import matplotlib.colors as mcolors

# Paths to all jsonl files
cities = [
    'Boston', 'Chicago', 'Dallas', 'Detroit', 'Los Angeles', 'Miami', 'New York', 'Philadelphia',
    'San Francisco', 'Seattle', 'Washington', 'Albuquerque', 'Austin', 'Baltimore', 'Charlotte',
    'Columbus', 'Denver', 'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas',
    'Memphis', 'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland', 'San Antonio',
    'San Diego', 'San Jose', 'Tucson'
]

base_path = "/data3/maruolong/VISAGE/data/mobility/visit_data"
jsonl_paths = [f"{base_path}/{city}_2019/{city}_2019_tract_segregation.jsonl" for city in cities]

# Collect all segregation values
segregation_values = []
for path in jsonl_paths:
    try:
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if "segregation" in data:
                    segregation_values.append(data["segregation"])
    except FileNotFoundError:
        print(f"File not found: {path}")


# Set plot style
sns.set(style="whitegrid", font="Arial", palette="muted", color_codes=True)

# Generate Chinese version plot (kept as requested, but comments are in English)
plt.figure(figsize=(10, 6))
sns.histplot(segregation_values, bins=50, kde=True, color='skyblue')
plt.title("31个美国城市中社区的收入隔离度分布", fontsize=16)
plt.xlabel("收入隔离度 (S)", fontsize=16)
plt.ylabel("社区数量", fontsize=16)
plt.tight_layout()
# plt.savefig("/data3/maruolong/segregation/Baseline/All_Cities/analysis/seg_distribution/Income_Segregation_Distribution_Chinese.png", dpi=300)
plt.close()

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Generate English version plot
plt.figure(figsize=(10, 6))
sns.histplot(segregation_values, bins=50, kde=True, color='#00008B')
plt.title("Distribution of Income Segregation across Census Tracts in 31 US Cities", fontsize=22)
plt.xlabel("Income Segregation (S)", fontsize=22)
plt.ylabel("Number of Census Tracts", fontsize=22)
plt.xticks(fontsize=18)  # Font size for x-axis ticks
plt.yticks(fontsize=18)  # Font size for y-axis ticks
plt.tight_layout()
plt.savefig("/data3/maruolong/segregation/Baseline/All_Cities/analysis/seg_distribution/Income_Segregation_Distribution.pdf", dpi=300)
plt.close()


# Store average segregation value for each city
city_avg_segs = []

for city, path in zip(cities, jsonl_paths):
    segs = []
    if not os.path.exists(path):
        print(f"Warning: {path} does not exist.")
        city_avg_segs.append(0)
        continue
    with open(path, 'r') as f:
        for line in f:
            record = json.loads(line)
            if 'segregation' in record:
                segs.append(record['segregation'])
    avg_seg = np.mean(segs) if segs else 0
    city_avg_segs.append(avg_seg)

# Set font to Times New Roman again to ensure consistency
plt.rcParams['font.family'] = 'Times New Roman'

# Sort cities and their averages (optional)
sorted_pairs = sorted(zip(cities, city_avg_segs), key=lambda x: x[1], reverse=True)
sorted_cities, sorted_avgs = zip(*sorted_pairs)

# Plotting the Bar Chart
plt.figure(figsize=(10, 6))

# Custom gradient color from deep cyan to dark blue
colors = ['#008B8B', '#00008B']  # Deep cyan to dark blue
cmap = mcolors.LinearSegmentedColormap.from_list("deep_cyan_to_blue", colors)

# Use custom gradient color
norm = plt.Normalize(vmin=min(sorted_avgs), vmax=max(sorted_avgs))
bars = plt.bar(sorted_cities, sorted_avgs, 
               color=cmap(norm(sorted_avgs)), 
               alpha=0.75)  # Set transparency

# Add value labels (commented out as in original)
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}', ha='center', va='bottom', fontsize=11)

plt.xticks(rotation=45, ha='right', fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Average Income Segregation", fontsize=22)
plt.title("Average Income Segregation Across 31 US Cities", fontsize=22)
plt.tight_layout()
plt.savefig("/data3/maruolong/segregation/Baseline/All_Cities/analysis/seg_distribution/segregation_city_means_en.pdf", dpi=300)
plt.close()

overall_avg_segregation = np.mean(segregation_values)
print(f"Average income segregation for all tracts in 31 cities: {overall_avg_segregation:.4f}")