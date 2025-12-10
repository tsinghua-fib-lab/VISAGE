import json
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Read visual description data
description_file = '/data3/maruolong/VISAGE/data/aggregated_description_by_tract.jsonl'
tract_descriptions = []

with open(description_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        tract_id = data['tract_id']
        descriptions = data['description']
        descriptions['tract_id'] = tract_id
        tract_descriptions.append(descriptions)

desc_df = pd.DataFrame(tract_descriptions)
desc_df['tract_id'] = desc_df['tract_id'].astype(str)

# # Remove MuralsSculpturesOrUrbanArt
# if 'MuralsSculpturesOrUrbanArt' in desc_df.columns:
#     desc_df.drop(columns=['MuralsSculpturesOrUrbanArt'], inplace=True)


# Read segregation data (concatenated for all cities)
segregation_dfs = []
cities = [
    "Seattle", "Philadelphia", "Boston", "Chicago", "Dallas",
    "Detroit", "Miami", "Los Angeles", "New York", "San Francisco", "Washington",
    'Albuquerque', 'Austin', 'Baltimore', 'Charlotte', 'Columbus', 'Denver',
    'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas', 'Memphis',
    'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland', 'San Antonio', 'San Diego',
    'San Jose', 'Tucson'
]

for city in cities:
    segregation_file = f"/data3/maruolong/segregation/All_time/visit_data/{city}_2019/{city}_2019_tract_segregation.jsonl"
    city_df = pd.read_json(segregation_file, lines=True)
    city_df['tract_id'] = city_df['tract_id'].astype(str)
    segregation_dfs.append(city_df)

# Merge segregation information for all cities
seg_df = pd.concat(segregation_dfs, ignore_index=True)

# Merge the two dataframes
merged_df = pd.merge(desc_df, seg_df[['tract_id', 'segregation']], on='tract_id', how='inner')

# Calculate Pearson correlation + p-value for each description item and segregation
correlation_results = {}
for col in merged_df.columns:
    if col not in ['tract_id', 'segregation']:
        corr, p_value = pearsonr(merged_df[col], merged_df['segregation'])
        correlation_results[col] = {
            'correlation': corr,
            'p_value': p_value
        }

# Sort (sort by correlation coefficient in descending order)
sorted_corr = dict(sorted(correlation_results.items(), key=lambda x: x[1]['correlation'], reverse=True))

# Print correlation results and significance
print("\n📊 Pearson correlation and significance (p-value) for each visual description item with segregation:\n")
for key, val in sorted_corr.items():
    signif = '✅' if val['p_value'] < 0.05 else '❌'
    print(f"{key}: r = {val['correlation']:.4f}, p = {val['p_value']:.4g} {signif}")


# === Visualize Correlation Plot === #
sns.set(style="whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(14, 8))

# Color settings: Blue for positive values, Red for negative values
# Note: Logic assumes positive correlation gets blue (or red based on specific need), checking value against 0
colors = ['#1f77b4' if v['correlation'] <= 0 else '#d62728' for v in sorted_corr.values()]

# Plotting
bars = sns.barplot(
    x=[v['correlation'] for v in sorted_corr.values()],
    y=list(sorted_corr.keys()),
    palette=colors
)

# Add labels
for i, (name, val) in enumerate(sorted_corr.items()):
    bars.text(val['correlation'] + (0.01 if val['correlation'] >= 0 else -0.01), i, f"{val['correlation']:.2f}",
              va='center', ha='left' if val['correlation'] >= 0 else 'right', fontsize=14, color='black')


# Beautify the plot
plt.title("Correlation Between Visual Descriptions and Income Segregation", fontsize=22)
plt.xlabel("Pearson Correlation with Segregation", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(-0.7, 0.7)  # ✅ X-axis range modified
plt.tight_layout()

# Save the plot, do not display
output_path = "/data3/maruolong/VISAGE/data/analysis/caption/Descriptions_segregation_correlation_plot.pdf"
plt.savefig(output_path, dpi=600)
plt.close()

print(f"\n✅ Plot saved to: {output_path}")