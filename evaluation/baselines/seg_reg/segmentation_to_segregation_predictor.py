"""
### 🏞️ Semantic Segmentation Regression Baseline (Seg_Reg)

This script implements the **Seg_Reg** baseline, which predicts the income segregation index 
of various US urban census tracts by leveraging **semantic information** extracted directly from 
street-level imagery using advanced image segmentation methods. 

**The process is structured as follows:**

1.  **Feature Extraction:** The percentage of **20+ distinct semantic elements** (e.g., 'building', 'sky', 'road', 'tree') 
    is calculated for each census tract based on pre-processed image segmentation results.
2.  **Feature Preparation:** The top 40 most frequent semantic elements are selected as features.
3.  **Regression Modeling:** We use **ElasticNet Regression** to model the relationship between 
    these segmented visual features and the segregation score (the target variable).
4.  **Robust Evaluation:** The model's performance is rigorously assessed using **5-Fold Cross-Validation (CV)**.
5.  **Statistical Analysis:** Mean differences and significance tests (T-test, Mann-Whitney U test) 
    are conducted to identify which specific semantic elements are most strongly associated with 
    high vs. low segregation areas.

This Seg_Reg script establishes a strong, interpretable benchmark by transforming raw visual data 
into quantifiable urban form elements for socio-economic prediction.
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold # 引入KFold进行5折交叉验证
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu, entropy

# 📚 List of cities for data reading
cities_file_path = [
    "Seattle", "Philadelphia", "Boston", "Chicago", "Dallas",
    "Detroit", "Miami", "Los Angeles", "New York", "San Francisco", "Washington", "Albuquerque", "Austin", "Baltimore", "Charlotte",
    "Columbus", "Denver", "El Paso", "Fort Worth", "Houston", "Jacksonville", "Las Vegas",
    "Memphis", "Milwaukee", "Oklahoma City", "Phoenix", "Portland", "San Antonio",
    "San Diego", "San Jose", "Tucson"
]
# List of cities for segregation file reading 
cities_segregation_path = [
    "Seattle", "Philadelphia", "Boston", "Chicago", "Dallas",
    "Detroit", "Miami", "Los Angeles", "New York", "San Francisco", "Washington", "Albuquerque", "Austin", "Baltimore", "Charlotte",
    "Columbus", "Denver", "El Paso", "Fort Worth", "Houston", "Jacksonville", "Las Vegas",
    "Memphis", "Milwaukee", "Oklahoma City", "Phoenix", "Portland", "San Antonio",
    "San Diego", "San Jose", "Tucson"
]

base_path = "/data3/maruolong/VISAGE/data/processed/baseline/seg"
averages_data = []

## 📊 **Part 1: Data Loading and Feature Preparation**

# ✅ 1️⃣ Read tract_averages.jsonl for all cities
print("1️⃣ Reading segmentation averages...")
for city in cities_file_path:
    averages_file = f"{base_path}/segment_results/{city}_segmentation/tract_averages.jsonl"
    try:
        with open(averages_file, "r") as f:
            for line in f:
                averages_data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Warning: Averages file not found for {city}. Skipping.")
        continue

# ✅ 2️⃣ Read segregation data for all cities
print("2️⃣ Reading segregation data...")
segregation_data = {}
segregation_base_path = "/data3/maruolong/VISAGE/data/raw/mobility/visit_data"

for city in cities_segregation_path:
    # Ensure consistency in path construction
    city_for_path = city.replace(" ", "_")
    segregation_file = f"{segregation_base_path}/{city_for_path}_2019/{city_for_path}_2019_tract_segregation.jsonl"
    try:
        with open(segregation_file, "r") as f:
            for line in f:
                data = json.loads(line)
                segregation_data[data["tract_id"]] = data["segregation"]
    except FileNotFoundError:
        print(f"Warning: Segregation file not found for {city}. Skipping.")
        continue

# ✅ 3️⃣ Convert data to DataFrame
df = pd.DataFrame(averages_data)
df["segregation"] = df["tract_id"].map(segregation_data)  # Link segregation data
df = df.dropna(subset=["segregation"])  # Drop tracts without segregation values

# ✅ 4️⃣ Extract category percentages and prepare for modeling
X = pd.DataFrame(df["averages"].tolist())  # Extract category percentages
X.fillna(0, inplace=True)  # Fill missing categories (not present in a tract) with 0
Y = df["segregation"].values # 转换为 numpy 数组

# Ensure all columns in X are numeric
X = X.apply(pd.to_numeric, errors="coerce")
X.fillna(0, inplace=True)

# Calculate category means and select top 40 categories
category_means = X.mean(axis=0)
top_40_categories = category_means.sort_values(ascending=False).head(40)

print("\n🔹 Top 40 Categories and their average percentage:")
for category, mean in top_40_categories.items():
    print(f"{category}: {mean:.4f}")

# Select only the top 40 categories for regression analysis
X_selected = X[top_40_categories.index] 
X_array = X_selected.values

# **Standardize Features**
scaler = StandardScaler()
X_scaled_all = scaler.fit_transform(X_array)


## 🚀 **Part 2: 5-Fold Cross-Validation and Evaluation**

# ✅ 5️⃣ Define Model and KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
elasticnet_model = ElasticNet(alpha=0.005, l1_ratio=0.3, random_state=42, max_iter=10000)

print("\n🚀 Starting 5-Fold CV for ElasticNet...")
fold = 1
all_Y_true, all_Y_pred = [], []
all_coefficients = []

for train_idx, test_idx in kf.split(X_scaled_all):
    X_train, X_test = X_scaled_all[train_idx], X_scaled_all[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # 训练模型
    elasticnet_model.fit(X_train, Y_train)
    Y_pred = elasticnet_model.predict(X_test)
    
    # 记录系数（用于后续平均分析）
    all_coefficients.append(elasticnet_model.coef_)

    # 记录真实值和预测值
    all_Y_true.extend(Y_test)
    all_Y_pred.extend(Y_pred)

    # 计算 Acc@0.1
    acc_0_1 = np.mean(np.abs(Y_test - Y_pred) < 0.1)

    print(f"Fold {fold}:")
    print(f"  R²:    {r2_score(Y_test, Y_pred):.4f}")
    print(f"  MAE:   {mean_absolute_error(Y_test, Y_pred):.4f}")
    print(f"  Acc@0.1: {acc_0_1:.4f}")
    fold += 1

# 计算整体 CV 指标
overall_r2 = r2_score(all_Y_true, all_Y_pred)
overall_mae = mean_absolute_error(all_Y_true, all_Y_pred)
overall_mse = mean_squared_error(all_Y_true, all_Y_pred)
overall_acc_0_1 = np.mean(np.abs(np.array(all_Y_true) - np.array(all_Y_pred)) < 0.1)

print("\nOverall 5-Fold CV Results for ElasticNet:")
print(f"  Overall R²: {overall_r2:.4f}")
print(f"  Overall MAE: {overall_mae:.4f}")
print(f"  Overall MSE: {overall_mse:.4f}")
print(f"  Overall Acc@0.1: {overall_acc_0_1:.4f}")

# 计算平均系数
mean_coefficients = np.mean(all_coefficients, axis=0)
print("\n🔹 Average Regression Coefficients across 5 Folds:")
for category, coef in zip(top_40_categories.index, mean_coefficients):
    print(f"{category}: {coef:.4f}")


# ✅ 6️⃣ Visualization (Density Scatter Plot)
def plot_density_scatter(Y_true, Y_pred, title, filename, r2_score_val):
    """Generates a density scatter plot for True vs Predicted values."""
    cmap = LinearSegmentedColormap.from_list("teal_shade", ["#a8e6cf", "#56c8d8", "#007c91"])
    plt.figure(figsize=(7, 7))
    # 使用 hexbin 创建密度散点图
    plt.hexbin(Y_true, Y_pred, gridsize=40, cmap=cmap, bins='log', linewidths=0)
    cb = plt.colorbar(label='Count in log scale')
    
    plt.xlabel("True Segregation", fontsize=14)
    plt.ylabel("Predicted Segregation", fontsize=14)
    plt.title(f"{title} (Overall R²={r2_score_val:.4f})", fontsize=16)
    # 添加 45 度线
    plt.axline([0, 0], slope=1, color="red", linestyle="--", linewidth=1.5)
    plt.tight_layout()
    os.makedirs(f"{base_path}/figures", exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()

output_path = f"{base_path}/figures/31_CV_density_elasticnet.png"
plot_density_scatter(np.array(all_Y_true), np.array(all_Y_pred),
                      "ElasticNet 5-Fold CV Segregation Prediction", output_path, overall_r2)
print(f"📊 5-Fold CV Density Plot saved to: {output_path}")


## 📈 **Part 3: Statistical Analysis (Using Full Dataset)**

# For statistical analysis (mean difference, t-test, diversity), we typically use the full dataset.

# Prepare DataFrame by extracting the top categories percentages (already done in preparation phase, ensuring consistency)
for category in top_40_categories.index:
    df[category] = df["averages"].apply(lambda x: x.get(category, 0))

# Define High and Low Segregation Tracts using the median
segregation_median = df["segregation"].median()
high_segregation = df[df["segregation"] > segregation_median].copy()
low_segregation = df[df["segregation"] <= segregation_median].copy()

# Calculate Mean Difference
mean_diff = high_segregation[top_40_categories.index].mean() - low_segregation[top_40_categories.index].mean()

# Statistical Testing
t_test_pvals = {}
u_test_pvals = {}

for feature in top_40_categories.index:
    # Use Welch's t-test (equal_var=False)
    t_stat, t_pval = ttest_ind(high_segregation[feature], low_segregation[feature], equal_var=False, nan_policy='omit')
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pval = mannwhitneyu(high_segregation[feature].dropna(), low_segregation[feature].dropna(), alternative='two-sided')
    t_test_pvals[feature] = t_pval
    u_test_pvals[feature] = u_pval

# Compile Results
stat_results = pd.DataFrame({
    "Mean Difference": mean_diff,
    "T-test p-value": pd.Series(t_test_pvals),
    "Mann-Whitney U p-value": pd.Series(u_test_pvals)
}).sort_values("Mean Difference", ascending=False)

# Filter for statistically significant features (p-value < 0.05, T-test)
significant_features = stat_results[stat_results["T-test p-value"] < 0.05]
print("\n--- Statistical Test Results (Top 40 Categories) ---")
print(stat_results)

print("\n📊 Statistically Significant Street View Elements Affecting Segregation (T-test p<0.05):")
print(significant_features)

# --- Visualization: Mean Difference Bar Plot ---
plt.figure(figsize=(12, 8))
plot_data = significant_features.sort_values("Mean Difference", ascending=False)
if plot_data.empty:
    plot_data = stat_results.head(10) # Fallback if none are significant
    
sns.barplot(x=plot_data.index, y=plot_data["Mean Difference"], palette="viridis")
plt.xticks(rotation=90)
plt.xlabel("Street View Elements")
plt.ylabel(f"Mean Percentage Difference (High Segregation Tracts - Low Segregation Tracts)")
plt.title("Mean Difference of Street View Elements Affecting Segregation")
plt.tight_layout()
plt.savefig(f"{base_path}/figures/31_influence_mean_difference_barplot.png")
plt.close()
print("✅ Mean Difference Bar Plot saved.")

print("\n🎯 All analysis and visualization completed!")