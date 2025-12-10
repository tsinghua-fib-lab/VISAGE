"""
### 📝 Description Frequency Regression Baseline (Desc_Reg)

This script establishes a baseline (Desc_Reg) for predicting urban income segregation scores 
using the **frequency of visual elements derived from image captions**. This method serves 
as a proxy for analyzing urban form and characteristics based on descriptive text extracted 
from street-level and remote sensing imagery.

**The process involves:**

1.  **Feature Compilation:** Reading and merging keyword/element frequencies (captions) 
    extracted from both street-view and remote sensing imagery for each census tract.
2.  **Dataset Construction:** Combining these element frequencies into a single feature vector (X) 
    and linking them with the corresponding segregation score (Y).
3.  **Cross-Validation & Regression:** Standardizing features and evaluating multiple regression models 
    (XGBoost, ElasticNet, SVR, KNN) using a robust **5-Fold Cross-Validation (CV)** approach.
4.  **Evaluation:** Reporting aggregated performance metrics (R², MAE, MSE) and generating 
    density plots for visualization.

This baseline assesses the predictive power of textual descriptions of the visual environment 
on socioeconomic metrics.
---
**⚠️ PREREQUISITE NOTE:**

To run this baseline, the feature files must be pre-generated using your large language model (LLM) 
cue extraction method. The required input files are:
* `street_caption_file = "/data3/maruolong/VISAGE/data/processed/cue_frequencies/aggregated_description_by_tract.jsonl"`
* `rs_caption_file = "/data3/maruolong/VISAGE/data/processed/cue_frequencies/rs_aggregated_description_by_tract.jsonl"`
---
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

# --- Data Paths ---
street_caption_file = "/data3/maruolong/VISAGE/data/processed/cue_frequencies/aggregated_description_by_tract.jsonl"
rs_caption_file = "/data3/maruolong/VISAGE/data/processed/cue_frequencies/rs_aggregated_description_by_tract.jsonl"
output_dir = "/data3/maruolong/VISAGE/data/processed/baseline/desc"
os.makedirs(output_dir, exist_ok=True)

# Segregation Data Files
segregation_files = [f"/data3/maruolong/VISAGE/data/raw/mobility/{city}_2019/{city}_2019_tract_segregation.jsonl"
                     for city in ['Boston', 'Chicago', 'Dallas', 'Detroit', 'Los Angeles', 'Miami', 'New York', 'Philadelphia',
                                  'San Francisco', 'Seattle', 'Washington', 'Albuquerque', 'Austin', 'Baltimore', 'Charlotte',
                                  'Columbus', 'Denver', 'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas',
                                  'Memphis', 'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland', 'San Antonio',
                                  'San Diego', 'San Jose', 'Tucson']]

# --- Data Loading Functions ---
def load_caption_file(file_path):
    """Loads a caption file (JSONL format) into a dictionary mapping tract_id to description features."""
    data = {}
    with open(file_path, "r") as f:
        for line in f:
            try:
                d = json.loads(line)
                data[d["tract_id"]] = d["description"]
            except json.JSONDecodeError:
                # Handle corrupted lines gracefully
                continue
    return data

# Load caption data
street_desc = load_caption_file(street_caption_file)
rs_desc = load_caption_file(rs_caption_file)

# --- Feature Preparation ---

# Find common tract IDs
common_tracts = set(street_desc.keys()) & set(rs_desc.keys())

# Merge descriptions
desc_by_tract = {}
for tract_id in common_tracts:
    # Merge dictionaries, remote sensing features overwrite street features if keys overlap (use {**d1, **d2} for merge)
    merged = {**street_desc.get(tract_id, {}), **rs_desc.get(tract_id, {})}
    desc_by_tract[tract_id] = merged

# Load segregation values
segregation_data = {}
for file in segregation_files:
    try:
        with open(file, "r") as f:
            for line in f:
                d = json.loads(line)
                segregation_data[d["tract_id"]] = d["segregation"]
    except FileNotFoundError:
        continue

# Build feature matrix (X) and target vector (Y)
X, Y = [], []
# Identify all possible keys across all tracts
all_keys = sorted({key for desc in desc_by_tract.values() for key in desc}) 

for tract_id in desc_by_tract:
    if tract_id in segregation_data:
        features = desc_by_tract[tract_id]
        # Create feature vector, filling missing keys with 0.0
        vec = [features.get(k, 0.0) for k in all_keys] 
        X.append(vec)
        Y.append(segregation_data[tract_id])

X = np.array(X)
Y = np.array(Y)

print(f"✅ Sample count: {len(X)}, Merged feature count: {X.shape[1]}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Regression Models and 5-Fold Cross-Validation ---

# Regression model dictionary
models = {
    "XGBoost": xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    "ElasticNet": ElasticNet(alpha=0.05, random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(n_neighbors=5),
}

# 5-Fold Cross-Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def plot_density_scatter(Y_true, Y_pred, model_name, r2_score_val, base_path):
    """Generates a density scatter plot for True vs Predicted values."""
    cmap = LinearSegmentedColormap.from_list("teal_shade", ["#a8e6cf", "#56c8d8", "#007c91"])
    plt.figure(figsize=(7, 7))
    plt.hexbin(Y_true, Y_pred, gridsize=40, cmap=cmap, bins='log', linewidths=0)
    cb = plt.colorbar(label='Count in log scale')
    
    plt.xlabel("True Segregation", fontsize=14)
    plt.ylabel("Predicted Segregation", fontsize=14)
    plt.title(f"{model_name} 5-Fold CV Prediction (R²={r2_score_val:.4f})", fontsize=16)
    plt.axline([0, 0], slope=1, color="red", linestyle="--", linewidth=1.5)
    plt.tight_layout()
    
    output_dir = os.path.join(base_path, "figures")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"31_CV_density_{model_name.lower().replace(' ', '')}.png")
    
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"📊 Plot saved to: {filename}")


# Run Cross-Validation for all models
for model_name, model in models.items():
    print(f"\n🚀 Starting 5-Fold CV for {model_name}...")
    fold = 1
    all_Y_true, all_Y_pred = [], []
    
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        all_Y_true.extend(Y_test)
        all_Y_pred.extend(Y_pred)

        print(f"Fold {fold}: R²: {r2_score(Y_test, Y_pred):.4f}, MAE: {mean_absolute_error(Y_test, Y_pred):.4f}")
        fold += 1
    
    # Calculate overall CV metrics
    overall_r2 = r2_score(all_Y_true, all_Y_pred)
    overall_mae = mean_absolute_error(all_Y_true, all_Y_pred)
    overall_mse = mean_squared_error(all_Y_true, all_Y_pred)

    print(f"\n📈 Overall 5-Fold CV Results for {model_name}:")
    print(f"R²: {overall_r2:.4f}")
    print(f"MAE: {overall_mae:.4f}")
    print(f"MSE: {overall_mse:.4f}")
    
    # Generate density plot
    plot_density_scatter(np.array(all_Y_true), np.array(all_Y_pred), model_name, overall_r2, os.path.dirname(output_dir))

print("\n✅ Regression analysis and visualization completed for all models!")