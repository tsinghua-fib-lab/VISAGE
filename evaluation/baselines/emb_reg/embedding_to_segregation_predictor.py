"""
### 🖼️ Image Embedding Regression Baseline (Emb_Reg)

This script implements the **Emb_Reg** baseline, which predicts the income segregation index 
of various US urban census tracts by directly regressing features extracted from street-level images. 
It represents a straightforward application of deep visual features to a socioeconomic prediction task.

**The process is divided into two main parts:**

1.  **Image Embedding Extraction (Feature Engineering):**
    * We use a pre-trained **ResNet50** model, a powerful Convolutional Neural Network (CNN), 
        to extract rich, high-dimensional **visual features (embeddings)** directly from 
        street-view images associated with each census tract.
    * The embeddings from all images within a tract are **averaged** to create a single, representative 
        "Visual Embedding" for that area, summarizing its visual characteristics.
    * These high-dimensional embeddings are saved for later analysis.

2.  **Model Training and 5-Fold Cross-Validation (Prediction & Evaluation):**
    * The high-dimensional embeddings are reduced to 30 components using **Principal Component Analysis (PCA)** to mitigate the curse of dimensionality.
    * We evaluate the predictive power of these visual features on the segregation index (the target variable).
    * Three standard Machine Learning models—**Support Vector Regression (SVR)**, **K-Nearest Neighbors (KNN)**, 
        and **Multi-layer Perceptron (MLP)**—are trained and tested using robust **5-Fold Cross-Validation (CV)**.
    * Performance metrics (R², MAE, MSE) are calculated and visualized using density plots to assess the strength of the visual baseline.

This Emb_Reg script provides a direct, feature-rich benchmark for predicting complex urban socio-economic metrics from raw image data.
"""
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import setproctitle

# For analysis and cross-validation
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

setproctitle.setproctitle('train@maruolong')

## 🚀 **Part 1: Image Embedding Extraction**

# ✅ **1. Select Pre-trained Model**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)  # Using ResNet50
model = nn.Sequential(*list(model.children())[:-1])  # Remove the final classification layer (AvgPool and FC remain)
model.to(device)
model.eval()

# ✅ **2. Image Preprocessing**
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# ✅ **3. Image Folders to Process**
image_folders = [f"/data3/maruolong/VISAGE/data/raw/imagery/{city}/images" for city in [
    'Boston', 'Chicago', 'Dallas', 'Detroit', 'Los_Angeles', 'Miami', 'New_York', 'Philadelphia',
    'San_Francisco', 'Seattle', 'Washington', 'Albuquerque', 'Austin', 'Baltimore', 'Charlotte',
    'Columbus', 'Denver', 'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas',
    'Memphis', 'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland', 'San Antonio',
    'San Diego', 'San Jose', 'Tucson'
]]

tract_embeddings = {}

def extract_image_embedding(image_path):
    """Extract embedding from a single image"""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(image).squeeze().cpu().numpy()
    
    return embedding.flatten()  # Flatten the embedding

# ✅ **4. Iterate through all tracts in folders**
for image_folder in image_folders:
    for tract_id in tqdm(os.listdir(image_folder), desc=f"Processing {image_folder}"):
        tract_path = os.path.join(image_folder, tract_id)
        if not os.path.isdir(tract_path):
            continue  

        embeddings = []
        for image_file in os.listdir(tract_path):
            if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(tract_path, image_file)
                try:
                    embeddings.append(extract_image_embedding(image_path))
                except Exception as e:
                    # print(f"Error processing {image_path}: {e}")
                    continue

        if embeddings:
            # Calculate the average embedding for the tract
            tract_embeddings[tract_id] = np.mean(embeddings, axis=0).tolist() 

# ✅ **5. Save Results**
embedding_output_path = "/data3/maruolong/VISAGE/data/processed/baseline/emb/31_tract_embeddings.json"
# Ensure the directory exists before saving
os.makedirs(os.path.dirname(embedding_output_path), exist_ok=True) 
with open(embedding_output_path, "w") as f:
    json.dump(tract_embeddings, f)

print(f"✅ Embeddings saved to {embedding_output_path}")


## 🔬 **Part 2: 5-Fold Cross-Validation and Analysis**

# The embedding is already available in 'tract_embeddings' dictionary from Part 1.

# ✅ **6. Segregation Data Files**
segregation_files = [f"/data3/maruolong/VISAGE/data/raw/mobility/visit_data/{city}_2019/{city}_2019_tract_segregation.jsonl" for city in [
    'Boston', 'Chicago', 'Dallas', 'Detroit', 'Los Angeles', 'Miami', 'New York', 'Philadelphia',
    'San Francisco', 'Seattle', 'Washington', 'Albuquerque', 'Austin', 'Baltimore', 'Charlotte',
    'Columbus', 'Denver', 'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas',
    'Memphis', 'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland', 'San Antonio',
    'San Diego', 'San Jose', 'Tucson'
]]

# ✅ **7. Read and Merge Segregation Data**
segregation_data = {}
for seg_file in segregation_files:
    try:
        with open(seg_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                segregation_data[data["tract_id"]] = data["segregation"]
    except FileNotFoundError:
        # print(f"Segregation file not found: {seg_file}")
        continue

# ✅ **8. Build Dataset**
X, Y, tract_ids = [], [], []
for tract_id, embedding in tract_embeddings.items():
    if tract_id in segregation_data:
        X.append(embedding)
        Y.append(segregation_data[tract_id])
        tract_ids.append(tract_id)

X = np.array(X)
Y = np.array(Y)

print(f"🔹 Sample size: {len(X)}, Original embedding dimension: {X.shape[1]}")

# ✅ **9. PCA Dimensionality Reduction**
pca = PCA(n_components=30)
X_reduced = pca.fit_transform(X)
print(f"🔹 Dimension after PCA: {X_reduced.shape[1]}")

# ✅ **10. Define Models**
models = {
    "SVR": SVR(kernel="rbf", C=1, gamma=0.01, epsilon=0.1),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "MLP": MLPRegressor(hidden_layer_sizes=(100, 50), activation="relu", solver="adam", max_iter=500, random_state=42),
}

# ✅ **11. Cross-Validation**
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    print(f"\n🚀 Starting 5-Fold CV for {model_name}...")
    fold = 1
    all_Y_true, all_Y_pred = [], []
    for train_idx, test_idx in kf.split(X_reduced):
        X_train, X_test = X_reduced[train_idx], X_reduced[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        all_Y_true.extend(Y_test)
        all_Y_pred.extend(Y_pred)

        print(f"Fold {fold}:")
        print(f"  R²:   {r2_score(Y_test, Y_pred):.4f}")
        print(f"  MAE:  {mean_absolute_error(Y_test, Y_pred):.4f}")
        print(f"  MSE:  {mean_squared_error(Y_test, Y_pred):.4f}")
        fold += 1
    
    # Calculate and print overall CV metrics
    overall_r2 = r2_score(all_Y_true, all_Y_pred)
    overall_mae = mean_absolute_error(all_Y_true, all_Y_pred)
    overall_mse = mean_squared_error(all_Y_true, all_Y_pred)
    print(f"\nOverall 5-Fold CV Results for {model_name}:")
    print(f"  Overall R²: {overall_r2:.4f}")
    print(f"  Overall MAE: {overall_mae:.4f}")
    print(f"  Overall MSE: {overall_mse:.4f}")


    # ✅ **12. Visualization (Density Scatter Plot)**
    def plot_density_scatter(Y_true, Y_pred, title, filename):
        cmap = LinearSegmentedColormap.from_list("teal_shade", ["#a8e6cf", "#56c8d8", "#007c91"])
        plt.figure(figsize=(7, 7))
        plt.hexbin(Y_true, Y_pred, gridsize=40, cmap=cmap, bins='log', linewidths=0)
        plt.xlabel("True Segregation", fontsize=18)
        plt.ylabel("Predicted Segregation", fontsize=18)
        plt.title(title, fontsize=20)
        plt.axline([0, 0], slope=1, color="red", linestyle="--", linewidth=1.5)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    
    output_path = f"/data3/maruolong/VISAGE/data/processed/baseline/emb/predict_result_density_{model_name}.png"
    plot_density_scatter(np.array(all_Y_true), np.array(all_Y_pred),
                          f"{model_name} CV Segregation Prediction (R²={overall_r2:.4f})", output_path)
    print(f"📊 Plot saved to: {output_path}")

print("\n✅ Embedding extraction, cross-validation, and analysis completed!")