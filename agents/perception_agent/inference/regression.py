import pickle
import json
import os
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# ✳️ 1️⃣ 加载两个 embedding .pkl 文件（train & test）
train_pkl_path = '/data3/maruolong/VISAGE/data/31_cities/data2/feature_vector_segregation_train.pkl'
test_pkl_path = '/data3/maruolong/VISAGE/data/31_cities/data2/feature_vector_segregation_test.pkl'

with open(train_pkl_path, 'rb') as f:
    train_embeddings = pickle.load(f)
with open(test_pkl_path, 'rb') as f:
    test_embeddings = pickle.load(f)

print(f"✅ Loaded {len(train_embeddings)} training tract embeddings.")
print(f"✅ Loaded {len(test_embeddings)} testing tract embeddings.")

# 2️⃣ 加载 segregation 真值
segregation_jsonl_paths = [f"/data3/maruolong/segregation/All_time/visit_data/{city}_2019/{city}_2019_tract_segregation.jsonl" for city in [
    'Boston', 'Chicago', 'Dallas', 'Detroit', 'Los Angeles', 'Miami', 'New York', 'Philadelphia',
    'San Francisco', 'Seattle', 'Washington', 'Albuquerque', 'Austin', 'Baltimore', 'Charlotte',
    'Columbus', 'Denver', 'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas',
    'Memphis', 'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland', 'San Antonio',
    'San Diego', 'San Jose', 'Tucson'
]]


segregation_data = {}
for path in segregation_jsonl_paths:
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            tract_id = str(item['tract_id'])
            segregation_value = item.get('segregation', None)
            if segregation_value is not None:
                segregation_data[tract_id] = segregation_value

print(f"✅ Loaded segregation truth for {len(segregation_data)} tracts.")

# 3️⃣ 构建训练集与测试集
train_tracts = [tid for tid in train_embeddings if tid in segregation_data]
test_tracts = [tid for tid in test_embeddings if tid in segregation_data]

X_train = np.array([train_embeddings[tid] for tid in train_tracts])
Y_train = np.array([segregation_data[tid] for tid in train_tracts])
X_test = np.array([test_embeddings[tid] for tid in test_tracts])
Y_test = np.array([segregation_data[tid] for tid in test_tracts])

print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}, Embedding dim: {X_train.shape[1]}")

# 4️⃣ PCA降维
pca = PCA(n_components=120)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"✅ PCA dim: {X_train_pca.shape[1]}")

# 🎯 模型评估函数
def evaluate_model(name, Y_true, Y_pred):
    return {
        f"{name} R²": r2_score(Y_true, Y_pred),
        f"{name} MAE": mean_absolute_error(Y_true, Y_pred),
        f"{name} MSE": mean_squared_error(Y_true, Y_pred)
    }

def print_metrics(metrics_dict):
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}")

# 5️⃣ SVR（GridSearchCV）
svr_param_grid = {
    'C': [0.1, 0.5, 1.0],
    'gamma': ['scale', 'auto', 0.01, 0.02, 0.05],
    'epsilon': [0.020, 0.040, 0.070]
}

svr_grid = GridSearchCV(SVR(kernel='rbf'), svr_param_grid, cv=5, n_jobs=-1, scoring='r2', verbose=1)
svr_grid.fit(X_train_pca, Y_train)

svr_best = svr_grid.best_estimator_
svr_metrics = {**evaluate_model("Train SVR", Y_train, svr_best.predict(X_train_pca)),
               **evaluate_model("Test SVR", Y_test, svr_best.predict(X_test_pca))}

print("\n🔹 SVR Results:")
print(f"Best Params: {svr_grid.best_params_}")
print_metrics(svr_metrics)

# ✅ 保存 SVR 模型在测试集上的预测结果为 jsonl 文件
output_jsonl_path = "/data3/maruolong/VISAGE/data/31_cities/data2/svr_prediction_result.jsonl"

Y_test_pred = svr_best.predict(X_test_pca)

with open(output_jsonl_path, 'w') as f:
    for tract_id, true_val, pred_val in zip(test_tracts, Y_test, Y_test_pred):
        json_line = {
            "tract_id": tract_id,
            "true_segregation": float(true_val),
            "predicted_segregation": float(pred_val)
        }
        f.write(json.dumps(json_line) + '\n')

print(f"✅ Saved SVR predictions to {output_jsonl_path}")

# 6️⃣ KNN（GridSearchCV）
knn_param_grid = {
    'n_neighbors': [7, 9, 10, 12],
    'weights': ['uniform', 'distance']
}

knn_grid = GridSearchCV(KNeighborsRegressor(), knn_param_grid, cv=5, n_jobs=-1, scoring='r2', verbose=1)
knn_grid.fit(X_train_pca, Y_train)

knn_best = knn_grid.best_estimator_
knn_metrics = {**evaluate_model("Train KNN", Y_train, knn_best.predict(X_train_pca)),
               **evaluate_model("Test KNN", Y_test, knn_best.predict(X_test_pca))}

print("\n🔹 KNN Results:")
print(f"Best Params: {knn_grid.best_params_}")
print_metrics(knn_metrics)

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

def plot_density_scatter(Y_true, Y_pred, title, filename):
    # 设置全局字体为 Times New Roman
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 16  # 全局字号

    # 自定义浅青色到深青色的渐变色图
    colors = ["#a8e6cf", "#56c8d8", "#007c91"]  # 浅青 -> 中青 -> 深青
    cmap = LinearSegmentedColormap.from_list("teal_shade", colors)

    plt.figure(figsize=(7, 7))
    plt.hexbin(Y_true, Y_pred, gridsize=40, cmap=cmap, bins='log', linewidths=0)

    # 坐标轴与标题样式
    plt.xlabel("True Segregation", fontsize=18)
    plt.ylabel("Predicted Segregation", fontsize=18)
    plt.title(title, fontsize=20)

    # 对角线（完美预测线）
    plt.axline([0, 0], slope=1, color="red", linestyle="--", linewidth=1.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# 可视化部分
# plot_density_scatter(Y_test, svr_best.predict(X_test_pca), 
#                      "UI-CoT-2 Income Segregation Prediction", 
#                      "/data3/maruolong/VISAGE/data/31_cities/data2/svr_result_density.pdf")

# plot_density_scatter(Y_test, knn_best.predict(X_test_pca), 
#                      "UI-CoT-2 Income Segregation Prediction", 
#                      "/data3/maruolong/VISAGE/data/31_cities/data2/knn_result_density.png")