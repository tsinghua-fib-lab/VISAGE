import json
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from glob import glob

# ✅ 1. 加载五个预测文件
prediction_files = glob("/data3/maruolong/VISAGE/data/31_cities/data*/svr_prediction_result.jsonl")
prediction_results = {}

for file_path in prediction_files:
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            tract_id = item["tract_id"]
            pred_value = item.get("predicted_segregation", None)
            if tract_id not in prediction_results and pred_value is not None:
                prediction_results[tract_id] = pred_value

print(f"✅ 已加载预测 tract 数: {len(prediction_results)}")

# ✅ 2. 加载所有城市的真实 segregation 值
segregation_files = [f"/data3/maruolong/segregation/All_time/visit_data/{city}_2019/{city}_2019_tract_segregation.jsonl" for city in [
    'Boston', 'Chicago', 'Dallas', 'Detroit', 'Los Angeles', 'Miami', 'New York', 'Philadelphia',
    'San Francisco', 'Seattle', 'Washington', 'Albuquerque', 'Austin', 'Baltimore', 'Charlotte',
    'Columbus', 'Denver', 'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas',
    'Memphis', 'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland', 'San Antonio',
    'San Diego', 'San Jose', 'Tucson'
]]

true_segregation = {}
for seg_file in segregation_files:
    with open(seg_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            true_segregation[data["tract_id"]] = data["segregation"]

print(f"✅ 已加载真实 segregation tract 数: {len(true_segregation)}")

# ✅ 3. 匹配 tract_id，构造真值与预测值列表
Y_true, Y_pred = [], []
for tract_id, pred in prediction_results.items():
    true_val = true_segregation.get(tract_id, None)
    if true_val is not None:
        Y_true.append(true_val)
        Y_pred.append(pred)

print(f"🔹 可用于绘图的匹配样本数: {len(Y_true)}")

# ✅ 4. 绘图函数
def plot_density_scatter(Y_true, Y_pred, title, filename):
    cmap = LinearSegmentedColormap.from_list("teal_shade", ["#a8e6cf", "#56c8d8", "#007c91"])
    plt.figure(figsize=(7, 7))
    plt.hexbin(Y_true, Y_pred, gridsize=40, cmap=cmap, bins='log', linewidths=0)
    plt.xlabel("True Segregation", fontsize=18)
    plt.ylabel("Predicted Segregation", fontsize=18)
    plt.title(title, fontsize=20)
    plt.axline([0, 0], slope=1, color="red", linestyle="--", linewidth=1.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()

# ✅ 5. 执行绘图
plot_density_scatter(
    Y_true, Y_pred,
    "UI-CoT-2 Income Segregation Prediction",
    "/data3/maruolong/VISAGE/data/31_cities/svr_5fold_prediction_result_density.pdf"
)
print("📊 图像保存成功！")
