
import os
import json
import pickle
import argparse
import numpy as np
from glob import glob
from typing import List, Optional

from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -------------------------------
# Default city list and helpers
# -------------------------------
default_cities = [
    'Boston', 'Chicago', 'Dallas', 'Detroit', 'Los Angeles', 'Miami',
    'New York', 'Philadelphia', 'San Francisco', 'Seattle', 'Washington',
    'Albuquerque', 'Austin', 'Baltimore', 'Charlotte', 'Columbus', 'Denver',
    'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas',
    'Memphis', 'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland',
    'San Antonio', 'San Diego', 'San Jose', 'Tucson'
]

def default_segregation_paths(cities: List[str] = None) -> List[str]:
    """Construct default segregation jsonl paths for given city list."""
    cities = cities or default_cities
    paths = [
        f"/data3/maruolong/VISAGE/data/mobility/visit_data/{city}_2019/{city}_2019_tract_segregation.jsonl"
        for city in cities
    ]
    return paths

# -------------------------------
# Utility Functions
# -------------------------------

def evaluate_model(name: str, Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    """Compute R2, MAE, MSE for given true and predicted values."""
    return {
        f"{name} R2": float(r2_score(Y_true, Y_pred)) if len(Y_true) > 0 else float("nan"),
        f"{name} MAE": float(mean_absolute_error(Y_true, Y_pred)) if len(Y_true) > 0 else float("nan"),
        f"{name} MSE": float(mean_squared_error(Y_true, Y_pred)) if len(Y_true) > 0 else float("nan")
    }

def plot_density_scatter(Y_true, Y_pred, title: str, filename: str):
    """Plot a hexbin density scatter (true vs predicted) and save to file."""
    cmap = LinearSegmentedColormap.from_list("teal_shade", ["#a8e6cf", "#56c8d8", "#007c91"])
    plt.figure(figsize=(7, 7))
    plt.hexbin(Y_true, Y_pred, gridsize=40, cmap=cmap, bins='log', linewidths=0)
    plt.xlabel("True Segregation", fontsize=14)
    plt.ylabel("Predicted Segregation", fontsize=14)
    plt.title(title, fontsize=16)
    # Perfect prediction line
    plt.axline([0, 0], slope=1, color="red", linestyle="--", linewidth=1.2)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# -------------------------------
# Main K-Fold Regression Routine
# -------------------------------

def run_kfold_regression(
    base_data_dir: str = "/data3/maruolong/VISAGE/data/processed/baseline/vlm/cross_train/31",
    num_folds: int = 5,
    output_dir: str = "/data3/maruolong/VISAGE/data/processed/baseline/vlm/regression_viz",
    segregation_jsonl_paths: Optional[List[str]] = None,
    cities: Optional[List[str]] = None,
    pca_n_components: int = 30,
    verbose: bool = True
) -> dict:
    """
    Run PCA -> SVR & KNN on each fold, save per-fold predictions,
    create combined visualization, and save aggregated metrics.

    Parameters:
        base_data_dir: directory containing fold subfolders data1, data2, ...
        num_folds: number of folds (k)
        output_dir: base directory to save outputs (predictions, plots, metrics)
        segregation_jsonl_paths: list of jsonl files with ground-truth segregation
        cities: optional list of city names to derive default segregation paths
        pca_n_components: maximum PCA components (will be clipped by embedding dim)
        verbose: whether to print progress

    Returns:
        A dictionary containing aggregated metrics for SVR and KNN across folds.
    """
    os.makedirs(output_dir, exist_ok=True)

    if segregation_jsonl_paths is None:
        segregation_jsonl_paths = default_segregation_paths(cities)

    # Load segregation ground truth
    segregation_data = {}
    for path in segregation_jsonl_paths:
        if not os.path.exists(path):
            if verbose:
                print(f"⚠️ Segregation path not found (skipping): {path}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                tid = str(item.get("tract_id") or item.get("TRACTID") or item.get("tract"))
                seg = item.get("segregation") if isinstance(item, dict) else None
                if tid and seg is not None:
                    segregation_data[tid] = seg

    if verbose:
        print(f"✅ Loaded segregation ground truth for {len(segregation_data)} tracts across provided files.")

    all_svr_metrics = []
    all_knn_metrics = []
    successful_folds = 0

    for fold in range(1, num_folds + 1):
        if verbose:
            print(f"\n🔹 Processing fold {fold}/{num_folds} ...")

        dataset_name = f"data{fold}"
        train_path = os.path.join(base_data_dir, dataset_name, "feature_vector_segregation_train.pkl")
        test_path = os.path.join(base_data_dir, dataset_name, "feature_vector_segregation_test.pkl")

        missing = []
        if not os.path.exists(train_path):
            missing.append(train_path)
        if not os.path.exists(test_path):
            missing.append(test_path)
        if missing:
            print(f"⚠️ Fold {fold} missing files, skipping this fold: {missing}")
            continue

        # Load embeddings (dict: tract_id -> vector)
        with open(train_path, 'rb') as f:
            train_emb = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_emb = pickle.load(f)

        # Filter tracts that have segregation ground truth
        train_tracts = [tid for tid in train_emb.keys() if tid in segregation_data]
        test_tracts = [tid for tid in test_emb.keys() if tid in segregation_data]

        if len(train_tracts) == 0 or len(test_tracts) == 0:
            print(f"⚠️ No overlapping tracts with ground truth in fold {fold}, skipping.")
            continue

        X_train = np.array([train_emb[tid] for tid in train_tracts])
        Y_train = np.array([segregation_data[tid] for tid in train_tracts])
        X_test = np.array([test_emb[tid] for tid in test_tracts])
        Y_test = np.array([segregation_data[tid] for tid in test_tracts])

        if verbose:
            print(f"   Train samples: {len(X_train)}, Test samples: {len(X_test)}, Embedding dim: {X_train.shape[1]}")

        # PCA (clip components)
        n_comp = min(pca_n_components, X_train.shape[1])
        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        if verbose:
            print(f"   PCA reduced dimension -> {X_train_pca.shape[1]}")

        # ----- SVR (GridSearchCV)
        svr_params = {
            "C": [0.1, 0.5, 1.0],
            "gamma": ["scale", "auto", 0.01, 0.02, 0.05],
            "epsilon": [0.02, 0.04, 0.07]
        }
        svr_grid = GridSearchCV(
            SVR(kernel='rbf'),
            svr_params,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        svr_grid.fit(X_train_pca, Y_train)
        svr_best = svr_grid.best_estimator_
        svr_pred = svr_best.predict(X_test_pca)
        svr_metrics = {
            **evaluate_model("Train SVR", Y_train, svr_best.predict(X_train_pca)),
            **evaluate_model("Test SVR", Y_test, svr_pred),
            "SVR Best Params": svr_grid.best_params_
        }
        all_svr_metrics.append(svr_metrics)
        if verbose:
            print("   SVR best params:", svr_grid.best_params_)

        # Save per-fold SVR predictions
        fold_pred_path = os.path.join(output_dir, f"fold{fold}_svr_prediction.jsonl")
        with open(fold_pred_path, 'w', encoding='utf-8') as f:
            for tid, t, p in zip(test_tracts, Y_test, svr_pred):
                f.write(json.dumps({
                    "tract_id": tid,
                    "true_segregation": float(t),
                    "predicted_segregation": float(p)
                }) + "\n")
        if verbose:
            print(f"   Saved SVR predictions -> {fold_pred_path}")

        # ----- KNN (GridSearchCV)
        knn_params = {
            "n_neighbors": [7, 9, 10, 12],
            "weights": ["uniform", "distance"]
        }
        knn_grid = GridSearchCV(
            KNeighborsRegressor(),
            knn_params,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        knn_grid.fit(X_train_pca, Y_train)
        knn_best = knn_grid.best_estimator_
        knn_pred = knn_best.predict(X_test_pca)
        knn_metrics = {
            **evaluate_model("Train KNN", Y_train, knn_best.predict(X_train_pca)),
            **evaluate_model("Test KNN", Y_test, knn_pred),
            "KNN Best Params": knn_grid.best_params_
        }
        all_knn_metrics.append(knn_metrics)
        if verbose:
            print("   KNN best params:", knn_grid.best_params_)

        successful_folds += 1

    # ---------------------- Combined Visualization
    pred_files = sorted(glob(os.path.join(output_dir, "fold*_svr_prediction.jsonl")))
    combined_true = []
    combined_pred = []
    for pf in pred_files:
        with open(pf, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                tid = obj.get("tract_id")
                pred_val = obj.get("predicted_segregation")
                if tid in segregation_data and pred_val is not None:
                    combined_true.append(segregation_data[tid])
                    combined_pred.append(pred_val)

    if combined_true and combined_pred:
        density_path = os.path.join(output_dir, "svr_5fold_density.pdf")
        plot_density_scatter(combined_true, combined_pred, "SVR 5-Fold Segregation Prediction-VLM Version", density_path)
        if verbose:
            print(f"✅ Saved combined density scatter -> {density_path}")
    else:
        if verbose:
            print("⚠️ No combined predictions found to plot.")

    # ---------------------- Save aggregated metrics
    out_metrics = {
        "successful_folds": successful_folds,
        "SVR_per_fold": all_svr_metrics,
        "KNN_per_fold": all_knn_metrics
    }
    metrics_path = os.path.join(output_dir, "kfold_regression_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(out_metrics, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"✅ Saved aggregated metrics -> {metrics_path}")

    return out_metrics

# -------------------------------
# Command-line interface
# -------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(description="K-fold Regression + Visualization for VISAGE embeddings")
    parser.add_argument("--base-data-dir", type=str, default="/data3/maruolong/VISAGE/data/processed/baseline/vlm/cross_train/31",
                        help="Base directory containing fold subfolders (data1, data2, ...)")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of folds (k)")
    parser.add_argument("--output-dir", type=str, default="/data3/maruolong/VISAGE/data/processed/baseline/vlm/regression_viz",
                        help="Base directory to save predictions, plots and metrics")
    parser.add_argument("--seg-paths", type=str, default=None,
                        help="Comma-separated list of segregation jsonl files. If not provided, defaults are used.")
    parser.add_argument("--cities", type=str, default=None,
                        help="Optional comma-separated city names to construct default segregation paths (overrides built-in list).")
    parser.add_argument("--pca-n", type=int, default=120, help="Max PCA components")
    parser.add_argument("--no-verbose", action="store_true", help="Disable verbose printing")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    if args.seg_paths:
        seg_paths = [p.strip() for p in args.seg_paths.split(",") if p.strip()]
    else:
        cities = [c.strip() for c in args.cities.split(",")] if args.cities else None
        seg_paths = default_segregation_paths(cities)

    metrics = run_kfold_regression(
        base_data_dir=args.base_data_dir,
        num_folds=args.num_folds,
        output_dir=args.output_dir,
        segregation_jsonl_paths=seg_paths,
        cities=None,
        pca_n_components=args.pca_n,
        verbose=not args.no_verbose
    )

    print("\nSummary (aggregated):")
    print(json.dumps(metrics, indent=2))
