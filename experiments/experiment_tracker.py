import os
import json
import uuid
import numpy as np
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List

# Try importing yaml for config saving; fallback to JSON if not installed
try:
    import yaml
except ImportError:
    yaml = None

# Default paths
DEFAULT_EXP_DIR = "/data3/maruolong/VISAGE/experiments"
DEFAULT_LOG_FILE = "visage_experiment_log.json"

class ExperimentTracker:
    """
    VISAGE Experiment Tracker (Structured Version).
    
    Features:
    - Master Log: Updates a single JSON file for quick comparison of all runs.
    - Run Directories: Creates structured folders 'runs/run_{id}/' containing:
        - config.yaml: The experiment configuration.
        - metrics.json: Detailed results including per-fold data.
        - artifacts/: Folder for outputs like plots and prediction files.
    """

    def __init__(self, root_dir: str = DEFAULT_EXP_DIR):
        """
        Initialize the Tracker.
        
        Parameters:
            root_dir: The root folder for experiments (e.g., /data3/.../experiments)
        """
        self.root_dir = root_dir
        self.runs_dir = os.path.join(root_dir, "runs")
        self.master_log_path = os.path.join(root_dir, DEFAULT_LOG_FILE)
        
        os.makedirs(self.runs_dir, exist_ok=True)
        print(f"🔬 Experiment Tracker Initialized.")
        print(f"   📂 Root: {self.root_dir}")
        print(f"   📜 Master Log: {self.master_log_path}")

    def _load_master_log(self) -> Dict[str, Any]:
        """Load the master log file from disk."""
        if not os.path.exists(self.master_log_path):
            return {}
        try:
            with open(self.master_log_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading master log: {e}")
            return {}

    def _save_master_log(self, logs: Dict[str, Any]):
        """Save the master log file to disk."""
        with open(self.master_log_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)

    def start_experiment(self, base_config: Dict[str, Any]) -> tuple[str, str]:
        """
        Start a new experiment run.
        
        Returns:
            experiment_id (str): Unique ID for the run.
            run_dir (str): Absolute path to the newly created run folder.
        """
        # 1. Generate Unique ID based on timestamp and UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = str(uuid.uuid4())[:8]
        experiment_id = f"EXP_{timestamp}_{unique_suffix}"
        
        # 2. Create Directory Structure
        # Pattern: experiments/runs/run_EXP_2025.../
        run_dir_name = f"run_{experiment_id}"
        self.current_run_dir = os.path.join(self.runs_dir, run_dir_name)
        self.current_artifacts_dir = os.path.join(self.current_run_dir, "artifacts")
        
        os.makedirs(self.current_artifacts_dir, exist_ok=True)

        # 3. Save Configuration (config.yaml)
        config_path = os.path.join(self.current_run_dir, "config.yaml")
        self._save_yaml(base_config, config_path)

        # 4. Update Master Log (Metadata only)
        metadata = {
            "experiment_id": experiment_id,
            "start_time": datetime.now().isoformat(),
            "status": "Running",
            "run_dir": self.current_run_dir,
            "config": base_config,
            "summary_metrics": {},
            "best_model_info": {}
        }
        logs = self._load_master_log()
        logs[experiment_id] = metadata
        self._save_master_log(logs)

        print(f"\n✨ Started Experiment: **{experiment_id}**")
        print(f"   📂 Run Directory: {self.current_run_dir}")
        
        return experiment_id, self.current_run_dir

    def log_results(self, experiment_id: str, detailed_results: Dict[str, Any], summary_metrics: Dict[str, Any], best_model_info: Dict[str, Any]):
        """
        Log results to both the master JSON and the specific run folder.
        """
        # 1. Update Master Log (Summary only, to keep it lightweight)
        logs = self._load_master_log()
        if experiment_id in logs:
            logs[experiment_id]['status'] = "Completed"
            logs[experiment_id]['end_time'] = datetime.now().isoformat()
            logs[experiment_id]['summary_metrics'] = summary_metrics
            logs[experiment_id]['best_model_info'] = best_model_info
            self._save_master_log(logs)

        # 2. Save Full Metrics to Run Folder (metrics.json)
        # Determine run directory (use current memory or fallback to master log record)
        run_dir = getattr(self, 'current_run_dir', None)
        if not run_dir or experiment_id not in run_dir:
            run_dir = logs.get(experiment_id, {}).get('run_dir')

        if run_dir and os.path.exists(run_dir):
            metrics_path = os.path.join(run_dir, "metrics.json")
            full_metrics_data = {
                "experiment_id": experiment_id,
                "end_time": datetime.now().isoformat(),
                "best_model": best_model_info,
                "summary": summary_metrics,
                "detailed_folds": detailed_results # Save heavy per-fold data here
            }
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(full_metrics_data, f, indent=4, ensure_ascii=False)
            print(f"   💾 Metrics saved to: {metrics_path}")
        else:
            print(f"⚠️ Could not find run directory for {experiment_id}, metrics.json not saved.")

        print(f"🎉 Results logged for **{experiment_id}**.")
        print(f"   🏆 Best Model: {best_model_info.get('model_name')} (R2: {best_model_info.get('r2_mean'):.4f})")

    def _save_yaml(self, data: Dict, path: str):
        """Helper to save YAML. Falls back to JSON-formatted text if PyYAML is missing."""
        try:
            if yaml:
                with open(path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            else:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print("⚠️ PyYAML not found, saved config as JSON-formatted YAML.")
        except Exception as e:
            print(f"⚠️ Error saving config: {e}")

    def compare_experiments(self, experiment_ids: Optional[List[str]] = None):
        """
        Print a comparison table of SVR and KNN performance.
        """
        logs = self._load_master_log()
        
        table_data = []
        
        for exp_id, record in logs.items():
            # Filter logic
            if record.get('status') != 'Completed':
                continue
            if experiment_ids and exp_id not in experiment_ids:
                continue

            summary = record.get('summary_metrics', {})
            config = record.get('config', {})
            
            # Extract key metrics with defaults
            row = {
                "ID": exp_id,
                "Date": record.get('start_time', '').split('T')[0],
                "Data": config.get('base_data_dir', 'N/A').split('/')[-1], # Short path
                
                # SVR Metrics
                "SVR R2": summary.get("Test SVR R2", float('nan')),
                "SVR RMSE": summary.get("Test SVR RMSE", float('nan')),
                
                # KNN Metrics
                "KNN R2": summary.get("Test KNN R2", float('nan')),
                "KNN RMSE": summary.get("Test KNN RMSE", float('nan'))
            }
            table_data.append(row)

        if not table_data:
            print("❌ No completed experiments to compare.")
            return

        print(f"\n📊 **Experiment Comparison (SVR vs KNN)**")
        print("-" * 85)
        
        # Formatting for the table
        header = f"| {'ID':<22} | {'Data':<15} | {'SVR R2':<8} | {'SVR RMSE':<8} | {'KNN R2':<8} | {'KNN RMSE':<8} |"
        print(header)
        print("|" + "-"*24 + "|" + "-"*17 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|")

        # Find best SVR R2 to highlight
        best_svr_r2 = -float('inf')
        for r in table_data:
            val = r["SVR R2"]
            if not np.isnan(val) and val > best_svr_r2:
                best_svr_r2 = val

        for row in table_data:
            # Format numbers
            svr_r2_str = f"{row['SVR R2']:.4f}" if not np.isnan(row['SVR R2']) else "N/A"
            svr_rmse_str = f"{row['SVR RMSE']:.4f}" if not np.isnan(row['SVR RMSE']) else "N/A"
            knn_r2_str = f"{row['KNN R2']:.4f}" if not np.isnan(row['KNN R2']) else "N/A"
            knn_rmse_str = f"{row['KNN RMSE']:.4f}" if not np.isnan(row['KNN RMSE']) else "N/A"
            
            # Simple highlight logic (add * if best)
            if row['SVR R2'] == best_svr_r2 and best_svr_r2 > -float('inf'):
                svr_r2_str = f"*{svr_r2_str}*"

            print(f"| {row['ID']:<22} | {row['Data']:<15} | {svr_r2_str:<8} | {svr_rmse_str:<8} | {knn_r2_str:<8} | {knn_rmse_str:<8} |")
        
        print("-" * 85)
        print("* denotes best SVR R2 performance.")

# Helper for calculation
def calculate_aggregated_metric(metrics_list: List[Dict[str, Any]], metric_key: str) -> Optional[float]:
    """Calculate mean of a metric across folds."""
    vals = [m.get(metric_key) for m in metrics_list if m.get(metric_key) is not None]
    vals = [x for x in vals if np.isfinite(x)]
    if vals:
        return sum(vals) / len(vals)
    return None