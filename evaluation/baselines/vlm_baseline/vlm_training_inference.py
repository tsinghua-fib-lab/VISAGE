
import json
import os
import sys
import subprocess
import argparse
import glob
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import time

# Ensure the necessary path is available for internal imports
sys.path.append("/data3/maruolong/VISAGE/agents/perception_agent/training")

class VISAGETrainingInference:
    """Integrated training and inference pipeline for VISAGE"""
    
    def __init__(self, 
                 base_output_dir: str = "/data3/maruolong/VISAGE/data/processed/baseline/vlm/cross_train/31",
                 base_data_dir: str = "/data3/maruolong/VISAGE/data/processed/training_data/31_cities",
                 initial_model: str = "/data2/zhangxin/model_zoo/Llama-3-VILA1.5-8B",   #  Change the path to your initial model (different to our pretrained model UrbanMLLM)
                 train_script: str = "/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/llava/train/train_mem.py",
                 inference_script: str = "/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/our_eval/model_vqa_loader_multiview_llm_forward.py",
                 deepspeed_config: str = "/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/scripts/zero3.json",
                 vision_tower: str = "/data2/zhangxin/model_zoo/google/siglip-so400m-patch14-384"):
        """
        Initialize VISAGE Training and Inference Pipeline
        
        Args:
            base_output_dir: Base directory for model outputs and checkpoints
            base_data_dir: Base directory where K-fold datasets (data1, data2, ...) are located
            initial_model: Path to initial pre-trained model for first fold
            train_script: Path to training script
            inference_script: Path to inference script
            deepspeed_config: Path to DeepSpeed configuration file
            vision_tower: Path to vision tower model
        """
        # Configuration paths (all configurable with defaults)
        self.base_output_dir = base_output_dir
        self.base_data_dir = base_data_dir
        self.initial_model = initial_model
        
        # Training script paths
        self.train_script = train_script
        self.inference_script = inference_script
        self.deepspeed_config = deepspeed_config
        self.vision_tower = vision_tower
        
        # Model configuration
        self.conv_mode = "llama_3"
        self.temperature = 0.2
        
        # Training parameters (Set fixed values for cross-validation consistency)
        self.batch_size = 1
        self.num_epochs = 1
        self.learning_rate = 1e-5
        self.gradient_accumulation_steps = 2
        
        # Execution environment setup
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["CURRENT_RANK"] = "0"

        self.vilaenv_path = "/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vilaenv"
        self.run_dir = "/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila"

    
    def detect_datasets(self, base_dir: str) -> List[str]:
        """
        Detect available datasets (e.g., data1, data2, ...) in the base directory
        
        Args:
            base_dir: Base directory to scan for datasets
            
        Returns:
            List of dataset directory names sorted numerically
        """
        datasets = []
        pattern = re.compile(r'data\d+')
        
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path) and pattern.match(item):
                    datasets.append(item)
        
        # Sort datasets numerically (data1, data2, data3, ...)
        datasets.sort(key=lambda x: int(re.search(r'data(\d+)', x).group(1)))
        return datasets
    
    def run_training_fold(self, dataset_name: str, model_path: str, output_dir: str, gpu_ids: List[int]) -> bool:
        """
        Run training for a single fold
        
        Args:
            dataset_name: Name of the dataset mixture defined in datasets_mixture.py
            model_path: Path to the input model (e.g., initial_model)
            output_dir: Output directory for this fold's checkpoint
            gpu_ids: List of GPU IDs to use
            
        Returns:
            True if training succeeded, False otherwise
        """

        # 1. Change the working directory to the training script's base directory
        original_dir = os.getcwd()
        os.chdir(self.run_dir)

        # 2. Update environment variables to use the vilaenv Python
        env = os.environ.copy()
        env["PATH"] = os.path.join(self.vilaenv_path, "bin") + ":" + env["PATH"]
        python_exe = os.path.join(self.vilaenv_path, "bin/python")

        print(f"🚀 Training fold: {dataset_name}")
        print(f"   Input model: {model_path}")
        print(f"   Output dir: {output_dir}")
        print(f"   Using GPUs: {gpu_ids}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        
        # Set CUDA visible devices
        gpu_str = ",".join(map(str, gpu_ids))
        env["CUDA_VISIBLE_DEVICES"] = gpu_str
        
        # Build training command (use torchrun + deepspeed)
        cmd = [
            "torchrun",
            f"--nnodes=1",
            f"--nproc_per_node={len(gpu_ids)}",
            f"--master_port=25005", # Unique port for multi-process
            f"--master_addr={env['MASTER_ADDR']}",
            f"--node_rank={env['CURRENT_RANK']}",
            self.train_script,
            f"--deepspeed", self.deepspeed_config,
            f"--model_name_or_path", model_path,
            f"--version", "llama_3",
            f"--data_mixture", dataset_name,
            f"--vision_tower", self.vision_tower,
            f"--mm_vision_select_feature", "cls_patch",
            f"--mm_projector", "mlp_downsample",
            f"--tune_vision_tower", "False",
            f"--tune_mm_projector", "True",
            f"--tune_language_model", "True",
            f"--mm_vision_select_layer", "-2",
            f"--mm_use_im_start_end", "False",
            f"--mm_use_im_patch_token", "False",
            f"--image_aspect_ratio", "resize",
            f"--bf16", "True",
            f"--output_dir", output_dir,
            f"--num_train_epochs", str(self.num_epochs),
            f"--per_device_train_batch_size", str(self.batch_size),
            f"--per_device_eval_batch_size", "1",
            f"--gradient_accumulation_steps", str(self.gradient_accumulation_steps),
            f"--evaluation_strategy", "no",
            f"--save_strategy", "steps",
            f"--save_steps", "500",
            f"--save_total_limit", "1",
            f"--learning_rate", str(self.learning_rate),
            f"--weight_decay", "0.",
            f"--warmup_ratio", "0.03",
            f"--lr_scheduler_type", "cosine",
            f"--logging_steps", "1",
            f"--tf32", "True",
            f"--model_max_length", "20000",
            f"--gradient_checkpointing", "True",
            f"--dataloader_num_workers", "8",
            f"--lazy_preprocess", "True",
            f"--vflan_no_system_prompt", "True",
            f"--report_to", "tensorboard"
        ]
        
        try:
            # We explicitly use the vilaenv python executable for the torchrun script
            cmd[0] = os.path.join(self.vilaenv_path, "bin/torchrun")
            
            print(f"   Running command: {' '.join(cmd[:10])}...") # Print first 10 args for brevity
            result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
            print(f"✅ Training completed for {dataset_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed for {dataset_name}")
            print(f"   Error: {e.stderr}")
            return False
        finally:
            os.chdir(original_dir) # Restore working directory
    
    def run_inference(self, dataset_folder: str, data_type: str, gpu_id: int) -> bool:
        """
        Run inference (feature extraction) for a specific dataset fold
        
        Args:
            dataset_folder: Folder name (e.g., "data1") where train/test data is located
            data_type: "train" or "test"
            gpu_id: GPU ID to use
            
        Returns:
            True if inference succeeded, False otherwise
        """

        # 1. Change the working directory
        original_dir = os.getcwd()
        os.chdir(self.run_dir)

        # 2. Update environment variables to use the vilaenv Python
        env = os.environ.copy()
        env["PATH"] = os.path.join(self.vilaenv_path, "bin") + ":" + env["PATH"]
        python_exe = os.path.join(self.vilaenv_path, "bin/python")

        # Define paths based on the dataset folder (e.g., "data1")
        model_path = f"{self.base_output_dir}/{dataset_folder}/vila_train_all_sisv_pretrain_textenhance_att_clean_8B_new"
        question_file = f"{self.base_output_dir}/{dataset_folder}/{data_type}_data.json"
        output_file = f"{self.base_output_dir}/{dataset_folder}/feature_vector_segregation_{data_type}.pkl"
        
        print(f"🔍 Running {data_type} inference for {dataset_folder} on GPU {gpu_id}")
        
        # Validation checks
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}. Skipping inference.")
            return False
        
        if not os.path.exists(question_file):
            print(f"❌ Question file not found: {question_file}. Skipping inference.")
            return False
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Set GPU
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Build inference command
        cmd = [
            python_exe, self.inference_script,
            f"--model-path", model_path,
            f"--question-file", question_file,
            f"--outputs-file", output_file,
            f"--image-folder", "",
            f"--temperature", str(self.temperature),
            f"--conv-mode", self.conv_mode
        ]
        
        try:
            result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
            print(f"✅ {data_type.capitalize()} inference completed for {dataset_folder} on GPU {gpu_id}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {data_type.capitalize()} inference failed for {dataset_folder} on GPU {gpu_id}")
            print(f"   Error: {e.stderr}")
            return False
        finally:
            os.chdir(original_dir) # Restore working directory
    
    def sequential_training(self, dataset_prefix: str, num_folds: int, start_fold: int, gpu_ids: List[int]) -> bool:
        """
        Run sequential K-fold training
        
        Args:
            dataset_prefix: Prefix for dataset names (e.g., "segregation_cot_40_31_data")
            num_folds: Number of folds to train (total count)
            start_fold: Starting fold index (e.g., 1)
            gpu_ids: List of GPU IDs to use
            
        Returns:
            True if all training succeeded, False otherwise
        """
        print("🔄 Starting sequential training...")
        
        # Determine the full dataset name for the training script (e.g., "segregation_cot_40_31_data1")
        datasets = [f"{dataset_prefix}{i}" for i in range(start_fold, start_fold + num_folds)]
        # Determine the folder name for the data/output (e.g., "data1")
        data_folders = [f"data{i}" for i in range(start_fold, start_fold + num_folds)]

        print(f"📊 Datasets to train: {datasets}")
        
        for i, dataset in enumerate(datasets):
            data_folder = data_folders[i]
            
            print(f"\n==========================================")
            print(f"🚀 Training fold {i+1}/{len(datasets)}: {dataset} (Data: {data_folder})")
            print(f"==========================================")
            
            # Determine input model path (always initial model for independent K-fold)
            model_path = self.initial_model
            
            # Create output directory path
            output_dir = f"{self.base_output_dir}/{data_folder}/vila_train_all_sisv_pretrain_textenhance_att_clean_8B_new"
            
            # Run training
            if not self.run_training_fold(dataset, model_path, output_dir, gpu_ids):
                return False
            
            print(f"✅ Completed {dataset}")
        
        print(f"\n🎉 All {len(datasets)} folds training completed successfully!")
        return True
    
    def sequential_inference(self, datasets: List[str], gpu_ids: List[int]) -> bool:
        """
        Run sequential inference
        
        Args:
            datasets: List of dataset folder names (e.g., "data1", "data2")
            gpu_ids: List of GPU IDs to use
            
        Returns:
            True if all inference succeeded, False otherwise
        """
        print("🔄 Starting sequential inference...")
        
        for i, dataset_folder in enumerate(datasets):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            
            print(f"\n📊 Processing dataset: {dataset_folder}")
            print(f"================================")
            
            # Train inference
            if not self.run_inference(dataset_folder, "train", gpu_id):
                return False
            
            # Test inference
            if not self.run_inference(dataset_folder, "test", gpu_id):
                return False
            
            print(f"✅ Completed all inference for {dataset_folder}")
        
        return True
    
    def parallel_inference(self, datasets: List[str], gpu_ids: List[int]) -> bool:
        """
        Run parallel inference
        
        Args:
            datasets: List of dataset folder names (e.g., "data1", "data2")
            gpu_ids: List of GPU IDs to use
            
        Returns:
            True if all inference succeeded, False otherwise
        """
        print("🔄 Starting parallel inference...")
        
        # Calculate distribution
        tasks = []
        for i, dataset_folder in enumerate(datasets):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            tasks.append((dataset_folder, gpu_id))
        
        datasets_per_gpu = (len(datasets) + len(gpu_ids) - 1) // len(gpu_ids)
        print(f"📊 Distribution: Up to {datasets_per_gpu} datasets per GPU")
        
        def run_dataset_inference(dataset_gpu_pair):
            dataset_folder, gpu_id = dataset_gpu_pair
            success = True
            
            print(f"🚀 Starting parallel job for {dataset_folder} on GPU {gpu_id}")
            
            # Train inference
            if not self.run_inference(dataset_folder, "train", gpu_id):
                success = False
            
            # Test inference
            if not self.run_inference(dataset_folder, "test", gpu_id):
                success = False
            
            if success:
                print(f"🎉 Completed parallel inference for {dataset_folder} on GPU {gpu_id}")
            else:
                print(f"❌ Failed parallel inference for {dataset_folder} on GPU {gpu_id}")
            
            return success
        
        # Run in parallel
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor: # Limit max_workers to number of available GPUs
            futures = {executor.submit(run_dataset_inference, task): task for task in tasks}
            
            results = []
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"❌ Exception in parallel execution for task {task}: {e}")
                    results.append(False)
        
        return all(results)


class VISAGECompletePipeline:
    """Pipeline integrating K-fold training and inference based on existing datasets"""
    
    def __init__(self, args):
        """
        Initialize complete pipeline
        
        Args:
            args: Command line arguments
        """
        self.args = args
        
        # Initialize training/inference pipeline
        self.training_pipeline = VISAGETrainingInference(
            base_output_dir=args.base_output_dir,
            base_data_dir=args.base_data_dir, # This is where the K-fold data folders (data1, data2, ...) are expected
            initial_model=args.initial_model,
            train_script=args.train_script,
            inference_script=args.inference_script,
            deepspeed_config=args.deepspeed_config,
            vision_tower=args.vision_tower
        )
    
    def run_pipeline(self):
        """Run complete pipeline"""
        print("🚀 Starting VISAGE K-fold Training and Inference Pipeline")
        print("=" * 60)
        
        # Parse GPU list
        gpu_ids = [int(gpu) for gpu in self.args.gpu_list.split(",")]
        print(f"🎮 Using GPUs: {gpu_ids}")
        
        # Get datasets for operation (e.g., data1, data2, ...)
        datasets = self.training_pipeline.detect_datasets(self.args.base_data_dir)
        if not datasets:
             print(f"❌ No K-fold datasets found in base data directory: {self.args.base_data_dir}")
             return

        print(f"📁 Found {len(datasets)} K-fold datasets for operation: {datasets}")
        
        # Step 1: Training
        if self.args.run_training:
            print("\n🏋️ STEP 1: Sequential K-fold Training")
            print("-" * 40)
            
            # The num_folds argument will now be derived from the detected folders,
            # but we use the command-line arguments for prefix and start fold for clarity in logging.
            
            training_success = self.training_pipeline.sequential_training(
                dataset_prefix=self.args.dataset_prefix,
                num_folds=len(datasets), # Use detected count
                start_fold=self.args.start_fold,
                gpu_ids=gpu_ids
            )
            
            if not training_success:
                print("❌ Training pipeline failed")
                return
        
        # Step 2: Inference
        if self.args.run_inference:
            print("\n🔍 STEP 2: Model Inference")
            print("-" * 40)
            
            # Filter datasets if a specific one is requested
            if self.args.specific_dataset:
                datasets_to_run = [self.args.specific_dataset]
                print(f"🎯 Running only for specific dataset: {self.args.specific_dataset}")
            else:
                datasets_to_run = datasets
                
            print(f"🏃 Running inference for {len(datasets_to_run)} dataset(s)")
            
            # Run inference based on mode
            if self.args.mode == "sequential":
                inference_success = self.training_pipeline.sequential_inference(datasets_to_run, gpu_ids)
            elif self.args.mode == "parallel":
                inference_success = self.training_pipeline.parallel_inference(datasets_to_run, gpu_ids)
            else:
                print(f"❌ Unknown mode: {self.args.mode}")
                return
            
            if inference_success:
                print("\n🎉 Inference completed successfully!")
            else:
                print("\n❌ Inference pipeline failed")
                return
        
        print("\n🎉 Complete pipeline execution finished successfully!")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="VISAGE K-fold Training and Inference Pipeline (Using Pre-generated Datasets)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE EXAMPLES AND DOCUMENTATION
================================================================================

This script runs cross-validation training and inference using datasets that 
must be pre-generated in the --base-data-dir (e.g., data1, data2, ... folders).

BASIC USAGE PATTERNS:

1. TRAINING + INFERENCE (Existing Datasets):
   python perception_workflow.py --run-training --run-inference

2. INFERENCE ONLY (Parallel Mode):
   python perception_workflow.py --run-inference --mode parallel

3. CUSTOM CONFIGURATION:
   python perception_workflow.py --run-training --run-inference \
        --dataset-prefix my_data --start-fold 1 --gpu-list 0,1,2

DETAILED PARAMETER EXPLANATION:

PIPELINE STEPS:
 --run-training         : Run training pipeline (K-fold sequential training)
 --run-inference        : Run inference pipeline

TRAINING PARAMETERS:
 --dataset-prefix PREFIX: Prefix for dataset names (e.g., 'segregation_cot_40_31_data') used in datasets_mixture.py
 --start-fold N         : Starting fold index (default: 1)

INFERENCE PARAMETERS:
 --mode MODE            : Execution mode - "sequential" or "parallel" (default: sequential)
 --specific-dataset NAME: Run only for specific dataset folder (e.g., 'data3')
 --gpu-list IDS         : Comma-separated GPU IDs (default: "0,1,2,3,4,5,6,7")

PATH CONFIGURATION:
 --base-output-dir PATH : Base directory for model outputs and checkpoints
 --base-data-dir PATH   : Base directory where K-fold data folders (data1, data2, ...) reside
 --initial-model PATH   : Path to initial pre-trained model for first fold
 --train-script PATH    : Path to training script
 --inference-script PATH: Path to inference script
 --deepspeed-config PATH: Path to DeepSpeed configuration file
 --vision-tower PATH    : Path to vision tower model

================================================================================
        """
    )
    
    # Pipeline step arguments
    # Removed --generate-datasets
    parser.add_argument("--run-training", action="store_true", 
                        help="Run training pipeline (K-fold sequential training)")
    parser.add_argument("--run-inference", action="store_true", 
                        help="Run inference pipeline")
    
    # Dataset arguments (only prefix and fold structure are relevant for existing data)
    parser.add_argument("--dataset-prefix", default="segregation_cot_40_31_data",
                        help="Prefix for dataset names (must match names in datasets_mixture.py)")
    
    # Removed --input-path, --datasets-mixture-path, --num-folds, --seed
    
    # Training arguments
    parser.add_argument("--start-fold", type=int, default=1,
                        help="Starting fold index (e.g., 1 for data1)")
    
    # Inference arguments
    parser.add_argument("--mode", choices=["sequential", "parallel"], default="sequential",
                        help="Inference execution mode")
    parser.add_argument("--specific-dataset", 
                        help="Run only for specific dataset folder (e.g., 'data3')")
    parser.add_argument("--gpu-list", default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs")
    
    # Path configuration arguments
    parser.add_argument("--base-output-dir", default="/data3/maruolong/VISAGE/data/processed/baseline/vlm/cross_train/31",
                        help="Base directory for model outputs and checkpoints")
    parser.add_argument("--base-data-dir", default="/data3/maruolong/VISAGE/data/processed/training_data/31_cities",
                        help="Base directory where K-fold datasets (data1, data2, ...) reside")
    parser.add_argument("--initial-model", default="/data2/zhangxin/model_zoo/Llama-3-VILA1.5-8B",   # Baseline model path
                        help="Path to initial pre-trained model")
    parser.add_argument("--train-script", default="/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/llava/train/train_mem.py",
                        help="Path to training script")
    parser.add_argument("--inference-script", default="/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/our_eval/model_vqa_loader_multiview_llm_forward.py",
                        help="Path to inference script")
    parser.add_argument("--deepspeed-config", default="/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/scripts/zero3.json",
                        help="Path to DeepSpeed configuration file")
    parser.add_argument("--vision-tower", default="/data2/zhangxin/model_zoo/google/siglip-so400m-patch14-384",
                        help="Path to vision tower model")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.run_training, args.run_inference]):
        print("❌ No pipeline steps specified. Use --run-training or --run-inference")
        return
    
    if args.run_training and not os.path.exists(args.initial_model):
        print(f"❌ Initial model not found: {args.initial_model}")
        print("Cannot run training without initial model.")
        return
    
    # Initialize and run complete pipeline
    pipeline = VISAGECompletePipeline(args)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()