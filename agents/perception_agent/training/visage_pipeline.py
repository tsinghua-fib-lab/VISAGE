#!/usr/bin/env python3
"""
VISAGE Training and Inference Pipeline
Integrated Python version of sequential_train.sh and inference_flexible.sh

Features:
- Sequential K-fold cross-validation training
- Flexible inference with sequential/parallel modes
- GPU management and resource allocation
- Error handling and progress tracking

Date: 2025-11-19
"""

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

class VISAGETrainingInference:
    """Integrated training and inference pipeline for VISAGE"""
    
    def __init__(self, 
                 base_output_dir: str = "/data3/maruolong/VISAGE/data/cross_train/31",
                 base_data_dir: str = "/data3/maruolong/VISAGE/data/31_cities",
                 initial_model: str = "/data3/zhangxin/wuwen/vila-1.5/last_checkpoint/pretrain_att_35w_clean_textenhance",
                 train_script: str = "/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/llava/train/train_mem.py",
                 inference_script: str = "/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/our_eval/model_vqa_loader_multiview_llm_forward.py",
                 deepspeed_config: str = "/data3/maruolong/UrbanMLLM/vila/scripts/zero3.json",
                 vision_tower: str = "/data2/zhangxin/model_zoo/google/siglip-so400m-patch14-384"):
        """
        Initialize VISAGE Training and Inference Pipeline
        
        Args:
            base_output_dir: Base directory for model outputs and checkpoints
            base_data_dir: Base directory for datasets
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
        
        # Training parameters
        self.batch_size = 1
        self.num_epochs = 1
        self.learning_rate = 1e-5
        self.gradient_accumulation_steps = 2
        
        # Environment setup
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["CURRENT_RANK"] = "0"
    
    def detect_datasets(self, base_dir: str) -> List[str]:
        """
        Detect available datasets in the base directory
        
        Args:
            base_dir: Base directory to scan for datasets
            
        Returns:
            List of dataset names sorted numerically
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
            dataset_name: Name of the dataset
            model_path: Path to the input model
            output_dir: Output directory for this fold
            gpu_ids: List of GPU IDs to use
            
        Returns:
            True if training succeeded, False otherwise
        """
        print(f"🚀 Training fold: {dataset_name}")
        print(f"   Input model: {model_path}")
        print(f"   Output dir: {output_dir}")
        print(f"   Using GPUs: {gpu_ids}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        
        # Set CUDA visible devices
        gpu_str = ",".join(map(str, gpu_ids))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_str
        
        # Build training command
        cmd = [
            "torchrun",
            f"--nnodes=1",
            f"--nproc_per_node={len(gpu_ids)}",
            "--master_port=25005",
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
            print(f"   Running command: {' '.join(cmd[:10])}...")  # Print first 10 args for brevity
            result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
            print(f"✅ Training completed for {dataset_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed for {dataset_name}")
            print(f"   Error: {e.stderr}")
            return False
    
    def run_inference(self, dataset: str, data_type: str, gpu_id: int) -> bool:
        """
        Run inference for a specific dataset and data type
        
        Args:
            dataset: Dataset name
            data_type: "train" or "test"
            gpu_id: GPU ID to use
            
        Returns:
            True if inference succeeded, False otherwise
        """
        model_path = f"{self.base_output_dir}/{dataset}/vila_train_all_sisv_pretrain_textenhance_att_clean_8B_new"
        question_file = f"{self.base_data_dir}/{dataset}/{data_type}_data.json"
        output_file = f"{self.base_data_dir}/{dataset}/feature_vector_segregation_{data_type}.pkl"
        
        print(f"🔍 Running {data_type} inference for {dataset} on GPU {gpu_id}")
        
        # Validation checks
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return False
        
        if not os.path.exists(question_file):
            print(f"❌ Question file not found: {question_file}")
            return False
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Set GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Build inference command
        cmd = [
            "python", self.inference_script,
            f"--model-path", model_path,
            f"--question-file", question_file,
            f"--outputs-file", output_file,
            f"--image-folder", "",
            f"--temperature", str(self.temperature),
            f"--conv-mode", self.conv_mode
        ]
        
        try:
            result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
            print(f"✅ {data_type.capitalize()} inference completed for {dataset} on GPU {gpu_id}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {data_type.capitalize()} inference failed for {dataset} on GPU {gpu_id}")
            print(f"   Error: {e.stderr}")
            return False
    
    def sequential_training(self, dataset_prefix: str, num_folds: int, start_fold: int, gpu_ids: List[int]) -> bool:
        """
        Run sequential K-fold training
        
        Args:
            dataset_prefix: Prefix for dataset names
            num_folds: Number of folds to train
            start_fold: Starting fold index
            gpu_ids: List of GPU IDs to use
            
        Returns:
            True if all training succeeded, False otherwise
        """
        print("🔄 Starting sequential training...")
        
        # Generate dataset list
        datasets = [f"{dataset_prefix}{i}" for i in range(start_fold, start_fold + num_folds)]
        print(f"📊 Datasets to train: {datasets}")
        
        for i, dataset in enumerate(datasets):
            print(f"\n==========================================")
            print(f"🚀 Training fold {i+1}/{len(datasets)}: {dataset}")
            print(f"==========================================")
            
            # Determine input model path
            if i == 0:
                model_path = self.initial_model
                print(f"   Using initial model: {model_path}")
            else:
                prev_dataset = datasets[i-1]
                model_path = f"{self.base_output_dir}/{prev_dataset}/vila_train_all_sisv_pretrain_textenhance_att_clean_8B_new"
                print(f"   Using previous model: {model_path}")
                
                # Check if previous model exists
                if not os.path.exists(model_path):
                    print(f"❌ Previous model not found: {model_path}")
                    return False
            
            # Create output directory path
            output_dir = f"{self.base_output_dir}/{dataset}/vila_train_all_sisv_pretrain_textenhance_att_clean_8B_new"
            
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
            datasets: List of dataset names
            gpu_ids: List of GPU IDs to use
            
        Returns:
            True if all inference succeeded, False otherwise
        """
        print("🔄 Starting sequential inference...")
        
        for i, dataset in enumerate(datasets):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            
            print(f"\n📊 Processing dataset: {dataset}")
            print(f"================================")
            
            # Train inference
            if not self.run_inference(dataset, "train", gpu_id):
                return False
            
            # Test inference
            if not self.run_inference(dataset, "test", gpu_id):
                return False
            
            print(f"✅ Completed all inference for {dataset}")
        
        return True
    
    def parallel_inference(self, datasets: List[str], gpu_ids: List[int]) -> bool:
        """
        Run parallel inference
        
        Args:
            datasets: List of dataset names
            gpu_ids: List of GPU IDs to use
            
        Returns:
            True if all inference succeeded, False otherwise
        """
        print("🔄 Starting parallel inference...")
        
        # Calculate distribution
        datasets_per_gpu = (len(datasets) + len(gpu_ids) - 1) // len(gpu_ids)
        print(f"📊 Distribution: Up to {datasets_per_gpu} datasets per GPU")
        
        def run_dataset_inference(dataset_gpu_pair):
            dataset, gpu_id = dataset_gpu_pair
            success = True
            
            print(f"🚀 Starting parallel job for {dataset} on GPU {gpu_id}")
            
            # Train inference
            if not self.run_inference(dataset, "train", gpu_id):
                success = False
            
            # Test inference
            if not self.run_inference(dataset, "test", gpu_id):
                success = False
            
            if success:
                print(f"🎉 Completed parallel inference for {dataset} on GPU {gpu_id}")
            else:
                print(f"❌ Failed parallel inference for {dataset} on GPU {gpu_id}")
            
            return success
        
        # Prepare tasks
        tasks = []
        for i, dataset in enumerate(datasets):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            tasks.append((dataset, gpu_id))
        
        # Run in parallel
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {executor.submit(run_dataset_inference, task): task for task in tasks}
            
            results = []
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"❌ Exception in parallel execution: {e}")
                    results.append(False)
        
        return all(results)
    
    def run_pipeline(self, args):
        """
        Main pipeline execution function
        
        Args:
            args: Command line arguments
        """
        print("🚀 Starting VISAGE Training and Inference Pipeline")
        print("=" * 60)
        
        # Parse GPU list
        gpu_ids = [int(gpu) for gpu in args.gpu_list.split(",")]
        print(f"🎮 Using GPUs: {gpu_ids}")
        
        # Handle training if requested
        if args.run_training:
            print("\n🏋️ STEP 1: Sequential K-fold Training")
            training_success = self.sequential_training(
                dataset_prefix=args.dataset_prefix,
                num_folds=args.num_folds,
                start_fold=args.start_fold,
                gpu_ids=gpu_ids
            )
            
            if not training_success:
                print("❌ Training pipeline failed")
                return
        
        # Handle inference
        print("\n🔍 STEP 2: Model Inference")
        
        # Get datasets for inference
        if args.specific_dataset:
            datasets = [args.specific_dataset]
            print(f"🎯 Running only for specific dataset: {args.specific_dataset}")
        else:
            datasets = self.detect_datasets(self.base_output_dir)
            if not datasets:
                print("❌ No datasets found for inference")
                return
        
        print(f"📁 Found {len(datasets)} datasets: {datasets}")
        
        # Run inference based on mode
        if args.mode == "sequential":
            inference_success = self.sequential_inference(datasets, gpu_ids)
        elif args.mode == "parallel":
            inference_success = self.parallel_inference(datasets, gpu_ids)
        else:
            print(f"❌ Unknown mode: {args.mode}")
            return
        
        if inference_success:
            print("\n🎉 Pipeline execution completed successfully!")
        else:
            print("\n❌ Inference pipeline failed")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="VISAGE Training and Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE EXAMPLES AND DOCUMENTATION
================================================================================

This script provides an integrated pipeline for K-fold cross-validation training
and inference with flexible GPU allocation and execution modes.

BASIC USAGE PATTERNS:

1. FULL PIPELINE (Training + Inference):
   python visage_pipeline.py --run-training --mode sequential

2. INFERENCE ONLY (Different Modes):
   python visage_pipeline.py --mode sequential                    # Sequential inference
   python visage_pipeline.py --mode parallel                      # Parallel inference
   python visage_pipeline.py --mode parallel --gpu-list 0,1,2,3  # Parallel with specific GPUs

3. CUSTOM DATASET AND FOLD CONFIGURATION:
   python visage_pipeline.py --run-training --dataset-prefix my_data --num-folds 3 --start-fold 1
   python visage_pipeline.py --mode sequential --specific-dataset data2 --gpu-list 0

4. CUSTOM PATHS CONFIGURATION:
   python visage_pipeline.py --run-training \
       --base-output-dir /path/to/output \
       --base-data-dir /path/to/data \
       --initial-model /path/to/initial/model \
       --train-script /path/to/train.py \
       --inference-script /path/to/inference.py

DETAILED PARAMETER EXPLANATION:

TRAINING PARAMETERS:
  --run-training          : Enable training pipeline (K-fold sequential training)
  --dataset-prefix PREFIX : Prefix for dataset names (e.g., "segregation_cot_40_31_data")
  --num-folds N          : Number of folds for cross-validation (default: 5)
  --start-fold N         : Starting fold index (default: 1)

INFERENCE PARAMETERS:
  --mode MODE            : Execution mode - "sequential" or "parallel" (default: sequential)
  --specific-dataset NAME: Run only for specific dataset (default: all detected datasets)
  --gpu-list IDS         : Comma-separated GPU IDs (default: "0,1,2,3,4,5,6,7")

PATH CONFIGURATION:
  --base-output-dir PATH : Base directory for model outputs and checkpoints
  --base-data-dir PATH   : Base directory for datasets
  --initial-model PATH   : Path to initial pre-trained model for first fold
  --train-script PATH    : Path to training script
  --inference-script PATH: Path to inference script
  --deepspeed-config PATH: Path to DeepSpeed configuration file
  --vision-tower PATH    : Path to vision tower model

EXECUTION MODES EXPLAINED:

1. Sequential Mode:
   - Processes one dataset at a time
   - Each dataset uses one GPU (round-robin assignment)
   - Suitable when GPU memory is limited
   - Better for debugging and monitoring

2. Parallel Mode:
   - Runs all datasets simultaneously
   - Distributes datasets across available GPUs
   - Each GPU can handle multiple datasets
   - Maximum parallelism for faster execution
   - Requires sufficient GPU memory

OUTPUT STRUCTURE:
  Training outputs: {base_output_dir}/{dataset_name}/vila_train_all_sisv_pretrain_textenhance_att_clean_8B_new/
  Inference outputs: {base_data_dir}/{dataset_name}/feature_vector_segregation_{train|test}.pkl

PREREQUISITES:
  - PyTorch and DeepSpeed installed
  - Pre-trained initial model available (for training)
  - Datasets prepared in the expected format
  - Sufficient GPU memory for model training/inference

TROUBLESHOOTING:
  - Check GPU availability with nvidia-smi
  - Verify all file paths exist before running
  - Ensure sufficient disk space for model checkpoints
  - Monitor GPU memory usage during parallel execution

================================================================================
        """
    )
    
    # Training arguments
    parser.add_argument("--run-training", action="store_true", 
                       help="Run training pipeline (K-fold sequential training)")
    parser.add_argument("--dataset-prefix", default="segregation_cot_40_31_data",
                       help="Prefix for dataset names (default: segregation_cot_40_31_data)")
    parser.add_argument("--num-folds", type=int, default=5,
                       help="Number of folds for training (default: 5)")
    parser.add_argument("--start-fold", type=int, default=1,
                       help="Starting fold index (default: 1)")
    
    # Inference arguments
    parser.add_argument("--mode", choices=["sequential", "parallel"], default="sequential",
                       help="Inference execution mode (default: sequential)")
    parser.add_argument("--specific-dataset", 
                       help="Run only for specific dataset (default: all detected datasets)")
    parser.add_argument("--gpu-list", default="0,1,2,3,4,5,6,7",
                       help="Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)")
    
    # Path configuration arguments
    parser.add_argument("--base-output-dir", default="/data3/maruolong/VISAGE/data/cross_train/31",
                       help="Base directory for model outputs and checkpoints")
    parser.add_argument("--base-data-dir", default="/data3/maruolong/VISAGE/data/31_cities",
                       help="Base directory for datasets")
    parser.add_argument("--initial-model", default="/data3/zhangxin/wuwen/vila-1.5/last_checkpoint/pretrain_att_35w_clean_textenhance",
                       help="Path to initial pre-trained model for first fold")
    parser.add_argument("--train-script", default="/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/llava/train/train_mem.py",
                       help="Path to training script")
    parser.add_argument("--inference-script", default="/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/our_eval/model_vqa_loader_multiview_llm_forward.py",
                       help="Path to inference script")
    parser.add_argument("--deepspeed-config", default="/data3/maruolong/UrbanMLLM/vila/scripts/zero3.json",
                       help="Path to DeepSpeed configuration file")
    parser.add_argument("--vision-tower", default="/data2/zhangxin/model_zoo/google/siglip-so400m-patch14-384",
                       help="Path to vision tower model")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.run_training and not os.path.exists(args.initial_model):
        print(f"❌ Initial model not found: {args.initial_model}")
        print("Cannot run training without initial model.")
        return
    
    # Initialize pipeline with custom paths
    pipeline = VISAGETrainingInference(
        base_output_dir=args.base_output_dir,
        base_data_dir=args.base_data_dir,
        initial_model=args.initial_model,
        train_script=args.train_script,
        inference_script=args.inference_script,
        deepspeed_config=args.deepspeed_config,
        vision_tower=args.vision_tower
    )
    
    # Run pipeline
    pipeline.run_pipeline(args)

if __name__ == "__main__":
    main()