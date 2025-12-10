#!/usr/bin/env python3
"""
Unified VISAGE Pipeline Interface with Command Line Support

This script integrates:
1. Literature Analysis (literature_agent)
2. VISAGE Data Processing (experiment_agent)
3. K-Fold Dataset Generation + Training + Inference (perception_agent)
4. Regression Visualization (predictor)

All important paths are configurable, with default values, and documented in English.
"""

import os
import sys
import argparse
from typing import List, Dict, Optional


# -----------------------------
# Import Modules
# -----------------------------

from literature_review_workflow import LiteratureAnalysisPipeline
from experiment_workflow import experiment_agent_run
from perception_workflow import VISAGECompletePipeline

# -----------------------------
# Default Paths for Experiment Agent
# -----------------------------
default_cities = [        
        'Boston', 'Chicago', 'Dallas', 'Detroit', 'Los Angeles', 'Miami',
        'New York', 'Philadelphia', 'San Francisco', 'Seattle', 'Washington',
        'Albuquerque', 'Austin', 'Baltimore', 'Charlotte', 'Columbus', 'Denver',
        'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas',
        'Memphis', 'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland',
        'San Antonio', 'San Diego', 'San Jose', 'Tucson']  # Example cities

default_paths = {
    "base_image_folder": "/data3/maruolong/VISAGE/data/raw/imagery",  
    # Base folder containing all city-level street and RS images

    "image_folders": [f"/data3/maruolong/VISAGE/data/raw/imagery/{city}/images" for city in default_cities],  
    # List of city-specific image folders for processing

    "json1_path": "/data3/maruolong/Train_Data/Urbanarea_dominant_race_data_all.json",  
    # JSON file containing urban area demographic data

    "segregation_jsonl_paths": [
        f"/data3/maruolong/VISAGE/data/raw/mobility/visit_data/{city}_2019/{city}_2019_tract_segregation.jsonl"
        for city in default_cities
    ],  
    # JSONL files with tract-level segregation data per city

    "income_distribution_paths": [
        f"/data3/maruolong/VISAGE/data/raw/mobility/visit_data/{city}_2019/{city}_2019_tract_income_distribution.jsonl"
        for city in default_cities
    ],  
    # JSONL files with tract-level income distribution data per city

    "rs_image_base_dir": "/data3/maruolong/VISAGE/data/raw/imagery/rs_image_new/merged",  
    # Folder containing merged remote sensing images

    "codebook_path": "/data3/maruolong/VISAGE/agents/literature_agent/knowledge_base/codebooks.json",  
    # Knowledge base codebook used in literature agent

    "rs_cue_detection": "/data3/maruolong/VISAGE/data/processed/cue_frequecies/rs_image_cue_detection_results.jsonl",  
    # Remote sensing cue detection results

    "street_cue_detection_base": "/data3/maruolong/VISAGE/data/processed/cue_frequecies/street_view_cue_detection",  
    # Folder containing street view cue detection results

    "rs_aggregated_desc": "/data3/maruolong/VISAGE/data/processed/cue_frequecies/rs_aggregated_description_by_tract.jsonl",  
    # Aggregated remote sensing description by tract

    "street_aggregated_desc": "/data3/maruolong/VISAGE/data/processed/cue_frequecies/street_aggregated_description_by_tract.jsonl",  
    # Aggregated street view description by tract

    "original_qa": "/data3/maruolong/VISAGE/data/processed/training_data/31_original_qa.json",  
    # Original QA dataset

    "final_qa_with_caption": "/data3/maruolong/VISAGE/data/processed/training_data/31_qa_with_caption.json"  
    # Final QA dataset with captions included
}

# -----------------------------
# Unified Pipeline Function
# -----------------------------
def run_visage_pipeline(
    base_dir: str = "/data3/maruolong/VISAGE/data/processed/literature",
    base_data_dir: str = "/data3/maruolong/VISAGE/data/processed/training_data/31_cities",
    base_output_dir: str = "/data3/maruolong/VISAGE/data/processed/training_data/cross_train/31",
    # Literature Agent
    literature_base_dir: Optional[str] = None,
    # Experiment agent
    experiment_paths: Optional[Dict[str, str]] = None,
    api_key: str = "your_api_key_here",
    base_url: str = "your_base_url_here",
    model: str = "your_model_name_here",
    max_tokens: int = 500,
    max_retries: int = 10,
    max_wait: int = 600,
    # Perception Agent
    input_json_path: str = default_paths["final_qa_with_caption"],
    datasets_mixture_path: str = "/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/llava/data/datasets_mixture.py",
    dataset_prefix: str = "segregation_cot_40_31_data",
    num_folds: int = 5,
    seed: int = 42,
    start_fold: int = 1,
    initial_model: str = "/data3/zhangxin/wuwen/vila-1.5/last_checkpoint/pretrain_att_35w_clean_textenhance",
    train_script: str = "/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/llava/train/train_mem.py",
    inference_script: str = "/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/our_eval/model_vqa_loader_multiview_llm_forward.py",
    deepspeed_config: str = "/data3/maruolong/UrbanMLLM/vila/scripts/zero3.json",
    vision_tower: str = "/data2/zhangxin/model_zoo/google/siglip-so400m-patch14-384",
    gpu_list: List[int] = [0,1,2,3,4,5,6,7],
    # Control flags
    run_literature: bool = True,
    run_experiment: bool = True,
    generate_datasets: bool = True,
    run_training: bool = True,
    run_inference: bool = True,
    inference_mode: str = "sequential",
    specific_dataset: Optional[str] = None,
    run_regression_viz: bool = True,
):
    """
    Unified VISAGE pipeline interface.
    """
    results = {}

    # -----------------------------
    # 1. Literature Pipeline
    # -----------------------------
    if run_literature:
        literature_base_dir = literature_base_dir or base_dir
        print("\n==============================")
        print("📚 Running Literature Analysis")
        print("==============================")
        literature_pipeline = LiteratureAnalysisPipeline(base_dir=literature_base_dir)
        literature_results = literature_pipeline.run_full_pipeline()
        results['literature'] = literature_results

    # -----------------------------
    # 2. Experiment Agent
    # -----------------------------
    if run_experiment:
        print("\n==============================")
        print("🏙️ Running VISAGE Data Processing")
        print("==============================")
        exp_paths = experiment_paths or default_paths
        experiment_results = experiment_agent_run(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            max_retries=max_retries,
            max_wait=max_wait,
            custom_paths=exp_paths
        )
        results['experiment'] = experiment_results

    # -----------------------------
    # 3. Dataset Generation + Training + Inference
    # -----------------------------
    print("\n==============================")
    print("🖥️ Running Dataset Generation + Training + Inference")
    print("==============================")

    class Args:
        def __init__(self):
            self.input_path = input_json_path
            self.datasets_mixture_path = datasets_mixture_path
            self.dataset_prefix = dataset_prefix
            self.num_folds = num_folds
            self.seed = seed
            self.start_fold = start_fold
            self.base_output_dir = base_output_dir
            self.base_data_dir = base_data_dir
            self.initial_model = initial_model
            self.train_script = train_script
            self.inference_script = inference_script
            self.deepspeed_config = deepspeed_config
            self.vision_tower = vision_tower
            self.generate_datasets = generate_datasets
            self.run_training = run_training
            self.run_inference = run_inference
            self.mode = inference_mode
            self.specific_dataset = specific_dataset
            self.gpu_list = ",".join(map(str, gpu_list))

    pipeline_args = Args()
    complete_pipeline = VISAGECompletePipeline(pipeline_args)
    complete_pipeline.run_pipeline()

    results['training_inference'] = {
        "base_data_dir": base_data_dir,
        "base_output_dir": base_output_dir,
        "dataset_prefix": dataset_prefix,
        "num_folds": num_folds,
        "gpu_list": gpu_list
    }

    # ----------------------------------------------------------------------
    # 4. Regression Visualization (NEW)
    # ----------------------------------------------------------------------
    if run_regression_viz:
        print("\n==============================")
        print("📈 Running Regression Visualization")
        print("==============================")

        try:
            from predictor import run_kfold_regression
        except ImportError:
            print("❌ predictor.py not found. Please check its path.")
            run_kfold_regression = None

        if run_kfold_regression is not None:
            # Output directory for regression visualization
            viz_save_dir = os.path.join(base_data_dir, "regression_viz")
            os.makedirs(viz_save_dir, exist_ok=True)

            # Call run_kfold_regression from predictor.py
            metrics_viz = run_kfold_regression(
                base_data_dir=base_data_dir,
                num_folds=num_folds,
                output_dir=viz_save_dir,
                segregation_jsonl_paths=experiment_paths.get("segregation_jsonl_paths"),
                verbose=True
            )

            results["regression_viz"] = metrics_viz
            print(f"✅ Regression visualization done, metrics saved to {viz_save_dir}")

    print("\n🎉 VISAGE Unified Pipeline Completed Successfully!")
    return results

# -----------------------------
# Command Line Interface
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the VISAGE Unified Pipeline")

    # Main pipeline options
    parser.add_argument("--base_dir", type=str, default="/data3/maruolong/VISAGE/data/processed/literature", help="Base directory for VISAGE data")
    parser.add_argument("--base_data_dir", type=str, default="/data3/maruolong/VISAGE/data/processed/training_data/31_cities", help="Base folder for training data")
    parser.add_argument("--base_output_dir", type=str, default="/data3/maruolong/VISAGE/data/processed/training_data/cross_train/31", help="Folder for outputs")
    parser.add_argument("--run_literature", action="store_true", help="Run literature analysis step")
    parser.add_argument("--run_experiment", action="store_true", help="Run experiment agent data processing")
    parser.add_argument("--generate_datasets", action="store_true", help="Generate K-fold datasets")
    parser.add_argument("--run_training", action="store_true", help="Run training step")
    parser.add_argument("--run_inference", action="store_true", help="Run inference step")
    parser.add_argument("--gpu_list", type=int, nargs="+", default=[0,1,2,3,4,5,6,7], help="List of GPU IDs to use")
    parser.add_argument("--inference_mode", type=str, default="sequential", choices=["sequential","parallel"], help="Inference mode")
    parser.add_argument("--run_regression_viz", action="store_true", help="Run regression visualization step")

    # Experiment agent paths
    for key in default_paths.keys():
        parser.add_argument(f"--{key}", type=str, default=default_paths[key], help=f"Override path for {key} (default: {default_paths[key]})")

    args = parser.parse_args()

    # Collect overridden paths
    experiment_paths = {key: getattr(args, key) for key in default_paths.keys()}

    run_visage_pipeline(
        base_dir=args.base_dir,
        base_data_dir=args.base_data_dir,
        base_output_dir=args.base_output_dir,
        run_literature=args.run_literature,
        run_experiment=args.run_experiment,
        generate_datasets=args.generate_datasets,
        run_training=args.run_training,
        run_inference=args.run_inference,
        gpu_list=args.gpu_list,
        inference_mode=args.inference_mode,
        experiment_paths=experiment_paths,
        run_regression_viz=args.run_regression_viz,
    )


"""
==============================
VISAGE Unified Pipeline
==============================

Functionality Overview:
-----------------------
1. Literature Analysis
   - Uses the `literature_agent` to analyze literature data and generate a knowledge base.
   - Main input path: `literature_base_dir`
   - Outputs can be used in downstream processing.

2. Experiment Agent (Data Processing)
   - Uses `experiment_agent_run` to process city image data, remote sensing (RS) images, and street view images.
   - All input/output paths can be overridden via command-line arguments.
   - Main processing steps:
       * City street view image folders (`image_folders`)
       * Remote sensing images (`rs_image_base_dir`)
       * Street view images and cue detection (`street_cue_detection_base`)
       * Analysis results JSONL files (`segregation_jsonl_paths`, `income_distribution_paths`)
       * QA data integration (`original_qa`, `final_qa_with_caption`)
   - Outputs are ready for model training or further analysis.

3. Dataset Generation, Training, Inference, and Regression Visualization
   - Generates K-fold datasets for training and validation.
   - Supports training using the `perception_agent` script with multi-GPU support.
   - Inference can be run in sequential or parallel mode.
   - After training and inference, optionally runs tract-level regression analysis (SVR & KNN)
     to predict segregation metrics from VISAGE embeddings.
   - Aggregates fold-wise predictions, visualizes combined results, and saves evaluation metrics.
   - Key parameters:
       * `input_json_path`: Path to QA dataset
       * `datasets_mixture_path`: Path to datasets mixture configuration
       * `output_dir`: Directory to save trained models or inference results
       * `run_regression_viz`: Flag to enable regression visualization
       * `base_data_dir`: Directory containing fold-wise data for regression
       * `segregation_jsonl_paths`: List of JSONL files with ground-truth segregation metrics

Command-Line Usage Example:
---------------------------
# Run the full pipeline with default paths
python closed_loop_workflow.py

# Override specific paths via command-line arguments
python closed_loop_workflow.py \
    --base_image_folder "/new/path/to/base_images" \
    --rs_image_base_dir "/new/path/to/rs_images" \
    --original_qa "/new/path/to/qa.json" \
    --final_qa_with_caption "/new/path/to/qa_with_caption.json"

# Run only experiment agent with a custom city list
python closed_loop_workflow.py --run_experiment --cities "NYC,LA,Chicago"

# Generate K-fold datasets for training and optionally run regression visualization
python closed_loop_workflow.py --generate_datasets --k_fold 5 --run_regression_viz

Notes:
------
- All paths in `default_paths` have meaningful defaults for typical use cases.
- Command-line arguments always override defaults.
- The pipeline is modular: you can run only literature analysis, data processing, training/inference,
  or regression visualization separately.
"""


"""
Default Paths Explanation:
--------------------------

| Path Name                        | Default Value Example                                                                                       | Description                                                                 |
|----------------------------------|------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| base_image_folder                 | /data3/maruolong/VISAGE/data/raw/imagery                                                                    | Base folder containing city image datasets.                                 |
| image_folders                     | /data3/maruolong/VISAGE/data/raw/imagery/{city}/images                                                      | List of folders for each city's images.                                     |
| json1_path                        | /data3/maruolong/Train_Data/Urbanarea_dominant_race_data_all.json                                        | JSON file with urban area demographic data.                                 |
| segregation_jsonl_paths           | /data3/maruolong/VISAGE/data/raw/mobility/visit_data/{city}_2019/{city}_2019_tract_segregation.jsonl        | Segregation metrics per tract for each city.                                |
| income_distribution_paths         | /data3/maruolong/VISAGE/data/raw/mobility/visit_data/{city}_2019/{city}_2019_tract_income_distribution.jsonl | Income distribution per tract for each city.                                 |
| rs_image_base_dir                  | /data3/maruolong/VISAGE/data/raw/imagery/rs_image_new/merged                                                | Base directory for remote sensing images.                                   |
| codebook_path                      | /data3/maruolong/VISAGE/agents/literature_agent/knowledge_base/codebooks.json                            | Knowledge base codebooks used by literature agent.                          |
| rs_cue_detection                   | /data3/maruolong/VISAGE/data/processed/cue_frequencies/rs_image_cue_detection_results.jsonl                                        | Remote sensing cue detection results.                                       |
| street_cue_detection_base          | /data3/maruolong/VISAGE/data/processed/cue_frequencies/street_view_cue_detection                                                  | Base folder containing street view cue detection results.                   |
| rs_aggregated_desc                 | /data3/maruolong/VISAGE/data/processed/cue_frequencies/rs_aggregated_description_by_tract.jsonl                                     | Aggregated remote sensing descriptions by tract.                             |
| street_aggregated_desc             | /data3/maruolong/VISAGE/data/processed/cue_frequencies/aggregated_description_by_tract.jsonl                                        | Aggregated street view descriptions by tract.                                |
| original_qa                        | /data3/maruolong/VISAGE/data/processed/training_data/31_original_qa.json                                                          | Original question-answer dataset.                                           |
| final_qa_with_caption              | /data3/maruolong/VISAGE/data/processed/training_data/31_qa_with_caption.json                                                      | QA dataset enriched with image captions and processed data.                 |

Notes:
------
- Replace `{city}` in paths with the actual city name when running the pipeline.
- These paths can be overridden with command-line arguments for flexibility.
- All paths point to default locations used in the current VISAGE project setup.
"""

"""
Prerequisites / Required Inputs:
--------------------------------
Before running the VISAGE Unified Pipeline, ensure you have the following prepared:

1. City Image Datasets
   - Street view images for all tracts of the target cities.
   - Remote sensing (RS) images covering the same tracts.
   - Paths to these images should be set in:
       * `base_image_folder` (main folder containing city-level folders)
       * `image_folders` (list of per-city image folders)
       * `rs_image_base_dir` (merged remote sensing images)

2. Demographic & Segregation Data
   - Urban area tracts data JSON:
       * `json1_path` 
   - Tract-level segregation ground truth JSONL files per city:
       * `segregation_jsonl_paths`
   - Tract-level income distribution ground truth JSONL files per city:
       * `income_distribution_paths`

3. Street and RS Cue Detection (optional but recommended)
   - Street view cue detection folder: `street_cue_detection_base`
   - Remote sensing cue detection results JSONL: `rs_cue_detection`

4. QA Datasets for LLM Training/Inference
   - Original QA dataset: `original_qa`
   - Final QA dataset enriched with captions: `final_qa_with_caption`

5. LLM API / Model Access
   - API key for LLM inference: `api_key`
   - Base URL for the LLM API (if applicable): `base_url`
   - Name of the LLM model to use: `model`
   - Ensure your LLM account has sufficient tokens or rate limits for full processing.

6. Perception Agent Training Setup
   - Paths to dataset mixture configuration: `datasets_mixture_path`
   - Pretrained model checkpoint (if starting from previous training): `initial_model`
   - Paths to training/inference scripts: `train_script` and `inference_script`
   - GPU availability list: `gpu_list`

Notes:
------
- All paths in `default_paths` assume a standard VISAGE project folder structure.
- Replace `{city}` placeholders with actual city names when overriding paths.
- Missing or incorrectly prepared inputs will result in pipeline errors.
- Users can override any of the default paths via command-line arguments.
"""
