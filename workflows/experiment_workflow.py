"""
Unified VISAGE Data Processing Pipeline Entry Point
Function: experiment_agent_run

This module wraps the full VISAGE pipeline into a single callable function.
Users can provide their own API configuration and file paths. If not provided,
default paths will be used.

All user-facing parameters:
- api_key: Your API key
- base_url: Your API base URL
- model: Model name
- max_tokens: Max tokens for API calls
- max_retries: Retry count for API calls
- max_wait: Timeout for API calls
- custom_paths: (dict) Override any default path

All comments are in English.
"""

import sys
import os
import json

# Add project root to Python path
sys.path.append("/data3/maruolong/VISAGE/agents/experiment_agent/core")

from cue_detector_satellite import RemoteSensingAnalyzer
from cue_detector_street import StreetViewAnalyzer
from frequency_aggregator import CaptionProcessor
from original_qa_generator import QADataGenerator
from qa_generator import CaptionGenerator


def experiment_agent_run(
    api_key="your_api_key_here",            # <-- user should replace with their own API key
    base_url="your_base_url_here",          # <-- e.g., "https://api.siliconflow.cn/v1"
    model="your_model_name_here",           # <-- e.g., "Pro/Qwen/Qwen2.5-VL-7B-Instruct"
    max_tokens=500,
    max_retries=10,
    max_wait=600,
    custom_paths=None
):
    """
    Unified interface for running the VISAGE pipeline.

    Parameters:
        api_key (str): User's API key.
        base_url (str): API base URL.
        model (str): Large-model name.
        max_tokens (int): API token limit.
        max_retries (int): Retry attempts for API calls.
        max_wait (int): Maximum wait (seconds).
        custom_paths (dict): Optional dict to override default file paths.

    Returns:
        dict: Summary statistics and final output paths.
    """

    # -----------------------------
    # Default file paths
    # -----------------------------
    cities = [
        'Boston', 'Chicago', 'Dallas', 'Detroit', 'Los Angeles', 'Miami',
        'New York', 'Philadelphia', 'San Francisco', 'Seattle', 'Washington',
        'Albuquerque', 'Austin', 'Baltimore', 'Charlotte', 'Columbus', 'Denver',
        'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas',
        'Memphis', 'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland',
        'San Antonio', 'San Diego', 'San Jose', 'Tucson'
    ]

    default_paths = {
        "base_image_folder": f"/data3/maruolong/VISAGE/data/raw/imagery",
        "image_folders": [f"/data3/maruolong/VISAGE/data/raw/imagery/{city}/images" for city in cities],
        "json1_path": "/data3/maruolong/Train_Data/Urbanarea_dominant_race_data_all.json",
        "segregation_jsonl_paths": [f"/data3/maruolong/VISAGE/data/mobility/visit_data/{city}_2019/{city}_2019_tract_segregation.jsonl" for city in cities],
        "income_distribution_paths": [f"/data3/maruolong/VISAGE/data/mobility/visit_data/{city}_2019/{city}_2019_tract_income_distribution.jsonl" for city in cities],
        "rs_image_base_dir": "/data3/maruolong/VISAGE/data/raw/imagery/rs_image_new/merged",

        "codebook_path": "/data3/maruolong/VISAGE/agents/literature_agent/knowledge_base/codebooks.json",
        # "codebook_path": "/data3/maruolong/VISAGE/data/codebooks.json",

        "rs_cue_detection": "/data3/maruolong/VISAGE/data/processed/cue_frequecies/rs_image_cue_detection_results.jsonl",
        "street_cue_detection_base": "/data3/maruolong/VISAGE/data/processed/cue_frequecies/street_view_cue_detection",
        "rs_aggregated_desc": "/data3/maruolong/VISAGE/data/processed/cue_frequecies/rs_aggregated_description_by_tract.jsonl",
        "street_aggregated_desc": "/data3/maruolong/VISAGE/data/processed/cue_frequecies/street_aggregated_description_by_tract.jsonl",
        "original_qa": "/data3/maruolong/VISAGE/data/processed/training_data/31_original_qa.json",
        "final_qa_with_caption": "/data3/maruolong/VISAGE/data/processed/training_data/31_qa_with_caption.json"
    }

    # Merge user-provided paths
    if custom_paths:
        default_paths.update(custom_paths)

    print("\n🚀 Starting VISAGE Data Processing Pipeline (experiment_agent_run)")
    print("=" * 60)

    # -----------------------------
    # Step 1: Remote Sensing Cue Detection
    # -----------------------------
    print("\n📡 STEP 1: Processing Remote Sensing Images")
    rs_analyzer = RemoteSensingAnalyzer(
        codebook_path=default_paths["codebook_path"],
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        max_retries=max_retries,
        max_wait=max_wait
    )
    rs_stats = rs_analyzer.process_single_folder(
        input_folder=default_paths["rs_image_base_dir"],
        output_jsonl=default_paths["rs_cue_detection"]
    )

    # -----------------------------
    # Step 2: Street View Cue Detection
    # -----------------------------
    print("\n🏙️ STEP 2: Processing Street View Images")
    street_analyzer = StreetViewAnalyzer(
        codebook_path=default_paths["codebook_path"],
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        max_retries=max_retries,
        max_wait=max_wait
    )
    street_stats = street_analyzer.process_all_cities(
        base_image_folder=default_paths["base_image_folder"],
        output_base_folder=default_paths["street_cue_detection_base"]
    )

    # -----------------------------
    # Step 3: Caption Aggregation
    # -----------------------------
    print("\n📊 STEP 3: Aggregating Captions")
    caption_processor = CaptionProcessor(
        codebook_path=default_paths["codebook_path"]
    )
    caption_processor.run_all(
        rs_input_path=default_paths["rs_cue_detection"],
        rs_output_path=default_paths["rs_aggregated_desc"],
        street_input_root=default_paths["street_cue_detection_base"],
        street_final_output=default_paths["street_aggregated_desc"]
    )

    # -----------------------------
    # Step 4: Generate Original QA Dataset
    # -----------------------------
    print("\n🤖 STEP 4: Generating Original QA Dataset")
    qa_generator = QADataGenerator()
    qa_stats = qa_generator.generate_qa_data(
        image_folders=default_paths["image_folders"],
        output_json_path=default_paths["original_qa"],
        json1_path=default_paths["json1_path"],
        segregation_jsonl_paths=default_paths["segregation_jsonl_paths"],
        income_distribution_paths=default_paths["income_distribution_paths"],
        rs_image_base_dir=default_paths["rs_image_base_dir"]
    )

    # -----------------------------
    # Step 5: Final QA + Caption Integration
    # -----------------------------
    print("\n🎨 STEP 5: Creating Final QA with Captions")
    caption_generator = CaptionGenerator(
        codebook_path=default_paths["codebook_path"]
    )
    caption_generator.process_file(
        input_path=default_paths["original_qa"],
        output_path=default_paths["final_qa_with_caption"],
        remote_desc_path=default_paths["rs_aggregated_desc"],
        street_desc_path=default_paths["street_aggregated_desc"]
    )

    # -----------------------------
    # Summary
    # -----------------------------
    summary = {
        "original_qa": default_paths["original_qa"],
        "final_qa_with_caption": default_paths["final_qa_with_caption"],
        "rs_caption": default_paths["rs_aggregated_desc"],
        "street_caption": default_paths["street_aggregated_desc"],
        "qa_stats": qa_stats
    }

    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("Output Summary:")
    for k, v in summary.items():
        print(f"  - {k}: {v}")

    return summary
