# System libs
import os
import argparse
from distutils.version import LooseVersion
from collections import defaultdict
import json
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
# Our libs
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from mit_semseg.config import cfg

# --- Configuration & Data Loading ---
colors = loadmat('/data3/maruolong/VISAGE/evaluation/baselines/seg_reg/semantic-segmentation-pytorch/data/color150.mat')['colors']
names = {}
# Load object names from object150_info.csv
with open('/data3/maruolong/VISAGE/evaluation/baselines/seg_reg/semantic-segmentation-pytorch/data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

# List of all cities to process
CITY_LIST = [
    'Albuquerque', 'Baltimore', 'Charlotte', 'Columbus', 'Denver',
    'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas', 'Memphis',
    'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland', 'San Antonio',
    'San Diego', 'San Jose', 'Tucson', 'Boston', 'Chicago', 'Dallas', 'Detroit', 
    'Los Angeles', 'Miami', 'New York', 'Philadelphia', 'San Francisco', 
    'Seattle', 'Washington'
]
BASE_INPUT_ROOT = "/data3/maruolong/VISAGE/data/raw/imagery"
BASE_OUTPUT_ROOT = "/data3/maruolong/VISAGE/data/processed/baseline/seg"

# --- Segmentation Functions ---

def visualize_result(data, pred, output_jsonl):
    """
    Analyzes prediction results (pixel counts) and saves the category percentages
    to a JSONL file. Note: The name 'visualize' is historical; this function
    primarily extracts and saves prediction metrics.
    """
    (img, info) = data
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)

    result_data = {
        "image": info.split('/')[-1],
        "predictions": []
    }

    # print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        # unique_id is 0-indexed, names dict is 1-indexed
        name = names.get(uniques[idx] + 1, f"Unknown_{uniques[idx]}") 
        ratio = round(counts[idx] / pixs * 100, 2)
        if ratio > 0.1:
            # print("  {}: {:.2f}%".format(name, ratio))
            result_data["predictions"].append({"category": name, "percentage": ratio})

    with open(output_jsonl, "a") as jsonl_file:
        jsonl_file.write(json.dumps(result_data) + "\n")

def test(segmentation_module, loader, output_jsonl, gpu):
    """Performs segmentation inference on images loaded by the loader."""
    segmentation_module.eval()
    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0], batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        visualize_result((batch_data['img_ori'], batch_data['info']), pred, output_jsonl)
        pbar.update(1)

def process_tract_segmentation(tract_id, img_folder, output_jsonl, segmentation_module, gpu):
    """Configures dataset/loader and runs segmentation for a single census tract."""
    if not os.path.isdir(img_folder):
        print(f"⚠️ Directory {img_folder} not found, skipping tract {tract_id}")
        return

    print(f"📌 Processing tract {tract_id}, outputting to {output_jsonl}")

    # Build list of image paths for the current tract
    cfg.list_test = [{"fpath_img": os.path.join(img_folder, img)}
                     for img in os.listdir(img_folder)
                     if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not cfg.list_test:
        print(f"⚠️ No images found in {tract_id}, skipping.")
        return

    dataset_test = TestDataset(cfg.list_test, cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True
    )

    # Run segmentation
    test(segmentation_module, loader_test, output_jsonl, gpu)
    print(f"✅ Segmentation for {tract_id} completed.")

def process_city_segmentation(city_name, segmentation_module, gpu):
    """Iterates through all tracts in a city for segmentation."""
    print(f"🌆 Starting segmentation for city: {city_name}")
    input_dir = os.path.join(BASE_INPUT_ROOT, city_name, "images")
    output_dir = os.path.join(BASE_OUTPUT_ROOT, "segment_results", f"{city_name}_segmentation", "result")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        print(f"❌ Input directory not found for {city_name}. Skipping.")
        return

    for tract_id in sorted(os.listdir(input_dir)):
        tract_path = os.path.join(input_dir, tract_id)
        if os.path.isdir(tract_path):
            output_jsonl = os.path.join(output_dir, f"{tract_id}_segment_result.jsonl")
            process_tract_segmentation(tract_id, tract_path, output_jsonl, segmentation_module, gpu)

# --- Element Averaging Functions (Merged from Code 2) ---

def calculate_tract_averages(city_name):
    """
    Reads the segmentation results (JSONL files) for a city and computes the 
    average percentage of each semantic element across all images in a tract.
    """
    print(f"\n📊 Starting average calculation for city: {city_name}")
    input_dir = os.path.join(BASE_OUTPUT_ROOT, "segment_results", f"{city_name}_segmentation", "result")
    output_dir = os.path.join(BASE_OUTPUT_ROOT, "segment_results", f"{city_name}_segmentation")
    os.makedirs(output_dir, exist_ok=True)
    OUTPUT_FILE = os.path.join(output_dir, "tract_averages.jsonl")

    tract_files = [f for f in os.listdir(input_dir) if f.endswith("_segment_result.jsonl")]

    with open(OUTPUT_FILE, "w") as output_file:
        for tract_file in tqdm(tract_files, desc=f"Averaging {city_name}"):
            # Extract tract_id from filename (e.g., '12345678901_segment_result.jsonl')
            tract_id = tract_file.split("_")[0] 
            tract_path = os.path.join(input_dir, tract_file)

            category_totals = defaultdict(float)  # Accumulate percentages
            image_count = 0  # Count images in this tract

            with open(tract_path, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        image_count += 1
                        for pred in data["predictions"]:
                            category_totals[pred["category"]] += pred["percentage"]
                    except json.JSONDecodeError:
                        # Skip corrupted lines
                        continue

            # Calculate average (Total percentage / Image count)
            if image_count > 0:
                # Round to 2 decimal places
                category_averages = {cat: round(total / image_count, 2) 
                                     for cat, total in category_totals.items()}
            else:
                category_averages = {}

            tract_result = {"tract_id": tract_id, "averages": category_averages}
            output_file.write(json.dumps(tract_result) + "\n")

    print(f"✅ Averages computed and saved to {OUTPUT_FILE}")

# --- Main Execution ---

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Testing")
    parser.add_argument("--cfg", default="config/ade20k-resnet50dilated-ppm_deepsup.yaml", type=str, help="path to config file")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id for evaluation")
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    logger = setup_logger(distributed_rank=0)
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    cfg.MODEL.weights_encoder = os.path.join(cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exist!"

    torch.cuda.set_device(args.gpu)

    # 1. Build Segmentation Model
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder
    )
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True
    )
    crit = nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit).cuda()

    # 2. Loop through all cities: Segment + Average
    for city in CITY_LIST:
        # A. Semantic Segmentation
        # Disabled by default, assuming segmentation results are already generated
        # process_city_segmentation(city, segmentation_module, args.gpu) 
        
        # B. Calculate Averages from segmentation results
        calculate_tract_averages(city)

    print("🎉 All cities processed: Segmentation results saved and averages calculated.")