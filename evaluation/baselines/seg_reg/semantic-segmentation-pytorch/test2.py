# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
import json
# Our libs
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from mit_semseg.config import cfg

colors = loadmat('data/color150.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

CITY_LIST = [
    'Albuquerque', 'Baltimore', 'Charlotte', 'Columbus', 'Denver',
    'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas', 'Memphis',
    'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland', 'San Antonio',
    'San Diego', 'San Jose', 'Tucson'
]
BASE_INPUT_ROOT = "/data3/maruolong/segregation/Baseline"
BASE_OUTPUT_ROOT = "/data3/maruolong/segregation/Baseline"

def visualize_result(data, pred, output_jsonl):
    (img, info) = data
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)

    result_data = {
        "image": info.split('/')[-1],
        "predictions": []
    }

    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names.get(uniques[idx] + 1, f"Unknown_{uniques[idx]}")
        ratio = round(counts[idx] / pixs * 100, 2)
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))
            result_data["predictions"].append({"category": name, "percentage": ratio})

    with open(output_jsonl, "a") as jsonl_file:
        jsonl_file.write(json.dumps(result_data) + "\n")

def test(segmentation_module, loader, output_jsonl, gpu):
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

def process_tract(tract_id, img_folder, output_jsonl, segmentation_module, gpu):
    if not os.path.isdir(img_folder):
        print(f"⚠️ 目录 {img_folder} 不存在，跳过该 tract_id")
        return

    print(f"📌 处理 {tract_id}，结果保存在 {output_jsonl}")

    cfg.list_test = [{"fpath_img": os.path.join(img_folder, img)}
                     for img in os.listdir(img_folder)
                     if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not cfg.list_test:
        print(f"⚠️ {tract_id} 目录下没有图片，跳过")
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

    test(segmentation_module, loader_test, output_jsonl, gpu)
    print(f"✅ {tract_id} 处理完成，结果已保存")

def process_city(city_name, segmentation_module, gpu):
    print(f"🌆 开始处理城市：{city_name}")
    input_dir = os.path.join(BASE_INPUT_ROOT, city_name, "images")
    output_dir = os.path.join(BASE_OUTPUT_ROOT, city_name, "segmentation", "result")
    os.makedirs(output_dir, exist_ok=True)

    for tract_id in sorted(os.listdir(input_dir)):
        tract_path = os.path.join(input_dir, tract_id)
        output_jsonl = os.path.join(output_dir, f"{tract_id}_segment_result.jsonl")
        process_tract(tract_id, tract_path, output_jsonl, segmentation_module, gpu)

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

    for city in CITY_LIST:
        process_city(city, segmentation_module, args.gpu)

    print("🎉 所有城市处理完成，结果保存在各自 segmentation/result 目录中")
