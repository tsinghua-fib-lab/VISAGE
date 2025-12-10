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

# # 自定义输出文件夹
# OUTPUT_DIR = "/data3/maruolong/segregation/Try/Image_Segmentation/Test_Images"
# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)
# 设定输入文件夹和输出文件夹
INPUT_DIR = "/data3/maruolong/segregation/Baseline/Austin/images"  # 存放各 tract_id 目录的文件夹
OUTPUT_DIR = "/data3/maruolong/segregation/Baseline/Austin/segmentation/result"  # 存放各 tract_id 的 jsonl 文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 确保 segmentation_result 目录存在
# OUTPUT_JSONL = ""

def visualize_result(data, pred, output_jsonl):
    (img, info) = data

    # 计算类别比例
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)

    result_data = {
        "image": info.split('/')[-1],
        "predictions": []
    }

    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:  # 按比例降序排列
        name = names.get(uniques[idx] + 1, f"Unknown_{uniques[idx]}")
        ratio = round(counts[idx] / pixs * 100, 2)  # 计算百分比
        if ratio > 0.1:  # 过滤掉占比很小的类别
            print("  {}: {:.2f}%".format(name, ratio))
            result_data["predictions"].append({"category": name, "percentage": ratio})

    # 追加写入 JSONL 文件
    with open(output_jsonl, "a") as jsonl_file:
        jsonl_file.write(json.dumps(result_data) + "\n")



def test(segmentation_module, loader, output_jsonl, gpu):
    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
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

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        # visualization
        visualize_result(
            (batch_data['img_ori'], batch_data['info']),
            pred,
            output_jsonl
        )

        pbar.update(1)

def process_tract(tract_id, gpu):
    """ 处理单个 tract_id 目录，生成 JSONL 文件 """
    img_folder = os.path.join(INPUT_DIR, tract_id)
    output_jsonl = os.path.join(OUTPUT_DIR, f"{tract_id}_segment_result.jsonl")

    if not os.path.isdir(img_folder):
        print(f"⚠️ 目录 {img_folder} 不存在，跳过该 tract_id")
        return

    print(f"📌 处理 {tract_id}，结果保存在 {output_jsonl}")

    # 解析数据集
    cfg.list_test = [{"fpath_img": os.path.join(img_folder, img)} for img in os.listdir(img_folder) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
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

    # 加载模型
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
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.cuda()

    # 进行推理并保存结果
    test(segmentation_module, loader_test, output_jsonl, gpu)
    print(f"✅ {tract_id} 处理完成，结果已保存")



if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Testing")
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="GPU id for evaluation"
    )
    args = parser.parse_args()

    # 读取配置文件
    cfg.merge_from_file(args.cfg)

    logger = setup_logger(distributed_rank=0)
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # 绝对路径的模型权重
    cfg.MODEL.weights_encoder = os.path.join(cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

    assert os.path.exists(cfg.MODEL.weights_encoder) and \
           os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exist!"

    # 设定 GPU
    torch.cuda.set_device(args.gpu)

    # 遍历 `images/` 目录，处理每个 tract_id
    for tract_id in sorted(os.listdir(INPUT_DIR)):
        process_tract(tract_id, args.gpu)

    print("🎉 所有 tract 处理完成，JSONL 结果保存在 `segmentation/result/`")

# # System libs
# import os
# import argparse
# from distutils.version import LooseVersion
# # Numerical libs
# import numpy as np
# import torch
# import torch.nn as nn
# from scipy.io import loadmat
# import csv
# import json
# # Our libs
# from mit_semseg.dataset import TestDataset
# from mit_semseg.models import ModelBuilder, SegmentationModule
# from mit_semseg.utils import colorEncode, find_recursive, setup_logger
# from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
# from mit_semseg.lib.utils import as_numpy
# from PIL import Image
# from tqdm import tqdm
# from mit_semseg.config import cfg

# # 加载类别颜色和名称
# colors = loadmat('data/color150.mat')['colors']
# names = {}
# with open('data/object150_info.csv') as f:
#     reader = csv.reader(f)
#     next(reader)
#     for row in reader:
#         names[int(row[0])] = row[5].split(";")[0]

# # 自定义输出文件夹
# OUTPUT_DIR = "./my_output_images"
# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)

# OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "predictions.jsonl")  # 存类别比例的 jsonl 文件
# OUTPUT_TXT = os.path.join(OUTPUT_DIR, "predictions.txt")  # 存类别比例的 txt 文件


# def visualize_result(data, pred, cfg):
#     (img, info) = data

#     # 计算类别比例
#     pred = np.int32(pred)
#     pixs = pred.size
#     uniques, counts = np.unique(pred, return_counts=True)

#     result_data = {
#         "image": info.split('/')[-1],
#         "predictions": []
#     }

#     print("Predictions in [{}]:".format(info))
#     for idx in np.argsort(counts)[::-1]:  # 按比例降序排列
#         name = names[uniques[idx] + 1]
#         ratio = round(counts[idx] / pixs * 100, 2)  # 计算百分比
#         if ratio > 0.1:  # 过滤掉占比很小的类别
#             print("  {}: {:.2f}%".format(name, ratio))
#             result_data["predictions"].append({"category": name, "percentage": ratio})

#     # 颜色编码预测结果
#     pred_color = colorEncode(pred, colors).astype(np.uint8)

#     # 组合原图和预测图
#     im_vis = np.concatenate((img, pred_color), axis=1)

#     # 生成文件名
#     img_name = info.split('/')[-1].replace('.jpg', '.png')

#     # 保存预测图像到自定义文件夹
#     output_img_path = os.path.join(OUTPUT_DIR, img_name)
#     Image.fromarray(im_vis).save(output_img_path)
#     print(f"Saved result to {output_img_path}")

#     # 追加写入 JSONL 文件
#     with open(OUTPUT_JSONL, "a") as jsonl_file:
#         jsonl_file.write(json.dumps(result_data) + "\n")

#     # 追加写入 TXT 文件
#     with open(OUTPUT_TXT, "a") as txt_file:
#         txt_file.write(f"Image: {result_data['image']}\n")
#         for item in result_data["predictions"]:
#             txt_file.write(f"  {item['category']}: {item['percentage']}%\n")
#         txt_file.write("\n")


# def test(segmentation_module, loader, gpu):
#     segmentation_module.eval()

#     pbar = tqdm(total=len(loader))
#     for batch_data in loader:
#         # process data
#         batch_data = batch_data[0]
#         segSize = (batch_data['img_ori'].shape[0],
#                    batch_data['img_ori'].shape[1])
#         img_resized_list = batch_data['img_data']

#         with torch.no_grad():
#             scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
#             scores = async_copy_to(scores, gpu)

#             for img in img_resized_list:
#                 feed_dict = batch_data.copy()
#                 feed_dict['img_data'] = img
#                 del feed_dict['img_ori']
#                 del feed_dict['info']
#                 feed_dict = async_copy_to(feed_dict, gpu)

#                 # forward pass
#                 pred_tmp = segmentation_module(feed_dict, segSize=segSize)
#                 scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

#             _, pred = torch.max(scores, dim=1)
#             pred = as_numpy(pred.squeeze(0).cpu())

#         # visualization
#         visualize_result(
#             (batch_data['img_ori'], batch_data['info']),
#             pred,
#             cfg
#         )

#         pbar.update(1)


# def main(cfg, gpu):
#     torch.cuda.set_device(gpu)

#     # Network Builders
#     net_encoder = ModelBuilder.build_encoder(
#         arch=cfg.MODEL.arch_encoder,
#         fc_dim=cfg.MODEL.fc_dim,
#         weights=cfg.MODEL.weights_encoder)
#     net_decoder = ModelBuilder.build_decoder(
#         arch=cfg.MODEL.arch_decoder,
#         fc_dim=cfg.MODEL.fc_dim,
#         num_class=cfg.DATASET.num_class,
#         weights=cfg.MODEL.weights_decoder,
#         use_softmax=True)

#     crit = nn.NLLLoss(ignore_index=-1)

#     segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

#     # Dataset and Loader
#     dataset_test = TestDataset(
#         cfg.list_test,
#         cfg.DATASET)
#     loader_test = torch.utils.data.DataLoader(
#         dataset_test,
#         batch_size=cfg.TEST.batch_size,
#         shuffle=False,
#         collate_fn=user_scattered_collate,
#         num_workers=5,
#         drop_last=True)

#     segmentation_module.cuda()

#     # Main loop
#     test(segmentation_module, loader_test, gpu)

#     print('Inference done!')


# if __name__ == '__main__':
#     assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
#         'PyTorch>=0.4.0 is required'

#     parser = argparse.ArgumentParser(
#         description="PyTorch Semantic Segmentation Testing"
#     )
#     parser.add_argument(
#         "--imgs",
#         required=True,
#         type=str,
#         help="an image path, or a directory name"
#     )
#     parser.add_argument(
#         "--cfg",
#         default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
#         metavar="FILE",
#         help="path to config file",
#         type=str,
#     )
#     parser.add_argument(
#         "--gpu",
#         default=0,
#         type=int,
#         help="gpu id for evaluation"
#     )
#     parser.add_argument(
#         "opts",
#         help="Modify config options using the command-line",
#         default=None,
#         nargs=argparse.REMAINDER,
#     )
#     args = parser.parse_args()

#     cfg.merge_from_file(args.cfg)
#     cfg.merge_from_list(args.opts)

#     logger = setup_logger(distributed_rank=0)
#     logger.info("Loaded configuration file {}".format(args.cfg))
#     logger.info("Running with config:\n{}".format(cfg))

#     # generate testing image list
#     if os.path.isdir(args.imgs):
#         imgs = find_recursive(args.imgs)
#     else:
#         imgs = [args.imgs]
#     cfg.list_test = [{'fpath_img': x} for x in imgs]

#     if not os.path.isdir(cfg.TEST.result):
#         os.makedirs(cfg.TEST.result)

#     main(cfg, args.gpu)
