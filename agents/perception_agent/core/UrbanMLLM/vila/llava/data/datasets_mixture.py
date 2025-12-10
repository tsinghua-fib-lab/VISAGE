from dataclasses import dataclass, field


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed desciption of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )
    test_script: str = (None,)
    maintainer: str = (None,)


DATASETS = {}

import warnings


def add_dataset(dataset):
    if dataset.dataset_name in DATASETS:
        # make sure the data_name is unique
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    DATASETS.update({dataset.dataset_name: dataset})



def register_datasets_mixtures():

    # Align
    llava_1_5_mm_align = Dataset(
        dataset_name='llava_1_5_align',
        dataset_type='torch',
        data_path='',
        image_path=''
    )
    add_dataset(llava_1_5_mm_align)

    # Pretrain
    coyo_25m = Dataset(
        dataset_name='coyo',
        dataset_type='coyo',
        data_path='./playground/data/coyo-700m/pkl02-split')
    add_dataset(coyo_25m)

    mmc4core = Dataset(
        dataset_name='mmc4core',
        dataset_type='mmc4',
        data_path='./playground/data/mmc4-core/pkl-core')
    add_dataset(mmc4core)

    sharegpt4v_pretrain = Dataset(
        dataset_name="sharegpt4v_pretrain",
        dataset_type="torch",
        data_path="/data2/zhangxin/data/filter-share-captioner_coco_lcs_sam_1246k_1107_692k_noSAM.json",
        image_path="/data3/fengjie/init_ckpt/InternVL-Chat-V1-2-SFT-Data/data",
    )
    
    add_dataset(sharegpt4v_pretrain)

    # SFT
    sharegpt4v_gpt4_100k = Dataset(
        dataset_name="sharegpt4v_gpt4_100k",
        dataset_type="torch",
        data_path="./playground/datasharegpt_video/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k.json",
        image_path="./playground/datasharegpt_video/ShareGPT4V/data",
    )
    add_dataset(sharegpt4v_gpt4_100k)
    llava_instruct_150k = Dataset(
        dataset_name="llava_instruct_150k",
        dataset_type="torch",
        data_path="/data3/fengjie/init_ckpt/LLaVA-Instruct-150K/llava_instruct_150k.json",
        image_path="/data3/fengjie/init_ckpt/InternVL-Chat-V1-2-SFT-Data/data/coco/train2017",
        description="",
    )
    add_dataset(llava_instruct_150k)
    llava_instruct = Dataset(
        dataset_name="llava_instruct",
        dataset_type="torch",
        #data_path="/mnt/public/vila-1.5/data_sv/si_our_task_data.json",
        data_path="/data3/zhangxin/wuwen/vila-1.5/data/sft_data/si_sv_single_wuwen.json",
        #data_path="/mnt/public/vila-1.5/data_sv/all_si_sv_single.json",
        #data_path= "/mnt/public/vila-1.5/data/ft_sv_data.json",
        image_path="",
        #image_path="/mnt/public/",
        description="",
    )
    add_dataset(llava_instruct)


    multi_image_instruct = Dataset(
        dataset_name="multi_image_instruct",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/vila-1.5/data/sft_data/si_sv_multi_wuwen.json",
        #data_path= "/mnt/public/vila-1.5/data/ft_sv_data.json",
        image_path="",
        #image_path="/mnt/public/",
        description="",
    )
    add_dataset(multi_image_instruct)
    
    multi_image_instruct_6 = Dataset(
        dataset_name="multi_image_instruct_6",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/vila-1.5/data/sft_data/si_sv_multi_6_ip.json",
        #data_path= "/mnt/public/vila-1.5/data/ft_sv_data.json",
        image_path="",
        #image_path="/mnt/public/",
        description="",
    )
    add_dataset(multi_image_instruct_6)
    
    multi_image_instruct_4 = Dataset(
        dataset_name="multi_image_instruct_4",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/vila-1.5/data/sft_data/si_sv_multi_4_ip.json",
        #data_path= "/mnt/public/vila-1.5/data/ft_sv_data.json",
        image_path="",
        #image_path="/mnt/public/",
        description="",
    )
    add_dataset(multi_image_instruct_4)
    multi_image_instruct_2 = Dataset(
        dataset_name="multi_image_instruct_2",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/vila-1.5/data/sft_data/si_sv_multi_2_ip.json",
        #data_path= "/mnt/public/vila-1.5/data/ft_sv_data.json",
        image_path="",
        #image_path="/mnt/public/",
        description="",
    )
    add_dataset(multi_image_instruct_2)
    
    gini_data = Dataset(
        dataset_name="gini_data",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/inequality_data/CoT_and_Original_Data/Original_Data/gc_5008_train_calculate_gini_value.json",
        image_path="",
        description="",
    )
    add_dataset(gini_data)


    gini_cot_data = Dataset(
        dataset_name="gini_cot_data",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/inequality_data/CoT_and_Original_Data/CoT_Data/gc_CoT_train.json",
        image_path="",
        description="",
    )
     
    add_dataset(gini_cot_data)
        
    gini_cot_10_data = Dataset(
        dataset_name="gini_cot_10_data",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/inequality_data/change_path_number/gc/10_images_gc_CoT_train.json",
        image_path="",
        description="",
    )
    add_dataset(gini_cot_10_data)
    gini_cot_5_data = Dataset(
        dataset_name="gini_cot_5_data",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/inequality_data/change_path_number/gc/5_images_gc_CoT_train.json",
        image_path="",
        description="",
    )
    add_dataset(gini_cot_5_data)        

    dominant_race_cot_data = Dataset(
        dataset_name="dominant_race_cot_data",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/inequality_data/dr/Original/dr_5008_train.json",
        image_path="",
        description="",
    )
    add_dataset(dominant_race_cot_data)
    dominant_race_cot_10_data = Dataset(
        dataset_name="dominant_race_cot_10_data",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/inequality_data/change_path_number/dr/10_images_dr_CoT_train.json",
        image_path="",
        description="",
    )
    add_dataset(dominant_race_cot_10_data)
    dominant_race_cot_10_data = Dataset(
        dataset_name="dominant_race_cot_10_data",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/inequality_data/change_path_number/dr/10_images_dr_CoT_train.json",
        image_path="",
        description="",
    )
    add_dataset(dominant_race_cot_10_data)   
    
    dominant_race_cot_5_data = Dataset(
        dataset_name="dominant_race_cot_5_data",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/inequality_data/change_path_number/dr/5_images_dr_CoT_train.json",
        image_path="",
        description="",
    )
    add_dataset(dominant_race_cot_5_data)   
    dominant_race_data = Dataset(
        dataset_name="dominant_race_data",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/inequality_data/dr/Original/dr_5008_train.json",
        image_path="",
        description="",
    )
    add_dataset(dominant_race_data)
    
        
    white_black_income_ratio_data = Dataset(
        dataset_name="white_black_income_ratio_data",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/inequality_data/bw/Original/bw_5008_train_new_value.json",
        image_path="",
        description="",
    )
    add_dataset(white_black_income_ratio_data)

    white_black_income_ratio_cot_data = Dataset(
        dataset_name="white_black_income_ratio_cot_data",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/inequality_data/bw/CoT/bw_CoT_train_change_prompt.json",
        image_path="",
        description="",
    )
    add_dataset(white_black_income_ratio_cot_data)


    white_black_income_ratio_cot_10_data = Dataset(
        dataset_name="white_black_income_ratio_cot_10_data",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/inequality_data/change_path_number/bw/10_images_bw_CoT_train.json",
        image_path="",
        description="",
    )
    add_dataset(white_black_income_ratio_cot_10_data)

    white_black_income_ratio_cot_5_data = Dataset(
        dataset_name="white_black_income_ratio_cot_5_data",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/inequality_data/change_path_number/bw/5_images_bw_CoT_train.json",
        image_path="",
        description="",
    )
    add_dataset(white_black_income_ratio_cot_5_data)

    segregation_cot_40_data = Dataset(
        dataset_name="segregation_cot_40_data",
        dataset_type="torch",
        data_path="/data3/maruolong/segregation/Train/Try/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_data)

    segregation_cot_40_data_update1 = Dataset(
        dataset_name="segregation_cot_40_data_update1",
        dataset_type="torch",
        data_path="/data3/maruolong/segregation/Train/Try/update_train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_data_update1)

    segregation_cot_40_data_update2 = Dataset(
        dataset_name="segregation_cot_40_data_update2",
        dataset_type="torch",
        data_path="/data3/maruolong/segregation/Train/Try/update_prompt_train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_data_update2)

    segregation_cot_40_data_update3 = Dataset(
        dataset_name="segregation_cot_40_data_update3",
        dataset_type="torch",
        data_path="/data3/maruolong/segregation/Train/Try/add_caption/update_caption_train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_data_update3)

    segregation_cot_40_31_data2 = Dataset(
        dataset_name="segregation_cot_40_31_data2",
        dataset_type="torch",
        data_path="/data3/maruolong/segregation/Train/Try/update/31_cities/data2/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31_data2)

    segregation_cot_40_31_data1 = Dataset(
        dataset_name="segregation_cot_40_31_data1",
        dataset_type="torch",
        data_path="/data3/maruolong/segregation/Train/Try/update/31_cities/data1/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31_data1)

    segregation_cot_40_31_data3 = Dataset(
        dataset_name="segregation_cot_40_31_data3",
        dataset_type="torch",
        data_path="/data3/maruolong/segregation/Train/Try/update/31_cities/data3/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31_data3)

    segregation_cot_40_31_data4 = Dataset(
        dataset_name="segregation_cot_40_31_data4",
        dataset_type="torch",
        data_path="/data3/maruolong/segregation/Train/Try/update/31_cities/data4/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31_data4)

    segregation_cot_40_31_data5 = Dataset(
        dataset_name="segregation_cot_40_31_data5",
        dataset_type="torch",
        data_path="/data3/maruolong/segregation/Train/Try/update/31_cities/data5/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31_data5)

    segregation_cot_40_31no_data2 = Dataset(
        dataset_name="segregation_cot_40_31no_data2",
        dataset_type="torch",
        data_path="/data3/maruolong/segregation/Train/Try/update/31_cities_no/data2/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31no_data2)

    segregation_cot_40_31no_data1 = Dataset(
        dataset_name="segregation_cot_40_31no_data1",
        dataset_type="torch",
        data_path="/data3/maruolong/segregation/Train/Try/update/31_cities_no/data1/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31no_data1)

    segregation_cot_40_31no_data3 = Dataset(
        dataset_name="segregation_cot_40_31no_data3",
        dataset_type="torch",
        data_path="/data3/maruolong/segregation/Train/Try/update/31_cities_no/data2/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31no_data3)

    segregation_cot_40_31no_data4 = Dataset(
        dataset_name="segregation_cot_40_31no_data4",
        dataset_type="torch",
        data_path="/data3/maruolong/segregation/Train/Try/update/31_cities_no/data2/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31no_data4)

    segregation_cot_40_31no_data5 = Dataset(
        dataset_name="segregation_cot_40_31no_data5",
        dataset_type="torch",
        data_path="/data3/maruolong/segregation/Train/Try/update/31_cities_no/data2/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31no_data5)

    citybench_pop = Dataset(
        dataset_name="citybench_pop",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/vila-1.5/data/sft_data/citybench_pop.json",
        image_path="",
        description="",
    )
    add_dataset(citybench_pop)
    
    citybench_sv = Dataset(
        dataset_name="citybench_sv",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/vila-1.5/data/sft_data/citybench_sv.json",
        image_path="",
        description="",
    )
    add_dataset(citybench_sv)
    
    vqa_class =Dataset(
        dataset_name="vqa_class",
        dataset_type="torch",
        data_path="/data3/zhangxin/wuwen/vila-1.5/data/sft_data/vqa_class_1116.json",
        image_path="/data2/zhangxin/llm/",
        description="",
    )
    add_dataset(vqa_class)
    sharegpt4v_sft = Dataset(
        dataset_name='sharegpt4v_sft',
        dataset_type='torch',
        data_path='./playground/data/sharegpt4v/sharegpt4v_mix738k_remove_sa.json',
        image_path='./playground/data'
    )
    add_dataset(sharegpt4v_sft)

    dvqa_train_200k = Dataset(
        dataset_name="dvqa_train_200k",
        dataset_type="torch",
        data_path="./playground/data/dvqa_train_200k.jsonl",
        image_path="./playground/data/dvqa",
        description="",
    )
    add_dataset(dvqa_train_200k)

    chartqa_train_18k = Dataset(
        dataset_name="chartqa_train_18k",
        dataset_type="torch",
        data_path="./playground/data/chartqa_train_18k.jsonl",
        image_path="./playground/data/chartqa",
        description="",
    )
    add_dataset(chartqa_train_18k)

    ai2d_train_12k = Dataset(
        dataset_name="ai2d_train_12k",
        dataset_type="torch",
        data_path="./playground/data/ai2d_train_12k.jsonl",
        image_path="./playground/data/ai2d",
        description="",
    )
    add_dataset(ai2d_train_12k)

    docvqa_train_10k = Dataset(
        dataset_name="docvqa_train_10k",
        dataset_type="torch",
        data_path="./playground/data/docvqa_train_10k.jsonl",
        image_path="./playground/data/docvqa",
        description="",
    )
    add_dataset(docvqa_train_10k)

    geoqa = Dataset(
        dataset_name="geoqa",
        dataset_type="torch",
        data_path="./playground/data/geoqa+.jsonl",
        image_path="./playground/data/geoqa+",
        description="",
    )
    add_dataset(geoqa)

    synthdog_en = Dataset(
        dataset_name="synthdog_en",
        dataset_type="torch",
        data_path="./playground/data/synthdog_en.jsonl",
        image_path="./playground/data/synthdog-en",
        description="",
    )
    add_dataset(synthdog_en)

    vflan = Dataset(
        dataset_name='vflan',
        dataset_type='vflan',
        data_path='./playground/data/vlm-flan-clean-text1m-nosqa-sharded'
    )
    add_dataset(vflan)


    scienceqa = Dataset(
        dataset_name="scienceqa",
        dataset_type="torch",
        data_path="./playground/data/scienceqa/scienceqa_train_12k.json",
        image_path="./playground/data/scienceqa/images",
    )
    add_dataset(scienceqa)

    
    sherlock = Dataset(
        dataset_name="sherlock",
        dataset_type="torch",
        data_path="./playground/data/sherlock/processed/sherlock_317k.json",
        image_path="./playground/data/sherlock/images",
    )
    add_dataset(sherlock)
    math = Dataset(
        dataset_name="math",
        dataset_type="vflan",
        data_path="./playground/data/math",
    )
    add_dataset(math)

    wit_subset = Dataset(
        dataset_name="wit_subset",
        dataset_type="torch",
        data_path="./playground/data/WIT/wit_1_8m/wit_processed_538k.json",
        image_path="./playground/data/WIT/wit_1_8m/images"
    )
    add_dataset(wit_subset)

    youcook2 = Dataset(
        dataset_name="youcook2",
        dataset_type="torch",
        data_path="./playground/data/youcook2/youcook_filtered_v3.json",
        image_path="./playground/data/youcook2/video_data_clipped",
    )
    add_dataset(youcook2)
    
    vatex = Dataset(
        dataset_name="vatex",
        dataset_type="torch",
        data_path="./playground/data/vatex/vatex_filtered_v3.json",
        image_path="./playground/data/vatex/videos_clipped",
    )
    add_dataset(vatex)

    video_chatgpt = Dataset(
        dataset_name="video_chatgpt",
        dataset_type="torch",
        data_path="./playground/data/Video_ChatGPT/VideoInstruct-100K/VideoInstruct100K.json",
        image_path="./playground/data/Video_ChatGPT/activitynet_videos/",
    )
    add_dataset(video_chatgpt)

    shot2story_shotonly = Dataset(
        dataset_name="shot2story_shotonly",
        dataset_type="torch",
        data_path="./playground/data/shot2story/shot2story_shotonly.json",
        image_path="./playground/data/shot2story/Shot2Story/data/videos_extracted",
    )
    add_dataset(shot2story_shotonly)

    sharegpt_video = Dataset(
        dataset_name="sharegpt_video",
        dataset_type="torch",
        data_path="./playground/data/sharegpt_video/video_caption_pretrain.json",
        image_path="./playground/data/sharegpt_video/videos",
    )
    add_dataset(sharegpt_video)

    segregation_cot_40_31_data1 = Dataset(
        dataset_name="segregation_cot_40_31_data1",
        dataset_type="torch",
        data_path="/data3/maruolong/VISAGE/data/31_cities/data1/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31_data1)
    
    segregation_cot_40_31_data2 = Dataset(
        dataset_name="segregation_cot_40_31_data2",
        dataset_type="torch",
        data_path="/data3/maruolong/VISAGE/data/31_cities/data2/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31_data2)
    
    segregation_cot_40_31_data3 = Dataset(
        dataset_name="segregation_cot_40_31_data3",
        dataset_type="torch",
        data_path="/data3/maruolong/VISAGE/data/31_cities/data3/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31_data3)
    
    segregation_cot_40_31_data4 = Dataset(
        dataset_name="segregation_cot_40_31_data4",
        dataset_type="torch",
        data_path="/data3/maruolong/VISAGE/data/31_cities/data4/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31_data4)
    
    segregation_cot_40_31_data5 = Dataset(
        dataset_name="segregation_cot_40_31_data5",
        dataset_type="torch",
        data_path="/data3/maruolong/VISAGE/data/31_cities/data5/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31_data5)
    

    segregation_cot_40_31_data1 = Dataset(
        dataset_name="segregation_cot_40_31_data1",
        dataset_type="torch",
        data_path="/data3/maruolong/VISAGE/data/31_cities/data1/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31_data1)
    
    segregation_cot_40_31_data2 = Dataset(
        dataset_name="segregation_cot_40_31_data2",
        dataset_type="torch",
        data_path="/data3/maruolong/VISAGE/data/31_cities/data2/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31_data2)
    
    segregation_cot_40_31_data3 = Dataset(
        dataset_name="segregation_cot_40_31_data3",
        dataset_type="torch",
        data_path="/data3/maruolong/VISAGE/data/31_cities/data3/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31_data3)
    
    segregation_cot_40_31_data4 = Dataset(
        dataset_name="segregation_cot_40_31_data4",
        dataset_type="torch",
        data_path="/data3/maruolong/VISAGE/data/31_cities/data4/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31_data4)
    
    segregation_cot_40_31_data5 = Dataset(
        dataset_name="segregation_cot_40_31_data5",
        dataset_type="torch",
        data_path="/data3/maruolong/VISAGE/data/31_cities/data5/train_data.json",
        image_path="",
        description="",
    )
    add_dataset(segregation_cot_40_31_data5)
    




    