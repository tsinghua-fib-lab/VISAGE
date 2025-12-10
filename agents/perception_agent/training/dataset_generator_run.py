from datasetGenerator import KFoldDatasetGenerator

# 使用自定义前缀
generator = KFoldDatasetGenerator(
    input_path="/data3/maruolong/VISAGE/data/31_qa_with_caption.json",
    base_output_dir="/data3/maruolong/VISAGE/data/31_cities",
    datasets_mixture_path="/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/llava/data/datasets_mixture.py",
    dataset_prefix="segregation_cot_40_31_data",  # 自定义前缀
    k=5,
    seed=42
)
generator.run()

# 生成的数据集名称：my_custom_prefix1, my_custom_prefix2, 等