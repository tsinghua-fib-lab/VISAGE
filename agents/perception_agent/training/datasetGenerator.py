import json
import random
import os
import argparse
from pathlib import Path

class KFoldDatasetGenerator:
    """
    K-Fold Cross Validation Dataset Generator
    """
    
    def __init__(self, input_path, base_output_dir, datasets_mixture_path, dataset_prefix="segregation_cot_40_31_data", k=5, seed=None):
        """
        Initialize the generator
        
        Args:
            input_path: Path to input JSON file
            base_output_dir: Base path for output directory
            datasets_mixture_path: Path to dataset configuration file
            dataset_prefix: Prefix for dataset names
            k: Number of folds, default is 5
            seed: Random seed
        """
        self.input_path = input_path
        self.base_output_dir = base_output_dir
        self.datasets_mixture_path = datasets_mixture_path
        self.dataset_prefix = dataset_prefix
        self.k = k
        self.seed = seed
        
    def generate_datasets(self):
        """Generate K-fold datasets"""
        if self.seed is not None:
            random.seed(self.seed)
        
        # Read and shuffle data
        with open(self.input_path, "r") as f:
            data = json.load(f)

        random.shuffle(data)

        # Split data into k equal parts
        chunk_size = len(data) // self.k
        folds = [data[i*chunk_size : (i+1)*chunk_size] for i in range(self.k)]

        # Handle remainder if data cannot be divided equally
        remainder = len(data) % self.k
        for i in range(remainder):
            folds[i].append(data[self.k*chunk_size + i])

        # Generate train and test sets for each fold
        dataset_names = []
        for i in range(self.k):
            test_data = folds[i]
            train_data = [item for j, fold in enumerate(folds) if j != i for item in fold]

            folder_name = os.path.join(self.base_output_dir, f"data{i+1}")
            os.makedirs(folder_name, exist_ok=True)

            train_path = os.path.join(folder_name, "train_data.json")
            test_path = os.path.join(folder_name, "test_data.json")

            with open(train_path, "w") as f_train:
                json.dump(train_data, f_train, indent=4)
            
            with open(test_path, "w") as f_test:
                json.dump(test_data, f_test, indent=4)

            dataset_name = f"{self.dataset_prefix}{i+1}"
            dataset_names.append(dataset_name)
            print(f"✅ Fold {i+1} completed: Test set size {len(test_data)}, Train set size {len(train_data)}, saved to {folder_name}")
        
        return dataset_names
    
    def update_config_file(self, dataset_names):
        """Update dataset configuration file"""
        # Read existing file content
        with open(self.datasets_mixture_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the position of the last add_dataset call
        last_add_dataset_pos = content.rfind('add_dataset(')
        if last_add_dataset_pos == -1:
            raise ValueError("No add_dataset call found")
        
        # Find the end position of the last add_dataset call
        last_line_end = content.find('\n', last_add_dataset_pos)
        while last_line_end < len(content) - 1 and content[last_line_end + 1] in [' ', '\t']:
            last_line_end = content.find('\n', last_line_end + 1)
        
        # Build new dataset definitions to insert
        new_datasets_code = "\n"
        for dataset_name in dataset_names:
            # Extract the fold number from dataset name (e.g., "data1" from "segregation_cot_40_31_data1")
            fold_number = dataset_name.replace(self.dataset_prefix, "")
            data_path = os.path.join(self.base_output_dir, f"data{fold_number}", "train_data.json")
            new_datasets_code += f"""    {dataset_name} = Dataset(
        dataset_name="{dataset_name}",
        dataset_type="torch",
        data_path="{data_path}",
        image_path="",
        description="",
    )
    add_dataset({dataset_name})
    
"""
        
        # Insert new code
        new_content = content[:last_line_end + 1] + new_datasets_code + content[last_line_end + 1:]
        
        # Write back to file
        with open(self.datasets_mixture_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ Dataset configuration file updated: {self.datasets_mixture_path}")
    
    def run(self):
        """Execute the complete generation pipeline"""
        print(f"🚀 Starting dataset generation with prefix: {self.dataset_prefix}")
        dataset_names = self.generate_datasets()
        self.update_config_file(dataset_names)
        print(f"🎉 All operations completed! Generated {self.k} fold datasets with prefix '{self.dataset_prefix}' and updated configuration file.")


def parse_arguments():
    """Parse command line arguments"""
    """
    Command Line Usage Methods：
    python script.py --input_path /path/to/data.json --output_dir /path/to/output --dataset_prefix "urban_segregation_data" --k 5 --seed 42
    """

    parser = argparse.ArgumentParser(description='Generate K-fold cross validation datasets')
    parser.add_argument('--input_path', type=str, 
                       default="/data3/maruolong/VISAGE/data/31_qa_with_caption.json",
                       help='Path to input JSON file')
    parser.add_argument('--output_dir', type=str,
                       default="/data3/maruolong/VISAGE/data/31_cities",
                       help='Base path for output directory')
    parser.add_argument('--config_path', type=str,
                       default="/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/llava/data/datasets_mixture.py",
                       help='Path to dataset configuration file')
    parser.add_argument('--dataset_prefix', type=str, 
                       default="segregation_cot_40_31_data",
                       help='Prefix for dataset names (e.g., "segregation_cot_40_31_data" will create datasets like "segregation_cot_40_31_data1", "segregation_cot_40_31_data2", etc.)')
    parser.add_argument('--k', type=int, default=5, help='Number of folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def main():
    """Main function for command line execution"""
    args = parse_arguments()
    
    generator = KFoldDatasetGenerator(
        input_path=args.input_path,
        base_output_dir=args.output_dir,
        datasets_mixture_path=args.config_path,
        dataset_prefix=args.dataset_prefix,
        k=args.k,
        seed=args.seed
    )
    generator.run()


# Usage example when imported as module
def example_usage():
    """Example usage when imported as module"""
    generator = KFoldDatasetGenerator(
        input_path="/data3/maruolong/VISAGE/data/31_qa_with_caption.json",
        base_output_dir="/data3/maruolong/VISAGE/data/31_cities",
        datasets_mixture_path="/data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/llava/data/datasets_mixture.py",
        dataset_prefix="segregation_cot_40_31_data",
        k=5,
        seed=42
    )
    generator.run()


if __name__ == "__main__":
    main()