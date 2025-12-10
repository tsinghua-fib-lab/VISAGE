import os
import json
import glob
import random
import base64
import re
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

class QADataGenerator:
    """Q&A data generator for generating income segregation analysis question-answer pairs"""
    
    def __init__(self):
        """Initialize Q&A data generator"""
        # City list
        self.cities = [
            'Boston', 'Chicago', 'Dallas', 'Detroit', 'Los Angeles', 'Miami', 'New York', 'Philadelphia',
            'San Francisco', 'Seattle', 'Washington', 'Albuquerque', 'Austin', 'Baltimore', 'Charlotte',
            'Columbus', 'Denver', 'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas',
            'Memphis', 'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland', 'San Antonio',
            'San Diego', 'San Jose', 'Tucson'
        ]
        
        # City code mapping table
        self.city_prefix_map = {
            "250": "Boston", "170": "Chicago", "4808": "Dallas", "4811": "Dallas", "4812": "Dallas",
            "4825": "Dallas", "4839": "Dallas", "261": "Detroit", "0603": "Los Angeles", 
            "06075": "San Francisco", "1208": "Miami", "360": "New York", "421": "Philadelphia",
            "530": "Seattle", "110": "Washington", "4845": "Austin", "4849": "Austin", "245": "Baltimore",
            "371": "Charlotte", "390": "Columbus", "080": "Denver", "4814": "El Paso", "4843": "Fort Worth",
            "4815": "Houston", "4820": "Houston", "4833": "Houston", "1203": "Jacksonville",
            "550": "Milwaukee", "320": "Las Vegas", "471": "Memphis", "40": "Oklahoma City",
            "04013": "Phoenix", "410": "Portland", "4802": "San Antonio", "06073": "San Diego",
            "0608": "San Jose", "04019": "Tucson", "350": "Albuquerque"
        }
        
        # Prompt variants
        self.prompt_variants = self._generate_prompt_variants()
        
        # Default paths
        self.default_paths = {
            "image_folders": [f"/data3/maruolong/segregation/Baseline/{city}/images" for city in self.cities],
            "json1_path": "/data3/maruolong/Train_Data/Urbanarea_dominant_race_data_all.json",
            "segregation_jsonl_paths": [f"/data3/maruolong/segregation/All_time/visit_data/{city}_2019/{city}_2019_tract_segregation.jsonl" for city in self.cities],
            "income_distribution_paths": [f"/data3/maruolong/segregation/All_time/visit_data/{city}_2019/{city}_2019_tract_income_distribution.jsonl" for city in self.cities],
            "rs_image_base_dir": "/data3/maruolong/segregation/All_time/rs_image_new/merged",
            "output_json_path": "/data3/maruolong/segregation/Train/Try/update/31_update_only_prompt_all_data_add_rs.json"
        }

    def _generate_prompt_variants(self) -> List[str]:
        """Generate prompt variants"""
        base_prompts = [
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Based on these images, please estimate the income segregation (S) within the tract. Income segregation is measured by the income distribution of the people who visit or reside in the community, reflecting the separation between different income groups. The index (S) ranges from 0 to 1, where higher values indicate more segregation between income groups.\n\nSTEP 1: Using all the visual data, estimate the distribution of individuals among the four income groups.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Based on these images, please estimate the income segregation (S) within the tract. Income segregation reflects the income distribution of the residents and visitors, showing how separated the income groups are. The score ranges from 0 (more diverse income distribution) to 1 (more segregated income distribution).\n\nSTEP 1: Assess the proportion of people from each income group in this tract based on all the images.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Based on these images, estimate the income segregation (S) within the tract. Income segregation measures the distribution of income levels among the people who live in or visit the tract. A higher value means more income concentration in fewer groups, while a lower value means a greater diversity of income levels.\n\nSTEP 1: Using all the images, estimate the share of individuals from each income group.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Based on these images, estimate the income segregation (S) within the tract. Income segregation (S) reflects how much income groups are represented in the area. A higher value (closer to 1) indicates fewer income groups, while a lower value (closer to 0) indicates a more diverse income distribution.\n\nSTEP 1: Use all the images to estimate the proportion of people from each income group in the tract.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Based on these images, please estimate the income segregation (S) within the tract. Income segregation is a metric showing how diverse or concentrated income groups are in the community. A value close to 1 indicates high segregation, while a value close to 0 indicates a more equal representation of income groups.\n\nSTEP 1: Using all the images, estimate the distribution of individuals into the four income groups.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Based on these images, estimate the income segregation (S) within the tract. Income segregation measures how distinct the income groups are in the community, considering both residents and visitors. A higher index (closer to 1) indicates more separation between income groups, and a lower index (closer to 0) indicates a greater income diversity.\n\nSTEP 1: Analyze all the images and estimate the proportion of each income group.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Based on these images, estimate the income segregation (S) within the tract. Income segregation shows how separated income groups are in the area. The index ranges from 0 to 1, where a value close to 1 means greater segregation and a value close to 0 means more diverse income distribution.\n\nSTEP 1: Use the visual data from all the images to estimate the share of individuals in each income group.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Based on these images, assess the income segregation (S) within the tract. Income segregation measures the level of separation between different income groups. A value of 1 indicates full segregation, while 0 indicates an equal distribution across income groups.\n\nSTEP 1: Using all the visual information, estimate the distribution of individuals in the four income groups.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Based on these images, estimate the income segregation (S) within the tract. Income segregation refers to the distribution of income groups in the tract. Higher values (close to 1) indicate fewer income groups, while lower values (close to 0) indicate a more diverse income distribution.\n\nSTEP 1: Analyze all the images and estimate how individuals from different income groups are represented.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Based on these images, estimate the income segregation (S) within the tract. The segregation score is based on how people from different income groups are distributed across the area. A higher score indicates more concentration of similar income groups, while a lower score indicates more income diversity.\n\nSTEP 1: Using all the images, estimate the portion of individuals in each income bracket.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Your task is to estimate the **income segregation (S)** in this tract based on the visual data provided. This index reflects the separation between income groups within the area, with 1 indicating complete segregation and 0 indicating equal representation of all income groups.\n\nSTEP 1: Estimate the proportion of each income group in the tract using all the images.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Based on these images, please predict the income segregation (S) within the tract. Income segregation measures the level of separation of income groups in the community, ranging from 0 (equal distribution) to 1 (complete segregation).\n\nSTEP 1: Using all the images, estimate the distribution of people among the four income categories.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Your task is to predict the **income segregation (S)** based on the images provided. This index quantifies how different income groups are distributed in the area, with higher values indicating more income concentration and lower values indicating a more equal distribution.\n\nSTEP 1: Based on all the images, estimate the share of each income group.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Based on these images, estimate the income segregation (S) within the tract. Income segregation shows how unevenly income groups are spread across a community. The index (S) ranges from 0 to 1. A score of 1 means maximum segregation, and 0 means equal distribution.\n\nSTEP 1: Analyze all the images to estimate the distribution of individuals in the four income groups.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Based on these images, estimate the income segregation (S) within the tract. Income segregation indicates how income groups are distributed in the tract, with 0 representing equal distribution and 1 representing full segregation.\n\nSTEP 1: Using all the images, estimate how the population is divided among the income levels.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Your task is to determine the **income segregation (S)** within the tract based on the visual information provided. This score quantifies how separated the income groups are in the area, with 0 indicating a diverse income distribution and 1 indicating a more segregated community.\n\nSTEP 1: Using all the images, estimate the income group proportions within the tract.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Your task is to assess the **income segregation (S)** in this tract based on the images. The index reflects how much the income groups are separated from each other, with higher values indicating more segregation and lower values indicating a more diverse income distribution.\n\nSTEP 1: Analyze the images and estimate the share of people in each income group.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Based on these images, estimate the income segregation (S) within the tract. Income segregation is a measure of how income groups are distributed in the community. A value of 1 indicates complete segregation, while a value of 0 indicates equal distribution of income groups.\n\nSTEP 1: Using all the images, estimate the distribution of income groups in the tract.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Your task is to calculate the **income segregation (S)** from these images. The segregation score quantifies how much income groups are separated in the area. The index (S) ranges from 0 to 1, where 1 means total separation and 0 means full integration.\n\nSTEP 1: Using all the images, estimate how the population is distributed among the income groups.",
            "You are a geography inequality expert. Here are multiple satellite and street images from a U.S. census tract. Based on these images, estimate the income segregation (S) within the tract. The index represents how segregated or integrated the income groups are in the area. A value close to 1 indicates that the area is predominantly made up of a single income group, and a value close to 0 indicates greater diversity.\n\nSTEP 1: Using all the images, estimate the share of people in each of the income groups."
        ]

        
        suffix = """The income levels are as follows:\n- Low-Income (P1)\n- Lower-Middle-Income (P2)\n- Upper-Middle-Income (P3)\n- High-Income (P4)\n\nThese proportions should sum to 1.\n\nSTEP 2: Based on the proportions estimated in Step 1, calculate the income segregation (S) using the formula:  \nS = 2/3 * Σ(|p_i - 0.25|),\nwhere p_i represents the proportion of individuals in each income group (P1, P2, P3, P4).\n\nImportant: When you output, please include explanations, calculations, or analysis in your response. And provide the estimated values in the end in the following format:\n\nOutput format:\n- Income Group Proportions:\n- Low-Income (P1): 0.XX\n- Lower-Middle-Income (P2): 0.XX\n- Upper-Middle-Income (P3): 0.XX\n- High-Income (P4): 0.XX\n\n- Income Segregation Index (S): X.XX\n"""
        
        return [p + suffix for p in base_prompts]

    def _load_data_sources(self, 
                         json1_path: Optional[str] = None,
                         segregation_paths: Optional[List[str]] = None,
                         income_paths: Optional[List[str]] = None) -> Tuple[Dict, Dict, Dict]:
        """Load data sources"""
        json1_path = json1_path or self.default_paths["json1_path"]
        segregation_paths = segregation_paths or self.default_paths["segregation_jsonl_paths"]
        income_paths = income_paths or self.default_paths["income_distribution_paths"]
        
        # Read remote sensing image mapping
        tract_to_rs_image = {}
        with open(json1_path, "r") as f:
            json1_data = json.load(f)
            for item in json1_data:
                tract_id = item["sample_id"]
                if item["image"]:
                    tract_to_rs_image[tract_id] = item["image"][0]

        # Read segregation ground truth
        tract_to_segregation = {}
        for seg_path in segregation_paths:
            if os.path.exists(seg_path):
                with open(seg_path, "r") as f:
                    for line in f:
                        item = json.loads(line)
                        tract_to_segregation[item["tract_id"]] = item["segregation"]

        # Read income distribution data
        tract_to_income_distribution = {}
        for income_path in income_paths:
            if os.path.exists(income_path):
                with open(income_path, "r") as f:
                    for line in f:
                        item = json.loads(line)
                        tract_to_income_distribution[item["tract_id"]] = item["income_distribution"]

        return tract_to_rs_image, tract_to_segregation, tract_to_income_distribution

    def _generate_cot_summary(self, p1: float, p2: float, p3: float, p4: float, segregation_score: float) -> str:
        """Generate chain-of-thought summary"""
        q1 = abs(p1 - 0.25)
        q2 = abs(p2 - 0.25)
        q3 = abs(p3 - 0.25)
        q4 = abs(p4 - 0.25)
        
        return f"""### 1. **<Summary>**  
Based on the provided street view and remote sensing images, our goal is to estimate the level of income segregation within the given Census Tract (CT). Income segregation is measured by assessing the distribution of different income groups visiting or residing in the area. By analyzing key visual elements, we infer the likely income composition of individuals associated with this location. Once the income proportions are estimated, we compute the segregation index using the predefined formula.  

### 2. **<Caption>**  
1. **From the remote sensing images**, I focus on features such as **road network connectivity**, which indicates accessibility and urban development; **green space coverage**, reflecting environmental quality and public investment; **land use types** (commercial, residential, industrial), which help infer economic activity; and **residential density**, suggesting housing affordability and socioeconomic distribution.  
2. **From the street view images**, I observe **building types** (single-family homes, apartment complexes, or high-rises) to estimate economic status; **infrastructure quality** (road conditions, sidewalks, public transportation) as a proxy for investment; **public spaces** and **commercial areas** (shops, restaurants, and businesses) which reflect economic vibrancy; **vegetation and greenery**, indicating neighborhood maintenance; and **street conditions and pedestrian activity**, which hint at safety, walkability, and overall socioeconomic engagement.  
3. **Based on these observations**, I estimate the proportion of visitors or residents in the CT belonging to different income levels:  
   - **Low-Income (P1):** {p1:.2f}  
   - **Lower-Middle-Income (P2):** {p2:.2f}  
   - **Upper-Middle-Income (P3):** {p3:.2f}  
   - **High-Income (P4):** {p4:.2f}  

### 3. **<Calculation>**  
The segregation index (S) is computed using the formula:  
S = (2/3) * Σ (|p_i - 0.25|), where p_i represents the proportion of individuals belonging to each income group (P1, P2, P3, P4) in the given tract.  

Substituting the estimated values:  
S = (2/3) * (|{p1:.2f} - 0.25| + |{p2:.2f} - 0.25| + |{p3:.2f} - 0.25| + |{p4:.2f} - 0.25|)  
S = (2/3) * ({q1:.2f} + {q2:.2f} + {q3:.2f} + {q4:.2f})

After calculation:  
S = {segregation_score:.2f}  

### 4. **<Answer>**  
OUTPUT:  
- Income Group Proportions: 
- Low-Income (P1): {p1:.2f}
- Lower-Middle-Income (P2): {p2:.2f}
- Upper-Middle-Income (P3): {p3:.2f}
- High-Income (P4): {p4:.2f}
  
- Income Segregation Index (S): {segregation_score:.2f}  
"""

    def _add_city_to_prompt(self, output_data: List[Dict]) -> List[str]:
        """Add city name to prompt"""
        unmatched_sample_ids = []

        for entry in output_data:
            sample_id = entry["sample_id"]
            city = None

            # Try matching prefix from first 2 to 6 characters
            for prefix_len in range(2, 7):
                prefix = sample_id[:prefix_len]
                if prefix in self.city_prefix_map:
                    city = self.city_prefix_map[prefix]
                    break

            original_value = entry["conversations"][0]["value"]

            if city:
                updated_value = original_value.replace(
                    "Here are multiple satellite and street images from a U.S. census tract.",
                    f"Here are multiple satellite and street images from a U.S. census tract in {city}.", 1
                )
                entry["conversations"][0]["value"] = updated_value
            else:
                unmatched_sample_ids.append(sample_id)

        return unmatched_sample_ids

    def _add_rs_images(self, output_data: List[Dict], rs_image_base_dir: Optional[str] = None) -> List[str]:
        """Add remote sensing image paths"""
        rs_image_base_dir = rs_image_base_dir or self.default_paths["rs_image_base_dir"]
        missing_sample_ids = []
        updated_samples = []

        for sample in tqdm(output_data):
            sample_id = sample["sample_id"]
            image_list = sample["image"]
            conversation = sample["conversations"][0]

            if image_list:
                image_list.pop(0)
                # Remove one <image> tag accordingly
                conversation["value"] = conversation["value"].replace("<image>", "", 1)

            # Find remote sensing image directory
            candidate_dir = os.path.join(rs_image_base_dir, sample_id)
            found_images = []
            
            if os.path.isdir(candidate_dir):
                image_files = sorted([
                    os.path.join(candidate_dir, fname)
                    for fname in os.listdir(candidate_dir)
                    if fname.lower().endswith((".png", ".jpg", ".jpeg"))
                ])
                if image_files:
                    found_images = image_files

            if found_images:
                # Insert new remote sensing image paths
                sample["image"] = found_images + image_list
                # Insert <image> tags
                new_tags = "<image>" * len(found_images)
                conversation["value"] = new_tags + conversation["value"]
            else:
                missing_sample_ids.append(sample_id)

            updated_samples.append(sample)

        return missing_sample_ids, updated_samples
    
    def _remove_empty_base64_samples(self, json_path: str) -> Tuple[int, List[str]]:
        """Remove image paths with empty Base64"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        valid_samples = []
        removed_samples = []

        for sample in data:
            sample_id = sample.get('sample_id')
            image_paths = sample.get('image', [])  
            conversations = sample.get('conversations', [])
            invalid_paths_count = 0

            for image_path in image_paths[:]:  # Iterate using copy
                try:
                    if not os.path.exists(image_path):
                        image_paths.remove(image_path)
                        invalid_paths_count += 1
                        continue
                    
                    with Image.open(image_path) as img:
                        img.verify()

                    with open(image_path, "rb") as image_file:
                        base64_content = base64.b64encode(image_file.read()).decode('utf-8')

                    if not base64_content.strip():
                        image_paths.remove(image_path)
                        invalid_paths_count += 1
                        continue
                except (UnidentifiedImageError, IOError, Exception):
                    image_paths.remove(image_path)
                    invalid_paths_count += 1
                    continue

            # Modify <image> tag count in conversations
            if invalid_paths_count > 0 and conversations:
                value = conversations[0].get('value', '')
                conversations[0]['value'] = value[invalid_paths_count * len("<image>"):]

            valid_samples.append(sample)

            if invalid_paths_count > 0:
                removed_samples.append(sample_id)

        # Save updated data
        with open(json_path, 'w') as f:
            json.dump(valid_samples, f, indent=4)

        return len(removed_samples), removed_samples

    def _fix_image_tags(self, json_path: str) -> int:
        """Fix mismatched <image> tag count and actual image count"""
        def load_json(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)

        def save_json(data, filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

        def count_image_tags(value):
            first_line = value.split("\n", 1)[0] 
            return first_line.count("<image>")

        json_data = load_json(json_path)
        modified_count = 0

        for sample in json_data:
            if "conversations" in sample and sample["conversations"] and "image" in sample:
                first_value = sample["conversations"][0]["value"]
                image_list = sample["image"]

                current_image_count = count_image_tags(first_value)
                expected_image_count = len(image_list)

                if current_image_count != expected_image_count:
                    modified_count += 1
                    
                    first_line = re.sub(r"^(<image>)*", "", first_value.split("\n", 1)[0])  
                    rest_text = "\n".join(first_value.split("\n")[1:]) 

                    corrected_first_line = "<image>" * expected_image_count + first_line
                    corrected_value = corrected_first_line + ("\n" + rest_text if rest_text else "")
                    sample["conversations"][0]["value"] = corrected_value

        save_json(json_data, json_path)
        return modified_count

    def generate_qa_data(self,
                        image_folders: Optional[List[str]] = None,
                        json1_path: Optional[str] = None,
                        segregation_jsonl_paths: Optional[List[str]] = None,
                        income_distribution_paths: Optional[List[str]] = None,
                        rs_image_base_dir: Optional[str] = None,
                        output_json_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate Q&A data
        
        Args:
            image_folders: List of image folders
            json1_path: Remote sensing image mapping file path
            segregation_jsonl_paths: List of segregation data file paths
            income_distribution_paths: List of income distribution data file paths
            rs_image_base_dir: Remote sensing image base directory
            output_json_path: Output JSON file path
            
        Returns:
            Processing statistics
        """
        # Set paths
        image_folders = image_folders or self.default_paths["image_folders"]
        json1_path = json1_path or self.default_paths["json1_path"]
        segregation_paths = segregation_jsonl_paths or self.default_paths["segregation_jsonl_paths"]
        income_paths = income_distribution_paths or self.default_paths["income_distribution_paths"]
        rs_image_base_dir = rs_image_base_dir or self.default_paths["rs_image_base_dir"]
        output_json_path = output_json_path or self.default_paths["output_json_path"]
        
        print("🚀 Starting to generate Q&A data...")
        
        # 1. Load data sources
        print("📂 Loading data sources...")
        tract_to_rs_image, tract_to_segregation, tract_to_income_distribution = self._load_data_sources(
            json1_path, segregation_paths, income_paths
        )
        
        # 2. Generate base data
        print("🔧 Generating base data...")
        output_data = []
        for folder in image_folders:
            for tract_id in os.listdir(folder):
                tract_path = os.path.join(folder, tract_id)
                if os.path.isdir(tract_path):
                    street_images = sorted(glob.glob(os.path.join(tract_path, "*.jpg")))
                    if not street_images:
                        continue

                    # Get remote sensing image path
                    rs_image = tract_to_rs_image.get(tract_id)
                    if rs_image:
                        total_images = len(street_images) + 1
                        image_list = [rs_image] + street_images
                    else:
                        total_images = len(street_images)
                        image_list = street_images

                    # Get data
                    segregation_value = tract_to_segregation.get(tract_id, "unknown")
                    income_distribution = tract_to_income_distribution.get(tract_id, [0.0, 0.0, 0.0, 0.0])
                    p1, p2, p3, p4 = income_distribution
                    segregation_score = (2 / 3) * sum([abs(p - 0.25) for p in income_distribution])

                    # Generate CoT answer
                    cot_summary = self._generate_cot_summary(p1, p2, p3, p4, segregation_score)

                    # Construct JSON structure
                    entry = {
                        "sample_id": tract_id,
                        "sub_task": "segregation",
                        "conversations": [
                            {
                                "from": "human",
                                "value": "<image>" * total_images + "\n" + self.prompt_variants[0]
                            },
                            {
                                "from": "gpt",
                                "value": cot_summary
                            }
                        ],
                        "image": image_list,
                        "choice_list": "null",
                        "metadata": {
                            "dataset": "Socioeconomic",
                            "question_type": "open-ended"
                        }
                    }
                    output_data.append(entry)

        # 3. Add prompt variants
        print("🔄 Adding prompt variants...")
        for entry in output_data:
            original_prompt = entry["conversations"][0]["value"]
            if original_prompt.startswith("<image>"):
                image_part, _ = original_prompt.split("\n", 1)
            else:
                image_part = ""
            new_prompt_body = random.choice(self.prompt_variants)
            entry["conversations"][0]["value"] = image_part + "\n" + new_prompt_body

        # 4. Add city names
        print("🏙️ Adding city names...")
        unmatched_ids = self._add_city_to_prompt(output_data)
        print(f"❗ Unmatched sample_ids: {len(unmatched_ids)}")

        # 5. Add remote sensing image paths
        print("🖼️ Adding remote sensing image paths...")
        missing_ids, updated_data = self._add_rs_images(output_data, rs_image_base_dir)
        print(f"⚠️ Sample_ids missing remote sensing images: {len(missing_ids)}")

        # 6. Save initial data
        print("💾 Saving initial data...")
        with open(output_json_path, "w") as f:
            json.dump(updated_data, f, indent=2)

        # 7. Clean invalid images
        print("🧹 Cleaning invalid images...")
        removed_count, removed_ids = self._remove_empty_base64_samples(output_json_path)
        print(f"🗑️ Removed invalid image samples: {removed_count}")

        # 8. Fix image tags
        print("🔧 Fixing image tags...")
        modified_count = self._fix_image_tags(output_json_path)
        print(f"✅ Fixed image tag samples: {modified_count}")

        # Return statistics
        stats = {
            "total_samples": len(updated_data),
            "unmatched_city_ids": len(unmatched_ids),
            "missing_rs_image_ids": len(missing_ids),
            "removed_invalid_image_ids": removed_count,
            "modified_image_tag_samples": modified_count,
            "output_path": output_json_path
        }

        print(f"\n🎉 Q&A data generation completed!")
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Output path: {stats['output_path']}")

        return stats


# Usage example
if __name__ == "__main__":
    # Create generator instance
    generator = QADataGenerator()
    
    # Method 1: Use default paths
    stats = generator.generate_qa_data()
    
    # Method 2: Custom paths
    # stats = generator.generate_qa_data(
    #     image_folders=["/custom/image/folder1", "/custom/image/folder2"],
    #     output_json_path="/custom/output/path.json"
    # )
    
    print(f"\n📊 Generation statistics: {stats}")