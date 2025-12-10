import os
import json
import time
import base64
import random
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

class RemoteSensingAnalyzer:
    """Remote sensing image analyzer for analyzing visual elements in remote sensing images"""
    
    def __init__(self, 
                 api_key: str = "your key",
                 base_url: str = "https://api.siliconflow.cn/v1",
                 model: str = "Pro/Qwen/Qwen2.5-VL-7B-Instruct",
                 max_tokens: int = 500,
                 max_retries: int = 10,
                 max_wait: int = 600,
                 codebook_path: Optional[str] = None):
        """
        Initialize remote sensing image analyzer
        
        Args:
            api_key: API key
            base_url: API base URL
            model: Model name
            max_tokens: Maximum tokens
            max_retries: Maximum retry attempts
            max_wait: Maximum wait time in seconds
            codebook_path: Path to codebook JSON file containing visual element definitions
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_wait = max_wait
        self.client = None
        self.codebook_path = codebook_path
        self.remote_sensing_elements = self._load_remote_sensing_elements()
        self._initialize_client()
    
    def _load_remote_sensing_elements(self) -> List[str]:
        """
        Load remote sensing elements from codebook or use default values
        
        Returns:
            List of remote sensing element names
        """
        # Default remote sensing elements
        default_elements = [
            "LowDensityHousing", "CulDeSacsOrFencedBlocks", "MonofunctionalResidentialZones",
            "VacantOrDemolitionSites", "ScatteredLawnsOrPrivateGreenery", "IsolatedLargeLots",
            "DisconnectedStreetSegments", "FragmentedDevelopmentPatches", "UnpavedRoadsOrDirtTracks",
            "UniformHousingBlocks", "LowStreetConnectivity", "LargeFootprintBuildings", "TallBuildings",
            "CommercialActivityZones", "ParkingLots", "ParksSquaresOrSportsFields", "GridOrRadialStreetPatterns",
            "LargeOpenPublicSpaces", "StreetTreesAndUrbanGreenStrips", "OverpassesOrInterchanges",
            "RailwaysOrTrainTracks", "ExpresswaysOrHighways", "SchoolsOrEducationalCampuses",
            "WideStreetsWithMediansOrShoulders", "TransportHubsOrStations", "ColorfulRoofsOrArchitecturalDiversity"
        ]
        
        # Try to load elements from codebook if path is provided
        if self.codebook_path and os.path.exists(self.codebook_path):
            try:
                with open(self.codebook_path, 'r') as f:
                    codebook_data = json.load(f)
                
                # Extract remote sensing elements from codebook
                remote_sensing_codebook = codebook_data.get("remote_sensing_codebook", [])
                if remote_sensing_codebook:
                    codebook_elements = [item["Name"] for item in remote_sensing_codebook]
                    print(f"✅ Loaded {len(codebook_elements)} remote sensing elements from codebook: {self.codebook_path}")
                    return codebook_elements
                else:
                    print(f"⚠️ No remote sensing elements found in codebook, using default elements")
                    return default_elements
                    
            except Exception as e:
                print(f"❌ Error loading codebook from {self.codebook_path}: {e}")
                print("Using default remote sensing elements")
                return default_elements
        else:
            if self.codebook_path:
                print(f"⚠️ Codebook path not found: {self.codebook_path}, using default elements")
            else:
                print("ℹ️ No codebook path provided, using default remote sensing elements")
            return default_elements
    
    def _initialize_client(self) -> None:
        """Initialize OpenAI client"""
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def encode_image_to_base64(self, img_path: str) -> str:
        """
        Read local image and convert to Base64
        
        Args:
            img_path: Image file path
            
        Returns:
            Base64 encoded image data
        """
        with open(img_path, "rb") as image_file:
            return "data:image/png;base64," + base64.b64encode(image_file.read()).decode("utf-8")
    
    def get_remote_sensing_prompt(self) -> str:
        """
        Get remote sensing image analysis prompt with elements from codebook or defaults
        
        Returns:
            Prompt text with visual elements to detect
        """
        # Create JSON structure for elements with default value 0
        elements_json = ",\n".join([f'"{element}": 0' for element in self.remote_sensing_elements])
        
        return f'''
        You are an urban geography and spatial inequality expert.

        Given a U.S. satellite or aerial image from a Census Tract, your task is to detect the presence (1) or absence (0) of the following visual elements that are known to either increase or decrease income segregation.

        Please carefully examine the image for the following {len(self.remote_sensing_elements)} elements. For each, output 1 if clearly visible, otherwise 0.

        Return your answer strictly in one-line JSONL format, like:
        {{"Element1": 1, "Element2": 0, ..., "ElementN": 1}}

        Do not include any explanations.

        Here is the full list of elements to detect:

        {{
        {elements_json}
        }}
        '''
    
    def analyze_remote_sensing_image(self, img_path: str) -> Optional[str]:
        """
        Analyze remote sensing image
        
        Args:
            img_path: Image file path
            
        Returns:
            Analysis result JSON string
        """
        img_base64 = self.encode_image_to_base64(img_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.get_remote_sensing_prompt()},
                    {"type": "image_url", "image_url": {"url": img_base64, "detail": "low"}},
                ],
            }
        ]

        retry = 0
        while retry < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                wait = min(self.max_wait, 2 ** retry + random.uniform(1, 5))
                print(f"Error analyzing {img_path}: {e}, retrying in {wait:.2f} seconds...")
                time.sleep(wait)
                retry += 1
        
        print(f"Failed to analyze {img_path} after {self.max_retries} retries")
        return None
    
    def process_single_image(self, tract_id: str, image_file: str, folder_path: str) -> Optional[Dict[str, Any]]:
        """
        Process single remote sensing image
        
        Args:
            tract_id: Tract ID
            image_file: Image file name
            folder_path: Image folder path
            
        Returns:
            Processing result dictionary
        """
        image_path = os.path.join(folder_path, tract_id, image_file)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        description = self.analyze_remote_sensing_image(image_path)
        
        if description:
            return {
                "tract_id": tract_id,
                "image": image_file,
                "description": description
            }
        return None
    
    def process_single_folder(self, 
                            input_folder: str, 
                            output_jsonl: str, 
                            max_workers: int = 5,
                            supported_formats: List[str] = None) -> Dict[str, Any]:
        """
        Process all remote sensing images in a single folder
        
        Args:
            input_folder: Input folder path (contains all tract subfolders)
            output_jsonl: Output JSONL file path
            max_workers: Maximum worker threads
            supported_formats: Supported image formats
            
        Returns:
            Processing statistics
        """
        if supported_formats is None:
            supported_formats = [".jpg", ".jpeg", ".png"]
        
        if not os.path.exists(input_folder):
            print(f"❌ Input folder does not exist: {input_folder}")
            return {"success": False, "error": "Input folder not found"}
        
        tasks = []
        total_images = 0
        
        # Collect all tasks
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for tract_id in tqdm(os.listdir(input_folder), desc="Scanning tracts"):
                tract_path = os.path.join(input_folder, tract_id)
                if not os.path.isdir(tract_path):
                    continue

                image_files = [f for f in os.listdir(tract_path) 
                             if any(f.lower().endswith(fmt) for fmt in supported_formats)]
                
                total_images += len(image_files)
                
                for image_file in image_files:
                    tasks.append(executor.submit(
                        self.process_single_image, tract_id, image_file, input_folder
                    ))

            # Process completed tasks
            successful_count = 0
            failed_count = 0
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
            
            with open(output_jsonl, "w", encoding="utf-8") as jsonl_file:
                for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing images"):
                    try:
                        result = future.result()
                        if result:
                            jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                            successful_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        print(f"Task failed with error: {e}")
                        failed_count += 1
        
        stats = {
            "success": True,
            "input_folder": input_folder,
            "output_file": output_jsonl,
            "total_tasks": len(tasks),
            "successful_count": successful_count,
            "failed_count": failed_count,
            "total_images": total_images
        }
        
        print(f"✅ Processing completed!")
        print(f"   Input folder: {input_folder}")
        print(f"   Output file: {output_jsonl}")
        print(f"   Total tasks: {len(tasks)}")
        print(f"   Successful: {successful_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Total images: {total_images}")
        
        return stats


# Usage example
if __name__ == "__main__":
    # Define codebook path
    default_codebook_path = "/data3/maruolong/VISAGE/agents/literature_agent/knowledge_base/codebooks.json"  # Update this path


    # ========= Example parameters (for users to modify as needed) =========
    api_key = "your_api_key_here"                  # <-- replace with your own API key
    base_url = "your_base_url_here"                # <-- e.g., "https://api.siliconflow.cn/v1"
    model = "your_model_name_here"                 # <-- e.g., "Pro/Qwen/Qwen2.5-VL-7B-Instruct"
    max_tokens = 500
    max_retries = 10
    max_wait = 600

    common_params = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "max_tokens": max_tokens,
        "max_retries": max_retries,
        "max_wait": max_wait,
    }

    # Method 1: Use default paths with codebook
    analyzer1 = RemoteSensingAnalyzer(
        codebook_path=default_codebook_path,
        **common_params
    )
    default_input_folder = "/data3/maruolong/VISAGE/data/raw/imagery/rs_image_new/merged" 
    default_output_jsonl = "/data3/maruolong/segregation/Baseline/All_Cities/analysis/caption/rs_images/rs_image_caption.jsonl"
    
    stats = analyzer1.process_single_folder(default_input_folder, default_output_jsonl)
    
    # Method 2: Custom paths with codebook
    analyzer2 = RemoteSensingAnalyzer(
        codebook_path="/your/custom/codebook/path.json",
        **common_params
    )
    stats = analyzer2.process_single_folder(
        input_folder="/your/custom/input/folder",
        output_jsonl="/your/custom/output/file.jsonl",
        max_workers=10
    )
    
    # Method 3: Without codebook (use default elements)
    analyzer3 = RemoteSensingAnalyzer(
        **common_params
    )
    stats = analyzer3.process_single_folder(
        input_folder="/your/custom/input/folder",
        output_jsonl="/your/custom/output/file.jsonl"
    )
    
    # Method 4: Process single image with codebook
    analyzer4 = RemoteSensingAnalyzer(
        codebook_path=default_codebook_path,
        **common_params
    )
    result = analyzer4.analyze_remote_sensing_image("/path/to/image.jpg")
    print(result)
