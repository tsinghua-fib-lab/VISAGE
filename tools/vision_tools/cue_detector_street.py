import os
import json
import time
import base64
import random
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

class StreetViewAnalyzer:
    """Street view image analyzer for analyzing visual elements in street view images"""
    
    def __init__(self, 
                 api_key: str = "your key here",
                 base_url: str = "https://api.siliconflow.cn/v1",
                 model: str = "Pro/Qwen/Qwen2.5-VL-7B-Instruct",
                 max_tokens: int = 500,
                 max_retries: int = 10,
                 max_wait: int = 600,
                 codebook_path: Optional[str] = None):
        """
        Initialize street view image analyzer
        
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
        self.street_view_elements = self._load_street_view_elements()
        self._initialize_client()
        
        # City list
        self.all_cities = [
            'Boston', 'Chicago', 'Dallas', 'Detroit', 'Los Angeles', 'Miami', 'New York', 'Philadelphia', 
            'San Francisco', 'Seattle', 'Washington', 'Albuquerque', 'Austin', 'Baltimore', 'Charlotte', 
            'Columbus', 'Denver', 'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas', 
            'Memphis', 'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland', 'San Antonio', 'San Diego', 
            'San Jose', 'Tucson'
        ]
    
    def _load_street_view_elements(self) -> List[str]:
        """
        Load street view elements from codebook or use default values
        
        Returns:
            List of street view element names
        """
        # Default street view elements
        default_elements = [
            "DetachedSingleFamilyHouse", "BungalowOrLowRiseHouse", "HighFencesOrWalls", "BarbedWireOrGatedEnclosures",
            "VacantLotsOrAbandonedBuildings", "LargeEmptyLand", "DeterioratedStreets", "Graffiti", "LitterOrTrashPiles",
            "BoardedUpOrClosedShops", "LackOfCommercialActivity", "NarrowRoadsOrDeadEnds", "NoSidewalks",
            "MonofunctionalResidentialBlocks", "SparseOrShortShrubs", "IsolatedLawnsWithoutTrees",
            "RowHousesOrTownhouses", "ApartmentBuildingsOrHighRises", "VisibleShopsOrColorfulSigns",
            "ConvenienceStoresOrRestaurants", "ParksPlaygroundsOrCommunitySquares", "PublicFacilitiesBusStopsSchools",
            "ParkingLots", "WideMultiLaneRoads", "CrosswalksOrOverpasses", "Bridges", "LandmarkBuildingsOrReligiousSites",
            "MuralsSculpturesOrUrbanArt", "TallTreesOrTreeLines", "StreetTreesOrGreenBelts", "CommunityGardens",
            "ColorfulCleanFacadesOrWindows"
        ]
        
        # Try to load elements from codebook if path is provided
        if self.codebook_path and os.path.exists(self.codebook_path):
            try:
                with open(self.codebook_path, 'r') as f:
                    codebook_data = json.load(f)
                
                # Extract street view elements from codebook
                street_view_codebook = codebook_data.get("street_view_codebook", [])
                if street_view_codebook:
                    codebook_elements = [item["Name"] for item in street_view_codebook]
                    print(f"✅ Loaded {len(codebook_elements)} street view elements from codebook: {self.codebook_path}")
                    return codebook_elements
                else:
                    print(f"⚠️ No street view elements found in codebook, using default elements")
                    return default_elements
                    
            except Exception as e:
                print(f"❌ Error loading codebook from {self.codebook_path}: {e}")
                print("Using default street view elements")
                return default_elements
        else:
            if self.codebook_path:
                print(f"⚠️ Codebook path not found: {self.codebook_path}, using default elements")
            else:
                print("ℹ️ No codebook path provided, using default street view elements")
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
            return "data:image/jpg;base64," + base64.b64encode(image_file.read()).decode("utf-8")
    
    def get_street_view_prompt(self) -> str:
        """
        Get street view image analysis prompt with elements from codebook or defaults
        
        Returns:
            Prompt text with visual elements to detect
        """
        # Create JSON structure for elements with default value 0
        elements_json = ",\n".join([f'"{element}": 0' for element in self.street_view_elements])
        
        return f'''
        You are an urban geography and spatial inequality expert.

        Given a U.S. street view image from a Census Tract, your task is to detect the presence (1) or absence (0) of the following visual elements that are known to either increase or decrease income segregation.

        Please carefully examine the image for the following {len(self.street_view_elements)} elements. For each, output 1 if clearly visible, otherwise 0.

        Return your answer strictly in one-line JSONL format, like:
        {{"Element1": 1, "Element2": 0, ..., "ElementN": 1}}

        Do not include any explanations.

        Here is the full list of elements to detect:

        {{
        {elements_json}
        }}  
        '''
    
    def analyze_street_view_image(self, img_path: str) -> Optional[str]:
        """
        Analyze street view image
        
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
                    {"type": "text", "text": self.get_street_view_prompt()},
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
    
    def process_single_image(self, tract_id: str, image_file: str, images_folder: str) -> Optional[Dict[str, Any]]:
        """
        Process single street view image
        
        Args:
            tract_id: Tract ID
            image_file: Image file name
            images_folder: Image folder path
            
        Returns:
            Processing result dictionary
        """
        image_path = os.path.join(images_folder, tract_id, image_file)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        description = self.analyze_street_view_image(image_path)
        
        if description:
            return {
                "tract_id": tract_id,
                "image": image_file,
                "description": description
            }
        return None
    
    def process_city_images(self, 
                          city_name: str, 
                          base_image_folder: Optional[str] = None,
                          output_base_folder: Optional[str] = None,
                          max_workers: int = 5,
                          supported_formats: List[str] = None) -> Dict[str, Any]:
        """
        Process street view images for a single city
        
        Args:
            city_name: City name
            base_image_folder: Base image folder path, uses preset path by default
            output_base_folder: Output folder path, uses preset path by default
            max_workers: Maximum worker threads
            supported_formats: Supported image formats
            
        Returns:
            Processing statistics
        """
        if supported_formats is None:
            supported_formats = [".jpg", ".jpeg", ".png"]
        
        # Set paths (use provided parameters or default values)
        base_folder = base_image_folder or "/data3/maruolong/VISAGE/data/raw/imagery"
        output_folder = output_base_folder or "/data3/maruolong/segregation/Baseline/All_Cities/analysis/caption/result_new"
        
        city_filename = city_name.replace(" ", "_")
        city_image_folder = os.path.join(base_folder, city_filename, 'images')
        output_jsonl = os.path.join(output_folder, f"{city_filename}_image_caption.jsonl")
        
        if not os.path.exists(city_image_folder):
            print(f"❌ City image folder does not exist: {city_image_folder}")
            return {"success": False, "error": "City image folder not found"}
        
        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        tasks = []
        total_images = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor, open(output_jsonl, "w", encoding="utf-8") as jsonl_file:
            # Collect all tasks
            for tract_id in tqdm(os.listdir(city_image_folder), desc=f"[{city_name}] Scanning tracts"):
                tract_path = os.path.join(city_image_folder, tract_id)
                if not os.path.isdir(tract_path):
                    continue
                
                image_files = [f for f in os.listdir(tract_path) 
                             if any(f.lower().endswith(fmt) for fmt in supported_formats)]
                
                total_images += len(image_files)
                
                for image_file in image_files:
                    tasks.append(executor.submit(
                        self.process_single_image, tract_id, image_file, city_image_folder
                    ))

            # Process completed tasks
            successful_count = 0
            failed_count = 0
            
            for future in tqdm(as_completed(tasks), total=len(tasks), desc=f"[{city_name}] Processing images"):
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
            "city": city_name,
            "input_folder": city_image_folder,
            "output_file": output_jsonl,
            "total_tasks": len(tasks),
            "successful_count": successful_count,
            "failed_count": failed_count,
            "total_images": total_images
        }
        
        print(f"✅ {city_name} processing completed!")
        print(f"   Input folder: {city_image_folder}")
        print(f"   Output file: {output_jsonl}")
        print(f"   Total tasks: {len(tasks)}")
        print(f"   Successful: {successful_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Total images: {total_images}")
        
        return stats
    
    def process_all_cities(self,
                         base_image_folder: Optional[str] = None,
                         output_base_folder: Optional[str] = None,
                         max_workers_per_city: int = 5,
                         supported_formats: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Process street view images for all cities
        
        Args:
            base_image_folder: Base image folder path, uses preset path by default
            output_base_folder: Output folder path, uses preset path by default
            max_workers_per_city: Maximum worker threads per city
            supported_formats: Supported image formats
            
        Returns:
            Dictionary of processing statistics for each city
        """
        if supported_formats is None:
            supported_formats = [".jpg", ".jpeg", ".png"]
        
        # Set paths (use provided parameters or default values)
        base_folder = base_image_folder or "/data3/maruolong/segregation/Baseline"
        output_folder = output_base_folder or "/data3/maruolong/segregation/Baseline/All_Cities/analysis/caption/result_new"
        
        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        all_stats = {}
        
        for city in self.all_cities:
            print(f"\n🚀 Starting to process city: {city}")
            stats = self.process_city_images(
                city_name=city,
                base_image_folder=base_folder,
                output_base_folder=output_folder,
                max_workers=max_workers_per_city,
                supported_formats=supported_formats
            )
            all_stats[city] = stats
        
        # Summary statistics
        total_successful = sum(stats.get("successful_count", 0) for stats in all_stats.values())
        total_failed = sum(stats.get("failed_count", 0) for stats in all_stats.values())
        total_tasks = sum(stats.get("total_tasks", 0) for stats in all_stats.values())
        
        print(f"\n🎉 All cities processing completed!")
        print(f"   Total tasks: {total_tasks}")
        print(f"   Total successful: {total_successful}")
        print(f"   Total failed: {total_failed}")
        
        return all_stats


# Usage example
if __name__ == "__main__":

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

    # Optional: path to the codebook
    codebook_path = "/path/to/your/codebook.json"  # can be modified

    # Method 1: Process all cities
    analyzer1 = StreetViewAnalyzer(
        codebook_path=codebook_path,
        **common_params
    )
    all_stats = analyzer1.process_all_cities()

    # Method 2: Custom paths for processing all cities
    """
    analyzer2 = StreetViewAnalyzer(
        codebook_path=codebook_path,
        **common_params
    )
    all_stats = analyzer2.process_all_cities(
        base_image_folder="/your/base/image/folder",
        output_base_folder="/your/output/folder",
        max_workers_per_city=7
    )
    """

    # Method 3: Process a single city (using default paths)
    """
    analyzer3 = StreetViewAnalyzer(
        codebook_path=codebook_path,
        **common_params
    )
    stats = analyzer3.process_city_images("Beijing")
    """

    # Method 4: Custom paths + single city
    """
    analyzer4 = StreetViewAnalyzer(
        codebook_path=codebook_path,
        **common_params
    )
    stats = analyzer4.process_city_images(
        city_name="Shanghai",
        base_image_folder="/your/images",
        output_base_folder="/your/output"
    )
    """

    # Method 5: Analyze a single street-view image
    """
    analyzer5 = StreetViewAnalyzer(
        codebook_path=codebook_path,
        **common_params
    )
    result = analyzer5.analyze_street_view_image("/path/to/image.jpg")
    print(result)
    """

    # Method 6: Without codebook (use default elements)
    """
    analyzer6 = StreetViewAnalyzer(**common_params)
    all_stats = analyzer6.process_all_cities()
    """
