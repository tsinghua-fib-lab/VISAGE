import os
import json
import re
import glob
from collections import defaultdict
from typing import List, Dict, Any, Set, Optional
from tqdm import tqdm

class CaptionProcessor:
    """Caption processor for processing and analyzing image caption data"""
    
    def __init__(self, codebook_path: Optional[str] = None):
        """
        Initialize Caption processor
        
        Args:
            codebook_path: Path to codebook JSON file containing visual element definitions
        """
        self.regex_pattern = r'"([^"]+)":\s*(\d)'
        self.codebook_path = codebook_path
        
        # Load visual elements from codebook or use defaults
        self.remote_sensing_keys = self._load_remote_sensing_elements()
        self.street_view_keys = self._load_street_view_elements()
        
        # City list
        self.all_cities = [
            'Boston', 'Chicago', 'Dallas', 'Detroit', 'Los Angeles', 'Miami', 'New York', 'Philadelphia',
            'San Francisco', 'Seattle', 'Washington', 'Albuquerque', 'Austin', 'Baltimore', 'Charlotte',
            'Columbus', 'Denver', 'El Paso', 'Fort Worth', 'Houston', 'Jacksonville', 'Las Vegas',
            'Memphis', 'Milwaukee', 'Oklahoma City', 'Phoenix', 'Portland', 'San Antonio',
            'San Diego', 'San Jose', 'Tucson'
        ]
    
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

    def _extract_description_dict(self, description_str: str) -> Dict[str, int]:
        """
        Extract fields and values from description string
        """
        matches = re.findall(self.regex_pattern, description_str)
        description_dict = {}
        for key, value in matches:
            try:
                description_dict[key] = int(value)
            except ValueError:
                description_dict[key] = 0
        return description_dict

    def _process_single_file(self, input_file: str, output_file: str) -> None:
        """
        Process single file, convert description string to structured dictionary
        """
        processed_count = 0
        
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                try:
                    entry = json.loads(line)
                    description_str = entry["description"]
                    description_dict = self._extract_description_dict(description_str)
                    entry["description"] = description_dict
                    json.dump(entry, outfile)
                    outfile.write("\n")
                    processed_count += 1
                except Exception as e:
                    print(f"Skipping erroneous data: {e}")
        
        print(f"✅ Processing completed: {input_file} -> {output_file}")

    def _aggregate_single_file(self, input_file: str, output_file: str, target_keys: List[str]) -> Set[str]:
        """
        Aggregate data from single file
        """
        tract_descriptions = defaultdict(lambda: defaultdict(int))
        keys_in_file = set()

        with open(input_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                tract_id = data['tract_id']
                desc = data.get('description', {})

                for key in target_keys:
                    val = int(desc.get(key, '0'))
                    tract_descriptions[tract_id][key] += val
                    if key in desc:
                        keys_in_file.add(key)

        with open(output_file, 'w') as f:
            for tract_id, desc_counts in tract_descriptions.items():
                completed_desc = {key: desc_counts.get(key, 0) for key in target_keys}
                json.dump({
                    'tract_id': tract_id,
                    'description': completed_desc
                }, f)
                f.write('\n')

        return keys_in_file

    def process_remote_sensing_data(self, 
                                  input_path: Optional[str] = None,
                                  intermediate_path: Optional[str] = None,
                                  output_path: Optional[str] = None) -> None:
        """
        Process remote sensing image data
        
        Args:
            input_path: Input file path, uses preset path by default
            intermediate_path: Intermediate file path, uses preset path by default  
            output_path: Output file path, uses preset path by default
        """
        print("🚀 Starting to process remote sensing image data...")
        
        # Set paths (use provided parameters or default values)
        rs_input = input_path or "/data3/maruolong/segregation/Baseline/All_Cities/analysis/caption/rs_images/rs_image_caption.jsonl"
        rs_intermediate = intermediate_path or "/data3/maruolong/segregation/Baseline/All_Cities/analysis/caption/rs_images/processed_rs_image_caption.jsonl"
        rs_output = output_path or "/data3/maruolong/segregation/Baseline/All_Cities/analysis/caption/rs_images/rs_aggregated_description_by_tract.jsonl"
        
        print(f"Input file: {rs_input}")
        print(f"Intermediate file: {rs_intermediate}")
        print(f"Output file: {rs_output}")
        
        # Step 1: Process raw data
        self._process_single_file(rs_input, rs_intermediate)
        print("✅ Remote sensing data preprocessing completed!")
        
        # Step 2: Aggregate data
        actual_keys = self._aggregate_single_file(rs_intermediate, rs_output, self.remote_sensing_keys)
        
        print(f"\n✅ Remote sensing data aggregation completed, output written to: {rs_output}")
        print("📋 Actual target description fields found in file:")
        print(f"Number of fields found: {len(actual_keys)}")
        print(f"Field list: {sorted(list(actual_keys))}")

    def process_street_view_data(self,
                               input_root: Optional[str] = None,
                               output_root: Optional[str] = None,
                               final_output: Optional[str] = None) -> None:
        """
        Process street view image data
        
        Args:
            input_root: Input root directory, uses preset path by default
            output_root: Output root directory, uses preset path by default
            final_output: Final output file, uses preset path by default
        """
        print("🚀 Starting to process street view image data...")
        
        # Set paths (use provided parameters or default values)
        street_input_root = input_root or "/data3/maruolong/segregation/Baseline/All_Cities/analysis/caption/result_new"
        street_output_root = output_root or "/data3/maruolong/segregation/Baseline/All_Cities/analysis/caption/result_process"
        street_final_output = final_output or '/data3/maruolong/segregation/Baseline/All_Cities/analysis/caption/aggregated_description_by_tract.jsonl'
        
        print(f"Input root directory: {street_input_root}")
        print(f"Output root directory: {street_output_root}")
        print(f"Final output: {street_final_output}")
        
        # Step 1: Process individual city files
        os.makedirs(street_output_root, exist_ok=True)

        for city in self.all_cities:
            city_filename = city.replace(" ", "_")
            input_file = os.path.join(street_input_root, f"{city_filename}_image_caption.jsonl")
            output_file = os.path.join(street_output_root, f"{city_filename}_image_caption.jsonl")
            
            if os.path.exists(input_file):
                print(f"✅ Processing: {city}")
                self._process_single_file(input_file, output_file)
            else:
                print(f"⚠️ File not found: {input_file}")

        print("🎉 All cities processing completed!")

        # Step 2: Aggregate all street view data
        input_files = glob.glob(os.path.join(street_output_root, '*.jsonl'))

        tract_descriptions = defaultdict(lambda: defaultdict(int))
        file_description_keys = {}

        for file_path in tqdm(input_files, desc="Processing all files"):
            filename = os.path.basename(file_path)
            keys_in_file = set()

            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    tract_id = data['tract_id']
                    desc = data.get('description', {})

                    for key in self.street_view_keys:
                        val = int(desc.get(key, '0'))
                        tract_descriptions[tract_id][key] += val
                        if key in desc:
                            keys_in_file.add(key)

            file_description_keys[filename] = keys_in_file

        with open(street_final_output, 'w') as f:
            for tract_id, desc_counts in tract_descriptions.items():
                completed_desc = {key: desc_counts.get(key, 0) for key in self.street_view_keys}
                json.dump({
                    'tract_id': tract_id,
                    'description': completed_desc
                }, f)
                f.write('\n')

        print(f"\n✅ Street view data aggregation completed, output written to: {street_final_output}")
        print("\n📋 Actual target description fields found in each file:\n")

        for fname, keys in file_description_keys.items():
            print(f"{fname}:")
            print(f"  Number of fields found: {len(keys)}")
            print(f"  Field list: {sorted(list(keys))}\n")

    def run_all(self, 
               rs_input_path: Optional[str] = None,
               rs_intermediate_path: Optional[str] = None,
               rs_output_path: Optional[str] = None,
               street_input_root: Optional[str] = None,
               street_output_root: Optional[str] = None,
               street_final_output: Optional[str] = None) -> None:
        """
        Run all processing pipelines
        
        Args:
            All path parameters are optional, use default values if not provided
        """
        print("🎯 Starting to process all data...")
        
        # Process remote sensing data
        self.process_remote_sensing_data(
            input_path=rs_input_path,
            intermediate_path=rs_intermediate_path,
            output_path=rs_output_path
        )
        
        print("\n" + "="*50 + "\n")
        
        # Process street view data
        self.process_street_view_data(
            input_root=street_input_root,
            output_root=street_output_root,
            final_output=street_final_output
        )
        
        print("\n🎉 All data processing completed!")


# Usage example
if __name__ == "__main__":
    # Define codebook path
    default_codebook_path = "/data3/maruolong/VISAGE/agents/literature_agent/knowledge_base/codebooks.json"
    
    # Method 1: Use all default paths with codebook (simplest)
    processor1 = CaptionProcessor(codebook_path=default_codebook_path)
    processor1.run_all()
    
    # Method 2: Customize some paths with codebook
    processor2 = CaptionProcessor(codebook_path=default_codebook_path)
    processor2.run_all(
        rs_input_path="/data3/maruolong/segregation/Baseline/All_Cities/analysis/caption/rs_images/rs_image_caption.jsonl",
        street_input_root="/data3/maruolong/segregation/Baseline/All_Cities/analysis/caption/result_new"
    )
    
    # Method 3: Process separately with custom paths and codebook
    processor3 = CaptionProcessor(codebook_path=default_codebook_path)
    processor3.process_remote_sensing_data(
        input_path="/data3/maruolong/segregation/Baseline/All_Cities/analysis/caption/rs_images/rs_image_caption.jsonl",
        output_path="/data3/maruolong/segregation/Baseline/All_Cities/analysis/caption/rs_images/rs_aggregated_description_by_tract.jsonl"
    )
    
    # Method 4: Process separately with default paths and codebook
    processor4 = CaptionProcessor(codebook_path=default_codebook_path)
    processor4.process_street_view_data()
    
    # Method 5: Without codebook (use default elements)
    processor5 = CaptionProcessor()  # No codebook path provided
    processor5.run_all()