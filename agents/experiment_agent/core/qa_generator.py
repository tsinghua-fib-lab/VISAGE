import json
import random
from typing import Dict, List, Any, Optional

class CaptionGenerator:
    """Caption generator for updating the Caption section in chain-of-thought"""
    
    def __init__(self, codebook_path: Optional[str] = None):
        """
        Initialize Caption generator
        
        Args:
            codebook_path: Path to codebook JSON file containing visual element definitions
        """
        # Load element classifications from codebook or use defaults
        self.codebook_path = codebook_path
        self.increase_segregation_elements_remote, self.decrease_segregation_elements_remote = self._load_remote_sensing_elements()
        self.increase_segregation_elements_street, self.decrease_segregation_elements_street = self._load_street_view_elements()
        
        self.remote_visual_info = {}
        self.street_visual_info = {}
    
    def _load_remote_sensing_elements(self) -> tuple[List[str], List[str]]:
        """
        Load remote sensing elements from codebook or use default values
        
        Returns:
            Tuple of (increase_elements, decrease_elements)
        """
        # Default remote sensing elements
        default_increase = [
            "MonofunctionalResidentialZones", "VacantOrDemolitionSites",
            "GridOrRadialStreetPatterns", "UniformHousingBlocks",
            "RailwaysOrTrainTracks", "ExpresswaysOrHighways"
        ]
        
        default_decrease = [
            "LowDensityHousing", "LargeFootprintBuildings", "TallBuildings",
            "CommercialActivityZones", "ParkingLots", "ParksSquaresOrSportsFields",
            "LargeOpenPublicSpaces", "StreetTreesAndUrbanGreenStrips",
            "ColorfulRoofsOrArchitecturalDiversity"
        ]
        
        # Try to load elements from codebook if path is provided
        if self.codebook_path:
            try:
                with open(self.codebook_path, 'r') as f:
                    codebook_data = json.load(f)
                
                # Extract remote sensing elements from codebook
                remote_sensing_codebook = codebook_data.get("remote_sensing_codebook", [])
                if remote_sensing_codebook:
                    increase_elements = []
                    decrease_elements = []
                    
                    for item in remote_sensing_codebook:
                        if item.get("DirectionConsensus") == "+":
                            increase_elements.append(item["Name"])
                        elif item.get("DirectionConsensus") == "-":
                            decrease_elements.append(item["Name"])
                    
                    print(f"✅ Loaded {len(increase_elements)} increase and {len(decrease_elements)} decrease remote sensing elements from codebook")
                    return increase_elements, decrease_elements
                else:
                    print(f"⚠️ No remote sensing elements found in codebook, using default elements")
                    return default_increase, default_decrease
                    
            except Exception as e:
                print(f"❌ Error loading codebook from {self.codebook_path}: {e}")
                print("Using default remote sensing elements")
                return default_increase, default_decrease
        else:
            print("ℹ️ No codebook path provided, using default remote sensing elements")
            return default_increase, default_decrease
    
    def _load_street_view_elements(self) -> tuple[List[str], List[str]]:
        """
        Load street view elements from codebook or use default values
        
        Returns:
            Tuple of (increase_elements, decrease_elements)
        """
        # Default street view elements
        default_increase = [
            "DetachedSingleFamilyHouse", "BungalowOrLowRiseHouse", "HighFencesOrWalls",
            "BarbedWireOrGatedEnclosures", "VacantLotsOrAbandonedBuildings", "NoSidewalks",
            "LargeEmptyLand", "DeterioratedStreets", "Graffiti", "LitterOrTrashPiles",
            "BoardedUpOrClosedShops", "LackOfCommercialActivity", "NarrowRoadsOrDeadEnds",
            "MonofunctionalResidentialBlocks", "SparseOrShortShrubs",
            "IsolatedLawnsWithoutTrees", "RowHousesOrTownhouses"
        ]
        
        default_decrease = [
            "ApartmentBuildingsOrHighRises", 
            "ParksPlaygroundsOrCommunitySquares",
            "PublicFacilitiesBusStopsSchools", "ParkingLots", 
            "WideMultiLaneRoads", "CrosswalksOrOverpasses", "Bridges",
            "LandmarkBuildingsOrReligiousSites", "TallTreesOrTreeLines",
            "StreetTreesOrGreenBelts", "CommunityGardens",
            "ColorfulCleanFacadesOrWindows"
        ]
        
        # Try to load elements from codebook if path is provided
        if self.codebook_path:
            try:
                with open(self.codebook_path, 'r') as f:
                    codebook_data = json.load(f)
                
                # Extract street view elements from codebook
                street_view_codebook = codebook_data.get("street_view_codebook", [])
                if street_view_codebook:
                    increase_elements = []
                    decrease_elements = []
                    
                    for item in street_view_codebook:
                        if item.get("DirectionConsensus") == "+":
                            increase_elements.append(item["Name"])
                        elif item.get("DirectionConsensus") == "-":
                            decrease_elements.append(item["Name"])
                    
                    print(f"✅ Loaded {len(increase_elements)} increase and {len(decrease_elements)} decrease street view elements from codebook")
                    return increase_elements, decrease_elements
                else:
                    print(f"⚠️ No street view elements found in codebook, using default elements")
                    return default_increase, default_decrease
                    
            except Exception as e:
                print(f"❌ Error loading codebook from {self.codebook_path}: {e}")
                print("Using default street view elements")
                return default_increase, default_decrease
        else:
            print("ℹ️ No codebook path provided, using default street view elements")
            return default_increase, default_decrease
        
    def load_visual_data(self, remote_desc_path: str, street_desc_path: str) -> None:
        """
        Load visual element data
        
        Args:
            remote_desc_path: Remote sensing image description file path
            street_desc_path: Street view image description file path
        """
        # Load visual elements for remote sensing images
        self.remote_visual_info = {}
        with open(remote_desc_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.remote_visual_info[item['tract_id']] = item['description']

        # Load visual elements for street view images
        self.street_visual_info = {}
        with open(street_desc_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.street_visual_info[item['tract_id']] = item['description']
    
    def _summarize_elements(self, elements: List[str], desc: Dict[str, int], is_street: bool = False) -> List[str]:
        """
        Summarize element descriptions
        
        Args:
            elements: Element list
            desc: Element count description
            is_street: Whether it's street view image
            
        Returns:
            Formatted element description list
        """
        result = []
        for key in elements:
            count = desc.get(key, 0)
            display_key = key.replace('_', ' ')
            result.append(f"{count} image(s) show {display_key}")
        return result
    
    def generate_caption(self, remote_desc: Dict[str, int], street_desc: Dict[str, int], street_count: int) -> str:
        """
        Generate Caption description
        
        Args:
            remote_desc: Remote sensing image element description
            street_desc: Street view image element description
            street_count: Number of street view images
            
        Returns:
            Complete Caption text
        """
        remote_increase = self._summarize_elements(self.increase_segregation_elements_remote, remote_desc)
        remote_decrease = self._summarize_elements(self.decrease_segregation_elements_remote, remote_desc)

        street_increase = self._summarize_elements(self.increase_segregation_elements_street, street_desc, is_street=True)
        street_decrease = self._summarize_elements(self.decrease_segregation_elements_street, street_desc, is_street=True)

        remote_caption = (
            f"1. **Among the 7 remote sensing images**, "
            + ", ".join(remote_increase)
            + ", which may contribute to greater income segregation and reduced income diversity among visitors; "
            + ", ".join(remote_decrease)
            + ", which may help promote social inclusion and increase income diversity among visitors."
        )

        street_caption = (
            f"2. **Among the {street_count} street view images**, "
            + ", ".join(street_increase)
            + ", which may contribute to increased segregation; "
            + ", ".join(street_decrease)
            + ", which may help promote diversity and reduce income segregation."
        )

        return remote_caption + "   \n" + street_caption
    
    def replace_caption_section(self, full_cot_text: str, new_caption_block: str) -> str:
        """
        Replace Caption section in chain-of-thought
        
        Args:
            full_cot_text: Complete chain-of-thought text
            new_caption_block: New Caption content
            
        Returns:
            Updated chain-of-thought text
        """
        parts = full_cot_text.split("### 2. **<Caption>**")
        if len(parts) != 2:
            return full_cot_text  # fallback

        summary_part = parts[0].strip()
        rest_part = parts[1]
        caption_split = rest_part.split(", and overall socioeconomic engagement.  ")
        if len(caption_split) != 2:
            return full_cot_text  # fallback

        calculation_and_answer = caption_split[1]
        return f"{summary_part}\n\n### 2. **<Caption>**\n{new_caption_block.strip()}\n   {calculation_and_answer.strip()}"
    
    def process_single_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process single data entry
        
        Args:
            entry: Data entry
            
        Returns:
            Processed data entry
        """
        sample_id = entry.get("sample_id") or entry.get("tract_id")
        if not sample_id:
            return entry

        remote_desc = self.remote_visual_info.get(sample_id, {})
        street_desc = self.street_visual_info.get(sample_id, {})

        if not remote_desc or not street_desc:
            return entry

        # Street view image count = total image paths - 7 (remote sensing image count)
        total_images = len(entry.get("image", []))
        street_image_count = max(total_images - 7, 0)

        new_caption = self.generate_caption(remote_desc, street_desc, street_image_count)

        # Update Caption section in conversation
        for convo in entry.get("conversations", []):
            if convo["from"] == "gpt":
                old_cot = convo["value"]
                new_cot = self.replace_caption_section(old_cot, new_caption)
                convo["value"] = new_cot
        
        return entry
    
    def process_file(self, input_path: str, output_path: str, remote_desc_path: str, street_desc_path: str) -> None:
        """
        Process entire file
        
        Args:
            input_path: Input JSON file path
            output_path: Output JSON file path
            remote_desc_path: Remote sensing description file path
            street_desc_path: Street view description file path
        """
        # Load visual data
        self.load_visual_data(remote_desc_path, street_desc_path)
        
        # Load main data
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Process each entry
        processed_count = 0
        total_count = len(data)
        
        for i, entry in enumerate(data):
            processed_entry = self.process_single_entry(entry)
            data[i] = processed_entry
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processing progress: {processed_count}/{total_count}")
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Update completed, processed {processed_count} entries")
        print(f"✅ Output file: {output_path}")


# Usage example
if __name__ == "__main__":
    # Define codebook path
    default_codebook_path = "/data3/maruolong/VISAGE/agents/literature_agent/knowledge_base/codebooks_filter.json"
    
    # Create generator instance with codebook
    generator = CaptionGenerator(codebook_path=default_codebook_path)
    
    # Define file paths
    input_path = '/data3/maruolong/segregation/Train/Try/update/31_update_only_prompt_all_data_add_rs.json'
    output_path = '/data3/maruolong/VISAGE/data/31_all_cities_seg_update_caption.json'
    remote_desc_path = '/data3/maruolong/segregation/Baseline/All_Cities/analysis/caption/rs_images/rs_aggregated_description_by_tract.jsonl'
    street_desc_path = '/data3/maruolong/segregation/Baseline/All_Cities/analysis/caption/aggregated_description_by_tract.jsonl'
    
    # Process file
    generator.process_file(input_path, output_path, remote_desc_path, street_desc_path)
    
    # Can also process single entry separately
    # with open(input_path, 'r') as f:
    #     data = json.load(f)
    # 
    # sample_entry = data[0]  # Get first entry
    # processed_entry = generator.process_single_entry(sample_entry)
    # print("Processed entry:", processed_entry)
    
    # Method without codebook (use default elements)
    # generator_default = CaptionGenerator()  # No codebook path provided
    # generator_default.process_file(input_path, output_path, remote_desc_path, street_desc_path)