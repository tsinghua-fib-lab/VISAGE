import json
import re
from typing import List, Dict, Any, Optional
import logging

class ExtractorAgent:
    """
    Evidence extractor agent for identifying image-observable cues from academic papers.
    """
    
    def __init__(self, model_client=None, logger=None):
        """
        Initialize Extractor Agent.
        
        Args:
            model_client: LLM client instance
            logger: Optional logger instance
        """
        self.model_client = model_client
        self.logger = logger or logging.getLogger(__name__)
        
        # Provisional categories
        self.categories = [
            "access_permeability", "housing_diversity_type", "commercial_frontage", 
            "maintenance_vacancy", "greenery_shade", "road_structure", 
            "barriers_boundaries", "public_space", "transport_hubs", "land_use_mix"
        ]
    
    def load_curator_results(self, curator_table_path: str) -> List[Dict]:
        """
        Load papers from curator_table.json file.
        
        Args:
            curator_table_path: Path to curator_table.json file
            
        Returns:
            List of paper dictionaries
        """
        try:
            with open(curator_table_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                curator_table = data.get('curator_table', [])
                self.logger.info(f"Loaded {len(curator_table)} papers from {curator_table_path}")
                return curator_table
        except FileNotFoundError:
            self.logger.error(f"Curator table file not found: {curator_table_path}")
            return []
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON format in: {curator_table_path}")
            return []
        except Exception as e:
            self.logger.error(f"Error loading curator results: {e}")
            return []
    
    def extract_cues_from_paper(self, paper: Dict) -> List[Dict]:
        """
        Extract image-observable cues from a single paper.
        
        Args:
            paper: Paper metadata with full text
            
        Returns:
            List of extracted cues
        """
        # Since we don't have full text in curator output, use abstract and title for extraction
        abstract = paper.get('Abstract150w', '')
        title = paper.get('Title', '')
        combined_text = f"{title}. {abstract}"
        
        if not combined_text.strip():
            self.logger.warning(f"No text content available for paper: {paper.get('Title')}")
            return []
        
        paper_id = paper.get('DOI') or paper.get('Title', 'unknown')
        
        # Use rule-based approach for extraction
        extracted = self._rule_based_extraction(combined_text, paper_id)
        
        # Filter only image-observable cues
        cues = []
        for cue in extracted:
            if self._is_image_observable(cue):
                cues.append(cue)
        
        self.logger.info(f"Extracted {len(cues)} cues from paper: {paper.get('Title')}")
        return cues
    
    def _rule_based_extraction(self, text: str, paper_id: str) -> List[Dict]:
        """
        Rule-based extraction of cues (placeholder for LLM implementation).
        """
        cues = []
        
        # Define patterns for different cue types
        patterns = {
            'building_height': r'(building height|building scale|tall buildings|high-rises?|skyscrapers?)',
            'green_space': r'(green space|parks?|vegetation|trees?|greenery|gardens?|lawns?)',
            'road_width': r'(road width|street width|wide streets?|narrow streets?|broad avenues?)',
            'commercial_signage': r'(commercial signage|store signs?|business signs?|shop fronts?|retail displays?)',
            'fence_wall': r'(fences?|walls?|barriers?|gates?|enclosures?)',
            'public_transport': r'(bus stops?|transit stations?|public transport|train stations?|subway stations?)',
            'parking_lots': r'(parking lots?|parking areas?|car parks?)',
            'mixed_use': r'(mixed.use|mixed use|commercial.residential|multipurpose)',
            'sidewalk_quality': r'(sidewalks?|pavements?|footpaths?|walkability)',
            'building_density': r'(building density|development density|urban density)',
            'public_amenities': r'(public amenities|community facilities|public spaces?)',
            'housing_type': r'(housing types?|dwelling types?|residential buildings?)'
        }
        
        for cue_name, pattern in patterns.items():
            if re.search(pattern, text.lower()):
                cue = {
                    "paper_id": paper_id,
                    "CueNameRaw": cue_name,
                    "ImageryPerspective": self._infer_perspective(cue_name),
                    "ProvisionalCategory": self._categorize_cue(cue_name),
                    "Direction": self._infer_direction(cue_name, text),
                    "Rationale40w": self._generate_rationale(cue_name),
                    "EvidenceQuote": self._extract_quote(text, cue_name),
                    "EvidenceLocation": "abstract/title",
                    "AppearanceHintsRaw": self._get_appearance_hints(cue_name),
                    "Geography": self._infer_geography(text),
                    "EvidenceType": "observational",
                    "Confidence": self._calculate_confidence(cue_name, text)
                }
                cues.append(cue)
        
        return cues
    
    def _is_image_observable(self, cue: Dict) -> bool:
        """Check if cue is observable in street view or satellite imagery."""
        perspective = cue.get('ImageryPerspective')
        return perspective in ['street_view', 'remote_sensing', 'both']
    
    def _infer_perspective(self, cue_name: str) -> str:
        """Infer imagery perspective for cue."""
        street_view_cues = ['commercial_signage', 'fence_wall', 'public_transport', 'sidewalk_quality']
        remote_sensing_cues = ['building_height', 'road_width', 'parking_lots', 'mixed_use', 'building_density']
        both_cues = ['green_space', 'public_amenities', 'housing_type']
        
        if cue_name in street_view_cues:
            return "street_view"
        elif cue_name in remote_sensing_cues:
            return "remote_sensing" 
        elif cue_name in both_cues:
            return "both"
        else:
            return "unclear"
    
    def _categorize_cue(self, cue_name: str) -> str:
        """Categorize cue into fixed vocabulary."""
        category_mapping = {
            'building_height': 'housing_diversity_type',
            'building_density': 'housing_diversity_type',
            'housing_type': 'housing_diversity_type',
            'green_space': 'greenery_shade', 
            'road_width': 'road_structure',
            'sidewalk_quality': 'road_structure',
            'commercial_signage': 'commercial_frontage',
            'fence_wall': 'barriers_boundaries',
            'public_transport': 'transport_hubs',
            'parking_lots': 'land_use_mix',
            'mixed_use': 'land_use_mix',
            'public_amenities': 'public_space'
        }
        return category_mapping.get(cue_name, 'access_permeability')
    
    def _infer_direction(self, cue_name: str, text: str) -> str:
        """Infer direction of association with segregation."""
        # Simplified inference based on keyword analysis
        segregation_indicators = ['segregat', 'divide', 'separat', 'exclusiv', 'inequal', 'disparit']
        mixing_indicators = ['mix', 'integrat', 'diverse', 'inclusiv', 'cohes', 'interact']
        
        text_lower = text.lower()
        seg_count = sum(1 for indicator in segregation_indicators if indicator in text_lower)
        mix_count = sum(1 for indicator in mixing_indicators if indicator in text_lower)
        
        # Default directions based on urban studies literature
        default_directions = {
            'building_height': 'increase',
            'building_density': 'ambiguous',
            'green_space': 'decrease',
            'road_width': 'increase',
            'commercial_signage': 'decrease',
            'fence_wall': 'increase',
            'public_transport': 'decrease',
            'parking_lots': 'ambiguous',
            'mixed_use': 'decrease',
            'sidewalk_quality': 'decrease',
            'public_amenities': 'decrease',
            'housing_type': 'ambiguous'
        }
        
        if seg_count > mix_count:
            return "increase"
        elif mix_count > seg_count:
            return "decrease" 
        else:
            return default_directions.get(cue_name, "ambiguous")
    
    def _generate_rationale(self, cue_name: str) -> str:
        """Generate rationale for cue."""
        rationales = {
            'building_height': 'Tall buildings may indicate economic stratification and vertical segregation',
            'building_density': 'Building density affects population concentration and social interactions',
            'green_space': 'Green space distribution influences social gathering and community interactions',
            'road_width': 'Wide roads can create physical and psychological barriers between neighborhoods',
            'commercial_signage': 'Commercial presence supports diverse activities and social encounters',
            'fence_wall': 'Physical barriers limit visual and physical access between social groups',
            'public_transport': 'Transit access enables mobility and cross-neighborhood social mixing',
            'parking_lots': 'Parking infrastructure reflects land use priorities and accessibility patterns',
            'mixed_use': 'Mixed use supports diverse activities and around-the-clock social interactions',
            'sidewalk_quality': 'Walkable sidewalks promote pedestrian activity and street-level encounters',
            'public_amenities': 'Public facilities provide shared spaces for community interaction',
            'housing_type': 'Housing diversity accommodates different socioeconomic groups'
        }
        return rationales.get(cue_name, "Built environment feature affecting social exposure patterns")
    
    def _extract_quote(self, text: str, cue_name: str) -> str:
        """Extract relevant quote from text."""
        # Find sentences containing the cue concept
        sentences = re.split(r'[.!?]+', text)
        cue_keywords = cue_name.split('_')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check if sentence contains relevant keywords
            if any(keyword in sentence_lower for keyword in cue_keywords):
                words = sentence.strip().split()
                if len(words) <= 40:
                    return sentence.strip()
                else:
                    return ' '.join(words[:40]) + '...'
        
        # Fallback: return first 40 words of abstract
        words = text.split()[:40]
        return ' '.join(words) + '...'
    
    def _get_appearance_hints(self, cue_name: str) -> str:
        """Get appearance hints for cue."""
        hints = {
            'building_height': 'Vertical structures, shadow patterns, roof lines, number of floors',
            'building_density': 'Building spacing, lot coverage, urban fabric density',
            'green_space': 'Vegetation coverage, park areas, tree canopies, lawn spaces',
            'road_width': 'Pavement width, lane markings, traffic flow, crosswalk presence',
            'commercial_signage': 'Store signs, business names, advertising displays, window displays',
            'fence_wall': 'Barrier structures, gates, perimeter boundaries, security features',
            'public_transport': 'Bus stops, shelters, station infrastructure, signage, waiting areas',
            'parking_lots': 'Paved areas, vehicle parking, lot boundaries, parking structures',
            'mixed_use': 'Combined residential and commercial structures, ground-floor retail',
            'sidewalk_quality': 'Pavement condition, width, accessibility features, street furniture',
            'public_amenities': 'Community buildings, recreational facilities, public gathering spaces',
            'housing_type': 'Building styles, sizes, conditions, architectural features'
        }
        return hints.get(cue_name)
    
    def _infer_geography(self, text: str) -> str:
        """Infer geographic context from text."""
        text_lower = text.lower()
        if any(term in text_lower for term in ['united states', 'us', 'usa', 'american', 'u.s.']):
            return "US"
        elif any(term in text_lower for term in ['europe', 'european', 'eu ', 'uk', 'germany', 'france', 'london', 'berlin']):
            return "EU"
        elif any(term in text_lower for term in ['global', 'international', 'cross-national', 'multiple countries']):
            return "Global"
        else:
            return None
    
    def _calculate_confidence(self, cue_name: str, text: str) -> str:
        """Calculate confidence level for extracted cue."""
        text_lower = text.lower()
        cue_keywords = cue_name.split('_')
        
        # Count occurrences of cue-related terms
        keyword_count = sum(1 for keyword in cue_keywords if keyword in text_lower)
        
        if keyword_count >= 2:
            return "high"
        elif keyword_count == 1:
            return "medium"
        else:
            return "low"
    
    def process_curator_output(self, curator_table_path: str, output_path: str) -> Dict:
        """
        Process all papers from curator table file and save extracted cues.
        
        Args:
            curator_table_path: Path to curator_table.json file
            output_path: Path to save extracted cues JSON file
            
        Returns:
            Dictionary with extracted cues
        """
        self.logger.info("Starting Extractor workflow")
        
        # Load papers from curator output
        papers = self.load_curator_results(curator_table_path)
        if not papers:
            self.logger.error("No papers loaded from curator output")
            return {"extracted_cues": []}
        
        # Extract cues from all accepted papers
        extracted_cues = []
        accepted_papers = [paper for paper in papers if paper.get('Accepted', False)]
        
        self.logger.info(f"Processing {len(accepted_papers)} accepted papers")
        
        for paper in accepted_papers:
            cues = self.extract_cues_from_paper(paper)
            extracted_cues.extend(cues)
        
        # Create output structure
        output = {"extracted_cues": extracted_cues}
        
        # Save to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Extracted cues saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving extracted cues: {e}")
        
        self.logger.info(f"Extracted {len(extracted_cues)} cues from {len(accepted_papers)} papers")
        return output

# Example usage
if __name__ == "__main__":
    # Initialize extractor agent
    extractor = ExtractorAgent()
    
    # Process curator output and save extracted cues
    curator_table_path = "/data3/maruolong/VISAGE/data/curator_table.json"
    output_path = "/data3/maruolong/VISAGE/data/extracted_cues.json"
    
    result = extractor.process_curator_output(curator_table_path, output_path)
    
    print(f"✅ Extraction completed!")
    print(f"📊 Extracted {len(result['extracted_cues'])} cues total")
    
    # Print summary by category
    categories = {}
    for cue in result['extracted_cues']:
        category = cue['ProvisionalCategory']
        categories[category] = categories.get(category, 0) + 1
    
    print(f"\n📋 Cues by category:")
    for category, count in categories.items():
        print(f"   - {category}: {count} cues")