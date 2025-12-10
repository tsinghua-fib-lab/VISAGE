import json
import re
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
import logging

class CoderAgent:
    """
    Coder agent for consolidating extracted cues into machine-readable codebooks.
    """
    
    def __init__(self, logger=None):
        """
        Initialize Coder Agent.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Fixed category vocabulary
        self.categories = [
            "access_permeability", "housing_diversity_type", "commercial_frontage",
            "maintenance_vacancy", "greenery_shade", "road_structure", 
            "barriers_boundaries", "public_space", "transport_hubs", "land_use_mix"
        ]
        
        # Modality rubric
        self.street_view_modality = [
            "façade/storefront/signage", "sidewalk/crosswalk", "street furniture", 
            "bus stop/shelter", "window/door condition", "murals/graffiti", 
            "fence transparency/height", "tree trunks/lines"
        ]
        
        self.remote_sensing_modality = [
            "roof form/color", "parcel/lot boundaries", "block pattern", "cul-de-sacs",
            "interchanges/overpasses", "parking lot extents", "housing footprints/density", 
            "canopy polygons", "large open fields/squares"
        ]
        
        self.both_modality = [
            "parks/open spaces", "mixed-use frontage vs block-level commercial zones", 
            "greenery (trunks vs canopy)", "barriers (walls vs perimeter continuity)"
        ]
    
    def load_extracted_cues(self, extracted_cues_path: str) -> List[Dict]:
        """
        Load extracted cues from JSON file.
        
        Args:
            extracted_cues_path: Path to extracted_cues.json file
            
        Returns:
            List of extracted cue dictionaries
        """
        try:
            with open(extracted_cues_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                extracted_cues = data.get('extracted_cues', [])
                self.logger.info(f"Loaded {len(extracted_cues)} cues from {extracted_cues_path}")
                return extracted_cues
        except FileNotFoundError:
            self.logger.error(f"Extracted cues file not found: {extracted_cues_path}")
            return []
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON format in: {extracted_cues_path}")
            return []
        except Exception as e:
            self.logger.error(f"Error loading extracted cues: {e}")
            return []
    
    def string_similarity(self, a: str, b: str) -> float:
        """
        Calculate string similarity between two strings.
        
        Args:
            a: First string
            b: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def normalize_cue_name(self, name: str) -> str:
        """
        Normalize cue name (case, number, spelling).
        
        Args:
            name: Raw cue name
            
        Returns:
            Normalized cue name
        """
        # Convert to lowercase
        normalized = name.lower()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Basic spelling normalization (could be expanded)
        replacements = {
            'height': 'height',
            'heigth': 'height',
            'comercial': 'commercial',
            'signage': 'signage', 
            'signange': 'signage',
            'fence': 'fence',
            'fencing': 'fence',
            'parking': 'parking',
            'mixed': 'mixed',
            'public': 'public',
            'transport': 'transport'
        }
        
        for wrong, correct in replacements.items():
            normalized = normalized.replace(wrong, correct)
        
        # Title case for consistency
        normalized = normalized.title()
        
        return normalized
    
    def merge_similar_cues(self, cues: List[Dict], similarity_threshold: float = 0.9) -> List[Dict]:
        """
        Merge near-duplicate cues based on string similarity.
        
        Args:
            cues: List of extracted cues
            similarity_threshold: Threshold for merging (0-1)
            
        Returns:
            List of merged cues
        """
        merged = []
        used_indices = set()
        
        for i, cue1 in enumerate(cues):
            if i in used_indices:
                continue
                
            current_cue = cue1.copy()
            current_sources = set([current_cue['paper_id']])
            
            for j, cue2 in enumerate(cues[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                name1 = self.normalize_cue_name(current_cue['CueNameRaw'])
                name2 = self.normalize_cue_name(cue2['CueNameRaw'])
                
                similarity = self.string_similarity(name1, name2)
                
                if similarity >= similarity_threshold:
                    # Merge cues
                    current_sources.add(cue2['paper_id'])
                    
                    # Update direction with confidence weighting
                    current_cue = self._merge_cue_directions(current_cue, cue2)
                    
                    used_indices.add(j)
            
            current_cue['Sources'] = list(current_sources)
            merged.append(current_cue)
            used_indices.add(i)
        
        return merged
    
    def _merge_cue_directions(self, cue1: Dict, cue2: Dict) -> Dict:
        """
        Merge directions of two similar cues using confidence weighting.
        """
        confidence_weights = {
            "high": 1.0,
            "medium": 0.7, 
            "low": 0.4
        }
        
        dir1 = cue1.get('Direction', 'ambiguous')
        dir2 = cue2.get('Direction', 'ambiguous')
        conf1 = confidence_weights.get(cue1.get('Confidence', 'medium'), 0.7)
        conf2 = confidence_weights.get(cue2.get('Confidence', 'medium'), 0.7)
        
        # Simple voting for now - could be enhanced
        if dir1 == dir2:
            return cue1
        else:
            # In case of conflict, keep the one with higher confidence
            if conf1 >= conf2:
                return cue1
            else:
                return cue2
    
    def calculate_direction_consensus(self, cue: Dict) -> str:
        """
        Calculate direction consensus for a cue.
        
        Args:
            cue: Cue dictionary
            
        Returns:
            Direction consensus: "+", "-", or "?"
        """
        direction = cue.get('Direction', 'ambiguous')
        
        if direction == 'increase':
            return "+"
        elif direction == 'decrease':
            return "-"
        else:
            return "?"
    
    def finalize_imagery_perspective(self, cue: Dict) -> str:
        """
        Finalize imagery perspective, resolving "unclear" where possible.
        
        Args:
            cue: Cue dictionary
            
        Returns:
            Final perspective: "street_view", "remote_sensing", or "both"
        """
        perspective = cue.get('ImageryPerspective', 'unclear')
        
        if perspective != 'unclear':
            return perspective
        
        # Resolve unclear perspectives based on cue characteristics
        cue_name = cue.get('CueNameRaw', '').lower()
        
        street_view_dominant = ['signage', 'facade', 'storefront', 'sidewalk', 'fence', 'graffiti', 'window', 'door']
        remote_sensing_dominant = ['height', 'density', 'footprint', 'parcel', 'block', 'canopy', 'roof', 'lot']
        
        if any(term in cue_name for term in street_view_dominant):
            return "street_view"
        elif any(term in cue_name for term in remote_sensing_dominant):
            return "remote_sensing"
        else:
            return "both"
    
    def compose_definition(self, cue: Dict) -> str:
        """
        Compose concise definition for cue (≤ 25 words).
        
        Args:
            cue: Cue dictionary
            
        Returns:
            Concise definition
        """
        cue_name = cue.get('CueNameRaw', '').lower()
        rationale = cue.get('Rationale40w', '')
        
        definitions = {
            'building_height': 'Vertical scale of structures in urban environment',
            'building_density': 'Concentration of buildings in given area',
            'green_space': 'Areas with vegetation like parks and green corridors', 
            'road_width': 'Physical width and scale of roadway infrastructure',
            'commercial_signage': 'Business identification and advertising signs',
            'fence_wall': 'Physical barriers defining property boundaries',
            'public_transport': 'Infrastructure for mass transit accessibility',
            'parking_lots': 'Designated areas for vehicle parking',
            'mixed_use': 'Combination of different land uses in same area',
            'sidewalk_quality': 'Condition and features of pedestrian pathways',
            'public_amenities': 'Community facilities and shared public resources',
            'housing_type': 'Variety of residential building forms and styles'
        }
        
        if cue_name in definitions:
            return definitions[cue_name]
        
        # Fallback: use first 25 words of rationale
        words = rationale.split()[:25]
        return ' '.join(words) if words else "Urban built environment feature"
    
    def compose_detection_hints(self, cue: Dict) -> str:
        """
        Compose detection hints for cue (≤ 20 words).
        
        Args:
            cue: Cue dictionary
            
        Returns:
            Detection hints
        """
        appearance = cue.get('AppearanceHintsRaw', '')
        if appearance:
            words = appearance.split()[:20]
            return ' '.join(words)
        
        cue_name = cue.get('CueNameRaw', '').lower()
        
        hints = {
            'building_height': 'Look for multi-story structures and shadows',
            'building_density': 'Observe building spacing and lot coverage',
            'green_space': 'Identify vegetated areas and park spaces',
            'road_width': 'Observe lane count and pavement width',
            'commercial_signage': 'Notice business signs and advertisements',
            'fence_wall': 'Spot boundary walls and fencing materials',
            'public_transport': 'Find bus stops and station infrastructure',
            'parking_lots': 'Look for paved vehicle parking areas',
            'mixed_use': 'Identify combined residential-commercial buildings',
            'sidewalk_quality': 'Check pavement condition and walkway features',
            'public_amenities': 'Locate community facilities and gathering spaces',
            'housing_type': 'Note building styles and residential variety'
        }
        
        return hints.get(cue_name, 'Visual characteristics in urban imagery')
    
    def compose_visual_manifestation_rs(self, cue: Dict) -> str:
        """
        Compose remote sensing visual manifestation (≤ 20 words).
        
        Args:
            cue: Cue dictionary
            
        Returns:
            Visual manifestation description
        """
        cue_name = cue.get('CueNameRaw', '').lower()
        
        manifestations = {
            'building_height': 'Building shadows and roof patterns from above',
            'building_density': 'Building spacing and urban fabric density',
            'green_space': 'Vegetation coverage and park boundaries',
            'road_width': 'Pavement width and lane markings visible',
            'commercial_signage': 'Not typically visible in overhead view',
            'fence_wall': 'Property boundaries and barrier lines',
            'public_transport': 'Station footprints and parking areas',
            'parking_lots': 'Large paved surfaces with vehicle patterns',
            'mixed_use': 'Building diversity and land use patterns',
            'sidewalk_quality': 'Limited visibility in satellite imagery',
            'public_amenities': 'Building footprints and facility layouts',
            'housing_type': 'Roof forms and building footprint variety'
        }
        
        return manifestations.get(cue_name, 'Overhead visual characteristics')
    
    def process_extracted_cues_file(self, extracted_cues_path: str, output_path: str) -> Dict:
        """
        Process extracted cues from file and generate final codebooks.
        
        Args:
            extracted_cues_path: Path to extracted_cues.json file
            output_path: Path to save codebooks JSON file
            
        Returns:
            Dictionary with street view and remote sensing codebooks
        """
        self.logger.info("Starting Coder workflow")
        
        # Load extracted cues from file
        extracted_cues = self.load_extracted_cues(extracted_cues_path)
        if not extracted_cues:
            self.logger.error("No cues loaded from extracted cues file")
            return {
                "street_view_codebook": [],
                "remote_sensing_codebook": [],
                "conflicts": []
            }
        
        # Step 1: Normalize names and merge near-duplicates
        self.logger.info("Normalizing cue names and merging duplicates...")
        for cue in extracted_cues:
            cue['CueNameRaw'] = self.normalize_cue_name(cue['CueNameRaw'])
        
        merged_cues = self.merge_similar_cues(extracted_cues)
        self.logger.info(f"After merging: {len(merged_cues)} unique cues")
        
        # Step 2: Organize into codebooks
        street_view_codebook = []
        remote_sensing_codebook = []
        conflicts = []
        
        self.logger.info("Generating codebook entries...")
        for cue in merged_cues:
            # Finalize perspective
            final_perspective = self.finalize_imagery_perspective(cue)
            
            # Calculate direction consensus
            direction_consensus = self.calculate_direction_consensus(cue)
            
            # Create codebook entry
            codebook_entry = {
                "Name": cue['CueNameRaw'],
                "Category": cue['ProvisionalCategory'],
                "ImageryPerspective": final_perspective,
                "Definition": self.compose_definition(cue),
                "DetectionHints": self.compose_detection_hints(cue),
                "DirectionConsensus": direction_consensus,
                "Confidence": cue.get('Confidence', 'medium'),
                "Sources": cue.get('Sources', [cue['paper_id']])
            }
            
            # Add to appropriate codebook(s)
            if final_perspective in ['street_view', 'both']:
                street_view_codebook.append(codebook_entry)
            
            if final_perspective in ['remote_sensing', 'both']:
                rs_entry = codebook_entry.copy()
                rs_entry['VisualManifestationRS'] = self.compose_visual_manifestation_rs(cue)
                remote_sensing_codebook.append(rs_entry)
            
            # Check for conflicts
            if direction_consensus == "?":
                conflicts.append({
                    "Cue": cue['CueNameRaw'],
                    "Issue": "Ambiguous direction of association",
                    "Sources": cue.get('Sources', [])
                })
        
        output = {
            "street_view_codebook": street_view_codebook,
            "remote_sensing_codebook": remote_sensing_codebook,
            "conflicts": conflicts
        }
        
        # Save to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Codebooks saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving codebooks: {e}")
        
        self.logger.info(f"Generated {len(street_view_codebook)} street view and {len(remote_sensing_codebook)} remote sensing codebook entries")
        self.logger.info(f"Found {len(conflicts)} conflicts")
        
        return output

# Example usage
if __name__ == "__main__":
    # Initialize coder agent
    coder = CoderAgent()
    
    # Process extracted cues and generate codebooks
    extracted_cues_path = "/data3/maruolong/VISAGE/data/extracted_cues.json"
    output_path = "/data3/maruolong/VISAGE/data/codebooks.json"
    
    result = coder.process_extracted_cues_file(extracted_cues_path, output_path)
    
    print(f"✅ Coder workflow completed!")
    print(f"📊 Generated {len(result['street_view_codebook'])} street view codebook entries")
    print(f"📊 Generated {len(result['remote_sensing_codebook'])} remote sensing codebook entries")
    print(f"⚠️  Found {len(result['conflicts'])} conflicts requiring review")
    
    # Print summary by category
    street_view_categories = {}
    for entry in result['street_view_codebook']:
        category = entry['Category']
        street_view_categories[category] = street_view_categories.get(category, 0) + 1
    
    remote_sensing_categories = {}
    for entry in result['remote_sensing_codebook']:
        category = entry['Category']
        remote_sensing_categories[category] = remote_sensing_categories.get(category, 0) + 1
    
    print(f"\n📋 Street View Codebook by category:")
    for category, count in street_view_categories.items():
        print(f"   - {category}: {count} cues")
    
    print(f"\n📋 Remote Sensing Codebook by category:")
    for category, count in remote_sensing_categories.items():
        print(f"   - {category}: {count} cues")