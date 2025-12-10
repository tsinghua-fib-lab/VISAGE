import json
import os
import sys
import logging
from typing import Dict, List, Any
from datetime import datetime

sys.path.append("/data3/maruolong/VISAGE/agents/literature_agent/core")

class LiteratureAnalysisPipeline:
    """
    Literature Analysis Pipeline - Integrates Curator, Extractor, and Coder modules
    for comprehensive academic literature analysis.
    """
    
    def __init__(self, base_dir: str = "/data3/maruolong/VISAGE/data/processed/literature", logger=None):
        """
        Initialize the analysis pipeline.
        
        Args:
            base_dir: Base directory for data storage
            logger: Logger instance (optional)
        """
        self.base_dir = base_dir
        self.logger = logger or logging.getLogger(__name__)
        
        # Define file paths (consistent with original modules)
        self.curator_files = {
            'prisma_counts': f"{base_dir}/prisma_counts.json",
            'curator_table': f"{base_dir}/curator_table.json"
        }
        self.extractor_files = {
            'extracted_cues': f"{base_dir}/extracted_cues.json"
        }
        self.coder_files = {
            'codebooks': f"{base_dir}/codebooks.json"
        }
        
        self.logger.info("Literature Analysis Pipeline initialized")

    def run_curator_workflow(self) -> Dict:
        """
        Run the literature collection workflow (Curator).
        
        Returns:
            Dictionary containing PRISMA counts and curated papers
        """
        self.logger.info("🚀 Starting Literature Collection Workflow (Curator)")
        
        try:
            # Import curator module
            from curator import CuratorAgent
            # from curator_with_knownpapers import CuratorAgent
            # from curator_with_more_APIs import CuratorAgent
            
            # Initialize curator (consistent with original code)
            curator = CuratorAgent()
            
            # Execute workflow (consistent with original code)
            result = curator.execute_workflow()
            
            # Save PRISMA counts to file (consistent with original code)
            with open(self.curator_files['prisma_counts'], 'w', encoding='utf-8') as f:
                json.dump({"prisma_counts": result["prisma_counts"]}, f, indent=2, ensure_ascii=False)
            
            # Save curated items to file (consistent with original code)
            with open(self.curator_files['curator_table'], 'w', encoding='utf-8') as f:
                json.dump({"curator_table": result["curator_table"]}, f, indent=2, ensure_ascii=False)
            
            print("✅ PRISMA counts saved to:", self.curator_files['prisma_counts'])
            print("✅ Curated items saved to:", self.curator_files['curator_table'])
            
            # Print to console for verification (consistent with original code)
            print("\n📊 PRISMA Counts:")
            print(json.dumps(result["prisma_counts"], indent=2))
            
            print(f"\n📚 Curated Items ({len(result['curator_table'])} papers):")
            for i, paper in enumerate(result['curator_table']):
                print(f"  {i+1}. {paper['Title']}")
                print(f"     Authors: {', '.join(paper['Authors'][:2])}...")
                print(f"     Year: {paper['Year']}, Perspective: {paper['PerspectiveHint']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Literature collection workflow failed: {e}")
            raise

    def run_extractor_workflow(self) -> Dict:
        """
        Run the feature extraction workflow (Extractor).
        
        Returns:
            Dictionary containing extracted cues and metadata
        """
        self.logger.info("🔍 Starting Feature Extraction Workflow (Extractor)")
        
        try:
            # Import extractor module
            from extractor import ExtractorAgent
            
            # Initialize extractor (consistent with original code)
            extractor = ExtractorAgent()
            
            # Process curator output and save extracted cues (consistent with original code)
            curator_table_path = self.curator_files['curator_table']
            output_path = self.extractor_files['extracted_cues']
            
            result = extractor.process_curator_output(curator_table_path, output_path)
            
            print(f"✅ Extraction completed!")
            print(f"📊 Extracted {len(result['extracted_cues'])} cues total")
            
            # Print summary by category (consistent with original code)
            categories = {}
            for cue in result['extracted_cues']:
                category = cue['ProvisionalCategory']
                categories[category] = categories.get(category, 0) + 1
            
            print(f"\n📋 Cues by category:")
            for category, count in categories.items():
                print(f"   - {category}: {count} cues")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Feature extraction workflow failed: {e}")
            raise

    def run_coder_workflow(self) -> Dict:
        """
        Run the coding analysis workflow (Coder).
        
        Returns:
            Dictionary containing codebooks and analysis results
        """
        self.logger.info("📊 Starting Coding Analysis Workflow (Coder)")
        
        try:
            # Import coder module
            from coder import CoderAgent
            
            # Initialize coder (consistent with original code)
            coder = CoderAgent()
            
            # Process extracted cues and generate codebooks (consistent with original code)
            extracted_cues_path = self.extractor_files['extracted_cues']
            output_path = self.coder_files['codebooks']
            
            result = coder.process_extracted_cues_file(extracted_cues_path, output_path)
            
            print(f"✅ Coder workflow completed!")
            print(f"📊 Generated {len(result['street_view_codebook'])} street view codebook entries")
            print(f"📊 Generated {len(result['remote_sensing_codebook'])} remote sensing codebook entries")
            print(f"⚠️  Found {len(result['conflicts'])} conflicts requiring review")
            
            # Print summary by category (consistent with original code)
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
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Coding analysis workflow failed: {e}")
            raise

    def run_full_pipeline(self) -> Dict:
        """
        Run the complete literature analysis pipeline.
        
        Returns:
            Dictionary containing comprehensive analysis results
        """
        self.logger.info("🎯 Starting Complete Literature Analysis Pipeline")
        
        try:
            # Step 1: Literature Collection
            print("\n" + "="*60)
            print("📚 STEP 1: Literature Collection (Curator)")
            print("="*60)
            curator_results = self.run_curator_workflow()
            
            # Step 2: Feature Extraction
            print("\n" + "="*60)
            print("🔍 STEP 2: Feature Extraction (Extractor)")
            print("="*60)
            extractor_results = self.run_extractor_workflow()
            
            # Step 3: Coding Analysis
            print("\n" + "="*60)
            print("📊 STEP 3: Coding Analysis (Coder)")
            print("="*60)
            coder_results = self.run_coder_workflow()
            
            # Generate final report
            final_report = self._generate_final_report(curator_results, extractor_results, coder_results)
            
            print("\n" + "🎉" * 30)
            print("🎉 COMPLETE LITERATURE ANALYSIS PIPELINE FINISHED!")
            print("🎉" * 30)
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"❌ Complete pipeline execution failed: {e}")
            raise

    def _generate_final_report(self, curator: Dict, extractor: Dict, coder: Dict) -> Dict:
        """Generate final analysis report."""
        
        final_report = {
            "pipeline_execution": {
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            },
            "summary_statistics": {
                "papers_collected": curator["prisma_counts"]["included"],
                "cues_extracted": len(extractor["extracted_cues"]),
                "street_view_codebook_entries": len(coder["street_view_codebook"]),
                "remote_sensing_codebook_entries": len(coder["remote_sensing_codebook"]),
                "conflicts_requiring_review": len(coder["conflicts"])
            },
            "file_locations": {
                "prisma_counts": self.curator_files['prisma_counts'],
                "curator_table": self.curator_files['curator_table'],
                "extracted_cues": self.extractor_files['extracted_cues'],
                "codebooks": self.coder_files['codebooks']
            },
            "curator_results": curator["prisma_counts"],
            "extractor_summary": {
                "total_cues_extracted": len(extractor["extracted_cues"]),
                "categories_distribution": self._count_categories(extractor["extracted_cues"])
            },
            "coder_summary": {
                "street_view_categories": self._count_codebook_categories(coder["street_view_codebook"]),
                "remote_sensing_categories": self._count_codebook_categories(coder["remote_sensing_codebook"])
            }
        }
        
        # Save final report
        report_file = f"{self.base_dir}/pipeline_final_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Final report saved to: {report_file}")
        
        # Print final summary
        self._print_final_summary(final_report)
        
        return final_report

    def _count_categories(self, cues: List[Dict]) -> Dict:
        """Count category distribution of cues."""
        categories = {}
        for cue in cues:
            category = cue.get('ProvisionalCategory', 'Unknown')
            categories[category] = categories.get(category, 0) + 1
        return categories

    def _count_codebook_categories(self, codebook: List[Dict]) -> Dict:
        """Count category distribution in codebook."""
        categories = {}
        for entry in codebook:
            category = entry.get('Category', 'Unknown')
            categories[category] = categories.get(category, 0) + 1
        return categories

    def _print_final_summary(self, report: Dict):
        """Print final summary to console."""
        stats = report["summary_statistics"]
        
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   📚 Papers Collected: {stats['papers_collected']}")
        print(f"   🔍 Cues Extracted: {stats['cues_extracted']}")
        print(f"   🏙️  Street View Codebook Entries: {stats['street_view_codebook_entries']}")
        print(f"   🛰️  Remote Sensing Codebook Entries: {stats['remote_sensing_codebook_entries']}")
        print(f"   ⚠️  Conflicts Requiring Review: {stats['conflicts_requiring_review']}")

    def run_specific_workflow(self, workflow_name: str) -> Dict:
        """
        Run a specific workflow.
        
        Args:
            workflow_name: Workflow name ('curator', 'extractor', 'coder', 'full')
            
        Returns:
            Results from the specified workflow
        """
        workflows = {
            'curator': self.run_curator_workflow,
            'extractor': self.run_extractor_workflow,
            'coder': self.run_coder_workflow,
            'full': self.run_full_pipeline
        }
        
        if workflow_name not in workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}. Available: {list(workflows.keys())}")
        
        return workflows[workflow_name]()


# =============================================================================
# USAGE EXAMPLES AND DOCUMENTATION
# =============================================================================

"""
COMPREHENSIVE USAGE GUIDE FOR LITERATURE ANALYSIS PIPELINE

This pipeline integrates three specialized modules for academic literature analysis:
1. Curator: Literature collection and screening
2. Extractor: Feature and cue extraction from papers  
3. Coder: Systematic coding and codebook generation

BASIC USAGE:
------------

1. Command Line Usage:
   -------------------
   # Run complete pipeline (recommended)
   python literature_review.py --workflow full
   
   # Run specific workflow only
   python literature_review.py --workflow curator
   python literature_review.py --workflow extractor  
   python literature_review.py --workflow coder
   
   # Custom data directory
   python literature_review.py --workflow full --base_dir /path/to/your/data

2. Python Script Usage:
   --------------------
   from literature_review import LiteratureAnalysisPipeline
   
   # Initialize pipeline
   pipeline = LiteratureAnalysisPipeline(base_dir="/data3/maruolong/VISAGE/data")
   
   # Run complete pipeline
   results = pipeline.run_full_pipeline()
   
   # Run individual workflows
   curator_results = pipeline.run_curator_workflow()
   extractor_results = pipeline.run_extractor_workflow() 
   coder_results = pipeline.run_coder_workflow()
   
   # Run specific workflow by name
   results = pipeline.run_specific_workflow('extractor')

3. Advanced Usage Examples:
   ------------------------
   # Custom logging setup
   import logging
   logging.basicConfig(level=logging.DEBUG)
   pipeline = LiteratureAnalysisPipeline(logger=logging.getLogger("CustomLogger"))
   
   # Error handling with try-catch
   try:
       results = pipeline.run_full_pipeline()
   except Exception as e:
       print(f"Pipeline failed: {e}")
       # Handle error or retry specific workflows

OUTPUT FILES:
------------
The pipeline generates these output files:

- prisma_counts.json: PRISMA flow diagram statistics
- curator_table.json: Curated papers with metadata  
- extracted_cues.json: Extracted visual cues and features
- codebooks.json: Generated codebooks for street view and remote sensing
- pipeline_final_report.json: Comprehensive final report

WORKFLOW DESCRIPTION:
--------------------
1. CURATOR: 
   - Searches academic databases using boolean queries
   - Applies inclusion/exclusion criteria
   - Deduplicates and ranks papers by relevance
   - Output: PRISMA counts and curated paper table

2. EXTRACTOR:
   - Processes curated papers to extract visual cues
   - Categorizes features by type and perspective
   - Output: Structured cue database with provisional categories

3. CODER:
   - Generates systematic codebooks from extracted cues
   - Creates separate codebooks for street view and remote sensing
   - Identifies conflicts for manual review
   - Output: Final codebooks and conflict resolution guide

TROUBLESHOOTING:
---------------
- Ensure all dependencies are installed: 
  pip install requests biopython arxiv scholarly

- Check file permissions for data directory

- Verify API keys for academic databases (optional)

- Individual workflows can be run separately if one fails

SUPPORT:
-------
For issues or questions, check the module documentation or contact the development team.
"""

if __name__ == "__main__":
    import argparse
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Command line argument parsing
    parser = argparse.ArgumentParser(
        description='Literature Analysis Pipeline - Integrated workflow for academic literature analysis'
    )
    parser.add_argument(
        '--workflow', 
        type=str, 
        default='full',
        choices=['curator', 'extractor', 'coder', 'full'],
        help='Workflow to execute (default: full pipeline)'
    )
    parser.add_argument(
        '--base_dir', 
        type=str, 
        default='/data3/maruolong/VISAGE/data/processed/literature',
        help='Base directory for data storage'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = LiteratureAnalysisPipeline(base_dir=args.base_dir)
        
        # Execute specified workflow
        result = pipeline.run_specific_workflow(args.workflow)
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        exit(1)