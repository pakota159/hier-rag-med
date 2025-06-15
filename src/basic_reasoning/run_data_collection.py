"""Main script for running the data collection pipeline."""

import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional

from .data_collection.medreason_fetcher import MedReasonFetcher
from .data_collection.msdiagnosis_fetcher import MSDiagnosisFetcher
from .data_collection.pmc_patients_fetcher import PMCPatientsFetcher
from .data_collection.drugbank_fetcher import DrugBankFetcher
from .data_collection.data_validator import DataValidator
from .utils.logging_config import setup_logging
from .utils.progress_tracker import ProgressTracker
from .utils.file_handlers import FileHandler

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_data_collection(config: Dict, max_documents: Optional[int] = None) -> None:
    """
    Run the data collection pipeline.
    
    Args:
        config: Configuration dictionary
        max_documents: Optional maximum number of documents to fetch per dataset
    """
    # Setup logging
    setup_logging(config.get("log_dir", "logs"))
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker()
    
    # Initialize file handler
    file_handler = FileHandler()
    
    # Create output directory
    output_dir = Path(config["processing"]["output_dir"])
    file_handler.ensure_dir(output_dir)
    
    # Initialize fetchers
    fetchers = {
        "medreason": MedReasonFetcher(config["datasets"]["medreason"], checkpoint_dir=output_dir / "checkpoints"),
        "msdiagnosis": MSDiagnosisFetcher(config["datasets"]["msdiagnosis"], checkpoint_dir=output_dir / "checkpoints"),
        "pmc_patients": PMCPatientsFetcher(config["datasets"]["pmc_patients"], checkpoint_dir=output_dir / "checkpoints"),
        "drugbank": DrugBankFetcher(config["datasets"]["drugbank"], checkpoint_dir=output_dir / "checkpoints")
    }
    
    # Initialize validator
    validator = DataValidator(config)
    
    # Register datasets with progress tracker
    for name, fetcher in fetchers.items():
        info = fetcher.get_dataset_info()
        progress_tracker.register_dataset(name, info.expected_size)
    
    # Collect documents from each dataset
    all_documents = []
    for name, fetcher in fetchers.items():
        try:
            logger.info(f"Fetching documents from {name}")
            documents = fetcher.fetch_documents(max_documents)
            progress_tracker.update_progress(name, len(documents))
            all_documents.extend(documents)
            
            # Save dataset-specific output
            output_file = output_dir / f"{name}_documents.json"
            file_handler.save_json([doc.__dict__ for doc in documents], output_file)
            
        except Exception as e:
            logger.error(f"Error fetching {name} documents: {str(e)}")
            continue
    
    # Validate all documents
    logger.info("Validating collected documents")
    validation_stats = validator.validate_dataset(all_documents)
    validation_report = validator.get_validation_report(validation_stats)
    logger.info(validation_report)
    
    # Save validation report
    report_file = output_dir / "validation_report.txt"
    with open(report_file, 'w') as f:
        f.write(validation_report)
    
    # Save unified dataset
    unified_file = output_dir / "unified_dataset.json"
    file_handler.save_json([doc.__dict__ for doc in all_documents], unified_file)
    
    # Print final progress
    progress_tracker.print_progress()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run medical dataset collection pipeline")
    parser.add_argument("--config", default="config/datasets_config.yaml", help="Path to configuration file")
    parser.add_argument("--max-documents", type=int, help="Maximum number of documents to fetch per dataset")
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        run_data_collection(config, args.max_documents)
    except Exception as e:
        logger.error(f"Error running data collection: {str(e)}")
        raise

if __name__ == "__main__":
    main() 