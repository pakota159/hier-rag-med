#!/usr/bin/env python3
"""
Foundation Dataset Fetcher for HierRAGMed.

Fetches MedReason, MSDiagnosis, PMC-Patients, and DrugBank datasets
for the 95K Foundation Dataset phase.

Usage:
    python fetch_foundation_data.py [--max-results 1000] [--email your.email@example.com]
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# Add basic_reasoning to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from basic_reasoning import fetch_foundation_datasets, save_foundation_datasets


def setup_logging():
    """Setup logging for the fetcher."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "foundation_data_fetch.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Fetch Foundation Medical Datasets for HierRAGMed")
    parser.add_argument(
        "--max-results",
        type=int,
        default=1000,
        help="Maximum results per dataset (default: 1000)"
    )
    parser.add_argument(
        "--email",
        type=str,
        default="hierragmed@example.com",
        help="Email for API requests (default: hierragmed@example.com)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/foundation",  # Separate from kg_raw
        help="Output directory (default: data/foundation)"
    )
    parser.add_argument(
        "--medreason-only",
        action="store_true",
        help="Fetch only MedReason dataset"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with 100 documents per dataset"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("🚀 HierRAGMed Foundation Dataset Fetcher")
    logger.info(f"   Max results per dataset: {args.max_results}")
    logger.info(f"   Email: {args.email}")
    logger.info(f"   Output directory: {args.output_dir}")
    
    try:
        # Determine fetch amounts
        if args.quick:
            max_per_dataset = 100
            logger.info("🏃 Quick mode: 100 documents per dataset")
        else:
            max_per_dataset = args.max_results
        
        if args.medreason_only:
            logger.info("🧠 Fetching MedReason only")
            documents = fetch_foundation_datasets(
                max_medreason=max_per_dataset,
                max_msdiagnosis=0,
                max_pmc=0,
                max_drugbank=0,
                email=args.email
            )
        else:
            logger.info("📚 Fetching all foundation datasets")
            documents = fetch_foundation_datasets(
                max_medreason=max_per_dataset,
                max_msdiagnosis=max_per_dataset,
                max_pmc=max_per_dataset // 2,  # PMC is slower
                max_drugbank=max_per_dataset,
                email=args.email
            )
        
        # Save datasets
        output_dir = Path(args.output_dir)
        save_foundation_datasets(documents, output_dir)
        
        # Print summary
        logger.info("✅ Foundation dataset fetching completed successfully!")
        logger.info(f"📁 Data saved to: {output_dir}")
        logger.info(f"📊 Total documents: {len(documents)}")
        
        # Print breakdown by source
        source_counts = {}
        for doc in documents:
            source = doc["metadata"]["source"]
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info("📋 Dataset breakdown:")
        for source, count in source_counts.items():
            logger.info(f"   {source}: {count} documents")
        
        # Print next steps
        print("\n🎯 Next Steps:")
        print("1. Review fetched data in data/foundation/")
        print("2. Create hierarchical RAG system using this foundation data")
        print("3. Build src/hierarchical/ module to use data/foundation/")
        print("4. Keep kg system separate - it uses data/kg_raw/")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Foundation data fetching interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"❌ Foundation data fetching failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())