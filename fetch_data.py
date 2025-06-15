#!/usr/bin/env python3
"""
Medical Dataset Fetcher for HierRAGMed Knowledge Graph Extension.

Fetches PubMed abstracts, MTSamples transcriptions, and MeSH vocabulary
for building an extended medical knowledge base.

Usage:
    python fetch_data.py [--source pubmed|mtsamples|mesh|all] [--max-results 1000]
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict
import json

# Add kg module to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "kg"))

from data_fetchers import PubMedFetcher, MTSamplesFetcher, MeSHFetcher, save_dataset_to_files
from loguru import logger


def setup_directories() -> Dict[str, Path]:
    """Setup directory structure for knowledge graph data."""
    base_dir = project_root / "data" / "kg_raw"
    
    directories = {
        "base": base_dir,
        "pubmed": base_dir / "pubmed",
        "mtsamples": base_dir / "mtsamples", 
        "mesh": base_dir / "mesh",
        "combined": base_dir / "combined"
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Setup data directories in: {base_dir}")
    return directories


def fetch_pubmed_data(max_results_per_topic: int = 500) -> List[Dict]:
    """Fetch PubMed abstracts for multiple medical topics."""
    logger.info("🔬 Starting PubMed data fetch...")
    
    fetcher = PubMedFetcher()
    
    # Define medical topics for comprehensive coverage
    medical_topics = [
        "diabetes mellitus",
        "hypertension",
        "myocardial infarction",
        "pregnancy complications",
        "depression",
        "asthma",
        "chronic kidney disease",
        "stroke",
        "heart failure",
        "obesity"
    ]
    
    all_documents = []
    
    for topic in medical_topics:
        try:
            logger.info(f"📋 Fetching abstracts for: {topic}")
            topic_docs = fetcher.fetch_abstracts_by_topic(topic, max_results_per_topic)
            all_documents.extend(topic_docs)
            
            logger.info(f"✅ Fetched {len(topic_docs)} documents for {topic}")
            
        except Exception as e:
            logger.error(f"❌ Error fetching {topic}: {e}")
            continue
    
    logger.info(f"🔬 PubMed fetch complete: {len(all_documents)} total documents")
    return all_documents


def fetch_mtsamples_data() -> List[Dict]:
    """Fetch MTSamples medical transcription data."""
    logger.info("🏥 Starting MTSamples data fetch...")
    
    fetcher = MTSamplesFetcher()
    documents = fetcher.fetch_all_samples()
    
    logger.info(f"🏥 MTSamples fetch complete: {len(documents)} documents")
    return documents


def fetch_mesh_data() -> List[Dict]:
    """Fetch MeSH vocabulary data."""
    logger.info("📚 Starting MeSH data fetch...")
    
    fetcher = MeSHFetcher()
    all_documents = []
    
    # Fetch all major MeSH categories for comprehensive coverage
    categories = ["C", "E", "G", "F", "D"]  # Diseases, Procedures, Processes, Psychology, Drugs
    
    for category in categories:
        try:
            logger.info(f"📖 Fetching MeSH category: {category}")
            category_docs = fetcher.fetch_concepts_by_category(category)
            all_documents.extend(category_docs)
            
        except Exception as e:
            logger.error(f"❌ Error fetching MeSH category {category}: {e}")
            continue
    
    logger.info(f"📚 MeSH fetch complete: {len(all_documents)} documents")
    return all_documents


def combine_datasets(pubmed_docs: List[Dict], mtsamples_docs: List[Dict], 
                    mesh_docs: List[Dict]) -> List[Dict]:
    """Combine all datasets and add dataset statistics."""
    combined = pubmed_docs + mtsamples_docs + mesh_docs
    
    # Add global metadata
    stats = {
        "total_documents": len(combined),
        "pubmed_count": len(pubmed_docs),
        "mtsamples_count": len(mtsamples_docs), 
        "mesh_count": len(mesh_docs),
        "sources": ["pubmed", "mtsamples", "mesh"]
    }
    
    logger.info(f"📊 Dataset Statistics:")
    logger.info(f"   Total Documents: {stats['total_documents']}")
    logger.info(f"   PubMed: {stats['pubmed_count']}")
    logger.info(f"   MTSamples: {stats['mtsamples_count']}")
    logger.info(f"   MeSH: {stats['mesh_count']}")
    
    return combined, stats


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Fetch medical datasets for HierRAGMed")
    parser.add_argument(
        "--source", 
        choices=["pubmed", "mtsamples", "mesh", "all"],
        default="all",
        help="Which data source to fetch (default: all)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=500,
        help="Maximum results per topic for PubMed (default: 500)"
    )
    parser.add_argument(
        "--email",
        type=str,
        default="hierragmed@example.com",
        help="Email for PubMed API (required for higher rate limits)"
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 HierRAGMed Medical Dataset Fetcher")
    logger.info(f"   Source: {args.source}")
    logger.info(f"   Max results per topic: {args.max_results}")
    
    # Setup directories
    directories = setup_directories()
    
    # Initialize data containers
    pubmed_docs = []
    mtsamples_docs = []
    mesh_docs = []
    
    try:
        # Fetch data based on source selection
        if args.source in ["pubmed", "all"]:
            pubmed_docs = fetch_pubmed_data(args.max_results)
            save_dataset_to_files(pubmed_docs, directories["pubmed"], "pubmed")
        
        if args.source in ["mtsamples", "all"]:
            mtsamples_docs = fetch_mtsamples_data()
            save_dataset_to_files(mtsamples_docs, directories["mtsamples"], "mtsamples")
        
        if args.source in ["mesh", "all"]:
            mesh_docs = fetch_mesh_data()
            save_dataset_to_files(mesh_docs, directories["mesh"], "mesh")
        
        # Combine datasets if fetching all
        if args.source == "all":
            combined_docs, stats = combine_datasets(pubmed_docs, mtsamples_docs, mesh_docs)
            
            # Save combined dataset
            combined_file = directories["combined"] / "all_medical_data.json"
            with open(combined_file, "w") as f:
                json.dump(combined_docs, f, indent=2)
            
            # Save statistics
            stats_file = directories["combined"] / "dataset_statistics.json"
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"💾 Saved combined dataset: {combined_file}")
            logger.info(f"📊 Saved statistics: {stats_file}")
        
        logger.info("✅ Data fetching completed successfully!")
        logger.info(f"📁 Data saved to: {directories['base']}")
        
        # Print next steps
        print("\n🎯 Next Steps:")
        print("1. Review fetched data in data/kg_raw/")
        print("2. Process data with: python -m src.kg.processing")
        print("3. Build knowledge graph enhanced RAG system")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Data fetching interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"❌ Data fetching failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())