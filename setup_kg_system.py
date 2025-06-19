#!/usr/bin/env python3
"""
Setup script to create KG system collections with real data ONLY.
Replicates exactly what the KG Streamlit app does but in headless mode.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict
import json
import time
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.kg.config import Config
    from src.kg.processing import DocumentProcessor
    from src.kg.retrieval import Retriever
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running from the project root directory.")
    sys.exit(1)


def check_real_data_sources():
    """Check what real data sources are available - NO SAMPLE DATA."""
    data_sources = {
        "raw_data": Path("data/raw"),
        "kg_data": Path("data/kg_raw"), 
        "processed_data": Path("data/processed")
    }
    
    available_sources = {}
    
    for source_name, source_path in data_sources.items():
        if source_path.exists():
            if source_name == "raw_data":
                txt_files = list(source_path.glob("*.txt"))
                if txt_files:  # Only count if files exist
                    available_sources[source_name] = {
                        "path": source_path,
                        "files": txt_files,
                        "count": len(txt_files)
                    }
            elif source_name == "kg_data":
                combined_file = source_path / "combined" / "all_medical_data.json"
                if combined_file.exists():
                    available_sources[source_name] = {
                        "path": source_path,
                        "files": [combined_file],
                        "count": 1
                    }
            elif source_name == "processed_data":
                json_files = list(source_path.glob("*.json"))
                if json_files:  # Only count if files exist
                    available_sources[source_name] = {
                        "path": source_path,
                        "files": json_files,
                        "count": len(json_files)
                    }
    
    return available_sources


def load_raw_text_files(raw_data_path: Path) -> List[Dict]:
    """Load and process raw text files exactly like Streamlit app."""
    logger.info(f"📂 Loading raw text files from {raw_data_path}")
    
    config = Config()
    processor = DocumentProcessor(config.config["processing"])
    all_docs = []
    
    for txt_file in raw_data_path.glob("*.txt"):
        logger.info(f"📄 Processing {txt_file.name}...")
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Process exactly like Streamlit app does
            docs = processor.process_text(content, {
                "source": "medical_documents",
                "doc_id": txt_file.stem
            })
            all_docs.extend(docs)
            logger.info(f"   ✅ Processed {len(docs)} chunks from {txt_file.name}")
            
        except Exception as e:
            logger.error(f"   ❌ Could not read {txt_file.name}: {e}")
            raise
    
    if not all_docs:
        raise ValueError(f"No documents could be processed from {raw_data_path}")
    
    logger.info(f"✅ Loaded {len(all_docs)} total chunks from raw text files")
    return all_docs


def load_kg_data(kg_data_path: Path) -> List[Dict]:
    """Load KG enhanced data exactly like Streamlit app."""
    logger.info(f"📂 Loading KG data from {kg_data_path}")
    
    config = Config()
    processor = DocumentProcessor(config.config["processing"])
    
    # Load KG datasets exactly like Streamlit app
    all_docs = processor.load_kg_datasets(kg_data_path)
    
    if not all_docs:
        raise ValueError(f"No KG documents could be loaded from {kg_data_path}")
    
    logger.info(f"✅ Loaded {len(all_docs)} chunks from KG data")
    return all_docs


def load_processed_data(processed_file: Path) -> List[Dict]:
    """Load processed JSON data."""
    logger.info(f"📂 Loading processed data from {processed_file}")
    
    config = Config()
    processor = DocumentProcessor(config.config["processing"])
    
    all_docs = processor.load_documents(processed_file)
    
    if not all_docs:
        raise ValueError(f"No documents could be loaded from {processed_file}")
    
    logger.info(f"✅ Loaded {len(all_docs)} documents from processed file")
    return all_docs


def setup_kg_system():
    """Main setup function for KG system - REAL DATA ONLY."""
    start_time = time.time()
    
    logger.info("🚀 Starting KG System Setup (Real Data Only)")
    logger.info("=" * 60)
    
    # Initialize configuration
    try:
        config = Config()
        logger.info(f"✅ Loaded config: {config.get_device_info()}")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return False
    
    # Initialize components
    try:
        processor = DocumentProcessor(config.config["processing"])
        retriever = Retriever(config)
        logger.info("✅ Initialized KG components")
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        return False
    
    # Check existing collection
    collection_name = "kg_medical_docs"
    try:
        retriever.load_collection(collection_name)
        logger.info(f"✅ Collection '{collection_name}' already exists")
        
        # Check if we should recreate
        response = input(f"\n🔄 Collection '{collection_name}' already exists. Recreate? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            logger.info("✅ Using existing collection")
            return True
            
    except ValueError:
        logger.info(f"📝 Collection '{collection_name}' doesn't exist, will create new one")
    
    # Check available real data sources
    available_sources = check_real_data_sources()
    
    if not available_sources:
        logger.error("❌ NO REAL DATA SOURCES FOUND!")
        logger.error("📋 Required data sources:")
        logger.error("   1. data/raw/*.txt files (from original HierRAGMed documents)")
        logger.error("   2. data/kg_raw/combined/all_medical_data.json (from fetch_data.py)")
        logger.error("   3. data/processed/*.json files (from previous processing)")
        logger.error("")
        logger.error("🔧 To create real data:")
        logger.error("   # For KG enhanced data:")
        logger.error("   python fetch_data.py --source all --max-results 1000")
        logger.error("")
        logger.error("   # Or place .txt files in data/raw/")
        raise ValueError("No real data sources available. Setup data first.")
    
    logger.info("📊 Available real data sources:")
    for source_name, source_info in available_sources.items():
        logger.info(f"   {source_name}: {source_info['count']} files")
    
    # Load documents from available sources (priority order matches Streamlit app)
    all_docs = []
    
    # Priority 1: KG enhanced data (best quality)
    if "kg_data" in available_sources:
        logger.info("🎯 Using KG enhanced data (highest priority)")
        all_docs = load_kg_data(available_sources["kg_data"]["path"])
    
    # Priority 2: Raw text files 
    elif "raw_data" in available_sources:
        logger.info("🎯 Using raw text files")
        all_docs = load_raw_text_files(available_sources["raw_data"]["path"])
    
    # Priority 3: Processed data (fallback)
    elif "processed_data" in available_sources:
        logger.info("🎯 Using processed data")
        processed_file = available_sources["processed_data"]["files"][0]
        all_docs = load_processed_data(processed_file)
    
    if not all_docs:
        raise ValueError("Failed to load any documents from available sources")
    
    # Create collection and add documents exactly like Streamlit app
    try:
        logger.info(f"📚 Creating collection with {len(all_docs)} documents")
        
        # Create collection
        retriever.create_collection(collection_name)
        
        # Add documents in batches
        retriever.add_documents(all_docs)
        
        # Verify collection works
        retriever.load_collection(collection_name)
        
        # Test search functionality
        test_results = retriever.search("diabetes treatment", n_results=3)
        logger.info(f"✅ Collection created successfully")
        logger.info(f"🔍 Test search returned {len(test_results)} results")
        
        # Log first result for verification
        if test_results:
            first_result = test_results[0]
            logger.info(f"📄 Sample result: {first_result['text'][:100]}...")
        
        setup_time = time.time() - start_time
        logger.info(f"⏱️ KG setup completed in {setup_time:.1f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create collection: {e}")
        raise


def main():
    """Main entry point."""
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | KG | {message}"
    )
    
    try:
        success = setup_kg_system()
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 KG SYSTEM SETUP COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("✅ Collection 'kg_medical_docs' is ready for evaluation")
            print("🔬 You can now run: python src/evaluation/run_evaluation.py --quick --models kg_system")
            print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ KG SYSTEM SETUP FAILED!")
        print("=" * 60)
        print(f"🔧 Error: {e}")
        print("")
        print("📋 Setup requirements:")
        print("   1. Real medical data in data/raw/*.txt OR")
        print("   2. KG data from: python fetch_data.py --source all")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()