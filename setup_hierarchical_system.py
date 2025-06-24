#!/usr/bin/env python3
"""
Enhanced Setup script for Hierarchical Medical Q&A System
Updated to support Microsoft BiomedNLP-PubMedBERT medical embedding

File: setup_hierarchical_system.py
"""

import sys
import time
import json
import torch
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from loguru import logger
from src.basic_reasoning.config import Config
from src.basic_reasoning.processing import HierarchicalDocumentProcessor
from src.basic_reasoning.retrieval import HierarchicalRetriever


def validate_medical_embedding_setup() -> bool:
    """Validate that the system can load the medical embedding model."""
    logger.info("🏥 Validating medical embedding setup...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        
        # Test tokenizer loading
        logger.info(f"   📝 Testing tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Test model loading (just check if it can be initialized)
        logger.info(f"   🧠 Testing model loading: {model_name}")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=False
        )
        
        # Test a simple encoding
        test_text = "The patient presents with acute myocardial infarction."
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
            
        logger.info(f"   ✅ Successfully generated embedding with shape: {embedding.shape}")
        logger.info("🏥 Medical embedding validation completed successfully")
        
        # Clean up memory
        del model, tokenizer, embedding, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing required packages for medical embedding: {e}")
        logger.error("   💡 Run: pip install transformers>=4.35.0 torch>=2.1.0")
        return False
    except Exception as e:
        logger.error(f"❌ Medical embedding validation failed: {e}")
        logger.error("   💡 Check internet connection and model availability")
        return False


def check_system_requirements() -> Dict[str, bool]:
    """Check system requirements for medical embedding support."""
    logger.info("🔍 Checking system requirements...")
    
    requirements = {
        "torch": False,
        "transformers": False,
        "sentence_transformers": False,
        "chromadb": False,
        "sufficient_memory": False,
        "device_support": False
    }
    
    try:
        import torch
        requirements["torch"] = True
        logger.info(f"   ✅ PyTorch {torch.__version__}")
        
        # Check device support
        if torch.cuda.is_available():
            logger.info(f"   🎮 CUDA available: {torch.cuda.get_device_name()}")
            requirements["device_support"] = True
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("   🍎 MPS (Apple Silicon) available")
            requirements["device_support"] = True
        else:
            logger.info("   💻 Using CPU")
            requirements["device_support"] = True
        
        # Check memory (rough estimate)
        if torch.cuda.is_available():
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"   💾 GPU Memory: {memory_gb:.1f} GB")
            requirements["sufficient_memory"] = memory_gb >= 6  # Minimum for medical model
        else:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            logger.info(f"   💾 System Memory: {memory_gb:.1f} GB")
            requirements["sufficient_memory"] = memory_gb >= 8  # Minimum for CPU
            
    except ImportError:
        logger.warning("   ⚠️ PyTorch not installed")
    
    try:
        import transformers
        requirements["transformers"] = True
        logger.info(f"   ✅ Transformers {transformers.__version__}")
    except ImportError:
        logger.warning("   ⚠️ Transformers not installed")
    
    try:
        import sentence_transformers
        requirements["sentence_transformers"] = True
        logger.info(f"   ✅ Sentence Transformers {sentence_transformers.__version__}")
    except ImportError:
        logger.warning("   ⚠️ Sentence Transformers not installed")
    
    try:
        import chromadb
        requirements["chromadb"] = True
        logger.info(f"   ✅ ChromaDB {chromadb.__version__}")
    except ImportError:
        logger.warning("   ⚠️ ChromaDB not installed")
    
    all_good = all(requirements.values())
    if all_good:
        logger.info("🎉 All system requirements satisfied")
    else:
        failed = [k for k, v in requirements.items() if not v]
        logger.warning(f"⚠️ Missing requirements: {failed}")
    
    return requirements


def find_foundation_dataset() -> Path:
    """Find the foundation dataset file with enhanced search."""
    logger.info("🔍 Searching for foundation dataset...")
    
    search_paths = [
        Path("data/foundation_dataset/foundation_medical_data.json"),
        Path("data/foundation_dataset.json"),
        Path("data/foundation_dataset/unified_dataset.json"),
        Path("data/kg_raw/combined/all_medical_data.json"),
        Path("foundation_dataset.json")
    ]
    
    for path in search_paths:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Found dataset: {path} ({size_mb:.1f} MB)")
            return path
    
    logger.error("❌ No foundation dataset found!")
    logger.error("💡 Please run one of these first:")
    logger.error("   python fetch_foundation_data.py --max-results 50000")
    logger.error("   python fetch_data.py --source all --max-results 1000")
    raise FileNotFoundError("Foundation dataset not found")


def load_foundation_dataset(dataset_path: Path) -> tuple[List[Dict], Dict]:
    """Load and analyze foundation dataset."""
    logger.info(f"📚 Loading foundation dataset from {dataset_path}")
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different data formats
        if isinstance(data, list):
            all_docs = data
        elif isinstance(data, dict):
            if "documents" in data:
                all_docs = data["documents"]
            elif "data" in data:
                all_docs = data["data"]
            else:
                # Assume it's a single document
                all_docs = [data]
        else:
            raise ValueError("Unsupported data format")
        
        # Analyze dataset
        analysis = {
            "total_documents": len(all_docs),
            "type": "enhanced_validated_fetchers" if len(all_docs) > 20000 else "standard",
            "specialties": set(),
            "sources": set(),
            "avg_length": 0
        }
        
        total_length = 0
        for doc in all_docs:
            if isinstance(doc, dict):
                text = str(doc.get("text", ""))
                total_length += len(text)
                
                metadata = doc.get("metadata", {})
                if "medical_specialty" in metadata:
                    analysis["specialties"].add(metadata["medical_specialty"])
                if "source" in metadata:
                    analysis["sources"].add(metadata["source"])
        
        analysis["avg_length"] = total_length // len(all_docs) if all_docs else 0
        analysis["specialty_coverage"] = len(analysis["specialties"]) / max(20, len(analysis["specialties"]))
        analysis["data_quality"] = "High" if analysis["avg_length"] > 200 else "Medium"
        analysis["validated_fetcher_coverage"] = min(1.0, len(all_docs) / 25000)
        
        logger.info(f"   📊 Loaded {analysis['total_documents']:,} documents")
        logger.info(f"   🏥 Medical specialties: {len(analysis['specialties'])}")
        logger.info(f"   📋 Data sources: {len(analysis['sources'])}")
        logger.info(f"   📏 Average length: {analysis['avg_length']} chars")
        
        return all_docs, analysis
        
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        raise


def setup_enhanced_hierarchical_system():
    """Main setup function with medical embedding support."""
    logger.info("🚀 Starting Enhanced Hierarchical Medical Q&A System Setup")
    logger.info("🏥 Optimized for Microsoft BiomedNLP-PubMedBERT embedding")
    
    start_time = time.time()
    
    # Check system requirements
    requirements = check_system_requirements()
    if not all(requirements.values()):
        logger.error("❌ System requirements not met. Please install missing packages.")
        return False
    
    # Validate medical embedding
    if not validate_medical_embedding_setup():
        logger.error("❌ Medical embedding validation failed. Using fallback may reduce performance.")
        response = input("Continue with fallback model? (y/N): ")
        if response.lower() != 'y':
            return False
    
    # Initialize configuration
    try:
        config = Config()
        logger.info("✅ Configuration loaded")
        
        # Display embedding info
        device_info = config.get_device_info()
        logger.info("🔧 System Configuration:")
        logger.info(f"   🖥️  Environment: {device_info['environment']}")
        logger.info(f"   🎯 Device: {device_info['device']}")
        logger.info(f"   🧠 Embedding: {device_info['embedding_model']}")
        logger.info(f"   🏥 Medical optimized: {device_info.get('medical_optimized', False)}")
        
    except Exception as e:
        logger.error(f"❌ Configuration failed: {e}")
        return False
    
    # Initialize components
    try:
        processor = HierarchicalDocumentProcessor(config)
        retriever = HierarchicalRetriever(config)
        logger.info("✅ Components initialized")
    except Exception as e:
        logger.error(f"❌ Component initialization failed: {e}")
        return False
    
    # Check existing collections
    try:
        if retriever.load_hierarchical_collections():
            logger.info("📂 Found existing hierarchical collections")
            
            # Get collection stats
            stats = retriever.get_collection_stats()
            logger.info("📊 Collection Statistics:")
            logger.info(f"   Tier 1: {stats['tier1']['count']:,} documents")
            logger.info(f"   Tier 2: {stats['tier2']['count']:,} documents")  
            logger.info(f"   Tier 3: {stats['tier3']['count']:,} documents")
            logger.info(f"   Total: {stats['total']:,} documents")
            logger.info(f"   Model: {stats['embedding_model']}")
            logger.info(f"   Medical: {stats['medical_optimized']}")
            
            # Check if recreation is needed for medical embedding
            current_model = config.get_embedding_config()["name"]
            stored_model = stats.get("embedding_model", "")
            
            if "BiomedNLP" not in stored_model and "BiomedNLP" in current_model:
                logger.warning("⚠️ Collections were created with non-medical embedding")
                response = input("Recreate collections with medical embedding for better performance? (Y/n): ")
                if response.lower() != 'n':
                    logger.info("🔄 Recreating collections with medical embedding...")
                else:
                    logger.info("✅ Using existing collections")
                    elapsed_time = time.time() - start_time
                    logger.info(f"⏱️ Setup completed in {elapsed_time:.1f} seconds")
                    return True
            else:
                response = input("Recreate collections? (y/N): ")
                if response.lower() != 'y':
                    logger.info("✅ Using existing collections")
                    elapsed_time = time.time() - start_time
                    logger.info(f"⏱️ Setup completed in {elapsed_time:.1f} seconds")
                    return True
                    
    except Exception:
        logger.info("📝 No existing collections found, creating new ones...")
    
    # Load and process foundation dataset
    try:
        dataset_path = find_foundation_dataset()
        all_docs, foundation_analysis = load_foundation_dataset(dataset_path)
        
        logger.info("📊 Foundation Dataset Analysis:")
        logger.info(f"   📚 Total documents: {foundation_analysis['total_documents']:,}")
        logger.info(f"   🔬 Dataset type: {foundation_analysis['type']}")
        logger.info(f"   ⭐ Data quality: {foundation_analysis['data_quality']}")
        logger.info(f"   🏥 Medical specialties: {len(foundation_analysis['specialties'])}")
        
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        return False
    
    # Process documents for hierarchical organization
    try:
        logger.info("🔧 Processing documents for medical hierarchical organization...")
        all_docs = processor.preprocess_documents(all_docs)
        
        # Log tier distribution
        tier_counts = {1: 0, 2: 0, 3: 0}
        for doc in all_docs:
            tier = doc["metadata"].get("tier", 2)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        logger.info("📊 Tier assignment results:")
        logger.info(f"   Tier 1 (Pattern Recognition): {tier_counts.get(1, 0):,} documents")
        logger.info(f"   Tier 2 (Clinical Reasoning): {tier_counts.get(2, 0):,} documents") 
        logger.info(f"   Tier 3 (Evidence Confirmation): {tier_counts.get(3, 0):,} documents")
        
    except Exception as e:
        logger.error(f"❌ Document processing failed: {e}")
        return False
    
    # Create hierarchical collections
    try:
        logger.info("🏗️ Creating hierarchical collections with medical embedding...")
        retriever.create_hierarchical_collections()
        
        # Organize documents by tier
        tier_docs = {1: [], 2: [], 3: []}
        for doc in all_docs:
            tier = doc["metadata"].get("tier", 2)
            tier_docs[tier].append(doc)
        
        # Add documents to each tier
        for tier, docs in tier_docs.items():
            if docs:
                logger.info(f"📚 Adding {len(docs):,} documents to Tier {tier}...")
                retriever.add_documents_to_tier(docs, tier)
        
        logger.info("✅ Hierarchical collections created successfully")
        
    except Exception as e:
        logger.error(f"❌ Collection creation failed: {e}")
        return False
    
    # Final validation
    try:
        stats = retriever.get_collection_stats()
        logger.info("🎉 Setup Complete! Final Statistics:")
        logger.info(f"   📚 Total documents: {stats['total']:,}")
        logger.info(f"   🏥 Medical embedding: {stats['medical_optimized']}")
        logger.info(f"   🧠 Model: {stats['embedding_model']}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️ Total setup time: {elapsed_time:.1f} seconds")
        
        logger.info("🚀 Ready for medical Q&A evaluation!")
        logger.info("💡 Run: python src/evaluation/run_evaluation.py --models hierarchical_system")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Final validation failed: {e}")
        return False


if __name__ == "__main__":
    try:
        success = setup_enhanced_hierarchical_system()
        if success:
            logger.info("✅ Enhanced hierarchical system setup completed successfully")
            exit(0)
        else:
            logger.error("❌ Setup failed")
            exit(1)
    except KeyboardInterrupt:
        logger.info("⏹️ Setup interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        exit(1)