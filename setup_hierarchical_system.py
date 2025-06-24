#!/usr/bin/env python3
"""
Enhanced Setup script for Hierarchical Medical Q&A System
Updated to support Microsoft BiomedNLP-PubMedBERT medical embedding with PyTorch 2.6+ compatibility

File: setup_hierarchical_system.py
"""

import sys
import time
import json
import torch
from pathlib import Path
from typing import Dict, List, Any
import importlib

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from loguru import logger
from src.basic_reasoning.config import Config
from src.basic_reasoning.processing import HierarchicalDocumentProcessor
from src.basic_reasoning.retrieval import HierarchicalRetriever


def force_reload_modules():
    """Force reload of modules that might be cached incorrectly."""
    modules_to_reload = [
        'sentence_transformers',
        'transformers',
        'torch'
    ]
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            try:
                importlib.reload(sys.modules[module_name])
                logger.debug(f"Reloaded module: {module_name}")
            except Exception as e:
                logger.debug(f"Could not reload {module_name}: {e}")


def check_import_with_retry(module_name: str, max_retries: int = 3):
    """Try to import a module with retries and different approaches."""
    for attempt in range(max_retries):
        try:
            if module_name == "sentence_transformers":
                import sentence_transformers
                return sentence_transformers
            elif module_name == "transformers":
                import transformers
                return transformers
            elif module_name == "chromadb":
                import chromadb
                return chromadb
        except ImportError as e:
            if attempt < max_retries - 1:
                logger.debug(f"Import attempt {attempt + 1} failed for {module_name}: {e}")
                # Force reload modules and clear cache
                force_reload_modules()
                # Clear import cache
                if hasattr(importlib, 'invalidate_caches'):
                    importlib.invalidate_caches()
                time.sleep(1)  # Brief pause
            else:
                logger.error(f"All import attempts failed for {module_name}: {e}")
                return None
    
    return None


def validate_medical_embedding_setup() -> bool:
    """Validate that the system can load the medical embedding model with safetensors support."""
    logger.info("🏥 Validating medical embedding setup...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        
        # Test tokenizer loading
        logger.info(f"   📝 Testing tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Test model loading with safetensors preference
        logger.info(f"   🧠 Testing model loading: {model_name}")
        
        # Try with safetensors first
        try:
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=False,
                use_safetensors=True  # Prefer safetensors format
            )
        except Exception as safetensors_error:
            # Fallback to standard loading
            logger.warning("⚠️ Safetensors loading failed, trying standard loading...")
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
        logger.error("   💡 Run: pip install transformers>=4.35.0 torch>=2.6.0")
        return False
    except Exception as e:
        error_msg = str(e)
        if "torch.load" in error_msg and "CVE-2025-32434" in error_msg:
            logger.error(f"❌ PyTorch version too old for security requirements: {e}")
            logger.error("   💡 Run: pip install torch>=2.6.0 --upgrade")
            logger.error("   💡 Or use safetensors format by upgrading transformers>=4.35.0")
        else:
            logger.error(f"❌ Medical embedding validation failed: {e}")
            logger.error("   💡 Check internet connection and model availability")
        return False


def check_system_requirements() -> Dict[str, bool]:
    """Check system requirements for medical embedding support with PyTorch 2.6+ compatibility."""
    logger.info("🔍 Checking system requirements...")
    
    requirements = {
        "torch": False,
        "torch_version": False,
        "transformers": False,
        "sentence_transformers": False,
        "chromadb": False,
        "sufficient_memory": False,
        "device_support": False
    }
    
    # Force reload modules first
    force_reload_modules()
    
    try:
        import torch
        requirements["torch"] = True
        torch_version = torch.__version__
        logger.info(f"   ✅ PyTorch {torch_version}")
        
        # Check if PyTorch version is 2.6+ for CVE-2025-32434 security fix
        version_parts = torch_version.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        if major > 2 or (major == 2 and minor >= 6):
            requirements["torch_version"] = True
            logger.info(f"   ✅ PyTorch version {torch_version} meets security requirements")
        else:
            requirements["torch_version"] = False
            logger.warning(f"   ⚠️ PyTorch {torch_version} < 2.6.0 has security vulnerability CVE-2025-32434")
            logger.warning("   💡 Recommend upgrading: pip install torch>=2.6.0 --upgrade")
        
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
    
    # Check transformers with retry
    transformers_module = check_import_with_retry("transformers")
    if transformers_module:
        requirements["transformers"] = True
        logger.info(f"   ✅ Transformers {transformers_module.__version__}")
    else:
        logger.warning("   ⚠️ Transformers not installed")
    
    # Check sentence-transformers with retry
    sentence_transformers_module = check_import_with_retry("sentence_transformers")
    if sentence_transformers_module:
        requirements["sentence_transformers"] = True
        logger.info(f"   ✅ Sentence Transformers {sentence_transformers_module.__version__}")
    else:
        logger.warning("   ⚠️ Sentence Transformers import failed")
        # Try direct approach
        try:
            import sys
            import subprocess
            result = subprocess.run([sys.executable, "-c", "import sentence_transformers; print(sentence_transformers.__version__)"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"   ✅ Sentence Transformers {version} (verified via subprocess)")
                requirements["sentence_transformers"] = True
            else:
                logger.error(f"   ❌ Sentence Transformers subprocess check failed: {result.stderr}")
        except Exception as e:
            logger.error(f"   ❌ Sentence Transformers subprocess check error: {e}")
    
    # Check chromadb with retry
    chromadb_module = check_import_with_retry("chromadb")
    if chromadb_module:
        requirements["chromadb"] = True
        logger.info(f"   ✅ ChromaDB {chromadb_module.__version__}")
    else:
        logger.warning("   ⚠️ ChromaDB not installed")
    
    # Check if core requirements are met (allow torch_version to be false for fallback)
    core_requirements = ["torch", "transformers", "sentence_transformers", "chromadb", "sufficient_memory", "device_support"]
    all_good = all(requirements[req] for req in core_requirements)
    
    if all_good:
        logger.info("🎉 All system requirements satisfied")
    else:
        failed = [k for k, v in requirements.items() if not v and k in core_requirements]
        logger.warning(f"⚠️ Missing requirements: {failed}")
        
        # If sentence_transformers is still failing, provide specific guidance
        if "sentence_transformers" in failed:
            logger.error("💡 Sentence Transformers troubleshooting:")
            logger.error("   1. Try: pip uninstall sentence-transformers -y && pip install sentence-transformers>=2.2.2")
            logger.error("   2. Check Python path conflicts")
            logger.error("   3. Restart Python interpreter")
    
    return requirements


def find_foundation_dataset() -> Path:
    """Find the foundation dataset file with enhanced search."""
    search_paths = [
        # Correct paths for your directory structure
        Path("data/foundation_dataset/foundation_medical_data.json"),
        Path("data/foundation_dataset/unified_dataset.json"),
        # Original paths as fallbacks
        Path("data/foundation_data.json"),
        Path("data/raw/foundation_data.json"),
        Path("data/processed/foundation_data.json"),
        Path("foundation_data.json"),
        Path("data/enhanced_validated_fetchers_dataset.json"),
        Path("data/raw/enhanced_validated_fetchers_dataset.json"),
        Path("enhanced_validated_fetchers_dataset.json"),
        Path("data/dataset.json"),
        Path("data/raw/dataset.json"),
        Path("dataset.json")
    ]
    
    # Check regular paths first
    for path in search_paths:
        if path.exists():
            logger.info(f"📁 Found dataset: {path}")
            return path
    
    # Check glob patterns separately
    glob_dir = Path("data/foundation_dataset")
    if glob_dir.exists():
        for file_path in glob_dir.glob("foundation_specialty_rebalanced_*.json"):
            if file_path.exists():
                logger.info(f"📁 Found dataset: {file_path}")
                return file_path
    
    logger.error("❌ Foundation dataset not found in any of these locations:")
    for path in search_paths:
        logger.error(f"   - {path}")
    logger.error(f"   - data/foundation_dataset/foundation_specialty_rebalanced_*.json")
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
    """Main setup function with medical embedding support and PyTorch 2.6+ compatibility."""
    logger.info("🚀 Starting Enhanced Hierarchical Medical Q&A System Setup")
    logger.info("🏥 Optimized for Microsoft BiomedNLP-PubMedBERT embedding")
    logger.info("🔒 Enhanced security with PyTorch 2.6+ and safetensors support")
    
    start_time = time.time()
    
    # Check system requirements
    requirements = check_system_requirements()
    core_reqs = ["torch", "transformers", "sentence_transformers", "chromadb", "sufficient_memory", "device_support"]
    
    if not all(requirements[req] for req in core_reqs):
        failed_reqs = [req for req in core_reqs if not requirements[req]]
        logger.error(f"❌ Core system requirements not met: {failed_reqs}")
        
        # Special handling for sentence_transformers import issue
        if "sentence_transformers" in failed_reqs:
            logger.warning("🔧 Attempting to resolve sentence_transformers import issue...")
            
            # Try to reinstall sentence_transformers
            import subprocess
            try:
                logger.info("   📦 Reinstalling sentence-transformers...")
                subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "sentence-transformers", "-y"])
                subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers>=2.2.2", "--force-reinstall"])
                
                # Clear Python import cache
                if hasattr(importlib, 'invalidate_caches'):
                    importlib.invalidate_caches()
                
                # Try import again
                time.sleep(2)  # Brief pause
                try:
                    import sentence_transformers
                    logger.info(f"   ✅ Successfully imported sentence_transformers {sentence_transformers.__version__}")
                    requirements["sentence_transformers"] = True
                except ImportError as e:
                    logger.error(f"   ❌ Still cannot import sentence_transformers: {e}")
                    logger.error("   💡 Manual fix required: restart Python interpreter")
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"   ❌ Failed to reinstall sentence_transformers: {e}")
        
        # Re-check requirements
        if not all(requirements[req] for req in core_reqs):
            logger.error("❌ Requirements still not satisfied. Please fix manually and restart.")
            return False
    
    # Warn about PyTorch version but continue
    if not requirements["torch_version"]:
        logger.warning("⚠️ PyTorch version < 2.6.0 detected. Security vulnerability present.")
        logger.warning("   Continuing with safetensors fallback, but upgrade recommended.")
    
    # Validate medical embedding with fallback handling
    medical_validation_success = validate_medical_embedding_setup()
    if not medical_validation_success:
        logger.warning("⚠️ Medical embedding validation failed.")
        if not requirements["torch_version"]:
            logger.error("❌ PyTorch version too old and medical model failed to load.")
            logger.error("   Please upgrade PyTorch: pip install torch>=2.6.0 --upgrade")
            return False
        else:
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
        logger.info(f"   🏥 Medical optimized: {device_info.get('medical_optimized', medical_validation_success)}")
        logger.info(f"   🔒 PyTorch secure: {requirements['torch_version']}")
        
    except Exception as e:
        logger.error(f"❌ Configuration failed: {e}")
        return False
    
    # Initialize components
    try:
        processor = HierarchicalDocumentProcessor(config.config["processing"])
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
                response = input("Recreate collections with medical embedding for better performance? (y/N): ")
                if response.lower() == 'y':
                    logger.info("🔄 Recreating collections with medical embedding...")
                    retriever.clear_all_collections()
                else:
                    logger.info("📚 Continuing with existing collections")
                    elapsed_time = time.time() - start_time
                    logger.info(f"⏱️ Setup completed in {elapsed_time:.1f} seconds")
                    return True
            else:
                logger.info("📚 Collections compatible with current embedding model")
                elapsed_time = time.time() - start_time
                logger.info(f"⏱️ Setup completed in {elapsed_time:.1f} seconds")
                return True
        else:
            logger.info("📄 No existing collections found. Creating new ones...")
    except Exception as e:
        logger.error(f"❌ Collection loading failed: {e}")
        logger.info("🔄 Will create new collections...")
    
    # Find and load foundation dataset
    try:
        dataset_path = find_foundation_dataset()
        all_documents, dataset_analysis = load_foundation_dataset(dataset_path)
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        return False
    
    # Create hierarchical collections
    try:
        logger.info("🏗️ Creating hierarchical collections with medical embedding...")
        
        # Create collections using retriever (not processor)
        retriever.create_hierarchical_collections()
        
        # Process documents into tiers using processor
        organized_docs = processor.organize_by_reasoning_type(all_documents)
        
        # Add documents to tiers using retriever
        retriever.add_documents_to_tiers(organized_docs)
        
        # Get creation statistics
        stats = retriever.get_collection_stats()
        
        logger.info("✅ Hierarchical collections created successfully")
        logger.info("📊 Creation Statistics:")
        logger.info(f"   Tier 1: {stats['tier1']['count']:,} documents")
        logger.info(f"   Tier 2: {stats['tier2']['count']:,} documents")
        logger.info(f"   Tier 3: {stats['tier3']['count']:,} documents")
        logger.info(f"   Total: {stats['total']:,} documents")
        
    except Exception as e:
        logger.error(f"❌ Collection creation failed: {e}")
        return False
    
    # Final validation
    try:
        logger.info("🔍 Performing final system validation...")
        
        # Test retrieval
        test_query = "What are the symptoms of diabetes mellitus?"
        test_results = retriever.hierarchical_search(test_query, top_k=3)
        
        if test_results and len(test_results) > 0:
            logger.info(f"✅ Test retrieval successful: {len(test_results)} results")
        else:
            logger.warning("⚠️ Test retrieval returned no results")
        
        # Get final statistics
        stats = retriever.get_collection_stats()
        logger.info("📊 Final Statistics:")
        logger.info(f"   📚 Total documents: {stats['total']:,}")
        logger.info(f"   🏥 Medical embedding: {stats['medical_optimized']}")
        logger.info(f"   🧠 Model: {stats['embedding_model']}")
        logger.info(f"   🔒 Secure PyTorch: {requirements['torch_version']}")
        
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