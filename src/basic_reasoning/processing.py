"""
Enhanced Document Processing module for Basic Reasoning system.
Rewritten with improved semantic similarity and multi-factor tier assignment.

File: src/basic_reasoning/processing.py
"""

import re
import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from loguru import logger
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import time
import torch

from .config import Config


class HierarchicalDocumentProcessor:
    """Enhanced document processor with improved tier assignment and balanced distribution."""

    def __init__(self, config: Config):
        """Initialize document processor with improved medical and device optimizations."""
        self.config = config
        
        # Auto-detect environment and setup device optimization
        self.environment = self._detect_environment()
        self.device_info = self._setup_device_optimization()
        
        # Processing configurations
        processing_config = config.config["processing"]
        self.chunk_size = processing_config["chunk_size"]
        self.chunk_overlap = processing_config["chunk_overlap"]
        self.min_content_length = processing_config.get("min_content_length", 50)
        self.enable_medical_entity_recognition = processing_config.get("enable_medical_entity_recognition", True)
        self.preserve_medical_terminology = processing_config.get("preserve_medical_terminology", True)
        self.enable_tier_balancing = processing_config.get("enable_tier_balancing", False)  # Disabled forced balancing
        
        # Target tier distribution
        self.target_distribution = processing_config.get("target_tier_distribution", {
            "tier1": 0.30,
            "tier2": 0.40,
            "tier3": 0.30
        })
        
        # Enhanced content type classifiers
        self.content_classifiers = self._initialize_content_classifiers()
        
        # Improved tier prototypes (more distinct and non-overlapping)
        self.tier_prototypes = self._define_tier_prototypes()
        
        # Medical terminology patterns (lightweight fallback)
        self.medical_patterns = self._compile_medical_patterns()
        
        # Processing statistics
        self.processing_stats = {
            "total_processed": 0,
            "tier_assignments": {"tier1": 0, "tier2": 0, "tier3": 0},
            "failed_processing": 0,
            "medical_entities_found": 0,
            "semantic_tier_assignment": False,
            "device_used": self.device_info["device"],
            "batch_processing": self.device_info["supports_batching"],
            "classification_method": "multi_factor_semantic"
        }
        
        # Initialize embedding model for tier assignment
        self._tier_embedding_model = None
        self._tier_prototype_embeddings = None

    def _detect_environment(self) -> str:
        """Auto-detect the current environment following project patterns."""
        
        # Check for RunPod environment first (highest priority for GPU)
        if (os.path.exists("/workspace") or 
            "RUNPOD_POD_ID" in os.environ or 
            "RUNPOD_POD_HOSTNAME" in os.environ or
            "runpod" in os.environ.get("HOSTNAME", "").lower()):
            return "runpod_gpu"
        
        # Check for general CUDA availability
        elif torch.cuda.is_available():
            return "cuda_gpu"
        
        # Check for MPS (Apple Silicon) - only if not in RunPod and CUDA not available
        elif (sys.platform == "darwin" and 
              hasattr(torch.backends, 'mps') and 
              torch.backends.mps.is_available()):
            return "mps_local"
        
        # Default to CPU
        else:
            return "cpu_local"

    def _setup_device_optimization(self) -> Dict:
        """Setup device-specific optimizations following project patterns."""
        
        device_info = {
            "environment": self.environment,
            "device": "cpu",
            "batch_size": 4,
            "supports_batching": False,
            "mixed_precision": False,
            "memory_efficient": True
        }
        
        if self.environment in ["runpod_gpu", "cuda_gpu"]:
            # GPU optimizations
            device_info.update({
                "device": "cuda",
                "batch_size": 32 if self.environment == "runpod_gpu" else 16,
                "supports_batching": True,
                "mixed_precision": True,
                "memory_efficient": True
            })
            
            # Configure CUDA optimizations
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # GPU memory info
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                device_info["gpu_memory_gb"] = gpu_memory
                
                # Adjust batch size based on GPU memory
                if gpu_memory >= 20:  # RTX 4090 or better
                    device_info["batch_size"] = 64
                elif gpu_memory >= 12:  # RTX 3080/4070 tier
                    device_info["batch_size"] = 32
                else:  # Lower-end GPUs
                    device_info["batch_size"] = 16
                    
                logger.info(f"🎮 GPU setup: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
            
        elif self.environment == "mps_local":
            # Apple Silicon optimizations
            device_info.update({
                "device": "mps",
                "batch_size": 8,
                "supports_batching": True,
                "mixed_precision": False,  # MPS doesn't support mixed precision well
                "memory_efficient": True
            })
            logger.info("🍎 MPS (Apple Silicon) setup")
            
        else:
            # CPU fallback
            device_info.update({
                "device": "cpu",
                "batch_size": 4,
                "supports_batching": False,
                "mixed_precision": False,
                "memory_efficient": True
            })
            logger.info("💻 CPU setup")
        
        logger.info(f"🔧 Device optimization: {device_info['device']} (batch_size={device_info['batch_size']})")
        
        return device_info

    def _initialize_content_classifiers(self) -> Dict:
        """Initialize enhanced content type classifiers for better tier differentiation."""
        
        return {
            # Educational/Basic content indicators (Tier 1)
            "educational_indicators": {
                "definitive_markers": [
                    "definition", "is defined as", "refers to", "means that",
                    "basic concept", "fundamental principle", "introduction to",
                    "overview of", "what is", "basic anatomy", "normal physiology"
                ],
                "structural_markers": [
                    "chapter", "section", "textbook", "manual", "handbook",
                    "educational material", "learning objective", "basic knowledge"
                ],
                "language_style": [
                    "simple explanation", "easy to understand", "beginners",
                    "basic level", "foundation", "fundamentals"
                ]
            },
            
            # Clinical/Practical content indicators (Tier 2)  
            "clinical_indicators": {
                "diagnostic_markers": [
                    "patient presents", "clinical presentation", "signs and symptoms",
                    "diagnostic criteria", "differential diagnosis", "most likely diagnosis",
                    "workup includes", "examination reveals", "assessment shows"
                ],
                "treatment_markers": [
                    "treatment plan", "management approach", "therapeutic intervention",
                    "medication regimen", "surgical procedure", "clinical protocol",
                    "patient care", "nursing intervention", "follow-up"
                ],
                "practical_language": [
                    "clinical practice", "bedside manner", "practical approach",
                    "real-world application", "case-based", "hands-on"
                ]
            },
            
            # Research/Evidence content indicators (Tier 3)
            "research_indicators": {
                "study_markers": [
                    "randomized controlled trial", "systematic review", "meta-analysis",
                    "cohort study", "case-control study", "clinical trial",
                    "research methodology", "statistical analysis", "p-value"
                ],
                "evidence_markers": [
                    "evidence-based", "clinical evidence", "scientific evidence",
                    "guideline recommendation", "expert consensus", "best practice",
                    "level of evidence", "grade recommendation"
                ],
                "academic_language": [
                    "peer-reviewed", "journal publication", "research findings",
                    "study conclusion", "data analysis", "statistical significance"
                ]
            }
        }

    def _define_tier_prototypes(self) -> Dict:
        """Define improved tier prototypes with minimal overlap and distinct vocabularies."""
        
        return {
            1: """educational textbook foundation learning basic fundamental anatomy physiology 
                  structure function definition meaning concept principle introduction overview 
                  normal mechanism teaching material study guide manual handbook reference 
                  beginner student educational objective knowledge base terminology glossary""",
            
            2: """patient clinical diagnosis treatment management care intervention therapy 
                  procedure examination assessment workup presentation symptoms signs bedside 
                  practical application case scenario nursing medical practice protocol 
                  differential diagnostic therapeutic medication surgical intervention""",
            
            3: """research study trial evidence analysis statistical methodology systematic 
                  review meta-analysis guideline recommendation consensus peer-reviewed 
                  publication journal scientific investigation data results conclusion 
                  evidence-based clinical trial randomized controlled efficacy outcome"""
        }

    def _compile_medical_patterns(self) -> Dict[str, re.Pattern]:
        """Compile medical terminology patterns for fallback classification."""
        patterns = {
            # Educational/Basic patterns (Tier 1)
            "educational": re.compile(r'\b(?:textbook|manual|handbook|guide|tutorial|introduction|overview|basic|fundamental|definition|anatomy|physiology|structure|function)\b', re.IGNORECASE),
            "terminology": re.compile(r'\b(?:definition|defined|means|refers to|terminology|glossary|vocabulary|concept|principle)\b', re.IGNORECASE),
            
            # Clinical/Practical patterns (Tier 2)
            "clinical_practice": re.compile(r'\b(?:patient|clinical|diagnosis|treatment|management|therapy|care|intervention|procedure|bedside|nursing|medical practice)\b', re.IGNORECASE),
            "diagnostic": re.compile(r'\b(?:symptoms|signs|presentation|examination|assessment|workup|differential|diagnostic criteria|most likely)\b', re.IGNORECASE),
            
            # Research/Evidence patterns (Tier 3)
            "research_methodology": re.compile(r'\b(?:study|trial|research|investigation|methodology|analysis|systematic review|meta-analysis|randomized|controlled)\b', re.IGNORECASE),
            "evidence_based": re.compile(r'\b(?:evidence|guideline|recommendation|consensus|peer-reviewed|journal|publication|scientific|grade|level of evidence)\b', re.IGNORECASE)
        }
        return patterns

    def preprocess_documents(self, documents: List[Dict]) -> List[Dict]:
        """Preprocess documents with enhanced medical hierarchical organization using optimized processing."""
        
        logger.info(f"🔧 Enhanced preprocessing of {len(documents):,} documents for medical Q&A")
        logger.info(f"🎯 Using {self.device_info['device']} with batch_size={self.device_info['batch_size']}")
        logger.info(f"🎯 Target distribution: T1:{self.target_distribution['tier1']:.0%} T2:{self.target_distribution['tier2']:.0%} T3:{self.target_distribution['tier3']:.0%}")
        
        if not documents:
            return []
        
        processed_docs = []
        failed_count = 0
        
        # Determine processing strategy based on device capabilities
        if self.device_info["supports_batching"] and len(documents) > 100:
            # Use batch processing for GPU/MPS with large document sets
            logger.info("🚀 Using optimized batch processing with multi-factor tier assignment")
            processed_docs, failed_count = self._process_documents_batch_improved(documents)
        else:
            # Use individual processing for CPU or small document sets
            logger.info("📝 Using individual document processing with enhanced classification")
            processed_docs, failed_count = self._process_documents_individual_improved(documents)
        
        # Update processing statistics
        self.processing_stats.update({
            "total_processed": len(processed_docs),
            "medical_entities_found": sum(
                len(doc["metadata"].get("medical_entities", [])) for doc in processed_docs
            )
        })
        
        # Log comprehensive statistics
        self._log_processing_statistics(processed_docs, failed_count)
        
        return processed_docs

    def _process_documents_batch_improved(self, documents: List[Dict]) -> Tuple[List[Dict], int]:
        """Process documents using improved batch processing with better tier assignment."""
        
        processed_docs = []
        failed_count = 0
        batch_size = min(1000, len(documents) // 4) if len(documents) > 4000 else 1000
        
        # Initialize embedding model once for all batches
        self._initialize_tier_embedding_model()
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            batch_docs = documents[i:batch_end]
            
            logger.info(f"   📝 Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size} ({len(batch_docs)} documents)")
            
            try:
                # Process individual documents in batch
                batch_processed = []
                
                for doc_idx, doc in enumerate(batch_docs):
                    try:
                        processed_doc = self._process_single_document(doc, i + doc_idx)
                        if processed_doc:
                            batch_processed.append(processed_doc)
                        else:
                            failed_count += 1
                    except Exception as e:
                        logger.debug(f"❌ Failed to process document {i + doc_idx}: {e}")
                        failed_count += 1
                
                # Assign tiers using multi-factor approach
                if batch_processed:
                    tier_assignments = self._assign_hierarchical_tier_multi_factor_batch(batch_processed)
                    
                    # Apply tier assignments
                    for doc, tier in zip(batch_processed, tier_assignments):
                        doc["metadata"]["tier"] = tier
                        self.processing_stats["tier_assignments"][f"tier{tier}"] += 1
                    
                    processed_docs.extend(batch_processed)
                
            except Exception as e:
                logger.error(f"❌ Batch processing failed: {e}")
                failed_count += len(batch_docs)
            
            # Log progress and clear cache periodically
            if (i // batch_size + 1) % 5 == 0 or batch_end == len(documents):
                logger.info(f"   📊 Processed {len(processed_docs):,}/{len(documents):,} documents")
                
                # Clear GPU cache if using CUDA
                if self.device_info["device"] == "cuda":
                    torch.cuda.empty_cache()
        
        return processed_docs, failed_count

    def _process_documents_individual_improved(self, documents: List[Dict]) -> Tuple[List[Dict], int]:
        """Process documents individually with enhanced classification (CPU or small batches)."""
        
        processed_docs = []
        failed_count = 0
        
        # Initialize embedding model once
        self._initialize_tier_embedding_model()
        
        for i, doc in enumerate(documents):
            try:
                processed_doc = self._process_single_document(doc, i)
                if processed_doc:
                    # Assign tier using multi-factor approach
                    text = processed_doc.get("text", "")
                    metadata = processed_doc.get("metadata", {})
                    
                    tier = self._assign_hierarchical_tier_multi_factor_individual(text, metadata)
                    
                    processed_doc["metadata"]["tier"] = tier
                    self.processing_stats["tier_assignments"][f"tier{tier}"] += 1
                    
                    processed_docs.append(processed_doc)
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process document {i}: {str(e)}")
                failed_count += 1
                self.processing_stats["failed_processing"] += 1
                continue
            
            # Progress logging
            if (i + 1) % 1000 == 0:
                logger.info(f"   📝 Processed {i + 1:,}/{len(documents):,} documents")
        
        return processed_docs, failed_count

    def _process_single_document(self, doc: Dict, doc_index: int) -> Optional[Dict]:
        """Process a single document with medical enhancement."""
        if not isinstance(doc, dict):
            return None
        
        # Extract text content
        text = self._extract_text_content(doc)
        if not text or len(text.strip()) < self.min_content_length:
            return None
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Extract metadata
        metadata = doc.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        
        # Enhance metadata with medical information
        metadata = self._enhance_metadata(metadata, text, doc_index)
        
        # Extract medical entities if enabled
        if self.enable_medical_entity_recognition:
            medical_entities = self._extract_medical_entities(text)
            metadata["medical_entities"] = medical_entities
            if medical_entities:
                self.processing_stats["medical_entities_found"] += 1
        
        # Create processed document
        processed_doc = {
            "text": text,
            "metadata": metadata
        }
        
        return processed_doc

    def _extract_text_content(self, doc: Dict) -> str:
        """Extract text content from document with multiple fallbacks."""
        # Try different fields that might contain text
        for field in ["text", "content", "body", "abstract", "summary", "description"]:
            if field in doc and doc[field]:
                return str(doc[field])
        
        # If no text found in common fields, try to extract from metadata
        metadata = doc.get("metadata", {})
        if isinstance(metadata, dict):
            for field in ["title", "abstract", "content"]:
                if field in metadata and metadata[field]:
                    return str(metadata[field])
        
        return ""

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for medical processing."""
        if not text:
            return ""
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Preserve medical terminology if enabled
        if self.preserve_medical_terminology:
            # Don't split common medical compound terms
            medical_compounds = [
                "beta-blocker", "ACE inhibitor", "T-cell", "B-cell",
                "X-ray", "CT scan", "MRI scan", "PCR test"
            ]
            # These would be preserved as-is
        
        return text

    def _enhance_metadata(self, metadata: Dict, text: str, doc_index: int) -> Dict:
        """Enhance metadata with medical information and processing details."""
        enhanced = metadata.copy()
        
        # Add processing information
        enhanced.update({
            "doc_index": doc_index,
            "text_length": len(text),
            "processing_timestamp": time.time(),
            "medical_content_score": self._calculate_medical_content_score(text)
        })
        
        # Extract medical specialty if not present
        if "medical_specialty" not in enhanced:
            enhanced["medical_specialty"] = self._detect_medical_specialty(text)
        
        # Determine content type
        enhanced["content_type"] = self._classify_content_type(text, metadata)
        
        return enhanced

    def _calculate_medical_content_score(self, text: str) -> float:
        """Calculate a score indicating how medical the content is."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        medical_terms = 0
        total_terms = len(text.split())
        
        # Count medical terminology
        for pattern in self.medical_patterns.values():
            matches = pattern.findall(text_lower)
            medical_terms += len(matches)
        
        # Calculate ratio
        if total_terms > 0:
            return min(1.0, medical_terms / total_terms * 10)  # Scale appropriately
        return 0.0

    def _detect_medical_specialty(self, text: str) -> str:
        """Detect medical specialty from text content."""
        text_lower = text.lower()
        
        specialty_keywords = {
            "cardiology": ["heart", "cardiac", "cardiovascular", "echocardiogram", "ekg", "coronary"],
            "neurology": ["brain", "neurological", "seizure", "stroke", "neuronal", "cognitive"],
            "oncology": ["cancer", "tumor", "malignant", "chemotherapy", "radiation", "metastasis"],
            "pediatrics": ["children", "pediatric", "infant", "newborn", "adolescent"],
            "psychiatry": ["mental health", "depression", "anxiety", "psychiatric", "therapy"],
            "surgery": ["surgical", "operation", "incision", "suture", "anesthesia"],
            "emergency": ["emergency", "trauma", "acute", "critical", "urgent"],
            "internal": ["internal medicine", "general medicine", "primary care"]
        }
        
        for specialty, keywords in specialty_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return specialty
        
        return "general"

    def _classify_content_type(self, text: str, metadata: Dict) -> str:
        """Classify the type of medical content."""
        text_lower = text.lower()
        
        # Check for specific content types
        if any(term in text_lower for term in ["definition", "what is", "refers to"]):
            return "definition"
        elif any(term in text_lower for term in ["study", "trial", "research"]):
            return "research"
        elif any(term in text_lower for term in ["guideline", "protocol", "recommendation"]):
            return "guideline"
        elif any(term in text_lower for term in ["case", "patient", "clinical"]):
            return "clinical"
        else:
            return "general"

    def _extract_medical_entities(self, text: str) -> List[str]:
        """Extract medical entities using pattern matching (lightweight approach)."""
        entities = []
        
        # Simple pattern-based entity extraction
        # This is a lightweight fallback - in production, use proper NER
        medical_entity_patterns = {
            "condition": re.compile(r'\b(?:syndrome|disease|disorder|infection|cancer|diabetes|hypertension|pneumonia)\b', re.IGNORECASE),
            "medication": re.compile(r'\b(?:aspirin|metformin|insulin|antibiotics|ibuprofen|acetaminophen)\b', re.IGNORECASE),
            "anatomy": re.compile(r'\b(?:heart|lung|liver|kidney|brain|stomach|muscle|bone)\b', re.IGNORECASE)
        }
        
        for entity_type, pattern in medical_entity_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                entities.append(f"{entity_type}:{match.lower()}")
        
        return list(set(entities))  # Remove duplicates

    def _initialize_tier_embedding_model(self):
        """Initialize embedding model for tier assignment if not already loaded."""
        if self._tier_embedding_model is not None:
            return
        
        try:
            # Try to use the same embedding model as the main system
            embedding_config = self.config.get_embedding_config()
            model_name = embedding_config.get("name", "sentence-transformers/all-MiniLM-L6-v2")
            
            # For simplicity, use sentence transformers if available
            if "sentence-transformers" in model_name or "all-MiniLM" in model_name:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._tier_embedding_model = SentenceTransformer(model_name)
                    self.processing_stats["semantic_tier_assignment"] = True
                    logger.info(f"📊 Initialized semantic tier assignment with {model_name}")
                except ImportError:
                    logger.warning("📊 SentenceTransformers not available, using pattern-based assignment")
                    self._tier_embedding_model = None
            else:
                logger.info("📊 Using pattern-based tier assignment (medical model detected)")
                self._tier_embedding_model = None
                
        except Exception as e:
            logger.warning(f"📊 Failed to initialize embedding model for tier assignment: {e}")
            self._tier_embedding_model = None

    def _get_batch_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Get embeddings for a batch of texts."""
        if not self._tier_embedding_model or not texts:
            return None
        
        try:
            embeddings = self._tier_embedding_model.encode(
                texts,
                batch_size=min(32, len(texts)),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            logger.warning(f"📊 Batch embedding failed: {e}")
            return None

    def _assign_hierarchical_tier_multi_factor_batch(self, documents: List[Dict]) -> List[int]:
        """Assign tiers using multi-factor analysis for better balance."""
        
        try:
            # Extract texts and metadata
            texts = []
            metadatas = []
            valid_indices = []
            
            for i, doc in enumerate(documents):
                text = doc.get("text", "")
                metadata = doc.get("metadata", {})
                
                # Prepare text for processing
                text_sample = text[:512] if len(text) > 512 else text
                text_sample = text_sample.strip()
                
                if len(text_sample) >= 10:
                    texts.append(text_sample)
                    metadatas.append(metadata)
                    valid_indices.append(i)
            
            if not texts:
                return [2] * len(documents)  # Default to clinical
            
            logger.debug(f"📊 Multi-factor batch processing {len(texts)} documents...")
            start_time = time.time()
            
            # Get multi-factor tier assignments
            tier_assignments = self._process_multi_factor_tier_assignment_batch(texts, metadatas)
            
            # Create result array with fallbacks for invalid documents
            results = []
            valid_idx = 0
            
            for i, doc in enumerate(documents):
                if i in valid_indices:
                    results.append(tier_assignments[valid_idx])
                    valid_idx += 1
                else:
                    # Use intelligent fallback for short texts
                    results.append(self._assign_tier_intelligent_fallback(doc.get("text", ""), doc.get("metadata", {})))
            
            elapsed = time.time() - start_time
            logger.debug(f"📊 Multi-factor batch assignment completed in {elapsed:.2f}s ({len(texts)/elapsed:.1f} docs/sec)")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Multi-factor batch assignment failed: {e}")
            # Fallback to intelligent individual processing
            return [self._assign_tier_intelligent_fallback(doc.get("text", ""), doc.get("metadata", {})) for doc in documents]

    def _process_multi_factor_tier_assignment_batch(self, texts: List[str], metadatas: List[Dict]) -> List[int]:
        """Process tier assignment using multiple factors for better classification."""
        
        try:
            # Factor 1: Semantic similarity scores
            semantic_scores = self._get_semantic_similarity_scores_batch(texts)
            
            # Factor 2: Content type classification scores
            content_scores = self._get_content_type_scores_batch(texts, metadatas)
            
            # Factor 3: Source and metadata scores
            metadata_scores = self._get_metadata_scores_batch(metadatas)
            
            # Factor 4: Length and structure scores
            structure_scores = self._get_structure_scores_batch(texts)
            
            # Combine all factors with adaptive weights
            tier_assignments = []
            
            for i, (text, metadata) in enumerate(zip(texts, metadatas)):
                # Weighted combination of all factors
                combined_scores = {1: 0.0, 2: 0.0, 3: 0.0}
                
                # Weight factors based on content type detection
                semantic_weight = 0.4
                content_weight = 0.3
                metadata_weight = 0.2
                structure_weight = 0.1
                
                # Adjust weights based on content characteristics
                if self._has_strong_metadata_signals(metadata):
                    metadata_weight = 0.35
                    semantic_weight = 0.3
                
                if len(text) < 200:  # Short content - rely more on structure and metadata
                    structure_weight = 0.2
                    metadata_weight = 0.3
                    semantic_weight = 0.25
                    content_weight = 0.25
                
                # Combine scores
                for tier in [1, 2, 3]:
                    if semantic_scores and i < len(semantic_scores):
                        combined_scores[tier] += semantic_weight * semantic_scores[i].get(tier, 0.0)
                    
                    if content_scores and i < len(content_scores):
                        combined_scores[tier] += content_weight * content_scores[i].get(tier, 0.0)
                    
                    if metadata_scores and i < len(metadata_scores):
                        combined_scores[tier] += metadata_weight * metadata_scores[i].get(tier, 0.0)
                    
                    if structure_scores and i < len(structure_scores):
                        combined_scores[tier] += structure_weight * structure_scores[i].get(tier, 0.0)
                
                # Apply diversity promotion (reduce over-assignment to Tier 2)
                combined_scores = self._apply_diversity_promotion(combined_scores, text, metadata)
                
                # Assign tier with highest combined score
                assigned_tier = max(combined_scores, key=combined_scores.get)
                tier_assignments.append(assigned_tier)
                
                # Debug logging for first few documents
                if i < 3:
                    logger.debug(f"📊 Doc {i}: Combined scores T1:{combined_scores[1]:.3f} T2:{combined_scores[2]:.3f} T3:{combined_scores[3]:.3f} → Tier {assigned_tier}")
            
            return tier_assignments
            
        except Exception as e:
            logger.error(f"❌ Multi-factor processing failed: {e}")
            # Fallback to intelligent individual processing
            return [self._assign_tier_intelligent_fallback(text, metadata) for text, metadata in zip(texts, metadatas)]

    def _get_semantic_similarity_scores_batch(self, texts: List[str]) -> Optional[List[Dict[int, float]]]:
        """Get semantic similarity scores for batch of texts."""
        
        if not self._tier_embedding_model:
            return None
        
        try:
            # Get embeddings for all texts in batch
            text_embeddings = self._get_batch_embeddings(texts)
            
            if text_embeddings is None:
                return None
            
            # Get embeddings for tier prototypes (cache these)
            if self._tier_prototype_embeddings is None:
                prototype_texts = list(self.tier_prototypes.values())
                self._tier_prototype_embeddings = self._get_batch_embeddings(prototype_texts)
                
                if self._tier_prototype_embeddings is None:
                    return None
            
            # Calculate similarities in batch
            similarities_matrix = np.dot(text_embeddings, self._tier_prototype_embeddings.T)
            
            # Convert to list of dictionaries
            scores_list = []
            for i in range(len(texts)):
                scores = {
                    1: float(similarities_matrix[i, 0]),
                    2: float(similarities_matrix[i, 1]),
                    3: float(similarities_matrix[i, 2])
                }
                scores_list.append(scores)
            
            return scores_list
            
        except Exception as e:
            logger.warning(f"📊 Semantic similarity scoring failed: {e}")
            return None

    def _get_content_type_scores_batch(self, texts: List[str], metadatas: List[Dict]) -> List[Dict[int, float]]:
        """Get content type classification scores for batch of texts."""
        
        scores_list = []
        
        for text, metadata in zip(texts, metadatas):
            text_lower = text.lower()
            scores = {1: 0.0, 2: 0.0, 3: 0.0}
            
            # Tier 1: Educational content scoring
            tier1_score = 0.0
            for marker_type, markers in self.content_classifiers["educational_indicators"].items():
                for marker in markers:
                    if marker in text_lower:
                        tier1_score += 1.0 if marker_type == "definitive_markers" else 0.5
            
            # Tier 2: Clinical content scoring
            tier2_score = 0.0
            for marker_type, markers in self.content_classifiers["clinical_indicators"].items():
                for marker in markers:
                    if marker in text_lower:
                        tier2_score += 1.0 if marker_type == "diagnostic_markers" else 0.7
            
            # Tier 3: Research content scoring
            tier3_score = 0.0
            for marker_type, markers in self.content_classifiers["research_indicators"].items():
                for marker in markers:
                    if marker in text_lower:
                        tier3_score += 1.0 if marker_type == "study_markers" else 0.8
            
            # Normalize scores
            max_score = max(tier1_score, tier2_score, tier3_score)
            if max_score > 0:
                scores[1] = tier1_score / max_score
                scores[2] = tier2_score / max_score
                scores[3] = tier3_score / max_score
            else:
                # No strong indicators, use neutral distribution
                scores = {1: 0.33, 2: 0.34, 3: 0.33}
            
            scores_list.append(scores)
        
        return scores_list

    def _get_metadata_scores_batch(self, metadatas: List[Dict]) -> List[Dict[int, float]]:
        """Get metadata-based scores for batch of documents."""
        
        scores_list = []
        
        for metadata in metadatas:
            scores = {1: 0.0, 2: 0.0, 3: 0.0}
            
            # Source-based scoring
            source = metadata.get("source", "").lower()
            source_type = metadata.get("source_type", "").lower()
            
            # Tier 1: Educational sources
            if any(edu in source for edu in ["textbook", "manual", "handbook", "educational", "tutorial"]):
                scores[1] += 0.8
            if source_type in ["educational", "textbook"]:
                scores[1] += 0.6
            
            # Tier 3: Research sources
            if any(research in source for research in ["pubmed", "journal", "research", "study", "trial"]):
                scores[3] += 0.8
            if source_type in ["research", "high_quality", "guideline"]:
                scores[3] += 0.6
            
            # Tier 2: Clinical sources (default)
            if any(clinical in source for clinical in ["clinical", "hospital", "practice", "care"]):
                scores[2] += 0.7
            if source_type in ["clinical", "practice"]:
                scores[2] += 0.5
            
            # Evidence level consideration
            evidence_level = metadata.get("evidence_level", "").lower()
            if evidence_level in ["high", "systematic"]:
                scores[3] += 0.5
            elif evidence_level in ["medium", "clinical"]:
                scores[2] += 0.3
            
            # Medical specialty boost for clinical content
            specialty = metadata.get("medical_specialty", "")
            if specialty:
                scores[2] += 0.3
            
            # Normalize scores
            total_score = sum(scores.values())
            if total_score > 0:
                for tier in scores:
                    scores[tier] = scores[tier] / total_score
            else:
                scores = {1: 0.33, 2: 0.34, 3: 0.33}
            
            scores_list.append(scores)
        
        return scores_list

    def _get_structure_scores_batch(self, texts: List[str]) -> List[Dict[int, float]]:
        """Get structural feature scores for batch of texts."""
        
        scores_list = []
        
        for text in texts:
            scores = {1: 0.0, 2: 0.0, 3: 0.0}
            
            content_length = len(text)
            
            # Length-based scoring
            if content_length < 300:  # Short content often basic definitions
                scores[1] += 0.7
            elif content_length > 1500:  # Long content often detailed research
                scores[3] += 0.6
            else:  # Medium content often clinical
                scores[2] += 0.5
            
            # Structure analysis
            sentence_count = len([s for s in text.split('.') if s.strip()])
            
            if sentence_count <= 3:  # Very short, likely definitions
                scores[1] += 0.5
            elif sentence_count > 10:  # Many sentences, likely detailed content
                scores[3] += 0.4
            
            # Question format (educational)
            if '?' in text and any(q in text.lower() for q in ["what is", "define", "explain"]):
                scores[1] += 0.6
            
            # Lists and bullet points (often educational or clinical protocols)
            if any(marker in text for marker in ['\n-', '\n•', '\n1.', '\n2.']):
                scores[1] += 0.3
                scores[2] += 0.3
            
            # Normalize scores
            total_score = sum(scores.values())
            if total_score > 0:
                for tier in scores:
                    scores[tier] = scores[tier] / total_score
            else:
                scores = {1: 0.33, 2: 0.34, 3: 0.33}
            
            scores_list.append(scores)
        
        return scores_list

    def _has_strong_metadata_signals(self, metadata: Dict) -> bool:
        """Check if metadata has strong signals for tier assignment."""
        strong_signals = [
            "source_type", "evidence_level", "medical_specialty",
            "content_type", "publication_type"
        ]
        return any(signal in metadata and metadata[signal] for signal in strong_signals)

    def _apply_diversity_promotion(self, scores: Dict[int, float], text: str, metadata: Dict) -> Dict[int, float]:
        """Apply diversity promotion to reduce over-assignment to clinical tier."""
        
        # Boost Tier 1 if content has educational characteristics
        if any(marker in text.lower() for marker in ["definition", "basic", "fundamental", "introduction", "what is"]):
            scores[1] *= 1.4
        
        # Boost Tier 3 if content has research characteristics
        if any(marker in text.lower() for marker in ["study", "research", "evidence", "trial", "guideline"]):
            scores[3] *= 1.3
        
        # Reduce Tier 2 dominance for ambiguous content
        if abs(scores[1] - scores[2]) < 0.2 and abs(scores[2] - scores[3]) < 0.2:
            scores[2] *= 0.8  # Slight reduction for ambiguous cases
        
        return scores

    def _assign_hierarchical_tier_multi_factor_individual(self, text: str, metadata: Dict) -> int:
        """Assign tier using multi-factor analysis for individual document."""
        
        try:
            # Get individual scores
            semantic_scores = self._get_semantic_similarity_scores_individual(text)
            content_scores = self._get_content_type_scores_individual(text, metadata)
            metadata_scores = self._get_metadata_scores_individual(metadata)
            structure_scores = self._get_structure_scores_individual(text)
            
            # Combine scores with weights
            combined_scores = {1: 0.0, 2: 0.0, 3: 0.0}
            
            # Adaptive weights
            semantic_weight = 0.4 if semantic_scores else 0.0
            content_weight = 0.3
            metadata_weight = 0.35 if self._has_strong_metadata_signals(metadata) else 0.2
            structure_weight = 0.1
            
            # Normalize weights
            total_weight = semantic_weight + content_weight + metadata_weight + structure_weight
            if total_weight > 0:
                semantic_weight /= total_weight
                content_weight /= total_weight
                metadata_weight /= total_weight
                structure_weight /= total_weight
            
            # Combine scores
            for tier in [1, 2, 3]:
                if semantic_scores:
                    combined_scores[tier] += semantic_weight * semantic_scores.get(tier, 0.0)
                combined_scores[tier] += content_weight * content_scores.get(tier, 0.0)
                combined_scores[tier] += metadata_weight * metadata_scores.get(tier, 0.0)
                combined_scores[tier] += structure_weight * structure_scores.get(tier, 0.0)
            
            # Apply diversity promotion
            combined_scores = self._apply_diversity_promotion(combined_scores, text, metadata)
            
            # Return tier with highest score
            return max(combined_scores, key=combined_scores.get)
            
        except Exception as e:
            logger.warning(f"📊 Multi-factor individual assignment failed: {e}")
            return self._assign_tier_intelligent_fallback(text, metadata)

    def _get_semantic_similarity_scores_individual(self, text: str) -> Optional[Dict[int, float]]:
        """Get semantic similarity scores for individual text."""
        if not self._tier_embedding_model:
            return None
        
        try:
            # Get text embedding
            text_embedding = self._get_batch_embeddings([text])
            if text_embedding is None:
                return None
            
            # Get prototype embeddings
            if self._tier_prototype_embeddings is None:
                prototype_texts = list(self.tier_prototypes.values())
                self._tier_prototype_embeddings = self._get_batch_embeddings(prototype_texts)
                if self._tier_prototype_embeddings is None:
                    return None
            
            # Calculate similarities
            similarities = np.dot(text_embedding[0], self._tier_prototype_embeddings.T)
            
            return {
                1: float(similarities[0]),
                2: float(similarities[1]),
                3: float(similarities[2])
            }
            
        except Exception as e:
            logger.warning(f"📊 Individual semantic scoring failed: {e}")
            return None

    def _get_content_type_scores_individual(self, text: str, metadata: Dict) -> Dict[int, float]:
        """Get content type scores for individual text."""
        return self._get_content_type_scores_batch([text], [metadata])[0]

    def _get_metadata_scores_individual(self, metadata: Dict) -> Dict[int, float]:
        """Get metadata scores for individual document."""
        return self._get_metadata_scores_batch([metadata])[0]

    def _get_structure_scores_individual(self, text: str) -> Dict[int, float]:
        """Get structure scores for individual text."""
        return self._get_structure_scores_batch([text])[0]

    def _assign_tier_intelligent_fallback(self, text: str, metadata: Dict) -> int:
        """Intelligent fallback tier assignment using pattern matching."""
        
        if not text:
            return 2  # Default to clinical
        
        text_lower = text.lower()
        
        # Check for strong Tier 1 indicators (educational/basic)
        tier1_indicators = [
            "definition", "defined as", "refers to", "what is", "basic",
            "fundamental", "introduction", "overview", "anatomy", "physiology"
        ]
        tier1_score = sum(1 for indicator in tier1_indicators if indicator in text_lower)
        
        # Check for strong Tier 3 indicators (research/evidence)
        tier3_indicators = [
            "study", "trial", "research", "evidence", "systematic review",
            "meta-analysis", "clinical trial", "randomized", "p-value"
        ]
        tier3_score = sum(1 for indicator in tier3_indicators if indicator in text_lower)
        
        # Check for Tier 2 indicators (clinical practice)
        tier2_indicators = [
            "patient", "clinical", "diagnosis", "treatment", "management",
            "symptoms", "therapeutic", "procedure", "examination"
        ]
        tier2_score = sum(1 for indicator in tier2_indicators if indicator in text_lower)
        
        # Consider source information
        source = metadata.get("source", "").lower()
        if "textbook" in source or "educational" in source:
            tier1_score += 2
        elif "pubmed" in source or "journal" in source:
            tier3_score += 2
        elif "clinical" in source:
            tier2_score += 1
        
        # Consider content length
        content_length = len(text)
        if content_length < 200:
            tier1_score += 1  # Short content often definitions
        elif content_length > 1000:
            tier3_score += 1  # Long content often research
        
        # Assign tier based on highest score
        scores = {1: tier1_score, 2: tier2_score, 3: tier3_score}
        assigned_tier = max(scores, key=scores.get)
        
        # Default to Tier 2 if all scores are equal
        if tier1_score == tier2_score == tier3_score:
            assigned_tier = 2
        
        return assigned_tier

    def _log_processing_statistics(self, processed_docs: List[Dict], failed_count: int):
        """Log comprehensive processing statistics."""
        
        total_input = self.processing_stats["total_processed"] + failed_count
        
        # Tier distribution
        tier_distribution = {"tier1": 0, "tier2": 0, "tier3": 0}
        for doc in processed_docs:
            tier = doc["metadata"].get("tier", 2)
            tier_distribution[f"tier{tier}"] += 1
        
        logger.info("📊 Processing completed:")
        logger.info(f"   📝 Total input documents: {total_input:,}")
        logger.info(f"   ✅ Successfully processed: {len(processed_docs):,}")
        logger.info(f"   ❌ Failed processing: {failed_count:,}")
        logger.info(f"   🏥 Medical entities found: {self.processing_stats['medical_entities_found']:,}")
        logger.info(f"   🧠 Semantic tier assignment: {self.processing_stats['semantic_tier_assignment']}")
        logger.info(f"   🎯 Device used: {self.processing_stats['device_used']}")
        
        logger.info("📊 Tier distribution:")
        total_processed = len(processed_docs)
        for tier, count in tier_distribution.items():
            percentage = (count / total_processed * 100) if total_processed > 0 else 0
            target_percentage = self.target_distribution.get(tier, 0) * 100
            logger.info(f"   {tier.upper()}: {count:,} documents ({percentage:.1f}%, target: {target_percentage:.1f}%)")

    def get_processing_statistics(self) -> Dict:
        """Get comprehensive processing statistics."""
        return {
            "processing_stats": self.processing_stats.copy(),
            "target_distribution": self.target_distribution,
            "device_info": self.device_info,
            "content_classifiers_count": len(self.content_classifiers),
            "medical_patterns_count": len(self.medical_patterns),
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "min_content_length": self.min_content_length,
                "enable_medical_entity_recognition": self.enable_medical_entity_recognition,
                "preserve_medical_terminology": self.preserve_medical_terminology,
                "enable_tier_balancing": self.enable_tier_balancing
            }
        }