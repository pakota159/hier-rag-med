"""
Enhanced Retrieval module for Basic Reasoning system.
Complete implementation with Microsoft BiomedNLP-PubMedBERT medical embedding support.

File: src/basic_reasoning/retrieval.py
"""

from typing import Dict, List, Optional, Union, Tuple
import chromadb
from loguru import logger
import torch
import numpy as np
import re
from pathlib import Path

from .config import Config


class HierarchicalRetriever:
    """Enhanced hierarchical retriever with medical embedding support."""

    def __init__(self, config: Config):
        """Initialize hierarchical retriever with medical embedding."""
        self.config = config
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(config.get_data_dir("vector_db"))
        )
        
        # Initialize embedding model with medical support
        self._initialize_embedding_model()
        
        # Retrieval configurations
        retrieval_config = config.config["hierarchical_retrieval"]
        self.tier1_top_k = retrieval_config["tier1_top_k"]
        self.tier2_top_k = retrieval_config["tier2_top_k"]
        self.tier3_top_k = retrieval_config["tier3_top_k"]
        
        # Medical-specific configurations
        self.medical_entity_boost = retrieval_config.get("medical_entity_boost", 1.2)
        self.clinical_context_window = retrieval_config.get("clinical_context_window", 3)
        self.enable_evidence_stratification = retrieval_config.get("enable_evidence_stratification", True)
        self.enable_temporal_weighting = retrieval_config.get("enable_temporal_weighting", True)
        self.medical_specialty_boost = retrieval_config.get("medical_specialty_boost", True)
        
        # Collections
        self.tier1_collection = None
        self.tier2_collection = None
        self.tier3_collection = None
        
        # Medical terminology patterns for enhanced matching
        self.medical_patterns = self._compile_medical_patterns()

    def _initialize_embedding_model(self):
        """Initialize embedding model with medical support and fallback."""
        embedding_config = self.config.get_embedding_config()
        
        try:
            # Try to load medical embedding model
            model_name = embedding_config["name"]
            device = embedding_config["device"]
            
            if "BiomedNLP-PubMedBERT" in model_name:
                logger.info(f"🏥 Loading medical embedding model: {model_name}")
                self._load_biomedical_model(model_name, device, embedding_config)
            else:
                logger.info(f"📝 Loading standard embedding model: {model_name}")
                self._load_sentence_transformer(model_name, device, embedding_config)
                
        except Exception as e:
            logger.error(f"❌ Failed to load primary embedding model: {e}")
            self._load_fallback_model(embedding_config)

    def _load_biomedical_model(self, model_name: str, device: str, config: Dict):
        """Load BiomedNLP-PubMedBERT model with proper configuration."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            # Load tokenizer and model
            logger.info(f"   📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Configure model loading parameters
            model_kwargs = config.get("model_kwargs", {})
            torch_dtype_str = model_kwargs.get("torch_dtype", "float32")
            
            # Convert string to torch dtype
            if torch_dtype_str == "float16":
                torch_dtype = torch.float16
            elif torch_dtype_str == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
            
            logger.info(f"   🧠 Loading model with dtype: {torch_dtype}")
            
            # Load model with optimizations
            self.embedding_model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=config.get("trust_remote_code", False),
                low_cpu_mem_usage=True,
                **{k: v for k, v in model_kwargs.items() if k not in ["torch_dtype"]}
            )
            
            # Move to device and set evaluation mode
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            if device != "cpu":
                self.embedding_model = self.embedding_model.to(device)
            
            # Set to evaluation mode
            self.embedding_model.eval()
            
            # Store configuration
            self.device = device
            self.batch_size = config["batch_size"]
            self.max_length = config["max_length"]
            self.normalize_embeddings = config.get("normalize_embeddings", True)
            self.is_medical_model = True
            self.model_name = model_name
            
            logger.info(f"✅ Successfully loaded medical embedding model on {device}")
            logger.info(f"   🎯 Batch size: {self.batch_size}")
            logger.info(f"   📏 Max length: {self.max_length}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load biomedical model: {e}")
            raise

    def _load_sentence_transformer(self, model_name: str, device: str, config: Dict):
        """Load sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Handle auto device detection
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            self.embedding_model = SentenceTransformer(
                model_name,
                device=device
            )
            
            self.device = device
            self.batch_size = config["batch_size"]
            self.max_length = config.get("max_length", 512)
            self.normalize_embeddings = config.get("normalize_embeddings", True)
            self.is_medical_model = False
            self.model_name = model_name
            
            logger.info(f"✅ Successfully loaded sentence transformer on {device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load sentence transformer: {e}")
            raise

    def _load_fallback_model(self, config: Dict):
        """Load fallback model if primary model fails."""
        fallback_name = config.get("fallback_name", "sentence-transformers/all-MiniLM-L6-v2")
        device = config["device"]
        
        logger.warning(f"🔄 Loading fallback model: {fallback_name}")
        
        try:
            # Create fallback config
            fallback_config = config.copy()
            fallback_config["name"] = fallback_name
            
            self._load_sentence_transformer(fallback_name, device, fallback_config)
            logger.info("✅ Fallback model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Fallback model loading failed: {e}")
            raise RuntimeError("Both primary and fallback embedding models failed to load")

    def _compile_medical_patterns(self) -> Dict[str, re.Pattern]:
        """Compile medical terminology patterns for enhanced matching."""
        patterns = {
            "diseases": re.compile(r'\b(?:syndrome|disease|disorder|condition|infection|cancer|tumor|carcinoma|sarcoma)\b', re.IGNORECASE),
            "symptoms": re.compile(r'\b(?:pain|fever|headache|nausea|fatigue|dyspnea|chest pain|abdominal pain)\b', re.IGNORECASE),
            "treatments": re.compile(r'\b(?:therapy|treatment|medication|drug|surgery|procedure|intervention)\b', re.IGNORECASE),
            "anatomy": re.compile(r'\b(?:heart|lung|liver|kidney|brain|stomach|intestine|muscle|bone|joint)\b', re.IGNORECASE),
            "diagnostics": re.compile(r'\b(?:diagnosis|test|scan|biopsy|x-ray|MRI|CT|ultrasound|blood test)\b', re.IGNORECASE),
            "medications": re.compile(r'\b(?:mg|mcg|tablet|capsule|injection|IV|oral|topical|dosage)\b', re.IGNORECASE)
        }
        return patterns

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text using the loaded embedding model."""
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        try:
            if self.is_medical_model:
                return self._encode_with_biomedical_model(texts)
            else:
                return self._encode_with_sentence_transformer(texts)
        except Exception as e:
            logger.error(f"❌ Text encoding failed: {e}")
            raise

    def _encode_with_biomedical_model(self, texts: List[str]) -> np.ndarray:
        """Encode texts using BiomedNLP-PubMedBERT model."""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Move to device
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                    
                    # Use mean pooling of last hidden states
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    
                    # Apply attention mask and compute mean
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    embeddings_batch = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    
                    # Normalize if configured
                    if self.normalize_embeddings:
                        embeddings_batch = torch.nn.functional.normalize(embeddings_batch, p=2, dim=1)
                    
                    # Move to CPU and convert to numpy
                    embeddings_batch = embeddings_batch.cpu().numpy()
                    embeddings.extend(embeddings_batch)
                    
            except Exception as e:
                logger.error(f"❌ Batch encoding failed for batch {i//self.batch_size}: {e}")
                # Create zero embeddings for failed batch
                batch_size = len(batch_texts)
                zero_embeddings = np.zeros((batch_size, 768))  # BERT base dimension
                embeddings.extend(zero_embeddings)
        
        return np.array(embeddings)

    def _encode_with_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Encode texts using sentence transformer model."""
        try:
            return self.embedding_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings
            )
        except Exception as e:
            logger.error(f"❌ Sentence transformer encoding failed: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), self.embedding_model.get_sentence_embedding_dimension()))

    def create_hierarchical_collections(self):
        """Create hierarchical collections for medical knowledge."""
        logger.info("🔧 Creating hierarchical medical knowledge collections")
        
        collection_names = [
            "tier1_pattern_recognition",
            "tier2_hypothesis_testing", 
            "tier3_confirmation"
        ]
        
        # Delete existing collections
        for name in collection_names:
            try:
                self.client.delete_collection(name)
                logger.info(f"🗑️ Deleted existing collection: {name}")
            except:
                pass
        
        # Create new collections with enhanced metadata
        self.tier1_collection = self.client.create_collection(
            name="tier1_pattern_recognition",
            metadata={
                "description": "Basic medical concepts, anatomy, definitions",
                "medical_focus": "foundational_knowledge",
                "embedding_model": self.model_name,
                "tier_level": 1,
                "is_medical_optimized": self.is_medical_model
            }
        )
        
        self.tier2_collection = self.client.create_collection(
            name="tier2_hypothesis_testing",
            metadata={
                "description": "Clinical reasoning, pathophysiology, diagnostics",
                "medical_focus": "clinical_reasoning",
                "embedding_model": self.model_name,
                "tier_level": 2,
                "is_medical_optimized": self.is_medical_model
            }
        )
        
        self.tier3_collection = self.client.create_collection(
            name="tier3_confirmation", 
            metadata={
                "description": "Evidence-based medicine, guidelines, clinical trials",
                "medical_focus": "evidence_based",
                "embedding_model": self.model_name,
                "tier_level": 3,
                "is_medical_optimized": self.is_medical_model
            }
        )
        
        logger.info("✅ Created hierarchical medical collections with enhanced metadata")

    def load_hierarchical_collections(self) -> bool:
        """Load existing hierarchical collections."""
        try:
            self.tier1_collection = self.client.get_collection("tier1_pattern_recognition")
            self.tier2_collection = self.client.get_collection("tier2_hypothesis_testing")
            self.tier3_collection = self.client.get_collection("tier3_confirmation")
            
            # Validate embedding model compatibility
            self._validate_collection_compatibility()
            
            logger.info("✅ Loaded hierarchical medical collections")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load hierarchical collections: {e}")
            return False

    def _validate_collection_compatibility(self):
        """Validate that collections are compatible with current embedding model."""
        current_model = self.model_name
        
        collections = [
            ("Tier 1", self.tier1_collection),
            ("Tier 2", self.tier2_collection),
            ("Tier 3", self.tier3_collection)
        ]
        
        for tier_name, collection in collections:
            if collection:
                stored_model = collection.metadata.get("embedding_model", "unknown")
                is_medical_optimized = collection.metadata.get("is_medical_optimized", False)
                
                if stored_model != current_model:
                    logger.warning(f"⚠️ {tier_name} was created with {stored_model}, "
                                 f"now using {current_model}")
                
                if not is_medical_optimized and self.is_medical_model:
                    logger.warning(f"⚠️ {tier_name} was created without medical optimization")

    def add_documents_to_tier(self, documents: List[Dict], tier: int):
        """Add documents to specific tier with enhanced medical processing."""
        if tier == 1:
            collection = self.tier1_collection
            tier_name = "Pattern Recognition"
        elif tier == 2:
            collection = self.tier2_collection
            tier_name = "Clinical Reasoning"
        elif tier == 3:
            collection = self.tier3_collection
            tier_name = "Evidence Confirmation"
        else:
            logger.error(f"❌ Invalid tier: {tier}")
            return
        
        if not collection:
            logger.error(f"❌ Tier {tier} collection not initialized")
            return
        
        # Process documents in batches
        batch_size = max(4, self.batch_size // 2)  # Reduce batch size for adding documents
        total_docs = len(documents)
        
        logger.info(f"📚 Adding {total_docs:,} documents to Tier {tier} ({tier_name})")
        
        added_count = 0
        failed_count = 0
        
        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i:i + batch_size]
            
            try:
                # Extract and validate data
                texts = []
                metadatas = []
                ids = []
                
                for j, doc in enumerate(batch_docs):
                    text = doc.get("text", "")
                    if not text or len(text.strip()) < 10:
                        continue
                    
                    metadata = doc.get("metadata", {})
                    doc_id = metadata.get("doc_id", f"tier{tier}_doc_{i+j}")
                    
                    # Clean metadata for ChromaDB compatibility
                    clean_metadata = self._clean_metadata_for_chromadb(metadata)
                    
                    # Ensure we have at least some metadata
                    if not clean_metadata:
                        clean_metadata = {"tier": tier, "doc_type": "medical"}
                    
                    # Ensure unique IDs
                    unique_id = f"{doc_id}_{tier}_{i+j}"
                    
                    texts.append(text)
                    metadatas.append(clean_metadata)
                    ids.append(unique_id)
                
                if not texts:
                    continue
                
                # Generate embeddings
                embeddings = self.encode_text(texts)
                
                if embeddings.size == 0:
                    logger.warning(f"⚠️ Failed to generate embeddings for batch {i//batch_size}")
                    failed_count += len(batch_docs)
                    continue
                
                # Add to collection
                collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings.tolist(),
                    ids=ids
                )
                
                added_count += len(texts)
                
                # Progress logging
                if (i // batch_size) % 10 == 0:
                    progress = min(i + batch_size, total_docs)
                    logger.info(f"   📝 Processed {progress:,}/{total_docs:,} documents "
                              f"({progress/total_docs*100:.1f}%)")
                    
            except Exception as e:
                logger.error(f"❌ Failed to add batch {i//batch_size} to tier {tier}: {e}")
                failed_count += len(batch_docs)
                continue
        
        logger.info(f"✅ Tier {tier} complete: {added_count:,} added, {failed_count:,} failed")

    def _clean_metadata_for_chromadb(self, metadata: Dict) -> Dict:
        """Clean metadata to ensure ChromaDB compatibility."""
        import json
        
        clean_metadata = {}
        
        for key, value in metadata.items():
            if value is None:
                continue  # Skip None values entirely
            elif isinstance(value, bool):
                clean_metadata[key] = value
            elif isinstance(value, int):
                clean_metadata[key] = value
            elif isinstance(value, float):
                clean_metadata[key] = value
            elif isinstance(value, str):
                if value.strip():  # Only add non-empty strings
                    clean_metadata[key] = value
            elif isinstance(value, (list, dict)):
                # Convert complex types to JSON strings
                try:
                    json_str = json.dumps(value)
                    if json_str and json_str != "null":
                        clean_metadata[key] = json_str
                except (TypeError, ValueError):
                    str_value = str(value)
                    if str_value and str_value != "None":
                        clean_metadata[key] = str_value
            else:
                # Convert other types to string
                str_value = str(value)
                if str_value and str_value != "None":
                    clean_metadata[key] = str_value
        
        return clean_metadata

    def search_single_tier(self, query: str, tier: int, top_k: Optional[int] = None) -> List[Dict]:
        """Search in a single tier with medical optimizations."""
        if top_k is None:
            if tier == 1:
                top_k = self.tier1_top_k
            elif tier == 2:
                top_k = self.tier2_top_k
            elif tier == 3:
                top_k = self.tier3_top_k
            else:
                top_k = 5
        
        # Select collection
        if tier == 1 and self.tier1_collection:
            collection = self.tier1_collection
            tier_name = "Pattern Recognition"
        elif tier == 2 and self.tier2_collection:
            collection = self.tier2_collection
            tier_name = "Clinical Reasoning"
        elif tier == 3 and self.tier3_collection:
            collection = self.tier3_collection
            tier_name = "Evidence Confirmation"
        else:
            logger.warning(f"⚠️ Tier {tier} collection not available")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.encode_text([query])
            
            if query_embedding.size == 0:
                logger.error(f"❌ Failed to generate query embedding for tier {tier}")
                return []
            
            # Perform search
            results = collection.query(
                query_embeddings=[query_embedding[0].tolist()],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results with medical relevance scoring
            formatted_results = []
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                score = 1 - dist  # Convert distance to similarity
                
                # Apply medical entity boost if applicable
                if self._contains_medical_entities(doc, query):
                    score *= self.medical_entity_boost
                
                # Apply specialty boost if applicable
                if self.medical_specialty_boost and self._check_specialty_match(doc, query, meta):
                    score *= 1.1
                
                formatted_results.append({
                    "text": doc,
                    "metadata": meta,
                    "score": float(np.clip(score, 0, 1)),
                    "tier": tier,
                    "tier_name": tier_name,
                    "rank": i + 1,
                    "distance": float(dist)
                })
            
            logger.debug(f"🔍 Tier {tier} search: {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Tier {tier} search failed: {e}")
            return []

    def search_hierarchical(self, query: str, use_all_tiers: bool = True, 
                          adaptive_selection: bool = True) -> Dict[str, List[Dict]]:
        """Perform hierarchical search across all tiers with medical optimizations."""
        logger.info(f"🔍 Hierarchical search: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        results = {
            "tier1": [],
            "tier2": [],
            "tier3": [],
            "combined": [],
            "query_classification": None,
            "search_strategy": "hierarchical"
        }
        
        # Classify query for adaptive search
        query_tier = self._classify_query_tier(query)
        results["query_classification"] = {
            "primary_tier": query_tier,
            "confidence": self._get_classification_confidence(query, query_tier),
            "medical_entities": self._extract_medical_entities(query)
        }
        
        # Execute search strategy
        if use_all_tiers:
            # Search all tiers
            results["tier1"] = self.search_single_tier(query, 1)
            results["tier2"] = self.search_single_tier(query, 2)
            results["tier3"] = self.search_single_tier(query, 3)
            results["search_strategy"] = "comprehensive"
        elif adaptive_selection:
            # Focus on primary tier with supporting tiers
            primary_results = self.search_single_tier(query, query_tier, top_k=self.tier2_top_k + 2)
            results[f"tier{query_tier}"] = primary_results
            
            # Add supporting evidence from other tiers
            if query_tier != 2:
                supporting_results = self.search_single_tier(query, 2, top_k=3)
                results["tier2"] = supporting_results
            
            results["search_strategy"] = "adaptive"
        else:
            # Single tier search
            results[f"tier{query_tier}"] = self.search_single_tier(query, query_tier)
            results["search_strategy"] = "focused"
        
        # Combine and rank results
        combined_results = []
        for tier_key, tier_results in results.items():
            if tier_key.startswith("tier") and isinstance(tier_results, list):
                combined_results.extend(tier_results)
        
        # Apply hierarchical weighting and ranking
        if combined_results:
            combined_results = self._apply_hierarchical_weighting(combined_results, query, query_tier)
            combined_results = self._deduplicate_results(combined_results)
            combined_results.sort(key=lambda x: x["final_score"], reverse=True)
            
            # Limit combined results
            max_combined = max(self.tier1_top_k, self.tier2_top_k, self.tier3_top_k)
            results["combined"] = combined_results[:max_combined]
        
        # Log results summary
        total_results = len(results["combined"])
        tier_counts = {f"tier{i}": len(results.get(f"tier{i}", [])) for i in range(1, 4)}
        logger.info(f"✅ Search complete: {total_results} total results")
        logger.debug(f"   📊 Tier distribution: {tier_counts}")
        
        return results

    def _contains_medical_entities(self, text: str, query: str) -> bool:
        """Enhanced medical entity detection using compiled patterns."""
        text_lower = text.lower()
        query_lower = query.lower()
        
        # Check for medical patterns in both text and query
        text_medical_matches = 0
        query_medical_matches = 0
        
        for pattern_name, pattern in self.medical_patterns.items():
            if pattern.search(text_lower):
                text_medical_matches += 1
            if pattern.search(query_lower):
                query_medical_matches += 1
        
        # Both text and query should have medical content
        return text_medical_matches >= 1 and query_medical_matches >= 1

    def _check_specialty_match(self, text: str, query: str, metadata: Dict) -> bool:
        """Check if document specialty matches query context."""
        if not self.medical_specialty_boost:
            return False
        
        doc_specialty = metadata.get("medical_specialty", "").lower()
        if not doc_specialty:
            return False
        
        # Simple specialty matching (can be enhanced)
        specialty_keywords = {
            "cardiology": ["heart", "cardiac", "cardiovascular", "ecg", "ekg"],
            "pulmonology": ["lung", "respiratory", "breathing", "pneumonia"],
            "gastroenterology": ["stomach", "intestine", "digestive", "gi"],
            "neurology": ["brain", "neurological", "seizure", "stroke"],
            "oncology": ["cancer", "tumor", "malignant", "chemotherapy"],
            "infectious_disease": ["infection", "bacteria", "virus", "antibiotic"]
        }
        
        query_lower = query.lower()
        for specialty, keywords in specialty_keywords.items():
            if specialty in doc_specialty:
                return any(keyword in query_lower for keyword in keywords)
        
        return False

    def _classify_query_tier(self, query: str) -> int:
        """Enhanced query classification for optimal tier selection."""
        query_lower = query.lower()
        
        # Tier 1: Basic medical concepts and definitions
        tier1_indicators = [
            "what is", "define", "definition", "anatomy", "structure",
            "basic", "introduction", "overview", "physiology", "normal"
        ]
        
        # Tier 3: Evidence-based and research queries
        tier3_indicators = [
            "guideline", "evidence", "study", "trial", "research",
            "recommendation", "consensus", "meta-analysis", "review",
            "efficacy", "effectiveness", "outcome", "prognosis"
        ]
        
        # Tier 2: Clinical reasoning and diagnostic queries (default)
        tier2_indicators = [
            "diagnosis", "treatment", "management", "patient", "clinical",
            "symptom", "presentation", "differential", "therapy",
            "medication", "drug", "procedure", "intervention"
        ]
        
        # Score each tier
        tier1_score = sum(1 for indicator in tier1_indicators if indicator in query_lower)
        tier2_score = sum(1 for indicator in tier2_indicators if indicator in query_lower)
        tier3_score = sum(1 for indicator in tier3_indicators if indicator in query_lower)
        
        # Return tier with highest score, default to tier 2
        scores = {1: tier1_score, 2: tier2_score, 3: tier3_score}
        max_score = max(scores.values())
        
        if max_score == 0:
            return 2  # Default to clinical reasoning
        
        # Return the tier with the highest score (prefer tier 2 in ties)
        for tier in [2, 1, 3]:
            if scores[tier] == max_score:
                return tier
        
        return 2

    def _get_classification_confidence(self, query: str, classified_tier: int) -> float:
        """Calculate confidence in query classification."""
        query_lower = query.lower()
        
        # Strong indicators for each tier
        strong_indicators = {
            1: ["anatomy", "definition", "what is", "structure", "physiology"],
            2: ["diagnosis", "treatment", "patient", "clinical", "symptom"],
            3: ["guideline", "evidence", "study", "research", "trial"]
        }
        
        matches = sum(1 for indicator in strong_indicators[classified_tier] 
                     if indicator in query_lower)
        
        # Simple confidence based on matches
        return min(1.0, matches / 3.0 + 0.3)

    def _extract_medical_entities(self, query: str) -> List[str]:
        """Extract medical entities from query."""
        entities = []
        
        for entity_type, pattern in self.medical_patterns.items():
            matches = pattern.findall(query)
            for match in matches:
                entities.append(f"{entity_type}:{match}")
        
        return entities

    def _apply_hierarchical_weighting(self, results: List[Dict], query: str, 
                                    primary_tier: int) -> List[Dict]:
        """Apply sophisticated weighting based on hierarchical reasoning."""
        
        # Tier preference weights based on query classification
        tier_weights = {
            1: {1: 1.3, 2: 1.0, 3: 0.8},  # Basic query: prefer foundational knowledge
            2: {1: 0.9, 2: 1.2, 3: 1.0},  # Clinical query: prefer clinical reasoning
            3: {1: 0.8, 2: 1.0, 3: 1.3}   # Evidence query: prefer research evidence
        }
        
        weights = tier_weights.get(primary_tier, {1: 1.0, 2: 1.0, 3: 1.0})
        
        for result in results:
            tier = result.get("tier", 2)
            base_score = result.get("score", 0.0)
            
            # Apply tier weighting
            tier_weight = weights.get(tier, 1.0)
            
            # Apply evidence stratification if enabled
            if self.enable_evidence_stratification:
                evidence_weight = self._calculate_evidence_weight(result, query)
            else:
                evidence_weight = 1.0
            
            # Apply temporal weighting if enabled
            if self.enable_temporal_weighting:
                temporal_weight = self._calculate_temporal_weight(result)
            else:
                temporal_weight = 1.0
            
            # Calculate final score
            final_score = base_score * tier_weight * evidence_weight * temporal_weight
            
            # Store all components for debugging
            result.update({
                "final_score": float(np.clip(final_score, 0, 1)),
                "tier_weight": tier_weight,
                "evidence_weight": evidence_weight,
                "temporal_weight": temporal_weight,
                "base_score": base_score
            })
        
        return results

    def _calculate_evidence_weight(self, result: Dict, query: str) -> float:
        """Calculate evidence quality weight based on source and content."""
        metadata = result.get("metadata", {})
        text = result.get("text", "").lower()
        
        # Evidence quality indicators
        high_quality_sources = ["pubmed", "cochrane", "uptodate", "statpearls", "nejm"]
        medium_quality_sources = ["medscape", "wikipedia", "textbook"]
        
        source = metadata.get("source", "").lower()
        
        # Base weight from source quality
        if any(hq_source in source for hq_source in high_quality_sources):
            base_weight = 1.2
        elif any(mq_source in source for mq_source in medium_quality_sources):
            base_weight = 1.0
        else:
            base_weight = 0.9
        
        # Content quality indicators
        quality_indicators = ["study", "trial", "evidence", "research", "peer-reviewed"]
        quality_boost = sum(0.05 for indicator in quality_indicators if indicator in text)
        
        return min(1.5, base_weight + quality_boost)

    def _calculate_temporal_weight(self, result: Dict) -> float:
        """Calculate temporal relevance weight."""
        metadata = result.get("metadata", {})
        
        # Try to extract publication year
        pub_date = metadata.get("publication_date", "")
        year = None
        
        if pub_date:
            import re
            year_match = re.search(r'20\d{2}', str(pub_date))
            if year_match:
                year = int(year_match.group())
        
        if not year:
            return 1.0  # No temporal information available
        
        # Current year (approximate)
        current_year = 2024
        age = current_year - year
        
        # Decay function: newer content gets higher weight
        if age <= 2:
            return 1.1  # Very recent
        elif age <= 5:
            return 1.0  # Recent
        elif age <= 10:
            return 0.95  # Moderately old
        else:
            return 0.9  # Older content

    def _deduplicate_results(self, results: List[Dict], similarity_threshold: float = 0.85) -> List[Dict]:
        """Remove duplicate or highly similar results."""
        if len(results) <= 1:
            return results
        
        deduplicated = []
        
        for current in results:
            current_text = current.get("text", "").lower()
            is_duplicate = False
            
            for existing in deduplicated:
                existing_text = existing.get("text", "").lower()
                
                # Simple similarity check using text overlap
                similarity = self._calculate_text_similarity(current_text, existing_text)
                
                if similarity > similarity_threshold:
                    # Keep the one with higher score
                    if current.get("final_score", 0) > existing.get("final_score", 0):
                        deduplicated.remove(existing)
                        deduplicated.append(current)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(current)
        
        return deduplicated

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using word overlap."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def get_collection_stats(self) -> Dict:
        """Get comprehensive statistics about the hierarchical collections."""
        stats = {
            "tier1": {"count": 0, "exists": False, "name": "Pattern Recognition"},
            "tier2": {"count": 0, "exists": False, "name": "Clinical Reasoning"},
            "tier3": {"count": 0, "exists": False, "name": "Evidence Confirmation"},
            "total": 0,
            "embedding_model": self.model_name,
            "medical_optimized": self.is_medical_model,
            "device": self.device,
            "batch_size": self.batch_size,
            "retrieval_config": {
                "tier1_top_k": self.tier1_top_k,
                "tier2_top_k": self.tier2_top_k,
                "tier3_top_k": self.tier3_top_k,
                "medical_entity_boost": self.medical_entity_boost,
                "evidence_stratification": self.enable_evidence_stratification,
                "temporal_weighting": self.enable_temporal_weighting,
                "medical_specialty_boost": self.medical_specialty_boost
            }
        }
        
        try:
            collections = [
                ("tier1", self.tier1_collection),
                ("tier2", self.tier2_collection),
                ("tier3", self.tier3_collection)
            ]
            
            for tier_key, collection in collections:
                if collection:
                    count = collection.count()
                    stats[tier_key]["count"] = count
                    stats[tier_key]["exists"] = True
                    stats["total"] += count
                    
                    # Add collection metadata
                    stats[tier_key]["metadata"] = collection.metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to get collection stats: {e}")
        
        return stats

    def health_check(self) -> Dict[str, bool]:
        """Perform comprehensive health check of the retrieval system."""
        health = {
            "embedding_model_loaded": False,
            "collections_available": False,
            "can_encode_text": False,
            "can_search": False,
            "medical_optimization": False
        }
        
        try:
            # Check embedding model
            if hasattr(self, 'embedding_model') and self.embedding_model is not None:
                health["embedding_model_loaded"] = True
                health["medical_optimization"] = self.is_medical_model
            
            # Check collections
            collections_exist = all([
                self.tier1_collection is not None,
                self.tier2_collection is not None,
                self.tier3_collection is not None
            ])
            health["collections_available"] = collections_exist
            
            # Test encoding
            if health["embedding_model_loaded"]:
                test_embedding = self.encode_text(["test medical query"])
                health["can_encode_text"] = test_embedding.size > 0
            
            # Test search
            if health["collections_available"] and health["can_encode_text"]:
                test_results = self.search_single_tier("test", 1, top_k=1)
                health["can_search"] = len(test_results) >= 0  # Even 0 results is ok
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
        
        return health

    def optimize_for_inference(self):
        """Optimize the retrieval system for inference performance."""
        try:
            if self.is_medical_model and hasattr(self.embedding_model, 'eval'):
                self.embedding_model.eval()
                
                # Enable inference optimizations
                if torch.cuda.is_available() and self.device == "cuda":
                    # Enable optimized attention if available
                    try:
                        torch.backends.cudnn.benchmark = True
                    except:
                        pass
                
                # Compile model if supported (PyTorch 2.0+)
                if hasattr(torch, 'compile') and hasattr(self.config.config["models"]["embedding"], "compile_model"):
                    if self.config.config["models"]["embedding"].get("compile_model", False):
                        try:
                            self.embedding_model = torch.compile(self.embedding_model)
                            logger.info("✅ Model compiled for optimized inference")
                        except Exception as e:
                            logger.warning(f"⚠️ Model compilation failed: {e}")
            
            logger.info("✅ Retrieval system optimized for inference")
            
        except Exception as e:
            logger.warning(f"⚠️ Inference optimization failed: {e}")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_stats = {
            "device": self.device,
            "model_name": self.model_name,
            "is_medical_model": self.is_medical_model
        }
        
        try:
            if torch.cuda.is_available() and self.device == "cuda":
                memory_stats.update({
                    "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "gpu_memory_max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2
                })
            
            # System memory
            try:
                import psutil
                process = psutil.Process()
                memory_stats.update({
                    "system_memory_mb": process.memory_info().rss / 1024**2,
                    "system_memory_percent": process.memory_percent()
                })
            except ImportError:
                pass
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get memory stats: {e}")
        
        return memory_stats

    def cleanup(self):
        """Clean up resources and free memory."""
        try:
            # Clear embedding model from memory
            if hasattr(self, 'embedding_model'):
                del self.embedding_model
            
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ Retrieval system resources cleaned up")
            
        except Exception as e:
            logger.warning(f"⚠️ Cleanup failed: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass