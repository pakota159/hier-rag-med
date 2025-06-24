"""
Enhanced Document Processing module for Basic Reasoning system.
Updated to support Microsoft BiomedNLP-PubMedBERT medical embedding.

File: src/basic_reasoning/processing.py
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from loguru import logger
import numpy as np
from collections import defaultdict

from .config import Config


class HierarchicalDocumentProcessor:
    """Enhanced document processor for medical hierarchical organization."""

    def __init__(self, config: Config):
        """Initialize document processor with medical optimizations."""
        self.config = config
        
        # Processing configurations
        processing_config = config.config["processing"]
        self.chunk_size = processing_config["chunk_size"]
        self.chunk_overlap = processing_config["chunk_overlap"]
        self.min_content_length = processing_config.get("min_content_length", 50)
        self.enable_medical_entity_recognition = processing_config.get("enable_medical_entity_recognition", True)
        self.preserve_medical_terminology = processing_config.get("preserve_medical_terminology", True)
        
        # Target tier distribution
        self.target_distribution = processing_config.get("target_tier_distribution", {
            "tier1": 0.30,
            "tier2": 0.40,
            "tier3": 0.30
        })
        
        # Medical terminology patterns
        self.medical_patterns = self._compile_medical_patterns()
        
        # Tier classification rules
        self.tier_rules = self._initialize_tier_rules()
        
        # Processing statistics
        self.processing_stats = {
            "total_processed": 0,
            "tier_assignments": {"tier1": 0, "tier2": 0, "tier3": 0},
            "failed_processing": 0,
            "medical_entities_found": 0
        }

    def _compile_medical_patterns(self) -> Dict[str, re.Pattern]:
        """Compile medical terminology patterns for classification."""
        patterns = {
            # Basic medical concepts (Tier 1)
            "anatomy": re.compile(r'\b(?:anatomy|anatomical|structure|organ|tissue|cell|bone|muscle|nerve|blood|heart|lung|liver|kidney|brain|stomach)\b', re.IGNORECASE),
            "physiology": re.compile(r'\b(?:physiology|physiological|function|normal|homeostasis|metabolism|circulation|respiration|digestion)\b', re.IGNORECASE),
            "basic_terms": re.compile(r'\b(?:definition|basic|fundamental|introduction|overview|concept|principle)\b', re.IGNORECASE),
            
            # Clinical reasoning (Tier 2)
            "clinical": re.compile(r'\b(?:clinical|diagnosis|diagnostic|differential|symptom|sign|presentation|examination|assessment|patient|case)\b', re.IGNORECASE),
            "pathology": re.compile(r'\b(?:pathology|pathological|disease|disorder|condition|syndrome|infection|inflammation|lesion|abnormal)\b', re.IGNORECASE),
            "treatment": re.compile(r'\b(?:treatment|therapy|therapeutic|management|medication|drug|surgery|procedure|intervention|care)\b', re.IGNORECASE),
            
            # Evidence-based medicine (Tier 3)
            "research": re.compile(r'\b(?:study|trial|research|evidence|meta-analysis|systematic review|randomized|controlled|cohort|case-control)\b', re.IGNORECASE),
            "guidelines": re.compile(r'\b(?:guideline|recommendation|consensus|protocol|standard|best practice|evidence-based|clinical practice)\b', re.IGNORECASE),
            "outcomes": re.compile(r'\b(?:outcome|prognosis|mortality|morbidity|survival|efficacy|effectiveness|safety|adverse|complication)\b', re.IGNORECASE),
            
            # Medical specialties
            "specialties": re.compile(r'\b(?:cardiology|pulmonology|gastroenterology|neurology|oncology|psychiatry|pediatrics|geriatrics|emergency|surgery)\b', re.IGNORECASE),
            
            # Medical procedures and tests
            "procedures": re.compile(r'\b(?:biopsy|endoscopy|catheterization|intubation|ventilation|dialysis|transfusion|transplant)\b', re.IGNORECASE),
            "diagnostics": re.compile(r'\b(?:x-ray|CT|MRI|ultrasound|echocardiogram|electrocardiogram|blood test|laboratory|imaging)\b', re.IGNORECASE)
        }
        return patterns

    def _initialize_tier_rules(self) -> Dict[int, Dict]:
        """Initialize tier classification rules."""
        return {
            1: {  # Basic medical knowledge
                "primary_patterns": ["anatomy", "physiology", "basic_terms"],
                "weight": 1.0,
                "description": "Foundational medical knowledge, anatomy, basic concepts"
            },
            2: {  # Clinical reasoning
                "primary_patterns": ["clinical", "pathology", "treatment", "specialties", "procedures", "diagnostics"],
                "weight": 1.0,
                "description": "Clinical reasoning, diagnosis, treatment, procedures"
            },
            3: {  # Evidence-based medicine
                "primary_patterns": ["research", "guidelines", "outcomes"],
                "weight": 1.0,
                "description": "Evidence-based medicine, research, guidelines"
            }
        }

    def preprocess_documents(self, documents: List[Dict]) -> List[Dict]:
        """Preprocess documents with enhanced medical hierarchical organization."""
        logger.info(f"🔧 Enhanced preprocessing of {len(documents)} documents for medical Q&A")
        
        if not documents:
            return []
        
        processed_docs = []
        failed_count = 0
        
        for i, doc in enumerate(documents):
            try:
                processed_doc = self._process_single_document(doc, i)
                if processed_doc:
                    processed_docs.append(processed_doc)
                    self.processing_stats["total_processed"] += 1
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
        
        # Apply tier balancing
        processed_docs = self._balance_tier_distribution(processed_docs)
        
        # Log final statistics
        self._log_processing_statistics(processed_docs, failed_count)
        
        return processed_docs

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
        
        # Assign hierarchical tier
        tier = self._assign_hierarchical_tier(text, metadata)
        metadata["tier"] = tier
        self.processing_stats["tier_assignments"][f"tier{tier}"] += 1
        
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
        # Try different possible text fields
        text_fields = ["text", "content", "body", "abstract", "summary", "description"]
        
        for field in text_fields:
            if field in doc:
                text = doc[field]
                if isinstance(text, str) and text.strip():
                    return text.strip()
        
        # Try nested content
        if "data" in doc and isinstance(doc["data"], dict):
            for field in text_fields:
                if field in doc["data"]:
                    text = doc["data"][field]
                    if isinstance(text, str) and text.strip():
                        return text.strip()
        
        return ""

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text while preserving medical terminology."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Preserve medical abbreviations (don't split on periods)
        if self.preserve_medical_terminology:
            # Common medical abbreviations
            medical_abbrevs = [
                r'\bDr\.', r'\bMD\.', r'\bRN\.', r'\bPhD\.', r'\bDDS\.', r'\bDVM\.',
                r'\be\.g\.', r'\bi\.e\.', r'\bvs\.', r'\betc\.', r'\bmg\.', r'\bml\.',
                r'\bIV\.', r'\bPO\.', r'\bBID\.', r'\bTID\.', r'\bQID\.'
            ]
            
            # Temporarily replace abbreviations
            protected_abbrevs = {}
            for i, abbrev_pattern in enumerate(medical_abbrevs):
                placeholder = f"__ABBREV_{i}__"
                matches = re.findall(abbrev_pattern, text, re.IGNORECASE)
                for match in matches:
                    protected_abbrevs[placeholder] = match
                    text = re.sub(abbrev_pattern, placeholder, text, flags=re.IGNORECASE)
            
            # Clean text
            text = text.strip()
            
            # Restore abbreviations
            for placeholder, original in protected_abbrevs.items():
                text = text.replace(placeholder, original)
        
        return text.strip()

    def _enhance_metadata(self, metadata: Dict, text: str, doc_index: int) -> Dict:
        """Enhance metadata with medical information."""
        enhanced_metadata = metadata.copy()
        
        # Ensure required fields
        if "doc_id" not in enhanced_metadata:
            enhanced_metadata["doc_id"] = f"doc_{doc_index}"
        
        # Extract medical specialty if not present
        if "medical_specialty" not in enhanced_metadata:
            specialty = self._classify_medical_specialty(text)
            if specialty:
                enhanced_metadata["medical_specialty"] = specialty
        
        # Extract source type if not present
        if "source_type" not in enhanced_metadata:
            source_type = self._classify_source_type(text, enhanced_metadata)
            enhanced_metadata["source_type"] = source_type
        
        # Add content length
        enhanced_metadata["content_length"] = len(text)
        
        # Add medical content score
        enhanced_metadata["medical_content_score"] = self._calculate_medical_content_score(text)
        
        return enhanced_metadata

    def _classify_medical_specialty(self, text: str) -> Optional[str]:
        """Classify medical specialty based on text content."""
        text_lower = text.lower()
        
        specialty_keywords = {
            "cardiology": ["heart", "cardiac", "cardiovascular", "coronary", "artery", "ecg", "ekg"],
            "pulmonology": ["lung", "respiratory", "pneumonia", "asthma", "copd", "breathing"],
            "gastroenterology": ["stomach", "intestine", "digestive", "gi", "liver", "hepatic"],
            "neurology": ["brain", "neurological", "seizure", "stroke", "nervous", "cognitive"],
            "oncology": ["cancer", "tumor", "malignant", "chemotherapy", "radiation", "oncology"],
            "infectious_disease": ["infection", "bacteria", "virus", "antibiotic", "sepsis"],
            "endocrinology": ["diabetes", "hormone", "thyroid", "endocrine", "insulin"],
            "psychiatry": ["mental", "psychiatric", "depression", "anxiety", "therapy"],
            "emergency_medicine": ["emergency", "trauma", "critical", "urgent", "triage"],
            "pediatrics": ["child", "pediatric", "infant", "adolescent", "developmental"]
        }
        
        specialty_scores = {}
        for specialty, keywords in specialty_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                specialty_scores[specialty] = score
        
        if specialty_scores:
            return max(specialty_scores, key=specialty_scores.get)
        
        return None

    def _classify_source_type(self, text: str, metadata: Dict) -> str:
        """Classify the type of medical source."""
        text_lower = text.lower()
        source = metadata.get("source", "").lower()
        
        # High-quality sources
        if any(hq in source for hq in ["pubmed", "nejm", "cochrane", "uptodate", "statpearls"]):
            return "high_quality"
        
        # Research sources
        if any(research in text_lower for research in ["study", "trial", "research", "meta-analysis"]):
            return "research"
        
        # Guidelines
        if any(guideline in text_lower for guideline in ["guideline", "recommendation", "consensus"]):
            return "guideline"
        
        # Textbook/educational
        if any(edu in text_lower for edu in ["textbook", "chapter", "education", "learning"]):
            return "educational"
        
        return "general"

    def _calculate_medical_content_score(self, text: str) -> float:
        """Calculate medical content relevance score."""
        if not text:
            return 0.0
        
        total_matches = 0
        total_patterns = len(self.medical_patterns)
        
        for pattern in self.medical_patterns.values():
            matches = len(pattern.findall(text))
            total_matches += min(matches, 5)  # Cap matches per pattern
        
        # Normalize score
        max_possible_score = total_patterns * 5
        score = total_matches / max_possible_score if max_possible_score > 0 else 0.0
        
        return min(1.0, score)

    def _extract_medical_entities(self, text: str) -> List[str]:
        """Extract medical entities from text."""
        entities = []
        
        for entity_type, pattern in self.medical_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                entities.append(f"{entity_type}:{match.lower()}")
        
        # Remove duplicates and limit
        entities = list(set(entities))
        return entities[:20]  # Limit to top 20 entities

    def _assign_hierarchical_tier(self, text: str, metadata: Dict) -> int:
        """Assign document to hierarchical tier based on content analysis."""
        text_lower = text.lower()
        
        # Calculate tier scores
        tier_scores = {}
        
        for tier, rules in self.tier_rules.items():
            score = 0
            
            # Primary pattern matching
            for pattern_name in rules["primary_patterns"]:
                if pattern_name in self.medical_patterns:
                    matches = len(self.medical_patterns[pattern_name].findall(text_lower))
                    score += matches * rules["weight"]
            
            tier_scores[tier] = score
        
        # Source-based adjustments
        source_type = metadata.get("source_type", "general")
        source = metadata.get("source", "").lower()
        
        # Tier 1 adjustments (basic knowledge)
        if any(basic in source for basic in ["anatomy", "physiology", "textbook", "basic"]):
            tier_scores[1] *= 1.3
        
        # Tier 3 adjustments (evidence-based)
        if source_type in ["research", "guideline"] or any(evidence in source for evidence in ["pubmed", "cochrane", "trial"]):
            tier_scores[3] *= 1.4
        
        # Content length considerations
        content_length = len(text)
        if content_length < 200:  # Short content often basic definitions
            tier_scores[1] *= 1.2
        elif content_length > 1000:  # Long content often detailed clinical/research
            tier_scores[3] *= 1.1
        
        # Medical specialty considerations
        specialty = metadata.get("medical_specialty")
        if specialty in ["emergency_medicine", "surgery"]:
            tier_scores[2] *= 1.2  # Clinical focus
        
        # Determine tier with highest score
        if not any(tier_scores.values()):
            return 2  # Default to tier 2 for clinical reasoning
        
        assigned_tier = max(tier_scores, key=tier_scores.get)
        
        return assigned_tier

    def _balance_tier_distribution(self, documents: List[Dict]) -> List[Dict]:
        """Balance tier distribution according to target ratios."""
        if not documents:
            return documents
        
        # Count current distribution
        current_distribution = {"tier1": 0, "tier2": 0, "tier3": 0}
        for doc in documents:
            tier = doc["metadata"].get("tier", 2)
            current_distribution[f"tier{tier}"] += 1
        
        total_docs = len(documents)
        target_counts = {
            tier: int(total_docs * ratio) 
            for tier, ratio in self.target_distribution.items()
        }
        
        # Calculate needed adjustments
        adjustments_needed = {}
        for tier, target_count in target_counts.items():
            current_count = current_distribution[tier]
            adjustments_needed[tier] = target_count - current_count
        
        # Apply adjustments if significant imbalance
        max_imbalance = max(abs(adj) for adj in adjustments_needed.values())
        if max_imbalance > total_docs * 0.1:  # Only if >10% imbalance
            documents = self._redistribute_tiers(documents, adjustments_needed)
        
        return documents

    def _redistribute_tiers(self, documents: List[Dict], adjustments: Dict) -> List[Dict]:
        """Redistribute documents across tiers to achieve target distribution."""
        # Group documents by current tier
        tier_groups = {"tier1": [], "tier2": [], "tier3": []}
        
        for doc in documents:
            tier = doc["metadata"].get("tier", 2)
            tier_groups[f"tier{tier}"].append(doc)
        
        # Sort each tier by medical content score (candidates for reassignment)
        for tier in tier_groups:
            tier_groups[tier].sort(
                key=lambda x: x["metadata"].get("medical_content_score", 0.5)
            )
        
        # Perform reassignments
        for tier, adjustment in adjustments.items():
            if adjustment > 0:  # Need more documents in this tier
                # Take from over-represented tiers
                for other_tier in tier_groups:
                    if other_tier != tier and len(tier_groups[other_tier]) > 0:
                        # Move documents with appropriate content
                        target_tier_num = int(tier[-1])
                        moved = 0
                        
                        for doc in tier_groups[other_tier][:]:
                            if moved >= adjustment:
                                break
                            
                            # Check if document could reasonably belong to target tier
                            score = self._calculate_tier_suitability(doc, target_tier_num)
                            if score > 0.3:  # Reasonable fit
                                doc["metadata"]["tier"] = target_tier_num
                                tier_groups[other_tier].remove(doc)
                                tier_groups[tier].append(doc)
                                moved += 1
        
        # Reconstruct document list
        redistributed_docs = []
        for tier_docs in tier_groups.values():
            redistributed_docs.extend(tier_docs)
        
        return redistributed_docs

    def _calculate_tier_suitability(self, doc: Dict, target_tier: int) -> float:
        """Calculate how suitable a document is for a target tier."""
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        
        # Use the same scoring logic as tier assignment
        text_lower = text.lower()
        rules = self.tier_rules.get(target_tier, {})
        
        score = 0
        for pattern_name in rules.get("primary_patterns", []):
            if pattern_name in self.medical_patterns:
                matches = len(self.medical_patterns[pattern_name].findall(text_lower))
                score += matches
        
        # Normalize by text length
        text_length = len(text)
        if text_length > 0:
            score = score / (text_length / 100)  # Per 100 characters
        
        return min(1.0, score)

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
            "tier_rules": self.tier_rules,
            "target_distribution": self.target_distribution,
            "medical_patterns_count": len(self.medical_patterns),
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "min_content_length": self.min_content_length,
                "enable_medical_entity_recognition": self.enable_medical_entity_recognition,
                "preserve_medical_terminology": self.preserve_medical_terminology
            }
        }