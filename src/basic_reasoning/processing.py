"""
Enhanced Hierarchical Document Processing for Medical Q&A
File: src/basic_reasoning/processing.py

Implements intelligent tier assignment based on medical content analysis
and reasoning types for optimal multiple choice question answering.
"""

from typing import Dict, List
import json
import re
from pathlib import Path
from loguru import logger


class HierarchicalDocumentProcessor:
    """Enhanced document processor for hierarchical medical reasoning."""

    def __init__(self, config: Dict):
        """Initialize processor with enhanced medical content analysis."""
        self.config = config
        
        # Enhanced medical content classifiers
        self.tier1_patterns = {
            # Basic medical knowledge patterns
            "definitions": [
                "definition", "defined as", "refers to", "is a", "means",
                "terminology", "called", "known as", "term for"
            ],
            "anatomy": [
                "anatomy", "structure", "located", "consists of", "composed of",
                "organ", "tissue", "cell", "bone", "muscle", "nerve"
            ],
            "physiology": [
                "function", "works by", "process", "mechanism", "physiology",
                "normal", "homeostasis", "regulation", "metabolism"
            ],
            "basic_concepts": [
                "classification", "types of", "categories", "forms of",
                "variants", "subtypes", "kinds of", "classes"
            ],
            "symptoms_signs": [
                "symptom", "sign", "presentation", "manifests", "appears as",
                "characterized by", "typical", "common", "usual"
            ]
        }
        
        self.tier2_patterns = {
            # Clinical reasoning patterns
            "pathophysiology": [
                "pathophysiology", "pathogenesis", "develops when", "caused by",
                "mechanism of disease", "how disease", "process of"
            ],
            "diagnostic_criteria": [
                "diagnosis", "diagnostic criteria", "criteria for", "diagnosed when",
                "features include", "characterized by", "findings"
            ],
            "clinical_reasoning": [
                "differential diagnosis", "consider", "rule out", "distinguish",
                "likely", "probable", "most common", "typically"
            ],
            "treatment_principles": [
                "treatment", "management", "therapy", "approach", "protocol",
                "guidelines recommend", "first-line", "standard care"
            ],
            "disease_mechanisms": [
                "leads to", "results in", "causes", "mechanism", "pathway",
                "cascade", "triggers", "initiates", "progression"
            ]
        }
        
        self.tier3_patterns = {
            # Evidence-based medicine patterns
            "guidelines": [
                "guideline", "recommendation", "consensus", "standard",
                "protocol", "evidence-based", "best practice"
            ],
            "research_evidence": [
                "study showed", "research indicates", "evidence suggests",
                "meta-analysis", "systematic review", "clinical trial"
            ],
            "clinical_outcomes": [
                "outcome", "prognosis", "survival", "mortality", "efficacy",
                "effectiveness", "results", "response rate"
            ],
            "authoritative_sources": [
                "who", "fda", "cdc", "aha", "acc", "esc", "acog", "idsa",
                "guidelines", "consensus", "statement", "position"
            ],
            "evidence_levels": [
                "grade a", "grade b", "level 1", "level 2", "strong evidence",
                "moderate evidence", "high quality", "randomized"
            ]
        }

    def preprocess_documents(self, documents: List[Dict]) -> List[Dict]:
        """Enhanced preprocessing with intelligent tier assignment."""
        logger.info(f"🔧 Enhanced preprocessing of {len(documents)} documents for medical Q&A")
        
        processed_docs = []
        tier_stats = {"content_based": 0, "source_based": 0, "fallback": 0}
        
        for i, doc in enumerate(documents):
            try:
                # Create unique document ID
                doc_id = f"doc_{i:06d}"
                
                # Validate required fields
                if "text" not in doc or not doc["text"]:
                    continue
                
                if "metadata" not in doc:
                    doc["metadata"] = {}
                
                # Convert text to string and validate
                text = str(doc["text"]) if doc["text"] else ""
                if len(text.strip()) < 50:  # Increased minimum length for medical content
                    continue
                
                # Clean metadata for ChromaDB compatibility
                clean_metadata = self._clean_metadata(doc["metadata"])
                
                # Enhanced tier assignment
                tier, assignment_method = self._assign_medical_tier_enhanced(doc)
                tier_stats[assignment_method] += 1
                
                # Determine medical specialty more accurately
                specialty = self._determine_medical_specialty_enhanced(doc)
                
                processed_doc = {
                    "text": text,
                    "metadata": {
                        **clean_metadata,
                        "doc_id": doc_id,
                        "tier": tier,
                        "tier_chunk_id": f"tier{tier}_{i}",
                        "medical_specialty": specialty,
                        "assignment_method": assignment_method
                    }
                }
                
                processed_docs.append(processed_doc)
                
            except Exception as e:
                logger.warning(f"Failed to process document {i}: {e}")
                continue
        
        logger.info(f"✅ Enhanced preprocessing complete: {len(processed_docs)} documents")
        logger.info(f"📊 Tier assignment methods: {tier_stats}")
        return processed_docs

    def _assign_medical_tier_enhanced(self, doc: Dict) -> tuple[int, str]:
        """Enhanced tier assignment with content analysis and balanced distribution."""
        metadata = doc.get("metadata", {})
        text = str(doc.get("text", "")).lower()
        title = str(metadata.get("title", "")).lower()
        
        combined_text = f"{title} {text}"
        
        # CONTENT-ONLY ANALYSIS (no source bias)
        tier1_score = self._calculate_pattern_score(combined_text, self.tier1_patterns)
        tier2_score = self._calculate_pattern_score(combined_text, self.tier2_patterns)
        tier3_score = self._calculate_pattern_score(combined_text, self.tier3_patterns)
        
        # Content-based assignment with balanced distribution
        max_score = max(tier1_score, tier2_score, tier3_score)
        
        # If clear content-based classification exists
        if max_score >= 2:
            if tier1_score == max_score:
                return self._balanced_tier_assignment(1), "content_based"
            elif tier3_score == max_score:
                return self._balanced_tier_assignment(3), "content_based"
            else:
                return self._balanced_tier_assignment(2), "content_based"
        
        # For unclear content, use balanced random assignment based on text characteristics
        text_length = len(combined_text)
        complexity_indicators = sum(1 for word in [
            'pathophysiology', 'mechanism', 'etiology', 'diagnosis', 
            'treatment', 'management', 'therapy', 'clinical', 'patient'
        ] if word in combined_text)
        
        # Balanced assignment based on content characteristics
        if text_length < 200 or any(word in combined_text for word in [
            'definition', 'what is', 'overview', 'introduction', 'basic'
        ]):
            return self._balanced_tier_assignment(1), "content_analysis"
        elif complexity_indicators >= 3 or any(word in combined_text for word in [
            'evidence', 'study', 'trial', 'research', 'guideline', 'recommendation'
        ]):
            return self._balanced_tier_assignment(3), "content_analysis"
        else:
            return self._balanced_tier_assignment(2), "content_analysis"
    
    def _balanced_tier_assignment(self, preferred_tier: int) -> int:
        """Ensure balanced tier distribution (target: 30/40/30)."""
        if not hasattr(self, '_tier_counts'):
            self._tier_counts = {1: 0, 2: 0, 3: 0}
            self._total_processed = 0
        
        self._total_processed += 1
        
        # Target distribution percentages
        target_ratios = {1: 0.30, 2: 0.40, 3: 0.30}
        
        # Calculate current ratios
        current_ratios = {
            tier: count / self._total_processed 
            for tier, count in self._tier_counts.items()
        }
        
        # Check if preferred tier is under its target
        if current_ratios[preferred_tier] < target_ratios[preferred_tier]:
            self._tier_counts[preferred_tier] += 1
            return preferred_tier
        
        # Find most under-represented tier
        deficits = {
            tier: target_ratios[tier] - current_ratios[tier]
            for tier in [1, 2, 3]
        }
        
        # Assign to most deficient tier
        assigned_tier = max(deficits.keys(), key=lambda k: deficits[k])
        self._tier_counts[assigned_tier] += 1
        return assigned_tier

    def _calculate_pattern_score(self, text: str, patterns: Dict[str, List[str]]) -> int:
        """Calculate weighted pattern score for tier assignment."""
        total_score = 0
        
        for category, pattern_list in patterns.items():
            category_score = 0
            for pattern in pattern_list:
                # Use word boundaries for more accurate matching
                if re.search(r'\b' + re.escape(pattern) + r'\b', text):
                    category_score += 1
            
            # Weight categories differently
            if category in ["definitions", "guidelines", "research_evidence"]:
                total_score += category_score * 2  # Higher weight
            else:
                total_score += category_score
        
        return total_score

    def _determine_medical_specialty_enhanced(self, doc: Dict) -> str:
        """Enhanced medical specialty determination."""
        metadata = doc.get("metadata", {})
        text = str(doc.get("text", "")).lower()
        title = str(metadata.get("title", "")).lower()
        
        # Check existing specialty
        existing_specialty = metadata.get("medical_specialty")
        if existing_specialty and existing_specialty != "Unknown":
            return existing_specialty
        
        combined_text = f"{title} {text}"
        
        # Enhanced specialty mapping
        specialty_keywords = {
            "Cardiology": [
                "heart", "cardiac", "cardiovascular", "coronary", "myocardial",
                "arrhythmia", "hypertension", "blood pressure", "ecg", "echo"
            ],
            "Endocrinology": [
                "diabetes", "insulin", "glucose", "thyroid", "hormone", "endocrine",
                "metabolism", "pituitary", "adrenal", "pancreas"
            ],
            "Pulmonology": [
                "lung", "pulmonary", "respiratory", "asthma", "copd", "pneumonia",
                "breathing", "ventilation", "oxygen", "bronchial"
            ],
            "Neurology": [
                "brain", "neurological", "stroke", "seizure", "dementia", "migraine",
                "nervous system", "spinal", "cognitive", "memory"
            ],
            "Infectious Disease": [
                "infection", "antibiotic", "antimicrobial", "sepsis", "bacterial",
                "viral", "fungal", "pathogen", "microorganism", "resistance"
            ],
            "Oncology": [
                "cancer", "tumor", "malignancy", "chemotherapy", "oncology",
                "metastasis", "carcinoma", "lymphoma", "leukemia", "radiation"
            ],
            "Gastroenterology": [
                "gastrointestinal", "digestive", "liver", "stomach", "intestine",
                "bowel", "gastric", "hepatic", "colon", "pancreatic"
            ],
            "Nephrology": [
                "kidney", "renal", "dialysis", "urine", "creatinine", "filtration",
                "nephron", "proteinuria", "uremia", "electrolyte"
            ],
            "Obstetrics/Gynecology": [
                "pregnancy", "pregnant", "delivery", "birth", "gynecologic",
                "obstetric", "uterine", "ovarian", "cervical", "menstrual"
            ],
            "Pediatrics": [
                "pediatric", "child", "infant", "neonatal", "adolescent",
                "growth", "development", "vaccination", "congenital"
            ],
            "Psychiatry": [
                "psychiatric", "mental health", "depression", "anxiety", "psychotic",
                "behavioral", "cognitive therapy", "antidepressant", "mood"
            ],
            "Emergency Medicine": [
                "emergency", "trauma", "acute", "critical", "resuscitation",
                "triage", "shock", "cpr", "defibrillation", "urgent"
            ]
        }
        
        # Score each specialty
        specialty_scores = {}
        for specialty, keywords in specialty_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                specialty_scores[specialty] = score
        
        # Return highest scoring specialty
        if specialty_scores:
            return max(specialty_scores, key=specialty_scores.get)
        
        return "General Medicine"

    def _clean_metadata(self, metadata: Dict) -> Dict:
        """Clean metadata for ChromaDB compatibility."""
        clean_meta = {}
        
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                if isinstance(value, str) and value.strip() == "":
                    continue
                clean_meta[key] = value
            elif isinstance(value, list):
                if value:
                    clean_meta[key] = ", ".join(str(item) for item in value if item is not None)
            elif isinstance(value, dict):
                if value:
                    clean_meta[key] = str(value)
            else:
                str_value = str(value).strip()
                if str_value and str_value.lower() not in ["none", "null", ""]:
                    clean_meta[key] = str_value
        
        return clean_meta

    def organize_by_reasoning_type(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """Organize documents by enhanced medical reasoning tiers."""
        logger.info(f"📊 Organizing {len(documents)} documents by enhanced medical reasoning tiers")
        
        organized = {
            "pattern_recognition": [],  # Tier 1 - Basic medical knowledge
            "hypothesis_testing": [],   # Tier 2 - Clinical reasoning
            "confirmation": []          # Tier 3 - Evidence-based medicine
        }
        
        tier_counts = {1: 0, 2: 0, 3: 0}
        specialty_counts = {}
        assignment_methods = {"content_based": 0, "source_based": 0, "fallback": 0}
        
        for doc in documents:
            tier = doc["metadata"].get("tier", 2)
            specialty = doc["metadata"].get("medical_specialty", "Unknown")
            method = doc["metadata"].get("assignment_method", "unknown")
            
            # Count statistics
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            specialty_counts[specialty] = specialty_counts.get(specialty, 0) + 1
            assignment_methods[method] = assignment_methods.get(method, 0) + 1
            
            # Organize by tier
            if tier == 1:
                organized["pattern_recognition"].append(doc)
            elif tier == 3:
                organized["confirmation"].append(doc)
            else:  # tier == 2 or any other value
                organized["hypothesis_testing"].append(doc)
        
        # Enhanced logging
        total_docs = sum(tier_counts.values())
        logger.info(f"📊 Enhanced medical knowledge organization:")
        logger.info(f"   Tier 1 (Pattern Recognition): {tier_counts.get(1, 0)} docs ({tier_counts.get(1, 0)/total_docs*100:.1f}%)")
        logger.info(f"   Tier 2 (Clinical Reasoning): {tier_counts.get(2, 0)} docs ({tier_counts.get(2, 0)/total_docs*100:.1f}%)")
        logger.info(f"   Tier 3 (Evidence Confirmation): {tier_counts.get(3, 0)} docs ({tier_counts.get(3, 0)/total_docs*100:.1f}%)")
        
        logger.info(f"📊 Assignment methods:")
        for method, count in assignment_methods.items():
            logger.info(f"   {method}: {count} docs ({count/total_docs*100:.1f}%)")
        
        logger.info(f"📊 Top medical specialties:")
        top_specialties = sorted(specialty_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        for specialty, count in top_specialties:
            logger.info(f"   {specialty}: {count} docs")
        
        # Validate distribution for medical Q&A effectiveness
        self._validate_medical_qa_distribution(tier_counts, total_docs)
        
        return organized

    def _validate_medical_qa_distribution(self, tier_counts: Dict[int, int], total_docs: int):
        """Validate tier distribution for optimal medical Q&A performance."""
        if total_docs == 0:
            logger.error("❌ No documents found after organization")
            return
        
        tier1_pct = (tier_counts.get(1, 0) / total_docs) * 100
        tier2_pct = (tier_counts.get(2, 0) / total_docs) * 100
        tier3_pct = (tier_counts.get(3, 0) / total_docs) * 100
        
        # Optimal distribution for medical Q&A
        if tier1_pct < 15:
            logger.warning(f"⚠️ Low Tier 1 content ({tier1_pct:.1f}%) - may affect basic concept questions")
        if tier2_pct < 25:
            logger.warning(f"⚠️ Low Tier 2 content ({tier2_pct:.1f}%) - may affect clinical reasoning questions")
        if tier3_pct < 15:
            logger.warning(f"⚠️ Low Tier 3 content ({tier3_pct:.1f}%) - may affect evidence-based questions")
        
        # Check for severe imbalance
        max_tier_pct = max(tier1_pct, tier2_pct, tier3_pct)
        if max_tier_pct > 75:
            logger.warning(f"⚠️ Severely imbalanced distribution (max: {max_tier_pct:.1f}%)")
            logger.warning("   This may reduce hierarchical reasoning effectiveness")
        else:
            logger.info("✅ Balanced tier distribution for medical Q&A")

    def load_foundation_dataset(self, foundation_dir: Path) -> List[Dict[str, str]]:
        """Load foundation dataset for enhanced medical knowledge processing."""
        logger.info(f"📂 Loading foundation dataset from {foundation_dir}")
        
        # Try multiple file patterns
        dataset_files = [
            foundation_dir / "foundation_medical_data.json",
            foundation_dir / "foundation_specialty_rebalanced.json",
            foundation_dir / "unified_dataset.json"
        ]
        
        for dataset_file in dataset_files:
            if dataset_file.exists():
                logger.info(f"   Found dataset: {dataset_file}")
                
                with open(dataset_file, 'r') as f:
                    data = json.load(f)
                
                # Handle different data formats
                if isinstance(data, list):
                    documents = data
                elif isinstance(data, dict) and 'documents' in data:
                    documents = data['documents']
                else:
                    documents = []
                
                logger.info(f"✅ Loaded {len(documents)} documents for enhanced processing")
                return documents
        
        logger.error(f"❌ No foundation dataset found in {foundation_dir}")
        return []