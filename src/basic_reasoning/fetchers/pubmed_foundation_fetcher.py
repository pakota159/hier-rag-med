#!/usr/bin/env python3
"""
Specialty-Rebalanced PubMed Foundation Fetcher (REAL DATA ONLY)
File: src/basic_reasoning/fetchers/pubmed_foundation_fetcher.py

SPECIALTY REBALANCING FOCUS:
- Reduces "General" categorization through specific MeSH terms
- Increases specialty-specific coverage
- Adds missing MIRAGE specialties: Surgery, OB/GYN, Anesthesiology, Pathology
- Enhanced evidence quality targeting

Uses legitimate PubMed E-utilities API for all data fetching.
"""

import logging
import time
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PubMedFoundationFetcher:
    """Specialty-rebalanced PubMed fetcher focusing on specific medical specialties."""
    
    def __init__(self, email: str = "hierragmed@example.com"):
        self.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        # SPECIALTY-REBALANCED medical searches with specific quotas
        # Focus: Reduce "General" categorization, increase specialty-specific content
        self.specialty_searches = {
            # CARDIOLOGY (High MIRAGE weight)
            "cardiology": {
                "quota": 150,
                "searches": [
                    "myocardial infarction[MeSH] AND acute management",
                    "heart failure[MeSH] AND therapy",
                    "atrial fibrillation[MeSH] AND anticoagulation",
                    "hypertension[MeSH] AND guideline",
                    "coronary artery disease[MeSH] AND intervention",
                    "cardiomyopathy[MeSH] AND treatment",
                    "cardiac catheterization[MeSH] AND procedure",
                    "arrhythmia[MeSH] AND management"
                ]
            },
            
            # SURGERY (Missing from MIRAGE - CRITICAL)
            "surgery": {
                "quota": 120,
                "searches": [
                    "surgical procedures operative[MeSH] AND complications",
                    "laparoscopy[MeSH] AND technique",
                    "appendectomy[MeSH] AND procedure",
                    "cholecystectomy[MeSH] AND laparoscopic",
                    "hernia repair[MeSH] AND surgical",
                    "postoperative complications[MeSH] AND prevention",
                    "surgical wound infection[MeSH] AND management",
                    "perioperative care[MeSH] AND protocol",
                    "anesthesia[MeSH] AND surgical procedure",
                    "minimally invasive surgical procedures[MeSH]"
                ]
            },
            
            # OBSTETRICS/GYNECOLOGY (Missing from MIRAGE - CRITICAL)
            "obstetrics_gynecology": {
                "quota": 100,
                "searches": [
                    "pregnancy[MeSH] AND complications",
                    "delivery obstetric[MeSH] AND management",
                    "prenatal care[MeSH] AND guidelines",
                    "preeclampsia[MeSH] AND treatment",
                    "gestational diabetes[MeSH] AND management",
                    "cesarean section[MeSH] AND indications",
                    "menstrual disorders[MeSH] AND therapy",
                    "contraception[MeSH] AND methods",
                    "gynecologic surgical procedures[MeSH]",
                    "ovarian cysts[MeSH] AND treatment"
                ]
            },
            
            # EMERGENCY MEDICINE (MIRAGE Critical)
            "emergency_medicine": {
                "quota": 100,
                "searches": [
                    "emergency medicine[MeSH] AND protocols",
                    "trauma[MeSH] AND resuscitation",
                    "cardiac arrest[MeSH] AND management",
                    "sepsis[MeSH] AND emergency treatment",
                    "stroke[MeSH] AND acute care",
                    "anaphylaxis[MeSH] AND emergency",
                    "poisoning[MeSH] AND antidote",
                    "emergency department[MeSH] AND triage",
                    "shock[MeSH] AND emergency management"
                ]
            },
            
            # PSYCHIATRY (MIRAGE Critical)
            "psychiatry": {
                "quota": 100,
                "searches": [
                    "major depressive disorder[MeSH] AND therapy",
                    "bipolar disorder[MeSH] AND treatment",
                    "schizophrenia[MeSH] AND antipsychotic",
                    "anxiety disorders[MeSH] AND therapy",
                    "suicide[MeSH] AND prevention",
                    "ADHD[Title/Abstract] AND medication",
                    "autism spectrum disorder[MeSH] AND intervention",
                    "PTSD[Title/Abstract] AND treatment",
                    "substance abuse[MeSH] AND therapy"
                ]
            },
            
            # DERMATOLOGY (MIRAGE Critical)
            "dermatology": {
                "quota": 80,
                "searches": [
                    "skin neoplasms[MeSH] AND diagnosis",
                    "melanoma[MeSH] AND treatment",
                    "psoriasis[MeSH] AND therapy",
                    "dermatitis atopic[MeSH] AND management",
                    "acne vulgaris[MeSH] AND treatment",
                    "dermatologic surgical procedures[MeSH]",
                    "skin diseases[MeSH] AND therapy"
                ]
            },
            
            # GASTROENTEROLOGY (MIRAGE Critical)
            "gastroenterology": {
                "quota": 80,
                "searches": [
                    "inflammatory bowel diseases[MeSH] AND therapy",
                    "liver cirrhosis[MeSH] AND management",
                    "gastrointestinal hemorrhage[MeSH] AND treatment",
                    "peptic ulcer[MeSH] AND therapy",
                    "hepatitis[MeSH] AND treatment",
                    "colonoscopy[MeSH] AND screening",
                    "gastroesophageal reflux[MeSH] AND therapy"
                ]
            },
            
            # PEDIATRICS (MIRAGE Critical)
            "pediatrics": {
                "quota": 80,
                "searches": [
                    "pediatrics[MeSH] AND emergency",
                    "child[MeSH] AND fever",
                    "asthma[MeSH] AND child",
                    "vaccination[MeSH] AND pediatric",
                    "pediatric surgery[MeSH] AND procedures",
                    "child development disorders[MeSH]",
                    "pediatric intensive care[MeSH]"
                ]
            },
            
            # ONCOLOGY (Enhanced)
            "oncology": {
                "quota": 100,
                "searches": [
                    "breast neoplasms[MeSH] AND therapy",
                    "lung neoplasms[MeSH] AND treatment",
                    "colorectal neoplasms[MeSH] AND surgery",
                    "prostatic neoplasms[MeSH] AND therapy",
                    "lymphoma[MeSH] AND chemotherapy",
                    "neoplasm staging[MeSH] AND guidelines",
                    "antineoplastic agents[MeSH] AND therapy",
                    "radiation oncology[MeSH] AND treatment",
                    "palliative care[MeSH] AND cancer"
                ]
            },
            
            # ENDOCRINOLOGY (Enhanced)
            "endocrinology": {
                "quota": 80,
                "searches": [
                    "diabetes mellitus type 2[MeSH] AND management",
                    "diabetic ketoacidosis[MeSH] AND treatment",
                    "thyroid diseases[MeSH] AND therapy",
                    "adrenal insufficiency[MeSH] AND diagnosis",
                    "osteoporosis[MeSH] AND treatment",
                    "insulin[MeSH] AND therapy",
                    "hypoglycemia[MeSH] AND management"
                ]
            },
            
            # NEPHROLOGY (Enhanced)
            "nephrology": {
                "quota": 80,
                "searches": [
                    "acute kidney injury[MeSH] AND management",
                    "chronic kidney disease[MeSH] AND therapy",
                    "renal dialysis[MeSH] AND complications",
                    "kidney transplantation[MeSH] AND outcomes",
                    "proteinuria[MeSH] AND treatment",
                    "hypertension renal[MeSH] AND therapy",
                    "nephritis[MeSH] AND treatment"
                ]
            },
            
            # INFECTIOUS DISEASE (Enhanced)
            "infectious_disease": {
                "quota": 80,
                "searches": [
                    "bacterial infections[MeSH] AND antibiotics",
                    "viral infections[MeSH] AND antiviral",
                    "tuberculosis[MeSH] AND treatment",
                    "HIV infections[MeSH] AND therapy",
                    "pneumonia[MeSH] AND management",
                    "drug resistance bacterial[MeSH]",
                    "infection control[MeSH] AND hospital"
                ]
            },
            
            # HEMATOLOGY (Enhanced)
            "hematology": {
                "quota": 70,
                "searches": [
                    "anemia[MeSH] AND therapy",
                    "thrombocytopenia[MeSH] AND treatment",
                    "leukemia[MeSH] AND chemotherapy",
                    "anticoagulants[MeSH] AND therapy",
                    "bleeding disorders[MeSH] AND management",
                    "blood transfusion[MeSH] AND safety",
                    "coagulation disorders[MeSH]"
                ]
            },
            
            # RHEUMATOLOGY (Enhanced)
            "rheumatology": {
                "quota": 70,
                "searches": [
                    "arthritis rheumatoid[MeSH] AND therapy",
                    "lupus erythematosus systemic[MeSH] AND treatment",
                    "osteoarthritis[MeSH] AND management",
                    "gout[MeSH] AND therapy",
                    "fibromyalgia[MeSH] AND treatment",
                    "autoimmune diseases[MeSH] AND therapy",
                    "arthritis[MeSH] AND inflammation"
                ]
            },
            
            # NEUROLOGY (Enhanced)
            "neurology": {
                "quota": 80,
                "searches": [
                    "stroke[MeSH] AND acute treatment",
                    "epilepsy[MeSH] AND therapy",
                    "alzheimer disease[MeSH] AND management",
                    "parkinson disease[MeSH] AND treatment",
                    "multiple sclerosis[MeSH] AND therapy",
                    "migraine disorders[MeSH] AND treatment",
                    "seizures[MeSH] AND management"
                ]
            },
            
            # PULMONOLOGY (Enhanced)
            "pulmonology": {
                "quota": 70,
                "searches": [
                    "asthma[MeSH] AND therapy",
                    "pulmonary disease chronic obstructive[MeSH] AND management",
                    "pneumonia[MeSH] AND treatment",
                    "pulmonary embolism[MeSH] AND therapy",
                    "lung diseases[MeSH] AND diagnosis",
                    "respiratory failure[MeSH] AND management"
                ]
            },
            
            # ANESTHESIOLOGY (Missing from MIRAGE - NEW)
            "anesthesiology": {
                "quota": 60,
                "searches": [
                    "anesthesia[MeSH] AND complications",
                    "anesthesia general[MeSH] AND management",
                    "anesthesia spinal[MeSH] AND technique",
                    "pain management[MeSH] AND anesthesia",
                    "perioperative care[MeSH] AND anesthesia",
                    "airway management[MeSH] AND anesthesia"
                ]
            },
            
            # PATHOLOGY (Missing from MIRAGE - NEW)
            "pathology": {
                "quota": 60,
                "searches": [
                    "pathology clinical[MeSH] AND diagnosis",
                    "biopsy[MeSH] AND interpretation",
                    "histopathology[MeSH] AND diagnosis",
                    "cytology[MeSH] AND diagnostic",
                    "autopsy[MeSH] AND findings",
                    "laboratory medicine[MeSH] AND diagnosis"
                ]
            },
            
            # CRITICAL CARE (Enhanced)
            "critical_care": {
                "quota": 60,
                "searches": [
                    "critical care[MeSH] AND management",
                    "intensive care units[MeSH] AND protocols",
                    "respiratory failure[MeSH] AND ventilation",
                    "shock[MeSH] AND critical care",
                    "sepsis[MeSH] AND intensive care",
                    "multiple organ failure[MeSH] AND treatment"
                ]
            }
        }
        
        # Total specialty quota calculation
        self.total_specialty_quota = sum(spec["quota"] for spec in self.specialty_searches.values())
        
        # Evidence quality indicators for better stratification
        self.tier3_keywords = [
            "guideline", "guidelines", "recommendation", "consensus", "systematic review", 
            "meta-analysis", "cochrane", "evidence-based", "clinical practice guideline",
            "treatment guideline", "management guideline", "practice parameter"
        ]
        
        self.tier2_keywords = [
            "randomized controlled trial", "clinical trial", "rct", "comparative effectiveness",
            "prospective study", "intervention study", "treatment outcome", "efficacy study"
        ]

    def fetch_pubmed_foundation(self, max_abstracts: int = 2000) -> List[Dict]:
        """Fetch specialty-rebalanced foundation dataset from PubMed."""
        logger.info("📚 FETCHING SPECIALTY-REBALANCED PUBMED FOUNDATION")
        logger.info("=" * 60)
        logger.info(f"Target abstracts: {max_abstracts}")
        logger.info(f"Medical specialties: {len(self.specialty_searches)} specific areas")
        logger.info(f"Total specialty quota: {self.total_specialty_quota}")
        logger.info("🎯 Focus: Reduce 'General' categorization, increase specialty-specific content")
        logger.info("📡 Data source: PubMed E-utilities API (eutils.ncbi.nlm.nih.gov)")
        
        all_documents = []
        
        # Calculate scaling factor if max_abstracts differs from total quota
        scaling_factor = max_abstracts / self.total_specialty_quota if self.total_specialty_quota > 0 else 1
        
        for specialty, config in self.specialty_searches.items():
            target_quota = int(config["quota"] * scaling_factor)
            if target_quota == 0:
                continue
                
            logger.info(f"🔬 Fetching {specialty} (target: {target_quota})")
            
            specialty_docs = []
            searches = config["searches"]
            docs_per_search = max(1, target_quota // len(searches))
            
            for search_query in searches:
                if len(specialty_docs) >= target_quota:
                    break
                    
                try:
                    search_docs = self._fetch_specialty_abstracts(
                        search_query, specialty, docs_per_search
                    )
                    specialty_docs.extend(search_docs)
                    
                    # Rate limiting for PubMed API
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.debug(f"Failed search '{search_query}': {e}")
                    continue
            
            # Trim to target quota
            specialty_docs = specialty_docs[:target_quota]
            all_documents.extend(specialty_docs)
            
            logger.info(f"✅ {specialty}: {len(specialty_docs)} specialized documents")
        
        logger.info(f"🎉 SPECIALTY-REBALANCED PUBMED COMPLETE: {len(all_documents)} documents")
        logger.info(f"📊 Specialties covered: {len(self.specialty_searches)}")
        logger.info(f"🎯 Specialty-specific content: ~{len(all_documents)} (reduced 'General' categorization)")
        
        return all_documents

    def _fetch_specialty_abstracts(self, search_query: str, specialty: str, max_count: int) -> List[Dict]:
        """Fetch abstracts for specific specialty with enhanced search strategy."""
        # Enhanced search with evidence quality filters
        enhanced_query = f'({search_query}) AND ("last 10 years"[PDat]) AND (hasabstract[text]) AND (english[lang])'
        
        # Search for PMIDs using real PubMed API
        search_url = f"{self.base_url}esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": enhanced_query,
            "retmax": min(max_count * 3, 100),  # Get extra for quality filtering
            "email": self.email,
            "tool": "hierragmed_specialty_rebalanced",
            "sort": "relevance"
        }
        
        try:
            search_response = requests.get(search_url, params=search_params, timeout=30)
            if search_response.status_code != 200:
                return []
            
            # Parse PMIDs
            root = ET.fromstring(search_response.content)
            pmids = [id_elem.text for id_elem in root.findall(".//Id")]
            
            if not pmids:
                return []
            
            # Fetch abstracts and apply quality filtering
            return self._fetch_and_filter_abstracts(pmids[:max_count * 2], specialty, max_count)
            
        except Exception as e:
            logger.debug(f"Search failed for '{search_query}': {e}")
            return []

    def _fetch_and_filter_abstracts(self, pmids: List[str], specialty: str, max_count: int) -> List[Dict]:
        """Fetch abstracts and apply quality filtering for better evidence stratification."""
        if not pmids:
            return []
        
        # Fetch abstracts using real PubMed API
        fetch_url = f"{self.base_url}efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
            "email": self.email,
            "tool": "hierragmed_specialty_rebalanced"
        }
        
        try:
            fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
            if fetch_response.status_code != 200:
                return []
            
            root = ET.fromstring(fetch_response.content)
            raw_documents = self._parse_specialty_abstracts(root, specialty)
            
            # Apply quality filtering and prioritization
            filtered_docs = self._apply_quality_filtering(raw_documents)
            
            return filtered_docs[:max_count]
            
        except Exception as e:
            logger.debug(f"Failed to fetch abstracts: {e}")
            return []

    def _parse_specialty_abstracts(self, root: ET.Element, specialty: str) -> List[Dict]:
        """Parse abstracts with specialty-specific metadata extraction."""
        documents = []
        
        for article in root.findall(".//PubmedArticle"):
            try:
                abstract_data = self._extract_enhanced_metadata(article)
                if not abstract_data or not abstract_data.get("abstract"):
                    continue
                
                if len(abstract_data["abstract"]) < 150:  # Quality filter
                    continue
                
                doc = self._create_specialty_document(abstract_data, specialty)
                documents.append(doc)
                
            except Exception as e:
                logger.debug(f"Failed to process article: {e}")
                continue
        
        return documents

    def _apply_quality_filtering(self, documents: List[Dict]) -> List[Dict]:
        """Apply quality filtering to prioritize high-evidence documents."""
        def get_quality_score(doc):
            metadata = doc.get("metadata", {})
            score = 0
            
            # Prioritize high-tier evidence
            tier = metadata.get("tier", 1)
            score += tier * 10
            
            # Prioritize high-evidence levels
            evidence_level = metadata.get("evidence_level", "standard")
            if evidence_level == "high":
                score += 20
            elif evidence_level == "medium":
                score += 10
            
            # Prioritize high-impact journals
            journal = metadata.get("journal", "").lower()
            if any(high_journal.lower() in journal for high_journal in [
                "nejm", "lancet", "jama", "nature", "bmj", "circulation", "chest"
            ]):
                score += 15
            
            return score
        
        # Sort by quality score (highest first)
        return sorted(documents, key=get_quality_score, reverse=True)

    def _extract_enhanced_metadata(self, article: ET.Element) -> Optional[Dict]:
        """Extract comprehensive metadata from PubMed article."""
        try:
            # Extract article information
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            
            abstract_elem = article.find(".//AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            if not abstract or len(abstract) < 100:
                return None
            
            # Extract publication metadata
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""
            
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""
            
            year_elem = article.find(".//PubDate/Year")
            year = year_elem.text if year_elem is not None else ""
            
            # Extract authors
            authors = []
            for author in article.findall(".//Author"):
                lastname = author.find("LastName")
                firstname = author.find("ForeName")
                if lastname is not None and firstname is not None:
                    authors.append(f"{firstname.text} {lastname.text}")
            
            # Extract publication types
            pub_types = []
            for pt in article.findall(".//PublicationType"):
                if pt.text:
                    pub_types.append(pt.text)
            
            # Extract MeSH terms
            mesh_terms = []
            for mesh in article.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text)
            
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "authors": authors,
                "publication_types": pub_types,
                "mesh_terms": mesh_terms
            }
            
        except Exception as e:
            logger.debug(f"Failed to extract metadata: {e}")
            return None

    def _create_specialty_document(self, abstract_data: Dict, specialty: str) -> Dict:
        """Create document with specialty-specific metadata and enhanced categorization."""
        tier = self._determine_evidence_tier(abstract_data)
        evidence_level = self._determine_evidence_level(abstract_data)
        therapeutic_area = self._map_specialty_to_therapeutic_area(specialty)
        
        # Create document text optimized for specialty identification
        text = f"""
Medical Specialty: {specialty.replace('_', ' ').title()}

Title: {abstract_data['title']}

Abstract: {abstract_data['abstract']}

Clinical Specialty Information:
- Primary Specialty: {specialty.replace('_', ' ').title()}
- Therapeutic Area: {therapeutic_area.replace('_', ' ').title()}
- Evidence Level: {evidence_level}

Publication Details:
- Journal: {abstract_data['journal']}
- Year: {abstract_data['year']}
- PMID: {abstract_data['pmid']}
- Publication Types: {', '.join(abstract_data['publication_types'])}

Medical Subject Headings (MeSH):
{', '.join(abstract_data['mesh_terms'][:8])}{'...' if len(abstract_data['mesh_terms']) > 8 else ''}
"""

        return {
            "text": text.strip(),
            "metadata": {
                "doc_id": f"pubmed_specialty_{specialty}_{abstract_data['pmid']}",
                "source": "pubmed",
                "title": abstract_data['title'],
                "pmid": abstract_data['pmid'],
                "journal": abstract_data['journal'],
                "year": abstract_data['year'],
                "medical_specialty": specialty,  # Specific specialty assignment
                "therapeutic_area": therapeutic_area,
                "evidence_level": evidence_level,
                "tier": tier,
                "organization": "NCBI",
                "publication_types": abstract_data['publication_types'],
                "mesh_terms": abstract_data['mesh_terms'],
                "authors": abstract_data['authors'],
                "chunk_id": 0,
                "reasoning_type": "specialty_specific_evidence",
                "data_source": "pubmed_eutils_api",
                "specialty_rebalanced": True  # Flag for tracking rebalanced content
            }
        }

    def _determine_evidence_tier(self, abstract_data: Dict) -> int:
        """Determine evidence tier with enhanced criteria for better stratification."""
        title_abstract = f"{abstract_data['title']} {abstract_data['abstract']}".lower()
        pub_types = [pt.lower() for pt in abstract_data['publication_types']]
        journal = abstract_data.get('journal', '').lower()
        
        # Tier 3: High-quality evidence (enhanced detection)
        tier3_pub_types = ["systematic review", "meta-analysis", "practice guideline", "clinical guideline"]
        tier3_journals = ["nejm", "new england journal", "lancet", "jama", "nature medicine", "bmj"]
        
        if (any(keyword in title_abstract for keyword in self.tier3_keywords) or
            any(pt in tier3_pub_types for pt in pub_types) or
            any(journal_name in journal for journal_name in tier3_journals)):
            return 3
        
        # Tier 2: Clinical studies and trials
        tier2_pub_types = ["randomized controlled trial", "clinical trial", "comparative study", "multicenter study"]
        
        if (any(keyword in title_abstract for keyword in self.tier2_keywords) or
            any(pt in tier2_pub_types for pt in pub_types)):
            return 2
        
        # Tier 1: Other research
        return 1

    def _determine_evidence_level(self, abstract_data: Dict) -> str:
        """Determine evidence level with enhanced stratification."""
        pub_types = [pt.lower() for pt in abstract_data['publication_types']]
        title_abstract = f"{abstract_data['title']} {abstract_data['abstract']}".lower()
        
        # High evidence
        if (any(pt in ["systematic review", "meta-analysis"] for pt in pub_types) or
            any(keyword in title_abstract for keyword in ["meta-analysis", "systematic review", "cochrane"])):
            return "high"
        
        # Medium evidence  
        if any(pt in ["randomized controlled trial", "clinical trial"] for pt in pub_types):
            return "medium"
        
        # Standard evidence
        return "standard"

    def _map_specialty_to_therapeutic_area(self, specialty: str) -> str:
        """Map medical specialty to specific therapeutic area (reduces 'General' categorization)."""
        mapping = {
            "cardiology": "cardiovascular",
            "surgery": "surgical",
            "obstetrics_gynecology": "obstetric_gynecologic",
            "emergency_medicine": "emergency",
            "psychiatry": "mental_health",
            "dermatology": "dermatological",
            "gastroenterology": "gastrointestinal",
            "pediatrics": "pediatric",
            "oncology": "oncology",
            "endocrinology": "endocrine",
            "nephrology": "renal",
            "infectious_disease": "infectious_disease",
            "hematology": "hematological",
            "rheumatology": "rheumatological",
            "neurology": "neurological",
            "pulmonology": "respiratory",
            "anesthesiology": "anesthetic",
            "pathology": "pathological",
            "critical_care": "critical_care"
        }
        return mapping.get(specialty, specialty)  # Use specialty name if not in mapping