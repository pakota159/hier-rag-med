# src/basic_reasoning/fetchers/pubmed_foundation_fetcher.py
"""
PubMed Foundation Fetcher for HierRAGMed
Fetches PubMed abstracts with automatic tier assignment for hierarchical system
"""

import time
import requests
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
import xml.etree.ElementTree as ET


class PubMedFoundationFetcher:
    """Fetcher for PubMed abstracts with hierarchical tier assignment."""
    
    def __init__(self, email: str = "hierragmed@example.com"):
        """Initialize PubMed fetcher."""
        self.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        # Medical topics for comprehensive coverage
        self.medical_topics = [
            "diabetes mellitus",
            "hypertension", 
            "myocardial infarction",
            "pneumonia",
            "breast cancer",
            "depression",
            "asthma",
            "chronic kidney disease",
            "stroke",
            "atrial fibrillation",
            "heart failure",
            "copd chronic obstructive pulmonary disease",
            "alzheimer disease",
            "sepsis",
            "covid-19"
        ]
        
        # Tier 3 keywords (meta-analyses, systematic reviews, guidelines)
        self.tier3_keywords = [
            "meta-analysis", "systematic review", "clinical practice guideline",
            "consensus statement", "practice parameter", "evidence-based",
            "cochrane review", "clinical guideline", "treatment guideline"
        ]
        
        # Tier 2 keywords (clinical trials, comparative studies)
        self.tier2_keywords = [
            "randomized controlled trial", "clinical trial", "comparative study",
            "prospective study", "cohort study", "diagnostic accuracy",
            "treatment protocol", "clinical effectiveness", "therapeutic"
        ]
        
        # High-impact medical journals (for evidence level)
        self.high_impact_journals = [
            "N Engl J Med", "Lancet", "JAMA", "BMJ", "Nature Medicine",
            "Cell", "Science", "Nature", "Circulation", "Journal of Clinical Oncology"
        ]

    def fetch_pubmed_foundation(self, max_abstracts: int = 3000) -> List[Dict]:
        """Fetch PubMed abstracts for foundation dataset."""
        logger.info(f"📚 FETCHING PUBMED FOUNDATION")
        logger.info("=" * 50)
        logger.info(f"Target abstracts: {max_abstracts}")
        logger.info(f"Topics: {len(self.medical_topics)} medical areas")
        
        all_documents = []
        abstracts_per_topic = max_abstracts // len(self.medical_topics)
        
        for topic in self.medical_topics:
            try:
                logger.info(f"🔍 Fetching: {topic} (max {abstracts_per_topic})")
                
                # Search for papers
                search_results = self._search_pubmed(topic, abstracts_per_topic)
                if not search_results:
                    continue
                
                # Fetch detailed abstracts
                abstracts = self._fetch_abstracts(search_results)
                
                # Process into foundation format
                documents = self._process_abstracts(abstracts, topic)
                all_documents.extend(documents)
                
                logger.info(f"✅ {topic}: {len(documents)} abstracts")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error fetching {topic}: {e}")
                continue
        
        logger.info(f"🎉 PUBMED FOUNDATION COMPLETE: {len(all_documents)} abstracts")
        return all_documents

    def _search_pubmed(self, topic: str, max_results: int) -> List[str]:
        """Search PubMed for topic and return PMIDs."""
        search_url = f"{self.base_url}/esearch.fcgi"
        
        params = {
            "db": "pubmed",
            "term": f'"{topic}"[MeSH Terms] OR "{topic}"[Title/Abstract]',
            "retmax": max_results,
            "retmode": "xml",
            "email": self.email,
            "sort": "relevance",
            "datetype": "pdat",
            "mindate": "2020/01/01",  # Recent papers for quality
            "maxdate": "2024/12/31"
        }
        
        try:
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            pmids = []
            
            for id_elem in root.findall(".//Id"):
                pmids.append(id_elem.text)
            
            return pmids
            
        except Exception as e:
            logger.error(f"Error searching PubMed for {topic}: {e}")
            return []

    def _fetch_abstracts(self, pmids: List[str]) -> List[Dict]:
        """Fetch detailed abstracts for PMIDs."""
        if not pmids:
            return []
        
        fetch_url = f"{self.base_url}/efetch.fcgi"
        
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "email": self.email
        }
        
        try:
            response = requests.get(fetch_url, params=params, timeout=60)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            abstracts = []
            
            for article in root.findall(".//PubmedArticle"):
                try:
                    abstract_data = self._parse_article(article)
                    if abstract_data:
                        abstracts.append(abstract_data)
                except Exception as e:
                    logger.debug(f"Error parsing article: {e}")
                    continue
            
            return abstracts
            
        except Exception as e:
            logger.error(f"Error fetching abstracts: {e}")
            return []

    def _parse_article(self, article) -> Optional[Dict]:
        """Parse individual PubMed article."""
        try:
            # Extract basic info
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else "unknown"
            
            # Title
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "No title"
            
            # Abstract
            abstract_elem = article.find(".//AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            if not abstract:
                return None  # Skip articles without abstracts
            
            # Journal
            journal_elem = article.find(".//Title")
            journal = journal_elem.text if journal_elem is not None else "Unknown"
            
            # Publication date
            pub_date = self._extract_publication_date(article)
            
            # Authors
            authors = self._extract_authors(article)
            
            # Publication types
            pub_types = self._extract_publication_types(article)
            
            # MeSH terms
            mesh_terms = self._extract_mesh_terms(article)
            
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "publication_date": pub_date,
                "authors": authors,
                "publication_types": pub_types,
                "mesh_terms": mesh_terms
            }
            
        except Exception as e:
            logger.debug(f"Error parsing article: {e}")
            return None

    def _extract_publication_date(self, article) -> str:
        """Extract publication date."""
        try:
            date_elem = article.find(".//PubDate")
            if date_elem is not None:
                year = date_elem.find("Year")
                month = date_elem.find("Month")
                
                if year is not None:
                    year_text = year.text
                    month_text = month.text if month is not None else "01"
                    return f"{year_text}-{month_text}-01"
            
            return "2023-01-01"  # Default
            
        except Exception:
            return "2023-01-01"

    def _extract_authors(self, article) -> List[str]:
        """Extract author names."""
        try:
            authors = []
            for author in article.findall(".//Author"):
                last_name = author.find("LastName")
                first_name = author.find("ForeName")
                
                if last_name is not None:
                    name = last_name.text
                    if first_name is not None:
                        name = f"{first_name.text} {name}"
                    authors.append(name)
            
            return authors[:5]  # Limit to first 5 authors
            
        except Exception:
            return []

    def _extract_publication_types(self, article) -> List[str]:
        """Extract publication types."""
        try:
            pub_types = []
            for pub_type in article.findall(".//PublicationType"):
                if pub_type.text:
                    pub_types.append(pub_type.text)
            return pub_types
            
        except Exception:
            return []

    def _extract_mesh_terms(self, article) -> List[str]:
        """Extract MeSH terms."""
        try:
            mesh_terms = []
            for mesh in article.findall(".//MeshHeading"):
                descriptor = mesh.find("DescriptorName")
                if descriptor is not None and descriptor.text:
                    mesh_terms.append(descriptor.text)
            return mesh_terms[:10]  # Limit to first 10
            
        except Exception:
            return []

    def _process_abstracts(self, abstracts: List[Dict], topic: str) -> List[Dict]:
        """Process abstracts into foundation format with tier assignment."""
        documents = []
        
        for abstract_data in abstracts:
            try:
                # Combine title and abstract
                text = f"{abstract_data['title']}\n\n{abstract_data['abstract']}"
                
                # Assign tier based on content and publication type
                tier = self._assign_tier(abstract_data)
                
                # Determine medical specialty from topic and MeSH terms
                specialty = self._determine_specialty(topic, abstract_data['mesh_terms'])
                
                # Determine evidence level
                evidence_level = self._determine_evidence_level(abstract_data)
                
                # Create foundation document
                document = {
                    "text": text,
                    "metadata": {
                        "source": "pubmed",
                        "tier": tier,
                        "medical_specialty": specialty,
                        "evidence_level": evidence_level,
                        "reasoning_type": "clinical_reasoning",
                        "organization": "NCBI",
                        "therapeutic_area": self._map_therapeutic_area(topic),
                        "doc_id": f"pmid_{abstract_data['pmid']}",
                        "title": abstract_data['title'],
                        "journal": abstract_data['journal'],
                        "publication_date": abstract_data['publication_date'],
                        "authors": abstract_data['authors'][:3],  # First 3 authors
                        "publication_types": abstract_data['publication_types'],
                        "mesh_terms": abstract_data['mesh_terms'][:5]  # First 5 MeSH terms
                    }
                }
                
                documents.append(document)
                
            except Exception as e:
                logger.debug(f"Error processing abstract {abstract_data.get('pmid', 'unknown')}: {e}")
                continue
        
        return documents

    def _assign_tier(self, abstract_data: Dict) -> int:
        """Assign hierarchical tier based on study type and content."""
        title_abstract = f"{abstract_data['title']} {abstract_data['abstract']}".lower()
        pub_types = [pt.lower() for pt in abstract_data['publication_types']]
        
        # Tier 3: Meta-analyses, systematic reviews, guidelines
        tier3_indicators = any(keyword in title_abstract for keyword in self.tier3_keywords)
        tier3_pub_types = any(pt in ["meta-analysis", "systematic review", "practice guideline"] for pt in pub_types)
        
        if tier3_indicators or tier3_pub_types:
            return 3
        
        # Tier 2: Clinical trials, comparative studies
        tier2_indicators = any(keyword in title_abstract for keyword in self.tier2_keywords)
        tier2_pub_types = any(pt in ["randomized controlled trial", "clinical trial", "comparative study"] for pt in pub_types)
        
        if tier2_indicators or tier2_pub_types:
            return 2
        
        # Tier 1: Case reports, basic research, observational studies
        return 1

    def _determine_specialty(self, topic: str, mesh_terms: List[str]) -> str:
        """Determine medical specialty from topic and MeSH terms."""
        # Map topics to specialties
        specialty_mapping = {
            "diabetes mellitus": "endocrinology",
            "hypertension": "cardiology",
            "myocardial infarction": "cardiology",
            "pneumonia": "pulmonology",
            "breast cancer": "oncology",
            "depression": "psychiatry",
            "asthma": "pulmonology",
            "chronic kidney disease": "nephrology",
            "stroke": "neurology",
            "atrial fibrillation": "cardiology",
            "heart failure": "cardiology",
            "copd chronic obstructive pulmonary disease": "pulmonology",
            "alzheimer disease": "neurology",
            "sepsis": "critical_care",
            "covid-19": "infectious_disease"
        }
        
        # Check direct mapping first
        specialty = specialty_mapping.get(topic, "general_medicine")
        
        # Refine based on MeSH terms
        mesh_specialties = {
            "cardiology": ["heart", "cardiac", "cardiovascular", "myocardial"],
            "oncology": ["cancer", "tumor", "neoplasm", "carcinoma"],
            "neurology": ["brain", "neurological", "nervous system", "stroke"],
            "endocrinology": ["diabetes", "hormone", "endocrine", "insulin"]
        }
        
        for spec, keywords in mesh_specialties.items():
            if any(keyword in " ".join(mesh_terms).lower() for keyword in keywords):
                specialty = spec
                break
        
        return specialty

    def _determine_evidence_level(self, abstract_data: Dict) -> str:
        """Determine evidence level based on journal and study type."""
        journal = abstract_data['journal'].lower()
        pub_types = [pt.lower() for pt in abstract_data['publication_types']]
        
        # High impact journal
        high_impact = any(hj.lower() in journal for hj in self.high_impact_journals)
        
        # High evidence study types
        high_evidence_types = ["meta-analysis", "systematic review", "randomized controlled trial"]
        high_evidence = any(het in pub_types for het in high_evidence_types)
        
        if high_impact and high_evidence:
            return "high"
        elif high_impact or high_evidence:
            return "medium"
        else:
            return "medium"  # Default to medium for PubMed (peer-reviewed)

    def _map_therapeutic_area(self, topic: str) -> str:
        """Map topic to therapeutic area."""
        area_mapping = {
            "diabetes mellitus": "endocrine",
            "hypertension": "cardiovascular",
            "myocardial infarction": "cardiovascular",
            "pneumonia": "respiratory",
            "breast cancer": "oncology",
            "depression": "mental_health",
            "asthma": "respiratory",
            "chronic kidney disease": "renal",
            "stroke": "neurological",
            "atrial fibrillation": "cardiovascular",
            "heart failure": "cardiovascular",
            "copd chronic obstructive pulmonary disease": "respiratory",
            "alzheimer disease": "neurological",
            "sepsis": "infectious_disease",
            "covid-19": "infectious_disease"
        }
        
        return area_mapping.get(topic, "general_medicine")