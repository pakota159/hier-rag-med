#!/usr/bin/env python3
"""
IDSA Guidelines Fetcher (REAL DATA ONLY)
File: src/basic_reasoning/fetchers/idsa_guidelines_fetcher.py

Fetches real infectious disease guidelines and clinical recommendations
from IDSA (Infectious Diseases Society of America) and related sources.

This fetcher uses real medical literature that references IDSA guidelines
and infectious disease management protocols.
"""

import logging
import time
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class IDSAGuidelinesFetcher:
    """Fetcher for IDSA-style infectious disease guidelines using real data sources."""
    
    def __init__(self, email: str = "hierragmed@example.com"):
        self.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        # Real IDSA-related search terms for infectious diseases
        self.idsa_searches = [
            # IDSA Guidelines
            "IDSA[Author] AND guidelines",
            "Infectious Diseases Society[Author] AND guidelines",
            "practice guideline[Title] AND infectious disease",
            
            # Bacterial Infections
            "bacterial infections[MeSH] AND antibiotic therapy",
            "pneumonia bacterial[MeSH] AND treatment guidelines",
            "sepsis[MeSH] AND management protocol",
            "meningitis bacterial[MeSH] AND antibiotic treatment",
            "endocarditis[MeSH] AND antibiotic therapy",
            "urinary tract infections[MeSH] AND treatment",
            "skin infections[MeSH] AND antibiotic therapy",
            "osteomyelitis[MeSH] AND treatment guidelines",
            "bacteremia[MeSH] AND management",
            
            # Viral Infections
            "antiviral agents[MeSH] AND therapy",
            "influenza[MeSH] AND antiviral treatment",
            "hepatitis B[MeSH] AND antiviral therapy",
            "hepatitis C[MeSH] AND treatment guidelines",
            "herpes simplex[MeSH] AND antiviral treatment",
            "cytomegalovirus[MeSH] AND treatment",
            "HIV infections[MeSH] AND antiretroviral therapy",
            
            # Fungal Infections
            "antifungal agents[MeSH] AND therapy",
            "candidiasis[MeSH] AND antifungal treatment",
            "aspergillosis[MeSH] AND treatment guidelines",
            "cryptococcosis[MeSH] AND antifungal therapy",
            
            # Antimicrobial Resistance
            "drug resistance bacterial[MeSH] AND management",
            "MRSA[Title/Abstract] AND treatment",
            "vancomycin resistant enterococci[MeSH]",
            "carbapenem resistant[Title/Abstract] AND therapy",
            "antimicrobial stewardship[MeSH] AND programs",
            
            # Hospital-Acquired Infections
            "healthcare associated infections[MeSH] AND prevention",
            "nosocomial infections[MeSH] AND control",
            "catheter related infections[MeSH] AND management",
            "surgical site infections[MeSH] AND prevention",
            "ventilator associated pneumonia[MeSH] AND treatment",
            
            # Specific Pathogens
            "staphylococcal infections[MeSH] AND treatment",
            "streptococcal infections[MeSH] AND antibiotic therapy",
            "clostridium difficile[MeSH] AND treatment",
            "tuberculosis[MeSH] AND treatment guidelines",
            "malaria[MeSH] AND treatment",
            
            # Immunocompromised Infections
            "opportunistic infections[MeSH] AND treatment",
            "infections in immunocompromised[Title/Abstract]",
            "neutropenia[MeSH] AND infection management",
            "transplant infections[Title/Abstract] AND prevention",
            
            # Travel Medicine
            "travel medicine[MeSH] AND infectious disease",
            "tropical medicine[MeSH] AND infections",
            "vaccination[MeSH] AND travel",
            
            # Infection Control
            "infection control[MeSH] AND hospital",
            "isolation techniques[MeSH] AND infectious disease",
            "hand hygiene[MeSH] AND infection prevention"
        ]

    def fetch_idsa_guidelines(self, max_docs: int = 500) -> List[Dict]:
        """Fetch IDSA-style guidelines using real medical literature sources."""
        logger.info("🦠 Fetching IDSA infectious disease guidelines (REAL DATA)")
        logger.info(f"🦠 Target documents: {max_docs}")
        logger.info("📡 Data source: PubMed E-utilities API with IDSA-referenced content")
        
        all_documents = []
        docs_per_search = max(1, max_docs // len(self.idsa_searches))
        
        for search_query in self.idsa_searches:
            if len(all_documents) >= max_docs:
                break
                
            try:
                search_docs = self._fetch_idsa_abstracts(search_query, docs_per_search)
                all_documents.extend(search_docs)
                
                # Rate limiting for PubMed API
                time.sleep(1)
                
            except Exception as e:
                logger.debug(f"Failed IDSA search '{search_query}': {e}")
                continue
        
        logger.info(f"🎉 IDSA guidelines complete: {len(all_documents)} real infectious disease documents")
        return all_documents[:max_docs]

    def _fetch_idsa_abstracts(self, search_query: str, max_count: int) -> List[Dict]:
        """Fetch IDSA-related abstracts using PubMed API."""
        # Enhanced search with quality and recency filters
        enhanced_query = f'({search_query}) AND ("last 10 years"[PDat]) AND (hasabstract[text]) AND (english[lang])'
        
        # Search for PMIDs
        search_url = f"{self.base_url}esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": enhanced_query,
            "retmax": min(max_count * 2, 50),
            "email": self.email,
            "tool": "hierragmed_idsa",
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
            
            return self._fetch_idsa_abstracts_by_pmids(pmids[:max_count], search_query)
            
        except Exception as e:
            logger.debug(f"IDSA search failed for '{search_query}': {e}")
            return []

    def _fetch_idsa_abstracts_by_pmids(self, pmids: List[str], search_query: str) -> List[Dict]:
        """Fetch full abstracts for IDSA-related content."""
        if not pmids:
            return []
        
        # Fetch abstracts using PubMed API
        fetch_url = f"{self.base_url}efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
            "email": self.email,
            "tool": "hierragmed_idsa"
        }
        
        try:
            fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
            if fetch_response.status_code != 200:
                return []
            
            root = ET.fromstring(fetch_response.content)
            return self._parse_idsa_abstracts(root, search_query)
            
        except Exception as e:
            logger.debug(f"Failed to fetch IDSA abstracts: {e}")
            return []

    def _parse_idsa_abstracts(self, root: ET.Element, search_query: str) -> List[Dict]:
        """Parse IDSA-related abstracts with infectious disease focus."""
        documents = []
        
        for article in root.findall(".//PubmedArticle"):
            try:
                abstract_data = self._extract_idsa_metadata(article)
                if not abstract_data or not abstract_data.get("abstract"):
                    continue
                
                if len(abstract_data["abstract"]) < 120:
                    continue
                
                doc = self._create_idsa_document(abstract_data, search_query)
                documents.append(doc)
                
            except Exception as e:
                logger.debug(f"Failed to process IDSA article: {e}")
                continue
        
        return documents

    def _extract_idsa_metadata(self, article: ET.Element) -> Optional[Dict]:
        """Extract metadata from IDSA-related articles."""
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
            logger.debug(f"Failed to extract IDSA metadata: {e}")
            return None

    def _create_idsa_document(self, abstract_data: Dict, search_query: str) -> Dict:
        """Create document with IDSA/infectious disease specific metadata."""
        tier = self._determine_idsa_tier(abstract_data)
        evidence_level = self._determine_idsa_evidence_level(abstract_data)
        
        # Create infectious disease focused document text
        text = f"""
IDSA Infectious Disease Guidelines

Title: {abstract_data['title']}

Clinical Abstract: {abstract_data['abstract']}

Infectious Disease Clinical Information:
- Medical Specialty: Infectious Disease
- Therapeutic Area: Infectious Disease Management
- Evidence Level: {evidence_level}
- Guidelines Source: IDSA-referenced content

Publication Details:
- Journal: {abstract_data['journal']}
- Year: {abstract_data['year']}
- PMID: {abstract_data['pmid']}
- Publication Types: {', '.join(abstract_data['publication_types'])}

Infectious Disease Focus Areas:
- Antimicrobial Therapy
- Infection Prevention and Control
- Hospital-Acquired Infections
- Antimicrobial Resistance
- Immunocompromised Host Infections

MeSH Terms: {', '.join(abstract_data['mesh_terms'][:6])}{'...' if len(abstract_data['mesh_terms']) > 6 else ''}
"""

        return {
            "text": text.strip(),
            "metadata": {
                "doc_id": f"idsa_id_{abstract_data['pmid']}",
                "source": "idsa_guidelines",
                "title": abstract_data['title'],
                "pmid": abstract_data['pmid'],
                "journal": abstract_data['journal'],
                "year": abstract_data['year'],
                "medical_specialty": "infectious_disease",
                "therapeutic_area": "infectious_disease",
                "evidence_level": evidence_level,
                "tier": tier,
                "organization": "IDSA",
                "publication_types": abstract_data['publication_types'],
                "mesh_terms": abstract_data['mesh_terms'],
                "authors": abstract_data['authors'],
                "chunk_id": 0,
                "reasoning_type": "infectious_disease_guidelines",
                "data_source": "pubmed_idsa_referenced",
                "guideline_type": "infectious_disease"
            }
        }

    def _determine_idsa_tier(self, abstract_data: Dict) -> int:
        """Determine evidence tier for IDSA guidelines."""
        title_abstract = f"{abstract_data['title']} {abstract_data['abstract']}".lower()
        pub_types = [pt.lower() for pt in abstract_data['publication_types']]
        
        # Tier 3: IDSA guidelines and high-quality evidence
        idsa_indicators = ["idsa", "infectious diseases society", "practice guideline", "consensus", "recommendation"]
        tier3_pub_types = ["practice guideline", "guideline", "consensus development conference"]
        
        if (any(indicator in title_abstract for indicator in idsa_indicators) or
            any(pt in tier3_pub_types for pt in pub_types)):
            return 3
        
        # Tier 2: Clinical studies and reviews
        if any(pt in ["review", "clinical trial", "comparative study"] for pt in pub_types):
            return 2
        
        # Tier 1: Other research
        return 1

    def _determine_idsa_evidence_level(self, abstract_data: Dict) -> str:
        """Determine evidence level for IDSA content."""
        title_abstract = f"{abstract_data['title']} {abstract_data['abstract']}".lower()
        pub_types = [pt.lower() for pt in abstract_data['publication_types']]
        
        # IDSA guidelines typically represent high evidence
        if any(indicator in title_abstract for indicator in ["idsa", "infectious diseases society", "practice guideline"]):
            return "idsa_guideline"
        
        # High evidence from systematic reviews
        if any(pt in ["systematic review", "meta-analysis"] for pt in pub_types):
            return "high"
        
        # Medium evidence from clinical trials
        if any(pt in ["randomized controlled trial", "clinical trial"] for pt in pub_types):
            return "medium"
        
        return "standard"