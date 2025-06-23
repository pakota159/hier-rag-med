#!/usr/bin/env python3
"""
ACOG Guidelines Fetcher (REAL DATA ONLY)
File: src/basic_reasoning/fetchers/acog_guidelines_fetcher.py

Fetches real obstetrics and gynecology guidelines and clinical recommendations
from ACOG (American College of Obstetricians and Gynecologists) and related sources.

This fetcher simulates access to ACOG's practice bulletins and committee opinions
using publicly available medical literature that references ACOG guidelines.
"""

import logging
import time
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ACOGGuidelinesFetcher:
    """Fetcher for ACOG-style obstetrics and gynecology guidelines using real data sources."""
    
    def __init__(self, email: str = "hierragmed@example.com"):
        self.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        # Real ACOG-related search terms for obstetrics and gynecology
        self.acog_searches = [
            # Pregnancy and Obstetrics
            "ACOG[Author] AND pregnancy complications",
            "practice bulletin[Title] AND obstetric",
            "committee opinion[Title] AND obstetric", 
            "preeclampsia[MeSH] AND guidelines",
            "gestational diabetes[MeSH] AND management",
            "pregnancy[MeSH] AND prenatal care",
            "labor delivery[MeSH] AND guidelines",
            "cesarean section[MeSH] AND indications",
            "postpartum hemorrhage[MeSH] AND management",
            "fetal monitoring[MeSH] AND guidelines",
            "pregnancy high risk[MeSH] AND management",
            "breech presentation[MeSH] AND delivery",
            "placenta previa[MeSH] AND management",
            "maternal mortality[MeSH] AND prevention",
            
            # Gynecology
            "gynecology[MeSH] AND guidelines",
            "contraception[MeSH] AND methods",
            "menstrual disorders[MeSH] AND treatment",
            "endometriosis[MeSH] AND therapy",
            "ovarian cysts[MeSH] AND management",
            "cervical cancer[MeSH] AND screening",
            "breast cancer[MeSH] AND screening",
            "osteoporosis[MeSH] AND postmenopausal",
            "hormone replacement therapy[MeSH] AND guidelines",
            "pelvic inflammatory disease[MeSH] AND treatment",
            "infertility[MeSH] AND management",
            "gynecologic surgical procedures[MeSH] AND complications",
            "uterine fibroids[MeSH] AND treatment",
            "polycystic ovary syndrome[MeSH] AND therapy",
            
            # Women's Health
            "women's health[MeSH] AND guidelines",
            "reproductive health[MeSH] AND services",
            "family planning[MeSH] AND services",
            "sexual health[MeSH] AND women",
            "domestic violence[MeSH] AND screening",
            "postmenopausal[MeSH] AND health",
            
            # Maternal-Fetal Medicine
            "maternal fetal medicine[MeSH] AND guidelines",
            "fetal abnormalities[MeSH] AND diagnosis",
            "genetic counseling[MeSH] AND pregnancy",
            "amniocentesis[MeSH] AND indications",
            "ultrasound prenatal[MeSH] AND guidelines"
        ]

    def fetch_acog_guidelines(self, max_docs: int = 500) -> List[Dict]:
        """Fetch ACOG-style guidelines using real medical literature sources."""
        logger.info("🤰 Fetching ACOG obstetrics/gynecology guidelines (REAL DATA)")
        logger.info(f"🤰 Target documents: {max_docs}")
        logger.info("📡 Data source: PubMed E-utilities API with ACOG-referenced content")
        
        all_documents = []
        docs_per_search = max(1, max_docs // len(self.acog_searches))
        
        for search_query in self.acog_searches:
            if len(all_documents) >= max_docs:
                break
                
            try:
                search_docs = self._fetch_acog_abstracts(search_query, docs_per_search)
                all_documents.extend(search_docs)
                
                # Rate limiting for PubMed API
                time.sleep(1)
                
            except Exception as e:
                logger.debug(f"Failed ACOG search '{search_query}': {e}")
                continue
        
        logger.info(f"🎉 ACOG guidelines complete: {len(all_documents)} real OB/GYN documents")
        return all_documents[:max_docs]

    def _fetch_acog_abstracts(self, search_query: str, max_count: int) -> List[Dict]:
        """Fetch ACOG-related abstracts using PubMed API."""
        # Enhanced search with quality and recency filters
        enhanced_query = f'({search_query}) AND ("last 15 years"[PDat]) AND (hasabstract[text]) AND (english[lang])'
        
        # Search for PMIDs
        search_url = f"{self.base_url}esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": enhanced_query,
            "retmax": min(max_count * 2, 50),
            "email": self.email,
            "tool": "hierragmed_acog",
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
            
            return self._fetch_acog_abstracts_by_pmids(pmids[:max_count], search_query)
            
        except Exception as e:
            logger.debug(f"ACOG search failed for '{search_query}': {e}")
            return []

    def _fetch_acog_abstracts_by_pmids(self, pmids: List[str], search_query: str) -> List[Dict]:
        """Fetch full abstracts for ACOG-related content."""
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
            "tool": "hierragmed_acog"
        }
        
        try:
            fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
            if fetch_response.status_code != 200:
                return []
            
            root = ET.fromstring(fetch_response.content)
            return self._parse_acog_abstracts(root, search_query)
            
        except Exception as e:
            logger.debug(f"Failed to fetch ACOG abstracts: {e}")
            return []

    def _parse_acog_abstracts(self, root: ET.Element, search_query: str) -> List[Dict]:
        """Parse ACOG-related abstracts with obstetrics/gynecology focus."""
        documents = []
        
        for article in root.findall(".//PubmedArticle"):
            try:
                abstract_data = self._extract_acog_metadata(article)
                if not abstract_data or not abstract_data.get("abstract"):
                    continue
                
                if len(abstract_data["abstract"]) < 120:
                    continue
                
                doc = self._create_acog_document(abstract_data, search_query)
                documents.append(doc)
                
            except Exception as e:
                logger.debug(f"Failed to process ACOG article: {e}")
                continue
        
        return documents

    def _extract_acog_metadata(self, article: ET.Element) -> Optional[Dict]:
        """Extract metadata from ACOG-related articles."""
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
            logger.debug(f"Failed to extract ACOG metadata: {e}")
            return None

    def _create_acog_document(self, abstract_data: Dict, search_query: str) -> Dict:
        """Create document with ACOG/OB-GYN specific metadata."""
        tier = self._determine_acog_tier(abstract_data)
        evidence_level = self._determine_acog_evidence_level(abstract_data)
        
        # Create OB/GYN focused document text
        text = f"""
ACOG Obstetrics & Gynecology Guidelines

Title: {abstract_data['title']}

Clinical Abstract: {abstract_data['abstract']}

OB/GYN Clinical Information:
- Medical Specialty: Obstetrics and Gynecology
- Therapeutic Area: Obstetric and Gynecologic Care
- Evidence Level: {evidence_level}
- Guidelines Source: ACOG-referenced content

Publication Details:
- Journal: {abstract_data['journal']}
- Year: {abstract_data['year']}
- PMID: {abstract_data['pmid']}
- Publication Types: {', '.join(abstract_data['publication_types'])}

Women's Health Focus Areas:
- Pregnancy and Maternal Care
- Gynecologic Health and Procedures
- Women's Preventive Care
- Reproductive Health Services

MeSH Terms: {', '.join(abstract_data['mesh_terms'][:6])}{'...' if len(abstract_data['mesh_terms']) > 6 else ''}
"""

        return {
            "text": text.strip(),
            "metadata": {
                "doc_id": f"acog_obgyn_{abstract_data['pmid']}",
                "source": "acog_guidelines",
                "title": abstract_data['title'],
                "pmid": abstract_data['pmid'],
                "journal": abstract_data['journal'],
                "year": abstract_data['year'],
                "medical_specialty": "obstetrics_gynecology",
                "therapeutic_area": "obstetric_gynecologic",
                "evidence_level": evidence_level,
                "tier": tier,
                "organization": "ACOG",
                "publication_types": abstract_data['publication_types'],
                "mesh_terms": abstract_data['mesh_terms'],
                "authors": abstract_data['authors'],
                "chunk_id": 0,
                "reasoning_type": "obstetric_gynecologic_guidelines",
                "data_source": "pubmed_acog_referenced",
                "guideline_type": "obstetric_gynecologic"
            }
        }

    def _determine_acog_tier(self, abstract_data: Dict) -> int:
        """Determine evidence tier for ACOG guidelines."""
        title_abstract = f"{abstract_data['title']} {abstract_data['abstract']}".lower()
        pub_types = [pt.lower() for pt in abstract_data['publication_types']]
        
        # Tier 3: ACOG guidelines and high-quality evidence
        acog_indicators = ["acog", "practice bulletin", "committee opinion", "guideline", "recommendation"]
        tier3_pub_types = ["practice guideline", "guideline", "consensus development conference"]
        
        if (any(indicator in title_abstract for indicator in acog_indicators) or
            any(pt in tier3_pub_types for pt in pub_types)):
            return 3
        
        # Tier 2: Clinical studies and reviews
        if any(pt in ["review", "clinical trial", "comparative study"] for pt in pub_types):
            return 2
        
        # Tier 1: Other research
        return 1

    def _determine_acog_evidence_level(self, abstract_data: Dict) -> str:
        """Determine evidence level for ACOG content."""
        title_abstract = f"{abstract_data['title']} {abstract_data['abstract']}".lower()
        pub_types = [pt.lower() for pt in abstract_data['publication_types']]
        
        # ACOG guidelines typically represent high evidence
        if any(indicator in title_abstract for indicator in ["acog", "practice bulletin", "committee opinion"]):
            return "acog_guideline"
        
        # High evidence from systematic reviews
        if any(pt in ["systematic review", "meta-analysis"] for pt in pub_types):
            return "high"
        
        # Medium evidence from clinical trials
        if any(pt in ["randomized controlled trial", "clinical trial"] for pt in pub_types):
            return "medium"
        
        return "standard"