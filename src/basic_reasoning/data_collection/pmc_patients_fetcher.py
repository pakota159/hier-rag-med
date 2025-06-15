"""Fetcher for PMC Patients dataset."""

import logging
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from .base_fetcher import BaseFetcher, DatasetInfo, Document

logger = logging.getLogger(__name__)

class PMCPatientsFetcher(BaseFetcher):
    """Fetcher for PMC Patients dataset from PubMed Central."""
    
    def __init__(self, config: Dict[str, Any], checkpoint_dir: Optional[str] = None):
        """
        Initialize PMC Patients fetcher.
        
        Args:
            config: Dataset configuration
            checkpoint_dir: Optional directory for checkpoints
        """
        super().__init__(config, checkpoint_dir)
        self.rate_limit_delay = 1.0 / self.config.get("rate_limit", 3)  # Rate limit in requests per second
    
    def get_dataset_info(self) -> DatasetInfo:
        """
        Get information about the PMC Patients dataset.
        
        Returns:
            DatasetInfo object
        """
        return DatasetInfo(
            name="pmc_patients",
            source=self.config["source"],
            url=self.config["url"],
            format=self.config["format"],
            expected_size=self.config["expected_size"],
            description=self.config["description"],
            api_url=self.config["api_url"]
        )
    
    def _parse_pmc_article(self, article_xml: str) -> Optional[Dict[str, Any]]:
        """
        Parse PMC article XML into document format.
        
        Args:
            article_xml: XML string of the article
            
        Returns:
            Dictionary containing parsed article data or None if invalid
        """
        try:
            root = ET.fromstring(article_xml)
            
            # Extract basic metadata
            title = root.find(".//article-title")
            title = title.text if title is not None else ""
            
            authors = []
            for author in root.findall(".//contrib[@contrib-type='author']"):
                name = author.find("name")
                if name is not None:
                    given = name.find("given-names")
                    surname = name.find("surname")
                    if given is not None and surname is not None:
                        authors.append(f"{given.text} {surname.text}")
            
            pub_date = root.find(".//pub-date")
            date = ""
            if pub_date is not None:
                year = pub_date.find("year")
                month = pub_date.find("month")
                if year is not None:
                    date = year.text
                    if month is not None:
                        date += f"-{month.text}"
            
            # Extract case presentation
            case_presentation = root.find(".//sec[@sec-type='case-presentation']")
            if case_presentation is None:
                return None
            
            text = ""
            for p in case_presentation.findall(".//p"):
                if p.text:
                    text += p.text + "\n"
            
            if not text:
                return None
            
            return {
                "id": root.get("id", ""),
                "text": text.strip(),
                "metadata": {
                    "title": title,
                    "authors": authors,
                    "publication_date": date,
                    "medical_specialty": "Case Report",
                    "evidence_level": "Level 4",
                    "reasoning_chain": []
                },
                "source_dataset": "pmc_patients",
                "reasoning_type": "case_study"
            }
            
        except Exception as e:
            logger.error(f"Error parsing PMC article: {str(e)}")
            return None
    
    def fetch_documents(self, max_documents: Optional[int] = None) -> List[Document]:
        """
        Fetch patient case studies from PubMed Central.
        
        Args:
            max_documents: Optional maximum number of documents to fetch
            
        Returns:
            List of Document objects
        """
        # Try to load from checkpoint first
        checkpoint_docs = self._load_checkpoint("pmc_patients")
        if checkpoint_docs:
            logger.info(f"Loaded {len(checkpoint_docs)} documents from checkpoint")
            if max_documents:
                return checkpoint_docs[:max_documents]
            return checkpoint_docs
        
        documents = []
        try:
            # Search for case reports
            search_url = f"{self.config['api_url']}esearch.fcgi"
            search_params = {
                "db": "pmc",
                "term": "case report[Publication Type]",
                "retmax": max_documents if max_documents else self.config["expected_size"],
                "retmode": "json"
            }
            
            response = self._make_request(search_url, params=search_params)
            search_data = response.json()
            
            if "esearchresult" not in search_data or "idlist" not in search_data["esearchresult"]:
                raise ValueError("Invalid search response format")
            
            # Fetch full articles
            for pmcid in search_data["esearchresult"]["idlist"]:
                fetch_url = f"{self.config['api_url']}efetch.fcgi"
                fetch_params = {
                    "db": "pmc",
                    "id": pmcid,
                    "retmode": "xml"
                }
                
                response = self._make_request(fetch_url, params=fetch_params)
                article_data = self._parse_pmc_article(response.text)
                
                if article_data:
                    doc = Document(
                        text=article_data["text"],
                        metadata=article_data["metadata"],
                        source_dataset=article_data["source_dataset"],
                        doc_id=article_data["id"],
                        reasoning_type=article_data["reasoning_type"]
                    )
                    
                    if self._validate_document(doc):
                        documents.append(doc)
                    
                    # Save checkpoint every 100 documents
                    if len(documents) % 100 == 0:
                        self._save_checkpoint(documents, "pmc_patients")
                
                if max_documents and len(documents) >= max_documents:
                    break
            
            # Save final checkpoint
            self._save_checkpoint(documents, "pmc_patients")
            
        except Exception as e:
            logger.error(f"Error fetching PMC Patients data: {str(e)}")
            raise
        
        return documents 