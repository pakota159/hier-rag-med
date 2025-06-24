"""
NIH UMLS Metathesaurus Fetcher
File: src/basic_reasoning/fetchers/umls_fetcher.py

Fetches medical terminology and definitions from NIH UMLS Metathesaurus API.
Critical for medical terminology normalization and concept relationships.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
from datetime import datetime
import base64

logger = logging.getLogger(__name__)

class UMLSFetcher:
    """Fetch medical terminology from NIH UMLS Metathesaurus API."""
    
    def __init__(self, api_key: str, email: str = "hierragmed@example.com"):
        if not api_key:
            raise Exception("UMLS API key is required. Get one from: https://uts.nlm.nih.gov/uts/signup-login")
            
        self.source_name = "nih_umls_metathesaurus"
        self.api_key = api_key
        self.email = email
        self.base_url = "https://uts-ws.nlm.nih.gov/rest"
        self.auth_url = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
        
        # Rate limiting
        self.request_delay = 0.1  # 10 requests per second
        self.ticket = None
        self.ticket_expiry = None
        
    def _get_service_ticket(self) -> str:
        """Get service ticket for UMLS authentication."""
        current_time = time.time()
        
        # Check if we need a new ticket
        if self.ticket and self.ticket_expiry and current_time < self.ticket_expiry:
            return self.ticket
            
        logger.info("🔐 Authenticating with UMLS API")
        
        auth_params = {
            'apikey': self.api_key
        }
        
        response = requests.post(self.auth_url, data=auth_params, timeout=30)
        
        if response.status_code != 201:
            raise Exception(f"UMLS authentication failed: {response.status_code} - {response.text}")
            
        # Extract ticket from response headers
        ticket_url = response.headers.get('location')
        if not ticket_url:
            raise Exception("No ticket URL returned from UMLS authentication")
            
        self.ticket = ticket_url.split('ticket=')[1]
        self.ticket_expiry = current_time + 7200  # 2 hours
        
        return self.ticket
        
    def _make_request(self, url: str, params: Dict = None) -> requests.Response:
        """Make authenticated request to UMLS API."""
        time.sleep(self.request_delay)
        
        if params is None:
            params = {}
            
        # Add service ticket
        params['ticket'] = self._get_service_ticket()
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 401:
            # Ticket expired, get new one and retry
            self.ticket = None
            params['ticket'] = self._get_service_ticket()
            response = requests.get(url, params=params, timeout=30)
            
        if response.status_code != 200:
            raise Exception(f"UMLS API request failed: {response.status_code} - {response.text}")
            
        return response
        
    def search_concepts(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search UMLS concepts."""
        logger.info(f"🔍 Searching UMLS concepts for: {query}")
        
        search_url = f"{self.base_url}/search/current"
        params = {
            'string': query,
            'searchType': 'exact',
            'returnIdType': 'concept',
            'pageSize': min(max_results, 100)
        }
        
        response = self._make_request(search_url, params)
        data = response.json()
        
        if 'result' not in data or 'results' not in data['result']:
            raise Exception(f"Invalid UMLS search response: {data}")
            
        concepts = []
        for result in data['result']['results']:
            if 'ui' not in result or 'name' not in result:
                continue
                
            concepts.append({
                'cui': result['ui'],
                'name': result['name'],
                'semantic_types': result.get('semanticTypes', [])
            })
            
        logger.info(f"📖 Found {len(concepts)} UMLS concepts")
        return concepts
        
    def get_concept_details(self, cui: str) -> Dict:
        """Get detailed information for a UMLS concept."""
        concept_url = f"{self.base_url}/content/current/CUI/{cui}"
        
        response = self._make_request(concept_url)
        data = response.json()
        
        if 'result' not in data:
            raise Exception(f"Invalid UMLS concept response for {cui}: {data}")
            
        concept_data = data['result']
        
        # Get definitions
        definitions = self._get_concept_definitions(cui)
        
        # Get relations
        relations = self._get_concept_relations(cui)
        
        return {
            'cui': cui,
            'name': concept_data.get('name', ''),
            'semantic_types': concept_data.get('semanticTypes', []),
            'definitions': definitions,
            'relations': relations,
            'atoms': concept_data.get('atoms', ''),
            'source_vocabularies': concept_data.get('sourceVocabularies', [])
        }
        
    def _get_concept_definitions(self, cui: str) -> List[Dict]:
        """Get definitions for a UMLS concept."""
        def_url = f"{self.base_url}/content/current/CUI/{cui}/definitions"
        
        try:
            response = self._make_request(def_url)
            data = response.json()
            
            if 'result' not in data:
                return []
                
            definitions = []
            for definition in data['result']:
                if 'value' in definition:
                    definitions.append({
                        'definition': definition['value'],
                        'source': definition.get('rootSource', 'Unknown')
                    })
                    
            return definitions
        except:
            return []
            
    def _get_concept_relations(self, cui: str, max_relations: int = 10) -> List[Dict]:
        """Get relations for a UMLS concept."""
        rel_url = f"{self.base_url}/content/current/CUI/{cui}/relations"
        params = {'pageSize': max_relations}
        
        try:
            response = self._make_request(rel_url, params)
            data = response.json()
            
            if 'result' not in data:
                return []
                
            relations = []
            for relation in data['result']:
                if 'relatedId' in relation and 'relationLabel' in relation:
                    relations.append({
                        'related_cui': relation['relatedId'],
                        'relation_type': relation['relationLabel'],
                        'related_name': relation.get('relatedIdName', '')
                    })
                    
            return relations
        except:
            return []
        
    def fetch_umls_terminology(self, max_results: int = 1000) -> List[Dict]:
        """Fetch UMLS medical terminology."""
        logger.info(f"📚 Fetching UMLS terminology (max {max_results})")
        
        # Core medical terminology searches
        medical_terms = [
            "diabetes mellitus",
            "hypertension", 
            "myocardial infarction",
            "pneumonia",
            "heart failure",
            "asthma",
            "chronic kidney disease",
            "stroke",
            "sepsis",
            "depression",
            "cancer",
            "infection",
            "inflammation",
            "drug therapy",
            "diagnostic procedure",
            "surgical procedure",
            "symptom",
            "sign",
            "syndrome",
            "disease"
        ]
        
        all_concepts = []
        concepts_per_term = max_results // len(medical_terms)
        
        for term in medical_terms:
            try:
                # Search for concepts
                concepts = self.search_concepts(term, concepts_per_term)
                
                if not concepts:
                    logger.warning(f"No concepts found for term: {term}")
                    continue
                    
                # Get detailed information for each concept
                for concept in concepts:
                    try:
                        detailed_concept = self.get_concept_details(concept['cui'])
                        all_concepts.append(detailed_concept)
                        
                        if len(all_concepts) >= max_results:
                            break
                            
                    except Exception as e:
                        logger.error(f"Failed to get details for concept {concept['cui']}: {e}")
                        raise  # Stop execution on any failure
                        
                if len(all_concepts) >= max_results:
                    break
                    
            except Exception as e:
                logger.error(f"Failed to fetch concepts for term '{term}': {e}")
                raise  # Stop execution on any failure
                
        if not all_concepts:
            raise Exception("No UMLS concepts were successfully fetched")
            
        # Convert to required format
        documents = []
        for concept in all_concepts[:max_results]:
            try:
                # Determine medical specialty from semantic types
                specialty = self._determine_specialty_from_semantic_types(concept['semantic_types'])
                
                # Format definitions
                definitions_text = ""
                if concept['definitions']:
                    definitions_text = "\n\n".join([
                        f"Definition ({defn['source']}): {defn['definition']}"
                        for defn in concept['definitions']
                    ])
                
                # Format relations
                relations_text = ""
                if concept['relations']:
                    relations_text = "\n\n".join([
                        f"{rel['relation_type']}: {rel['related_name']} (CUI: {rel['related_cui']})"
                        for rel in concept['relations'][:5]  # Limit to top 5 relations
                    ])
                
                # Create document text
                text = f"""UMLS Medical Concept: {concept['name']}

Concept Unique Identifier (CUI): {concept['cui']}

Semantic Types: {', '.join([st.get('name', str(st)) for st in concept['semantic_types']]) if concept['semantic_types'] else 'Not specified'}

{definitions_text}

Related Concepts:
{relations_text}

Source Vocabularies: {', '.join(concept['source_vocabularies']) if concept['source_vocabularies'] else 'Not specified'}

Clinical Context:
This UMLS concept provides standardized medical terminology used across healthcare systems. UMLS integrates over 200 biomedical vocabularies and standards to enable interoperability and consistent meaning in medical informatics applications."""

                doc = {
                    "text": text,
                    "metadata": {
                        "title": concept['name'],
                        "source": self.source_name,
                        "medical_specialty": specialty,
                        "evidence_level": "high",
                        "publication_date": datetime.now().strftime("%Y-%m-%d"),
                        "cui": concept['cui'],
                        "semantic_types": concept['semantic_types'],
                        "source_vocabularies": concept['source_vocabularies'],
                        "tier": 2,  # Hypothesis Testing
                        "chunk_id": 0
                    }
                }
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Failed to process concept {concept.get('cui', 'unknown')}: {e}")
                raise  # Stop execution on any failure
                
        logger.info(f"📚 UMLS fetch complete: {len(documents)} documents")
        return documents
        
    def _determine_specialty_from_semantic_types(self, semantic_types: List[Dict]) -> str:
        """Determine medical specialty from UMLS semantic types."""
        if not semantic_types:
            return "General Medicine"
            
        # Map semantic types to specialties
        specialty_mapping = {
            "Disease or Syndrome": "Internal Medicine",
            "Neoplastic Process": "Oncology", 
            "Mental or Behavioral Dysfunction": "Psychiatry",
            "Congenital Abnormality": "Pediatrics",
            "Acquired Abnormality": "Internal Medicine",
            "Pathologic Function": "Pathology",
            "Pharmacologic Substance": "Pharmacology",
            "Therapeutic or Preventive Procedure": "General Medicine",
            "Diagnostic Procedure": "Radiology",
            "Laboratory Procedure": "Laboratory Medicine",
            "Body Part, Organ, or Organ Component": "Anatomy",
            "Organism Function": "Physiology"
        }
        
        for st in semantic_types:
            st_name = st.get('name', str(st))
            if st_name in specialty_mapping:
                return specialty_mapping[st_name]
                
        return "General Medicine"