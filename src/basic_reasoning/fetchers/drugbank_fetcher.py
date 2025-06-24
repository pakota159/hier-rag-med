"""
DrugBank.ca Official API Fetcher
File: src/basic_reasoning/fetchers/drugbank_fetcher.py

Fetches drug information including doses, mechanisms, and interactions from DrugBank.ca API.
Critical for pharmacological knowledge and drug safety information.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

class DrugBankFetcher:
    """Fetch drug information from DrugBank.ca official API."""
    
    def __init__(self, api_key: str, email: str = "hierragmed@example.com"):
        if not api_key:
            raise Exception("DrugBank API key is required. Get one from: https://go.drugbank.com/releases/latest")
            
        self.source_name = "drugbank_official"
        self.api_key = api_key
        self.email = email
        self.base_url = "https://go.drugbank.com/api/v1"
        
        # Rate limiting - DrugBank allows 300 requests per minute
        self.request_delay = 0.2  # 5 requests per second
        
        # Setup authentication headers
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': f'HierRAGMed/{email}'
        }
        
    def _make_request(self, endpoint: str, params: Dict = None) -> requests.Response:
        """Make authenticated request to DrugBank API."""
        time.sleep(self.request_delay)
        
        url = f"{self.base_url}/{endpoint}"
        
        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        
        if response.status_code == 401:
            raise Exception("DrugBank API authentication failed. Check your API key.")
        elif response.status_code == 429:
            raise Exception("DrugBank API rate limit exceeded. Reduce request frequency.")
        elif response.status_code != 200:
            raise Exception(f"DrugBank API request failed: {response.status_code} - {response.text}")
            
        return response
        
    def search_drugs(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search DrugBank for drugs."""
        logger.info(f"🔍 Searching DrugBank for: {query}")
        
        params = {
            'q': query,
            'size': min(max_results, 100)
        }
        
        response = self._make_request('drugs/search', params)
        data = response.json()
        
        if 'data' not in data:
            raise Exception(f"Invalid DrugBank search response: {data}")
            
        drugs = []
        for drug in data['data']:
            if 'id' not in drug or 'attributes' not in drug:
                continue
                
            attributes = drug['attributes']
            drugs.append({
                'drugbank_id': drug['id'],
                'name': attributes.get('name', ''),
                'description': attributes.get('description', ''),
                'indication': attributes.get('indication', ''),
                'pharmacodynamics': attributes.get('pharmacodynamics', ''),
                'mechanism_of_action': attributes.get('mechanism-of-action', ''),
                'toxicity': attributes.get('toxicity', ''),
                'half_life': attributes.get('half-life', ''),
                'protein_binding': attributes.get('protein-binding', ''),
                'route_of_elimination': attributes.get('route-of-elimination', ''),
                'dosage_forms': attributes.get('dosage-forms', []),
                'drug_interactions': attributes.get('drug-interactions', [])
            })
            
        logger.info(f"💊 Found {len(drugs)} drugs")
        return drugs
        
    def get_drug_details(self, drugbank_id: str) -> Dict:
        """Get detailed information for a specific drug."""
        response = self._make_request(f'drugs/{drugbank_id}')
        data = response.json()
        
        if 'data' not in data or 'attributes' not in data['data']:
            raise Exception(f"Invalid DrugBank drug response for {drugbank_id}: {data}")
            
        drug_data = data['data']['attributes']
        
        # Get additional details
        categories = self._get_drug_categories(drugbank_id)
        targets = self._get_drug_targets(drugbank_id)
        
        return {
            'drugbank_id': drugbank_id,
            'name': drug_data.get('name', ''),
            'description': drug_data.get('description', ''),
            'indication': drug_data.get('indication', ''),
            'pharmacodynamics': drug_data.get('pharmacodynamics', ''),
            'mechanism_of_action': drug_data.get('mechanism-of-action', ''),
            'toxicity': drug_data.get('toxicity', ''),
            'half_life': drug_data.get('half-life', ''),
            'protein_binding': drug_data.get('protein-binding', ''),
            'route_of_elimination': drug_data.get('route-of-elimination', ''),
            'dosage_forms': drug_data.get('dosage-forms', []),
            'drug_interactions': drug_data.get('drug-interactions', []),
            'categories': categories,
            'targets': targets,
            'cas_number': drug_data.get('cas-number', ''),
            'unii': drug_data.get('unii', ''),
            'average_mass': drug_data.get('average-mass', ''),
            'monoisotopic_mass': drug_data.get('monoisotopic-mass', '')
        }
        
    def _get_drug_categories(self, drugbank_id: str) -> List[Dict]:
        """Get drug categories."""
        try:
            response = self._make_request(f'drugs/{drugbank_id}/categories')
            data = response.json()
            
            if 'data' not in data:
                return []
                
            categories = []
            for category in data['data']:
                if 'attributes' in category:
                    categories.append({
                        'category': category['attributes'].get('category', ''),
                        'mesh_id': category['attributes'].get('mesh-id', ''),
                        'description': category['attributes'].get('description', '')
                    })
                    
            return categories
        except:
            return []
            
    def _get_drug_targets(self, drugbank_id: str) -> List[Dict]:
        """Get drug targets."""
        try:
            response = self._make_request(f'drugs/{drugbank_id}/targets')
            data = response.json()
            
            if 'data' not in data:
                return []
                
            targets = []
            for target in data['data']:
                if 'attributes' in target:
                    targets.append({
                        'name': target['attributes'].get('name', ''),
                        'organism': target['attributes'].get('organism', ''),
                        'actions': target['attributes'].get('actions', []),
                        'pharmacologically_active': target['attributes'].get('pharmacologically-active', False)
                    })
                    
            return targets
        except:
            return []
        
    def fetch_drugbank_data(self, max_results: int = 1000) -> List[Dict]:
        """Fetch DrugBank drug information."""
        logger.info(f"💊 Fetching DrugBank data (max {max_results})")
        
        # Core drug categories and classes
        drug_searches = [
            "antibiotics",
            "antihypertensives", 
            "antidiabetics",
            "analgesics",
            "antidepressants",
            "anticoagulants",
            "beta blockers",
            "ACE inhibitors",
            "statins",
            "diuretics",
            "anticonvulsants",
            "antipsychotics",
            "bronchodilators",
            "corticosteroids",
            "antihistamines",
            "proton pump inhibitors",
            "antivirals",
            "antifungals",
            "chemotherapy",
            "immunosuppressants"
        ]
        
        all_drugs = []
        drugs_per_search = max_results // len(drug_searches)
        
        for search_term in drug_searches:
            try:
                # Search for drugs
                drugs = self.search_drugs(search_term, drugs_per_search)
                
                if not drugs:
                    logger.warning(f"No drugs found for search: {search_term}")
                    continue
                    
                # Get detailed information for each drug
                for drug in drugs:
                    try:
                        detailed_drug = self.get_drug_details(drug['drugbank_id'])
                        all_drugs.append(detailed_drug)
                        
                        if len(all_drugs) >= max_results:
                            break
                            
                    except Exception as e:
                        logger.error(f"Failed to get details for drug {drug['drugbank_id']}: {e}")
                        raise  # Stop execution on any failure
                        
                if len(all_drugs) >= max_results:
                    break
                    
            except Exception as e:
                logger.error(f"Failed to fetch drugs for search '{search_term}': {e}")
                raise  # Stop execution on any failure
                
        if not all_drugs:
            raise Exception("No DrugBank drugs were successfully fetched")
            
        # Convert to required format
        documents = []
        for drug in all_drugs[:max_results]:
            try:
                # Determine medical specialty from categories
                specialty = self._determine_specialty_from_categories(drug['categories'])
                
                # Format drug interactions
                interactions_text = ""
                if drug['drug_interactions']:
                    interactions_text = "\n".join([
                        f"• {interaction.get('name', str(interaction))}: {interaction.get('description', 'Interaction noted')}"
                        for interaction in drug['drug_interactions'][:5]  # Limit to top 5
                    ])
                
                # Format targets
                targets_text = ""
                if drug['targets']:
                    targets_text = "\n".join([
                        f"• {target['name']} ({target['organism']}): {', '.join(target['actions'])}"
                        for target in drug['targets'][:3]  # Limit to top 3
                    ])
                
                # Format categories
                categories_text = ""
                if drug['categories']:
                    categories_text = ", ".join([cat['category'] for cat in drug['categories']])
                
                # Create document text
                text = f"""DrugBank Drug Information: {drug['name']}

DrugBank ID: {drug['drugbank_id']}
CAS Number: {drug['cas_number']}
UNII: {drug['unii']}

Description:
{drug['description']}

Clinical Indication:
{drug['indication']}

Mechanism of Action:
{drug['mechanism_of_action']}

Pharmacodynamics:
{drug['pharmacodynamics']}

Pharmacokinetics:
• Half-life: {drug['half_life']}
• Protein binding: {drug['protein_binding']}
• Route of elimination: {drug['route_of_elimination']}

Drug Categories:
{categories_text}

Molecular Targets:
{targets_text}

Drug Interactions:
{interactions_text}

Toxicity Information:
{drug['toxicity']}

Available Dosage Forms:
{', '.join([form.get('form', str(form)) for form in drug['dosage_forms']]) if drug['dosage_forms'] else 'Not specified'}

Clinical Context:
This DrugBank entry provides comprehensive pharmacological information for healthcare professionals. DrugBank is a comprehensive, freely accessible online database containing information on drugs and drug targets."""

                doc = {
                    "text": text,
                    "metadata": {
                        "title": drug['name'],
                        "source": self.source_name,
                        "medical_specialty": specialty,
                        "evidence_level": "high",
                        "publication_date": datetime.now().strftime("%Y-%m-%d"),
                        "drugbank_id": drug['drugbank_id'],
                        "cas_number": drug['cas_number'],
                        "unii": drug['unii'],
                        "drug_categories": drug['categories'],
                        "tier": 1,  # Pattern Recognition
                        "chunk_id": 0
                    }
                }
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Failed to process drug {drug.get('drugbank_id', 'unknown')}: {e}")
                raise  # Stop execution on any failure
                
        logger.info(f"💊 DrugBank fetch complete: {len(documents)} documents")
        return documents
        
    def _determine_specialty_from_categories(self, categories: List[Dict]) -> str:
        """Determine medical specialty from drug categories."""
        if not categories:
            return "Pharmacology"
            
        # Map drug categories to medical specialties
        specialty_mapping = {
            "Cardiovascular": "Cardiology",
            "Antihypertensive": "Cardiology", 
            "Anticoagulant": "Hematology",
            "Antidiabetic": "Endocrinology",
            "Antibiotic": "Infectious Disease",
            "Antiviral": "Infectious Disease",
            "Antifungal": "Infectious Disease",
            "Analgesic": "Pain Medicine",
            "Anti-inflammatory": "Rheumatology",
            "Antidepressant": "Psychiatry",
            "Antipsychotic": "Psychiatry",
            "Anticonvulsant": "Neurology",
            "Bronchodilator": "Pulmonology",
            "Corticosteroid": "Endocrinology",
            "Immunosuppressant": "Immunology",
            "Chemotherapy": "Oncology",
            "Antineoplastic": "Oncology"
        }
        
        for category in categories:
            cat_name = category.get('category', '')
            for key, specialty in specialty_mapping.items():
                if key.lower() in cat_name.lower():
                    return specialty
                    
        return "Pharmacology"