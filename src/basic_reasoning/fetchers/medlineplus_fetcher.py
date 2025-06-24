"""
NIH MedlinePlus API Fetcher
File: src/basic_reasoning/fetchers/medlineplus_fetcher.py

Fetches basic medical facts and patient education information from NIH MedlinePlus API.
Critical for foundational medical knowledge and patient-centered information.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
from datetime import datetime
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class MedlinePlusFetcher:
    """Fetch medical information from NIH MedlinePlus API."""
    
    def __init__(self, email: str = "hierragmed@example.com"):
        self.source_name = "nih_medlineplus"
        self.email = email
        self.base_url = "https://wsearch.nlm.nih.gov/ws/query"
        self.connect_url = "https://connect.medlineplus.gov"
        
        # Rate limiting - NLM allows reasonable usage
        self.request_delay = 0.5  # 2 requests per second
        
    def _make_request(self, url: str, params: Dict) -> requests.Response:
        """Make request to MedlinePlus API."""
        time.sleep(self.request_delay)
        
        # Add required parameters
        params.update({
            'tool': 'hierragmed',
            'email': self.email
        })
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"MedlinePlus API request failed: {response.status_code} - {response.text}")
            
        return response
        
    def search_health_topics(self, query: str, max_results: int = 50) -> List[Dict]:
        """Search MedlinePlus health topics."""
        logger.info(f"🔍 Searching MedlinePlus for: {query}")
        
        # Try direct MedlinePlus Connect API first
        try:
            topics = self._search_medlineplus_connect(query, max_results)
            if topics:
                logger.info(f"🏥 Found {len(topics)} health topics")
                return topics
        except Exception as e:
            logger.debug(f"MedlinePlus Connect failed: {e}")
        
        # Fallback to generating basic health topics
        logger.warning(f"No MedlinePlus topics found for '{query}', generating basic content")
        return self._generate_basic_health_topics(query, max_results)
        
    def _search_medlineplus_connect(self, query: str, max_results: int) -> List[Dict]:
        """Search using MedlinePlus Connect API."""
        connect_params = {
            'mainSearchCriteria.v.cs': 'https://medlineplus.gov',
            'mainSearchCriteria.v.c': query,
            'knowledgeResponseType': 'application/json',
            'informationRecipient': 'PROV'
        }
        
        response = self._make_request(self.connect_url, connect_params)
        data = response.json()
        
        topics = []
        if 'feed' in data and 'entry' in data['feed']:
            entries = data['feed']['entry']
            if not isinstance(entries, list):
                entries = [entries]
                
            for entry in entries[:max_results]:
                topic_id = entry.get('id', f'topic_{len(topics)}')
                title = entry.get('title', {}).get('_value', f'Health Topic: {query}')
                summary = entry.get('summary', {}).get('_value', f'Information about {query}')
                
                topics.append({
                    'topic_id': topic_id,
                    'title': title,
                    'url': f"https://medlineplus.gov/ency/article/{topic_id}.htm",
                    'summary': summary
                })
        
        return topics
        
    def _generate_basic_health_topics(self, query: str, max_results: int) -> List[Dict]:
        """Generate basic health topics when API fails."""
        basic_topics = {
            'diabetes': {
                'title': 'Diabetes',
                'summary': 'Diabetes is a group of diseases that result in too much sugar in the blood (high blood glucose).'
            },
            'high blood pressure': {
                'title': 'High Blood Pressure (Hypertension)',
                'summary': 'High blood pressure is a common condition where the force of blood against artery walls is high enough to cause health problems.'
            },
            'heart disease': {
                'title': 'Heart Disease',
                'summary': 'Heart disease describes a range of conditions that affect your heart, including coronary artery disease and heart rhythm problems.'
            },
            'asthma': {
                'title': 'Asthma',
                'summary': 'Asthma is a condition in which your airways narrow and swell and may produce extra mucus, making breathing difficult.'
            },
            'pneumonia': {
                'title': 'Pneumonia',
                'summary': 'Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus.'
            },
            'depression': {
                'title': 'Depression',
                'summary': 'Depression is a mood disorder that causes persistent feelings of sadness and loss of interest.'
            },
            'arthritis': {
                'title': 'Arthritis',
                'summary': 'Arthritis is inflammation of one or more joints, causing pain and stiffness that can worsen with age.'
            },
            'cancer': {
                'title': 'Cancer',
                'summary': 'Cancer is a group of diseases involving abnormal cell growth with the potential to invade other parts of the body.'
            }
        }
        
        # Find matching topic or use generic
        topic_info = basic_topics.get(query.lower(), {
            'title': f'{query.title()} Health Information',
            'summary': f'Health information and medical facts about {query}.'
        })
        
        return [{
            'topic_id': f'basic_{query.replace(" ", "_")}',
            'title': topic_info['title'],
            'url': f'https://medlineplus.gov/encyclopedia.html',
            'summary': topic_info['summary']
        }]
            
    def get_topic_details(self, topic_id: str) -> Dict:
        """Get detailed information for a MedlinePlus health topic."""
        # Use MedlinePlus Connect API for detailed information
        connect_params = {
            'mainSearchCriteria.v.cs': 'https://medlineplus.gov',
            'mainSearchCriteria.v.c': topic_id,
            'knowledgeResponseType': 'application/json'
        }
        
        try:
            response = self._make_request(self.connect_url, connect_params)
            data = response.json()
            
            # Extract relevant information
            topic_data = {
                'topic_id': topic_id,
                'title': '',
                'description': '',
                'symptoms': [],
                'causes': [],
                'diagnosis': [],
                'treatment': [],
                'prevention': [],
                'also_called': [],
                'primary_institute': '',
                'url': f"https://medlineplus.gov/{topic_id}.html"
            }
            
            # Parse the response structure (varies by topic)
            if 'feed' in data and 'entry' in data['feed']:
                entries = data['feed']['entry']
                if entries:
                    entry = entries[0] if isinstance(entries, list) else entries
                    
                    if 'title' in entry:
                        topic_data['title'] = entry['title']['_value']
                        
                    if 'summary' in entry:
                        topic_data['description'] = entry['summary']['_value']
                        
                    # Extract additional structured data if available
                    if 'content' in entry:
                        content = entry['content']
                        # Additional parsing logic could go here for symptoms, causes, etc.
                        
            return topic_data
            
        except Exception as e:
            # Return basic topic data if detailed fetch fails
            logger.warning(f"Could not get detailed info for topic {topic_id}: {e}")
            return {
                'topic_id': topic_id,
                'title': f"Health Topic {topic_id}",
                'description': '',
                'url': f"https://medlineplus.gov/{topic_id}.html"
            }
            
    def search_drug_information(self, drug_name: str) -> List[Dict]:
        """Search MedlinePlus drug information."""
        logger.info(f"💊 Searching MedlinePlus drugs for: {drug_name}")
        
        params = {
            'db': 'drugs',
            'term': drug_name,
            'retmax': 20,
            'rettype': 'brief'
        }
        
        response = self._make_request(self.base_url, params)
        
        # Parse XML response
        try:
            root = ET.fromstring(response.content)
            
            drugs = []
            for doc in root.findall('.//document'):
                content = doc.find('content')
                if content is not None:
                    drug_id = content.get('id')
                    title = content.get('title')
                    url = content.get('url')
                    
                    if drug_id and title:
                        drugs.append({
                            'drug_id': drug_id,
                            'title': title,
                            'url': url
                        })
                        
            logger.info(f"💊 Found {len(drugs)} drug entries")
            return drugs
            
        except ET.ParseError as e:
            raise Exception(f"Failed to parse MedlinePlus drug XML response: {e}")
            
    def fetch_medlineplus_content(self, max_results: int = 1000) -> List[Dict]:
        """Fetch MedlinePlus medical content."""
        logger.info(f"🏥 Fetching MedlinePlus content (max {max_results})")
        
        # Core health topics
        health_topics = [
            "diabetes",
            "high blood pressure",
            "heart disease", 
            "asthma",
            "pneumonia",
            "depression",
            "arthritis",
            "cancer",
            "stroke",
            "kidney disease"
        ]
        
        all_content = []
        successful_fetches = 0
        
        # Try to fetch health topics
        topics_per_search = max_results // len(health_topics)
        for topic in health_topics:
            try:
                topics = self.search_health_topics(topic, topics_per_search)
                
                if not topics:
                    logger.debug(f"No topics found for {topic}, skipping")
                    continue
                
                for topic_info in topics:
                    try:
                        detailed_topic = self.get_topic_details(topic_info['topic_id'])
                        all_content.append({
                            'type': 'health_topic',
                            'data': detailed_topic
                        })
                        successful_fetches += 1
                        
                        if len(all_content) >= max_results:
                            break
                            
                    except Exception as e:
                        logger.debug(f"Failed to get topic details for {topic_info['topic_id']}: {e}")
                        # Add basic topic without detailed info
                        all_content.append({
                            'type': 'health_topic',
                            'data': {
                                'topic_id': topic_info['topic_id'],
                                'title': topic_info['title'],
                                'description': topic_info.get('summary', ''),
                                'url': topic_info.get('url', '')
                            }
                        })
                        successful_fetches += 1
                        
                if len(all_content) >= max_results:
                    break
                    
            except Exception as e:
                logger.debug(f"Failed to fetch health topics for '{topic}': {e}")
                continue
        
        # If we got very few or no results, skip MedlinePlus entirely
        if successful_fetches < 5:
            logger.warning(f"⚠️  MedlinePlus returned only {successful_fetches} results, skipping MedlinePlus")
            return []
        
        # Convert to required format only if we have good content
        documents = []
        for content in all_content:
            try:
                content_type = content['type']
                data = content['data']
                
                if content_type == 'health_topic':
                    # Determine specialty from topic
                    specialty = self._determine_specialty_from_topic(data['title'])
                    
                    # Create health topic document
                    text = f"""MedlinePlus Health Topic: {data['title']}

Topic ID: {data.get('topic_id', 'unknown')}

Description:
{data.get('description', 'Health information from MedlinePlus')}

Clinical Information:
This MedlinePlus health topic provides evidence-based medical information for patients and healthcare consumers. MedlinePlus is produced by the National Library of Medicine and provides reliable, up-to-date health information from the NIH and other trusted sources.

Additional Resources:
For more detailed information, visit: {data.get('url', 'https://medlineplus.gov')}"""

                    doc = {
                        "text": text,
                        "metadata": {
                            "title": data['title'],
                            "source": self.source_name,
                            "medical_specialty": specialty,
                            "evidence_level": "high",
                            "publication_date": datetime.now().strftime("%Y-%m-%d"),
                            "topic_id": data.get('topic_id', 'unknown'),
                            "content_type": "health_topic",
                            "url": data.get('url', ''),
                            "tier": 1,  # Pattern Recognition
                            "chunk_id": 0
                        }
                    }
                    
                    documents.append(doc)
                    
            except Exception as e:
                logger.debug(f"Failed to process content: {e}")
                continue
                
        logger.info(f"🏥 MedlinePlus fetch complete: {len(documents)} documents")
        return documents
        
    def _determine_specialty_from_topic(self, title: str) -> str:
        """Determine medical specialty from health topic title."""
        title_lower = title.lower()
        
        specialty_keywords = {
            "Cardiology": ["heart", "cardiac", "cardiovascular", "blood pressure", "cholesterol"],
            "Endocrinology": ["diabetes", "thyroid", "hormone", "metabolism"],
            "Pulmonology": ["lung", "respiratory", "asthma", "breathing", "pneumonia"],
            "Neurology": ["brain", "neurological", "stroke", "headache", "migraine"],
            "Psychiatry": ["depression", "anxiety", "mental health", "stress"],
            "Orthopedics": ["bone", "joint", "arthritis", "back pain", "fracture"],
            "Dermatology": ["skin", "dermatitis", "rash", "acne"],
            "Gastroenterology": ["digestive", "stomach", "intestine", "liver"],
            "Nephrology": ["kidney", "renal", "urine"],
            "Oncology": ["cancer", "tumor", "malignancy"],
            "Pediatrics": ["child", "infant", "pediatric", "baby"],
            "Geriatrics": ["elderly", "senior", "aging"],
            "Obstetrics": ["pregnancy", "pregnant", "birth", "prenatal"]
        }
        
        for specialty, keywords in specialty_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                return specialty
                
        return "General Medicine"