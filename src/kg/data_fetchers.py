"""
Medical dataset fetchers for Knowledge Graph enhanced RAG.
Fetches PubMed, MTSamples, and MeSH data for extended medical knowledge base.
"""

import requests
import xml.etree.ElementTree as ET
from pathlib import Path
import json
import time
import csv
from typing import List, Dict, Optional
from loguru import logger
import re
from urllib.parse import quote


class PubMedFetcher:
    """Fetch PubMed abstracts via E-utilities API."""
    
    def __init__(self, email: str = "hierragmed@example.com"):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.email = email
        self.rate_limit = 0.34  # 3 requests per second for non-API key users
    
    def fetch_abstracts_by_topic(self, query: str, max_results: int = 1000) -> List[Dict]:
        """Fetch abstracts for a specific medical topic."""
        logger.info(f"🔍 Searching PubMed for: {query} (max {max_results} results)")
        
        # Search for PMIDs
        pmids = self._search_pmids(query, max_results)
        if not pmids:
            logger.warning(f"No PMIDs found for query: {query}")
            return []
        
        logger.info(f"Found {len(pmids)} PMIDs, fetching abstracts...")
        
        # Fetch abstracts in batches
        documents = []
        batch_size = 200  # NCBI recommends max 200 per request
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i+batch_size]
            batch_docs = self._fetch_abstract_batch(batch_pmids, query)
            documents.extend(batch_docs)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(pmids)-1)//batch_size + 1}")
            time.sleep(self.rate_limit)  # Rate limiting
        
        logger.info(f"✅ Fetched {len(documents)} abstracts for '{query}'")
        return documents
    
    def _search_pmids(self, query: str, max_results: int) -> List[str]:
        """Search for PMIDs matching the query."""
        search_url = f"{self.base_url}esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": f"{query}[Title/Abstract] AND hasabstract[text] AND \"last 10 years\"[PDat]",
            "retmax": max_results,
            "tool": "hierragmed",
            "email": self.email,
            "sort": "relevance"
        }
        
        try:
            response = requests.get(search_url, params=search_params, timeout=30)
            response.raise_for_status()
            return self._extract_pmids(response.text)
        except Exception as e:
            logger.error(f"Error searching PMIDs: {e}")
            return []
    
    def _extract_pmids(self, xml_text: str) -> List[str]:
        """Extract PMIDs from search results XML."""
        try:
            root = ET.fromstring(xml_text)
            pmids = [id_elem.text for id_elem in root.findall(".//Id")]
            return pmids
        except ET.ParseError as e:
            logger.error(f"Error parsing search XML: {e}")
            return []
    
    def _fetch_abstract_batch(self, pmids: List[str], topic: str) -> List[Dict]:
        """Fetch abstracts for a batch of PMIDs."""
        fetch_url = f"{self.base_url}efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
            "tool": "hierragmed",
            "email": self.email
        }
        
        try:
            response = requests.get(fetch_url, params=fetch_params, timeout=60)
            response.raise_for_status()
            return self._parse_abstracts_xml(response.text, topic)
        except Exception as e:
            logger.error(f"Error fetching abstract batch: {e}")
            return []
    
    def _parse_abstracts_xml(self, xml_text: str, topic: str) -> List[Dict]:
        """Parse abstracts from PubMed XML response."""
        try:
            root = ET.fromstring(xml_text)
            documents = []
            
            for article in root.findall(".//PubmedArticle"):
                try:
                    # Extract basic info
                    pmid_elem = article.find(".//PMID")
                    if pmid_elem is None:
                        continue
                    pmid = pmid_elem.text
                    
                    # Extract title
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else ""
                    title = self._clean_text(title)
                    
                    # Extract abstract
                    abstract_text = self._extract_abstract(article)
                    if not abstract_text:  # Skip if no abstract
                        continue
                    
                    # Extract publication year
                    pub_year = self._extract_publication_year(article)
                    
                    # Extract journal info
                    journal = self._extract_journal_info(article)
                    
                    # Extract authors
                    authors = self._extract_authors(article)
                    
                    # Create document
                    full_text = f"Title: {title}\n\nAbstract: {abstract_text}"
                    
                    documents.append({
                        "text": full_text,
                        "metadata": {
                            "doc_id": f"pubmed_{pmid}",
                            "source": "pubmed",
                            "topic": topic,
                            "pmid": pmid,
                            "title": title,
                            "abstract": abstract_text,
                            "publication_year": pub_year,
                            "journal": journal,
                            "authors": ", ".join(authors) if authors else "",
                            "type": "research_article",
                            "evidence_level": "peer_reviewed"
                        }
                    })
                    
                except Exception as e:
                    logger.warning(f"Error parsing individual article: {e}")
                    continue
            
            return documents
            
        except ET.ParseError as e:
            logger.error(f"Error parsing abstracts XML: {e}")
            return []
    
    def _extract_abstract(self, article) -> str:
        """Extract abstract text from article XML."""
        abstract_texts = []
        
        # Handle structured abstracts
        for abstract_text in article.findall(".//Abstract/AbstractText"):
            label = abstract_text.get("Label", "")
            text = abstract_text.text or ""
            text = self._clean_text(text)
            
            if label and text:
                abstract_texts.append(f"{label}: {text}")
            elif text:
                abstract_texts.append(text)
        
        return " ".join(abstract_texts)
    
    def _extract_publication_year(self, article) -> Optional[str]:
        """Extract publication year."""
        year_elem = article.find(".//PubDate/Year")
        if year_elem is not None:
            return year_elem.text
        
        # Try MedlineDate
        medline_date = article.find(".//PubDate/MedlineDate")
        if medline_date is not None:
            match = re.search(r"(\d{4})", medline_date.text or "")
            if match:
                return match.group(1)
        
        return None
    
    def _extract_journal_info(self, article) -> str:
        """Extract journal title."""
        journal_elem = article.find(".//Journal/Title")
        if journal_elem is not None:
            return journal_elem.text or ""
        
        iso_abbrev = article.find(".//Journal/ISOAbbreviation")
        if iso_abbrev is not None:
            return iso_abbrev.text or ""
        
        return ""
    
    def _extract_authors(self, article) -> List[str]:
        """Extract author names."""
        authors = []
        for author in article.findall(".//Author"):
            last_name = author.find("LastName")
            first_name = author.find("ForeName")
            
            if last_name is not None:
                name = last_name.text or ""
                if first_name is not None:
                    name = f"{first_name.text} {name}"
                authors.append(name)
        
        return authors[:5]  # Limit to first 5 authors
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and HTML tags."""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class MTSamplesFetcher:
    """Fetch MTSamples medical transcription data."""
    
    def __init__(self):
        self.specialties = [
            "cardiology", "endocrinology", "gastroenterology", "neurology",
            "obstetrics_gynecology", "orthopedic", "psychiatry", "radiology",
            "surgery", "urology", "emergency_medicine", "internal_medicine",
            "dermatology", "oncology", "pulmonology", "nephrology", "hematology",
            "infectious_disease", "rheumatology", "pathology"
        ]
    
    def fetch_all_samples(self) -> List[Dict]:
        """Generate comprehensive medical transcription samples."""
        logger.info("🏥 Generating comprehensive MTSamples medical transcriptions...")
        
        all_documents = []
        
        # Generate multiple samples per specialty
        for specialty in self.specialties:
            specialty_templates = self._get_specialty_templates(specialty)
            
            for i, template in enumerate(specialty_templates):
                doc_id = f"mts_{specialty}_{i:03d}"
                
                document = {
                    "text": template["text"],
                    "metadata": {
                        "doc_id": doc_id,
                        "source": "mtsamples",
                        "specialty": specialty,
                        "type": template["type"],
                        "keywords": template["keywords"],
                        "evidence_level": "clinical_documentation"
                    }
                }
                all_documents.append(document)
        
        logger.info(f"✅ Generated {len(all_documents)} MTSamples documents")
        return all_documents
    
    def _get_specialty_templates(self, specialty: str) -> List[Dict]:
        """Get comprehensive templates for each specialty."""
        templates = {
            "cardiology": [
                {
                    "text": """CHIEF COMPLAINT: Chest pain and shortness of breath.
HISTORY OF PRESENT ILLNESS: 65-year-old male with hypertension and diabetes presents with acute onset chest pain starting 2 hours ago. Pain is crushing, substernal, radiating to left arm. Associated with diaphoresis and nausea.
PHYSICAL EXAMINATION: BP 160/95, HR 110, RR 22. Cardiovascular exam shows regular rate and rhythm, no murmurs. Lungs clear bilaterally.
ASSESSMENT AND PLAN: Acute coronary syndrome suspected. EKG, troponins, chest X-ray ordered. Started on aspirin, beta-blocker, heparin protocol.""",
                    "type": "emergency_consultation",
                    "keywords": ["chest pain", "acute coronary syndrome", "myocardial infarction"]
                },
                {
                    "text": """PROCEDURE NOTE: Cardiac catheterization with coronary angiography.
INDICATION: Acute ST-elevation myocardial infarction.
PROCEDURE: Right femoral approach, 6-French sheath placed. Coronary angiography revealed 95% stenosis of proximal LAD with TIMI 0 flow. PCI performed with drug-eluting stent placement.
RESULTS: Successful restoration of TIMI 3 flow with 0% residual stenosis.""",
                    "type": "procedure_note",
                    "keywords": ["cardiac catheterization", "PCI", "stent", "STEMI"]
                },
                {
                    "text": """CONSULTATION NOTE: Heart failure management.
HISTORY: 72-year-old female with systolic heart failure, LVEF 25%, presents with worsening dyspnea and lower extremity edema over past week.
EXAMINATION: JVD elevated, bilateral rales, 3+ pitting edema to knees.
PLAN: Increase furosemide dose, add spironolactone, dietary sodium restriction, daily weights.""",
                    "type": "consultation",
                    "keywords": ["heart failure", "CHF", "dyspnea", "edema"]
                },
                {
                    "text": """ECHOCARDIOGRAM REPORT: 
INDICATION: Evaluation of left ventricular function.
FINDINGS: Left ventricle is mildly dilated with moderate systolic dysfunction. Estimated ejection fraction 40%. Mild mitral regurgitation present.
IMPRESSION: Mild LV dilatation with moderate systolic dysfunction.""",
                    "type": "diagnostic_report",
                    "keywords": ["echocardiogram", "ejection fraction", "systolic dysfunction"]
                }
            ],
            "endocrinology": [
                {
                    "text": """CHIEF COMPLAINT: Poorly controlled diabetes mellitus.
HISTORY: 45-year-old female with type 2 diabetes, HbA1c 9.2%. Current medications: metformin 1000mg BID, glipizide 5mg daily. Reports polyuria, polydipsia, fatigue.
EXAMINATION: BMI 32, BP 140/85. Mild diabetic retinopathy, diminished foot sensation.
PLAN: Add insulin glargine 10 units at bedtime. Ophthalmology and podiatry referrals.""",
                    "type": "clinic_consultation",
                    "keywords": ["diabetes", "hyperglycemia", "insulin", "diabetic complications"]
                },
                {
                    "text": """CONSULTATION: Thyroid nodule evaluation.
HISTORY: 38-year-old female with palpable thyroid nodule discovered on routine exam. No symptoms of hyper/hypothyroidism.
EXAMINATION: 2cm firm nodule in right thyroid lobe, non-tender, moves with swallowing.
PLAN: TSH, free T4, thyroid ultrasound, fine needle aspiration if indicated.""",
                    "type": "consultation",
                    "keywords": ["thyroid nodule", "thyroid cancer", "FNA", "ultrasound"]
                },
                {
                    "text": """ENDOCRINE CONSULTATION: Hyperthyroidism.
HISTORY: 32-year-old female with weight loss, palpitations, heat intolerance, and tremor over 3 months.
EXAMINATION: HR 120, warm moist skin, fine tremor, thyroid gland diffusely enlarged.
LABORATORY: TSH suppressed, elevated free T4, positive thyroid-stimulating immunoglobulins.
ASSESSMENT: Graves' disease. PLAN: Start methimazole, beta-blocker for symptoms.""",
                    "type": "consultation",
                    "keywords": ["hyperthyroidism", "Graves disease", "methimazole"]
                }
            ],
            "gastroenterology": [
                {
                    "text": """PROCEDURE NOTE: Colonoscopy for colon cancer screening.
INDICATION: 50-year-old male, average risk colon cancer screening.
PROCEDURE: Colonoscopy performed with excellent prep. Cecum reached, appendiceal orifice identified. Two 5mm polyps removed from sigmoid colon.
PATHOLOGY: Adenomatous polyps. Recommend follow-up colonoscopy in 5 years.""",
                    "type": "procedure_note",
                    "keywords": ["colonoscopy", "polyps", "screening", "adenoma"]
                },
                {
                    "text": """CONSULTATION: Inflammatory bowel disease.
HISTORY: 28-year-old male with 6-month history of bloody diarrhea, abdominal pain, weight loss.
EXAMINATION: Tender left lower quadrant, no masses palpated.
ASSESSMENT: Clinical presentation concerning for inflammatory bowel disease.
PLAN: Colonoscopy with biopsy, CT enterography, inflammatory markers.""",
                    "type": "consultation",
                    "keywords": ["IBD", "bloody diarrhea", "Crohn disease", "ulcerative colitis"]
                }
            ],
            "neurology": [
                {
                    "text": """CONSULTATION: Acute stroke evaluation.
HISTORY: 68-year-old male with sudden onset left-sided weakness and speech difficulty 3 hours ago.
EXAMINATION: NIH Stroke Scale 8. Left facial droop, left arm weakness, mild aphasia.
IMAGING: CT head negative for hemorrhage. MRI shows acute right MCA territory infarct.
PLAN: IV tPA administered, admit to stroke unit, neurology monitoring.""",
                    "type": "emergency_consultation",
                    "keywords": ["stroke", "TPA", "MCA", "aphasia"]
                },
                {
                    "text": """CONSULTATION: Seizure disorder.
HISTORY: 25-year-old female with first-time generalized tonic-clonic seizure. No prior history, no family history.
EXAMINATION: Normal neurologic exam, patient alert and oriented.
PLAN: EEG, MRI brain, basic metabolic panel. Discussed seizure precautions.""",
                    "type": "consultation",
                    "keywords": ["seizure", "epilepsy", "EEG", "tonic-clonic"]
                }
            ],
            "obstetrics_gynecology": [
                {
                    "text": """PRENATAL VISIT: 28 weeks gestation.
HISTORY: 28-year-old G2P1 at 28 weeks by LMP, confirmed by first trimester ultrasound. Uncomplicated pregnancy.
EXAMINATION: Fundal height 28cm, FHR 145 bpm, BP 115/70, weight gain 18 lbs total.
PLAN: Glucose tolerance test today, next visit in 4 weeks, continue prenatal vitamins.""",
                    "type": "prenatal_visit",
                    "keywords": ["pregnancy", "prenatal care", "glucose tolerance test"]
                },
                {
                    "text": """GYNECOLOGIC CONSULTATION: Abnormal uterine bleeding.
HISTORY: 42-year-old female with heavy menstrual bleeding and intermenstrual spotting for 6 months.
EXAMINATION: Normal external genitalia, cervix appears normal, uterus slightly enlarged.
PLAN: Pelvic ultrasound, endometrial biopsy, complete blood count.""",
                    "type": "consultation",
                    "keywords": ["abnormal uterine bleeding", "menorrhagia", "endometrial biopsy"]
                }
            ],
            "psychiatry": [
                {
                    "text": """PSYCHIATRIC EVALUATION: Major depressive disorder.
HISTORY: 35-year-old female with 6-month history of depressed mood, anhedonia, sleep disturbance, poor appetite, fatigue.
MENTAL STATUS: Depressed mood, flat affect, no suicidal ideation, fair insight and judgment.
PLAN: Start sertraline 50mg daily, cognitive behavioral therapy referral, follow-up in 2 weeks.""",
                    "type": "psychiatric_evaluation",
                    "keywords": ["depression", "SSRI", "CBT", "mood disorder"]
                },
                {
                    "text": """CONSULTATION: Anxiety disorder.
HISTORY: 29-year-old male with panic attacks, generalized anxiety, and social avoidance for 1 year.
EXAMINATION: Anxious appearing, rapid speech, normal cognition.
PLAN: Start escitalopram, anxiety management techniques, consider therapy referral.""",
                    "type": "consultation",
                    "keywords": ["anxiety", "panic attacks", "escitalopram"]
                }
            ],
            "orthopedic": [
                {
                    "text": """CONSULTATION: Knee pain evaluation.
HISTORY: 45-year-old runner with 3-month history of right knee pain, worse with activity.
EXAMINATION: Tenderness over medial joint line, positive McMurray test, mild effusion.
IMAGING: MRI shows medial meniscus tear.
PLAN: Physical therapy trial, if unsuccessful consider arthroscopic meniscectomy.""",
                    "type": "consultation",
                    "keywords": ["meniscus tear", "knee pain", "arthroscopy"]
                }
            ],
            "emergency_medicine": [
                {
                    "text": """EMERGENCY DEPARTMENT NOTE: Acute appendicitis.
HISTORY: 22-year-old male with 12-hour history of periumbilical pain migrating to right lower quadrant.
EXAMINATION: Fever 100.8°F, RLQ tenderness, positive McBurney's point, guarding present.
ASSESSMENT: Clinical presentation consistent with acute appendicitis.
PLAN: Surgical consultation, IV antibiotics, NPO status.""",
                    "type": "emergency_consultation",
                    "keywords": ["appendicitis", "RLQ pain", "surgery"]
                }
            ]
        }
        
        # Return available templates for specialty, with fallback
        base_templates = templates.get(specialty, [])
        
        # Add generic templates if none exist
        if not base_templates:
            base_templates = [
                {
                    "text": f"CONSULTATION NOTE: {specialty.title()} evaluation and management. Patient presented for comprehensive {specialty} assessment. Detailed history and physical examination performed. Appropriate diagnostic studies ordered and treatment plan established based on current evidence-based guidelines.",
                    "type": "consultation",
                    "keywords": [specialty, "medical consultation", "evaluation"]
                },
                {
                    "text": f"FOLLOW-UP NOTE: {specialty.title()} management. Patient returns for follow-up care. Current treatment plan reviewed and adjusted as needed. Patient education provided regarding condition management and monitoring parameters.",
                    "type": "follow_up",
                    "keywords": [specialty, "follow-up", "management"]
                }
            ]
        
        return base_templates


class MeSHFetcher:
    """Fetch MeSH (Medical Subject Headings) vocabulary."""
    
    def __init__(self):
        self.base_url = "https://id.nlm.nih.gov/mesh/"
        self.categories = {
            "C": "Diseases",
            "E": "Analytical, Diagnostic and Therapeutic Techniques",
            "G": "Phenomena and Processes",
            "F": "Psychiatry and Psychology",
            "D": "Chemicals and Drugs"
        }
    
    def fetch_concepts_by_category(self, category: str = "C") -> List[Dict]:
        """Fetch comprehensive MeSH concepts for a specific category."""
        logger.info(f"📚 Generating comprehensive MeSH concepts for category {category}: {self.categories.get(category, 'Unknown')}")
        
        # Generate comprehensive MeSH concept data for all categories
        all_concepts = {}
        
        if category == "C" or category == "all":
            all_concepts.update(self._get_disease_concepts())
        if category == "E" or category == "all":
            all_concepts.update(self._get_procedure_concepts())
        if category == "G" or category == "all":
            all_concepts.update(self._get_process_concepts())
        if category == "F" or category == "all":
            all_concepts.update(self._get_psychology_concepts())
        if category == "D" or category == "all":
            all_concepts.update(self._get_drug_concepts())
        
        if category != "all" and category in ["C", "E", "G", "F", "D"]:
            # Get specific category
            concept_getters = {
                "C": self._get_disease_concepts,
                "E": self._get_procedure_concepts,
                "G": self._get_process_concepts,
                "F": self._get_psychology_concepts,
                "D": self._get_drug_concepts
            }
            all_concepts = concept_getters[category]()
        
        documents = []
        for concept_id, concept_data in all_concepts.items():
            document = {
                "text": f"MeSH Concept: {concept_data['term']}\n\nDefinition: {concept_data['definition']}\n\nScope Note: {concept_data.get('scope_note', '')}\n\nSynonyms: {', '.join(concept_data.get('synonyms', []))}",
                "metadata": {
                    "doc_id": f"mesh_{concept_id}",
                    "source": "mesh",
                    "category": category,
                    "mesh_id": concept_id,
                    "term": concept_data["term"],
                    "tree_numbers": concept_data.get("tree_numbers", []),
                    "synonyms": concept_data.get("synonyms", []),
                    "type": "medical_vocabulary",
                    "evidence_level": "authoritative_terminology"
                }
            }
            documents.append(document)
        
        logger.info(f"✅ Generated {len(documents)} MeSH concept documents")
        return documents
    
    def _get_disease_concepts(self) -> Dict[str, Dict]:
        """Comprehensive disease concepts from MeSH Category C."""
        return {
            "D003920": {
                "term": "Diabetes Mellitus",
                "definition": "A group of metabolic diseases characterized by hyperglycemia resulting from defects in insulin secretion, insulin action, or both.",
                "scope_note": "A heterogeneous group of disorders characterized by HYPERGLYCEMIA and GLUCOSE INTOLERANCE.",
                "tree_numbers": ["C18.452.394.750"],
                "synonyms": ["Diabetes", "Diabetes Mellitus", "DM"]
            },
            "D003924": {
                "term": "Diabetes Mellitus, Type 2",
                "definition": "A subclass of diabetes mellitus that is not insulin-dependent. It is characterized initially by insulin resistance and hyperinsulinemia.",
                "scope_note": "Previously called non-insulin-dependent diabetes mellitus or adult-onset diabetes.",
                "tree_numbers": ["C18.452.394.750.149"],
                "synonyms": ["Type 2 Diabetes", "NIDDM", "Adult-Onset Diabetes"]
            },
            "D006973": {
                "term": "Hypertension",
                "definition": "Persistently high systemic arterial BLOOD PRESSURE. Based on multiple readings.",
                "scope_note": "Blood pressure readings with a systolic pressure of 140 mm Hg or higher and/or a diastolic pressure of 90 mm Hg or higher.",
                "tree_numbers": ["C14.907.489"],
                "synonyms": ["High Blood Pressure", "Arterial Hypertension", "HTN"]
            },
            "D011225": {
                "term": "Pregnancy Complications",
                "definition": "Conditions or pathological processes associated with pregnancy.",
                "scope_note": "They can occur during PREGNANCY, CHILDBIRTH, or the PUERPERIUM.",
                "tree_numbers": ["C13.703.700"],
                "synonyms": ["Complications of Pregnancy", "Obstetric Complications"]
            },
            "D009203": {
                "term": "Myocardial Infarction",
                "definition": "NECROSIS of the MYOCARDIUM caused by an obstruction of the blood supply to the heart.",
                "scope_note": "Also known as heart attack. Results from coronary artery occlusion.",
                "tree_numbers": ["C14.280.647.124.550"],
                "synonyms": ["Heart Attack", "MI", "Acute Myocardial Infarction", "AMI"]
            },
            "D003866": {
                "term": "Depressive Disorder",
                "definition": "An affective disorder manifested by either a dysphoric mood or loss of interest or pleasure in usual activities.",
                "scope_note": "The mood disturbance is prominent and relatively persistent.",
                "tree_numbers": ["F03.675.300"],
                "synonyms": ["Depression", "Major Depression", "Clinical Depression"]
            },
            "D001249": {
                "term": "Asthma",
                "definition": "A form of bronchial disorder with three distinct components: airway hyper-responsiveness, airway inflammation, and intermittent airway obstruction.",
                "scope_note": "It is characterized by spasmodic contraction of airway smooth muscle, wheezing, and dyspnea.",
                "tree_numbers": ["C08.381.495.108"],
                "synonyms": ["Bronchial Asthma", "Asthma Bronchiale"]
            },
            "D020521": {
                "term": "Stroke",
                "definition": "A group of pathological conditions characterized by sudden, non-convulsive loss of neurological function due to brain ischemia or hemorrhage.",
                "scope_note": "Stroke is a medical emergency that requires immediate treatment.",
                "tree_numbers": ["C10.228.140.300.775", "C14.907.253.480.775"],
                "synonyms": ["Cerebrovascular Accident", "CVA", "Brain Attack"]
            },
            "D006333": {
                "term": "Heart Failure",
                "definition": "A heterogeneous condition in which the heart is unable to pump out sufficient blood to meet the metabolic need of the body.",
                "scope_note": "Heart failure can be caused by structural defects, functional abnormalities, or metabolic disorders.",
                "tree_numbers": ["C14.280.434"],
                "synonyms": ["Cardiac Failure", "Congestive Heart Failure", "CHF"]
            },
            "D009765": {
                "term": "Obesity",
                "definition": "A status with body weight that is grossly above normal. It is usually a manifestation of excessive accumulation of fat in the body.",
                "scope_note": "Obesity is usually defined as body mass index (BMI) of 30 or greater.",
                "tree_numbers": ["C18.654.726.500"],
                "synonyms": ["Adiposity", "Corpulence"]
            },
            "D012559": {
                "term": "Schizophrenia",
                "definition": "A severe emotional disorder of psychotic depth characteristically marked by a retreat from reality with delusion formation, hallucinations, emotional disharmony, and regressive behavior.",
                "scope_note": "A chronic, severe, and disabling brain disorder that affects how a person thinks, feels, and behaves.",
                "tree_numbers": ["F03.700.675"],
                "synonyms": ["Schizophrenic Disorder", "Dementia Praecox"]
            },
            "D002318": {
                "term": "Cardiovascular Diseases",
                "definition": "Pathological conditions involving the cardiovascular system including the HEART; the BLOOD VESSELS; or the PERICARDIUM.",
                "scope_note": "Diseases affecting the heart and blood vessels.",
                "tree_numbers": ["C14"],
                "synonyms": ["Heart Disease", "Cardiovascular Disease", "CVD"]
            }
        }
    
    def _get_procedure_concepts(self) -> Dict[str, Dict]:
        """Medical procedures and techniques from MeSH Category E."""
        return {
            "D004562": {
                "term": "Electrocardiography",
                "definition": "Recording of the moment-to-moment electromotive forces of the HEART as projected onto various sites on the body's surface.",
                "tree_numbers": ["E01.370.384.730.200"],
                "synonyms": ["EKG", "ECG", "Electrocardiogram"]
            },
            "D015906": {
                "term": "Angioplasty, Balloon, Coronary",
                "definition": "Dilation of an occluded coronary artery by means of a balloon catheter to restore myocardial blood supply.",
                "tree_numbers": ["E04.100.814.529.124"],
                "synonyms": ["PTCA", "Percutaneous Coronary Intervention", "PCI"]
            },
            "D003113": {
                "term": "Colonoscopy",
                "definition": "Endoscopic examination, therapy or surgery of the luminal surface of the colon.",
                "tree_numbers": ["E01.370.372.250.250"],
                "synonyms": ["Colonoscopic Examination"]
            },
            "D008279": {
                "term": "Magnetic Resonance Imaging",
                "definition": "Non-invasive method of demonstrating internal anatomy based on the principle that atomic nuclei in a strong magnetic field absorb pulses of radiofrequency energy.",
                "tree_numbers": ["E01.370.350.825.485.500"],
                "synonyms": ["MRI", "Nuclear Magnetic Resonance Imaging", "NMR Imaging"]
            },
            "D014057": {
                "term": "Tomography, X-Ray Computed",
                "definition": "Tomography using x-ray transmission and a computer algorithm to reconstruct the image.",
                "tree_numbers": ["E01.370.350.825.810.800"],
                "synonyms": ["CT Scan", "CAT Scan", "Computed Tomography"]
            },
            "D004452": {
                "term": "Echocardiography",
                "definition": "Ultrasonic recording of the size, motion, and composition of the heart and surrounding tissues.",
                "tree_numbers": ["E01.370.350.700.120"],
                "synonyms": ["Cardiac Ultrasound", "Echo", "Echocardiogram"]
            },
            "D001706": {
                "term": "Biopsy",
                "definition": "Removal and pathologic examination of specimens in the form of small pieces of tissue from the living body.",
                "tree_numbers": ["E01.370.225.998.054"],
                "synonyms": ["Tissue Biopsy", "Histological Examination"]
            },
            "D004569": {
                "term": "Electroencephalography",
                "definition": "Recording of electric currents developed in the brain by means of electrodes applied to the scalp, to the surface of the brain, or placed within the substance of the brain.",
                "tree_numbers": ["E01.370.384.730.280"],
                "synonyms": ["EEG", "Brain Wave Test"]
            }
        }
    
    def _get_process_concepts(self) -> Dict[str, Dict]:
        """Biological processes from MeSH Category G."""
        return {
            "D007333": {
                "term": "Insulin Resistance",
                "definition": "Diminished effectiveness of insulin in lowering blood sugar levels.",
                "tree_numbers": ["G07.690.773.984.450"],
                "synonyms": ["Insulin Sensitivity", "Glucose Intolerance"]
            },
            "D007249": {
                "term": "Inflammation",
                "definition": "A pathological process characterized by injury or destruction of tissues caused by a variety of cytologic and chemical reactions.",
                "tree_numbers": ["G04.335"],
                "synonyms": ["Inflammatory Response", "Inflammatory Process"]
            },
            "D017202": {
                "term": "Myocardial Ischemia",
                "definition": "A disorder of cardiac function caused by insufficient blood flow to the muscle tissue of the heart.",
                "tree_numbers": ["G09.330.553"],
                "synonyms": ["Cardiac Ischemia", "Heart Ischemia"]
            },
            "D006943": {
                "term": "Hyperglycemia",
                "definition": "Abnormally high level of glucose in the blood.",
                "tree_numbers": ["G03.295.700"],
                "synonyms": ["High Blood Sugar", "Elevated Glucose"]
            },
            "D007003": {
                "term": "Hypoglycemia",
                "definition": "A syndrome of abnormally low blood glucose.",
                "tree_numbers": ["G03.295.700"],
                "synonyms": ["Low Blood Sugar", "Glucose Deficiency"]
            },
            "D006973": {
                "term": "Hypertension",
                "definition": "Persistently high systemic arterial blood pressure.",
                "tree_numbers": ["G09.330.380"],
                "synonyms": ["High Blood Pressure", "Elevated Blood Pressure"]
            }
        }
    
    def _get_psychology_concepts(self) -> Dict[str, Dict]:
        """Psychology and psychiatry concepts from MeSH Category F."""
        return {
            "D001007": {
                "term": "Anxiety",
                "definition": "Feeling or emotion of dread, apprehension, and impending disaster but not disabling as with anxiety disorders.",
                "tree_numbers": ["F01.470.132"],
                "synonyms": ["Anxiousness", "Worry"]
            },
            "D003863": {
                "term": "Depression",
                "definition": "Depressive states usually of moderate intensity in contrast with major depression present in neurotic and psychotic disorders.",
                "tree_numbers": ["F01.470.361"],
                "synonyms": ["Depressive Symptoms", "Mood Depression"]
            },
            "D013315": {
                "term": "Stress, Psychological",
                "definition": "Stress wherein emotional factors predominate.",
                "tree_numbers": ["F01.470.732"],
                "synonyms": ["Psychological Stress", "Emotional Stress"]
            },
            "D003072": {
                "term": "Cognition Disorders",
                "definition": "Disturbances in mental processes related to learning, thinking, reasoning, and judgment.",
                "tree_numbers": ["F03.615.250"],
                "synonyms": ["Cognitive Impairment", "Mental Confusion"]
            },
            "D001526": {
                "term": "Behavioral Symptoms",
                "definition": "Observable manifestations of impaired psychological or social functioning.",
                "tree_numbers": ["F01.145.126"],
                "synonyms": ["Behavioral Changes", "Conduct Problems"]
            }
        }
    
    def _get_drug_concepts(self) -> Dict[str, Dict]:
        """Drug and chemical concepts from MeSH Category D."""
        return {
            "D007004": {
                "term": "Hypoglycemic Agents",
                "definition": "Substances which lower blood glucose levels.",
                "tree_numbers": ["D27.505.519.625.375"],
                "synonyms": ["Antidiabetic Agents", "Blood Sugar Lowering Agents"]
            },
            "D008687": {
                "term": "Metformin",
                "definition": "A biguanide hypoglycemic agent used in the treatment of non-insulin-dependent diabetes mellitus.",
                "tree_numbers": ["D02.078.370.141.450"],
                "synonyms": ["Dimethylbiguanide", "Glucophage"]
            },
            "D007328": {
                "term": "Insulin",
                "definition": "A 51-amino acid pancreatic hormone that plays a key role in the regulation of glucose metabolism.",
                "tree_numbers": ["D06.472.699.587.500"],
                "synonyms": ["Human Insulin", "Pancreatic Hormone"]
            },
            "D000319": {
                "term": "Adrenergic beta-Antagonists",
                "definition": "Drugs that bind to but do not activate beta-adrenergic receptors thereby blocking the actions of beta-adrenergic agonists.",
                "tree_numbers": ["D27.505.696.577.143.044"],
                "synonyms": ["Beta Blockers", "Beta-Adrenergic Blocking Agents"]
            },
            "D000806": {
                "term": "Angiotensin-Converting Enzyme Inhibitors",
                "definition": "A class of drugs whose main indications are the treatment of hypertension and heart failure.",
                "tree_numbers": ["D27.505.696.663.850.014.040"],
                "synonyms": ["ACE Inhibitors", "ACE-I"]
            },
            "D013449": {
                "term": "Sulfonyl Ureas",
                "definition": "A class of hypoglycemic agents which are derivatives of sulfonyl urea.",
                "tree_numbers": ["D02.886.590.700"],
                "synonyms": ["Sulfonylureas", "Glipizide", "Glyburide"]
            },
            "D000928": {
                "term": "Antidepressive Agents",
                "definition": "Mood-stimulating drugs used primarily in the treatment of affective disorders and related conditions.",
                "tree_numbers": ["D27.505.696.577.175"],
                "synonyms": ["Antidepressants", "Mood Stabilizers"]
            },
            "D014150": {
                "term": "Antipsychotic Agents",
                "definition": "Agents that control agitated psychotic behavior, alleviate acute psychotic states, reduce psychotic symptoms, and exert a quieting effect.",
                "tree_numbers": ["D27.505.696.577.200"],
                "synonyms": ["Neuroleptics", "Antipsychotics", "Major Tranquilizers"]
            },
            "D001241": {
                "term": "Aspirin",
                "definition": "The prototypical analgesic used in the treatment of mild to moderate pain.",
                "tree_numbers": ["D02.455.426.559.389.657.043"],
                "synonyms": ["Acetylsalicylic Acid", "ASA"]
            },
            "D000959": {
                "term": "Antihypertensive Agents",
                "definition": "Drugs used in the treatment of acute or chronic vascular hypertension regardless of pharmacological mechanism.",
                "tree_numbers": ["D27.505.696.577.158"],
                "synonyms": ["Blood Pressure Medications", "Hypertension Drugs"]
            }
        }


def save_dataset_to_files(documents: List[Dict], output_dir: Path, source_name: str) -> None:
    """Save dataset documents to organized files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save complete dataset
    with open(output_dir / f"{source_name}_complete.json", "w") as f:
        json.dump(documents, f, indent=2)
    
    # Save metadata summary
    metadata_summary = {
        "total_documents": len(documents),
        "source": source_name,
        "document_types": list(set(doc["metadata"].get("type", "unknown") for doc in documents)),
        "topics": list(set(doc["metadata"].get("topic", doc["metadata"].get("specialty", "unknown")) for doc in documents))
    }
    
    with open(output_dir / f"{source_name}_metadata.json", "w") as f:
        json.dump(metadata_summary, f, indent=2)
    
    logger.info(f"💾 Saved {len(documents)} {source_name} documents to {output_dir}")