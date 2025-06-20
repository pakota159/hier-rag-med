"""
Enhanced PubMedQA Benchmark with Real Dataset Loading
REPLACE: src/evaluation/benchmarks/pubmedqa_benchmark.py
"""

import re
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from .base_benchmark import BaseBenchmark
from loguru import logger

from src.evaluation.data.data_loader import BenchmarkDataLoader

# Add project root to path for data loader import
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

class PubMedQABenchmark(BaseBenchmark):
    """PubMedQA benchmark with real dataset loading."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "PubMedQA"
        self.data_loader = BenchmarkDataLoader(config)
    
    def load_dataset(self) -> List[Dict]:
        """Load PubMedQA dataset using centralized data loader."""
        logger.info(f"🔄 Loading PubMedQA dataset using data loader...")
        
        try:
            # Use the centralized data loader
            data = self.data_loader.load_benchmark_data(
                benchmark_name="pubmedqa",
                split="test",
                max_samples=self.sample_size if self.sample_size < 1000 else None
            )
            
            if data and len(data) > 0:
                # Convert to PubMedQA format if needed
                formatted_data = []
                for item in data:
                    formatted_item = {
                        "id": item.get("question_id", item.get("id", f"pubmedqa_{len(formatted_data)}")),
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),
                        "context": item.get("context", ""),
                        "long_answer": item.get("long_answer", ""),
                        "reasoning_type": "evidence_based"
                    }
                    formatted_data.append(formatted_item)
                
                logger.info(f"✅ Loaded {len(formatted_data)} PubMedQA questions via data loader")
                return formatted_data
            
        except Exception as e:
            logger.error(f"❌ Data loader failed for PubMedQA: {e}")
        
        # Fallback to minimal synthetic data
        logger.warning("⚠️ Using minimal fallback dataset for PubMedQA")
        return self._generate_minimal_fallback()
    
    def _generate_minimal_fallback(self) -> List[Dict]:
        """Generate minimal fallback dataset if all else fails."""
        return [
            {
                "id": "pubmedqa_fallback_001",
                "question": "Does metformin reduce cardiovascular risk in type 2 diabetes?",
                "answer": "yes",
                "context": "Multiple studies show metformin reduces cardiovascular events."
            },
            {
                "id": "pubmedqa_fallback_002",
                "question": "Are statins effective in primary prevention?",
                "answer": "yes",
                "context": "Clinical trials demonstrate statin benefits in primary prevention."
            }
        ]
    
    def _generate_comprehensive_pubmedqa_dataset(self) -> List[Dict]:
        """Generate comprehensive synthetic PubMedQA dataset."""
        synthetic_data = []
        
        # Medical research question templates with evidence-based answers
        research_questions = {
            "cardiovascular": [
                ("Does aspirin reduce cardiovascular risk in primary prevention?", "yes", "Multiple large RCTs demonstrate aspirin reduces cardiovascular events in primary prevention."),
                ("Are statins effective for secondary prevention of cardiovascular disease?", "yes", "Landmark trials show statins significantly reduce recurrent cardiovascular events."),
                ("Does vitamin E supplementation prevent cardiovascular disease?", "no", "Large randomized trials have not shown cardiovascular benefits from vitamin E."),
                ("Is Mediterranean diet effective for cardiovascular prevention?", "yes", "PREDIMED trial demonstrated cardiovascular benefits of Mediterranean diet."),
                ("Do omega-3 supplements prevent heart disease?", "maybe", "Evidence is mixed, with some studies showing modest benefits."),
            ],
            "diabetes": [
                ("Does metformin reduce cardiovascular risk in type 2 diabetes?", "yes", "UKPDS and other studies show metformin reduces cardiovascular events in diabetes."),
                ("Is intensive glucose control beneficial in type 2 diabetes?", "maybe", "Benefits depend on patient factors, with some studies showing increased mortality."),
                ("Do SGLT2 inhibitors reduce heart failure risk?", "yes", "Multiple CVOTs demonstrate heart failure benefits with SGLT2 inhibitors."),
                ("Is bariatric surgery effective for diabetes remission?", "yes", "Studies show high rates of diabetes remission after bariatric surgery."),
                ("Do GLP-1 agonists reduce cardiovascular events?", "yes", "Several CVOTs demonstrate cardiovascular benefits of GLP-1 receptor agonists."),
            ],
            "oncology": [
                ("Does mammography screening reduce breast cancer mortality?", "yes", "Meta-analyses show mammography screening reduces breast cancer deaths."),
                ("Is PSA screening effective for prostate cancer?", "maybe", "Benefits exist but must be weighed against harms of overdiagnosis."),
                ("Do antioxidants prevent cancer?", "no", "Large trials have not shown cancer prevention benefits from antioxidant supplements."),
                ("Is HPV vaccination effective in preventing cervical cancer?", "yes", "Studies demonstrate significant reduction in cervical precancerous lesions."),
                ("Does low-dose aspirin prevent colorectal cancer?", "yes", "Long-term follow-up studies show reduced colorectal cancer risk."),
            ],
            "infectious_disease": [
                ("Do probiotics prevent antibiotic-associated diarrhea?", "yes", "Systematic reviews show probiotics reduce AAD risk."),
                ("Is azithromycin effective for COVID-19 treatment?", "no", "Large RCTs have not shown benefit of azithromycin for COVID-19."),
                ("Do face masks prevent respiratory infections?", "yes", "Studies and systematic reviews support mask effectiveness."),
                ("Is vitamin D supplementation protective against respiratory infections?", "maybe", "Some studies suggest benefit, but evidence is not conclusive."),
                ("Do zinc supplements reduce common cold duration?", "yes", "Meta-analyses show modest reduction in cold duration with zinc."),
            ],
            "mental_health": [
                ("Is cognitive behavioral therapy effective for depression?", "yes", "Numerous RCTs demonstrate CBT efficacy for depression treatment."),
                ("Do antidepressants prevent suicide?", "maybe", "Evidence is mixed, with some benefit in adults but concerns in adolescents."),
                ("Is exercise effective as treatment for depression?", "yes", "Studies show exercise can be as effective as medication for mild-moderate depression."),
                ("Do omega-3 supplements improve mood disorders?", "maybe", "Some studies suggest benefit, but evidence quality is variable."),
                ("Is mindfulness-based therapy effective for anxiety?", "yes", "RCTs demonstrate effectiveness of mindfulness interventions for anxiety."),
            ],
            "pediatrics": [
                ("Does breastfeeding reduce childhood obesity risk?", "yes", "Systematic reviews show breastfeeding is associated with reduced obesity risk."),
                ("Are probiotics effective for infant colic?", "maybe", "Some studies suggest benefit, but evidence quality is limited."),
                ("Does early peanut introduction prevent peanut allergy?", "yes", "LEAP trial demonstrated early introduction prevents peanut allergy."),
                ("Is paracetamol safe in pregnancy?", "yes", "Large studies support paracetamol safety in pregnancy when used appropriately."),
                ("Do vitamin D supplements prevent rickets?", "yes", "Vitamin D supplementation is established prevention for rickets."),
            ]
        }
        
        # Generate questions for each specialty
        question_id = 1
        for specialty, questions in research_questions.items():
            for question, answer, context in questions:
                synthetic_data.append({
                    "id": f"pubmedqa_{question_id:03d}",
                    "question": question,
                    "answer": answer,
                    "context": context,
                    "reasoning_type": "evidence_based",
                    "medical_specialty": specialty
                })
                question_id += 1
        
        # Add more complex research questions
        complex_questions = [
            {
                "id": f"pubmedqa_{question_id:03d}",
                "question": "Does intensive blood pressure control reduce cardiovascular events in elderly patients?",
                "answer": "yes",
                "context": "SPRINT trial and meta-analyses show benefits of intensive BP control, though bleeding risk increases.",
                "reasoning_type": "evidence_based"
            },
            {
                "id": f"pubmedqa_{question_id+1:03d}",
                "question": "Are PCSK9 inhibitors cost-effective for cardiovascular prevention?",
                "answer": "maybe",
                "context": "PCSK9 inhibitors are effective but cost-effectiveness depends on pricing and patient selection.",
                "reasoning_type": "evidence_based"
            }
        ]
        
        synthetic_data.extend(complex_questions)
        question_id += len(complex_questions)
        
        # Generate additional questions to reach 500 total
        specialties = list(research_questions.keys())
        answer_options = ["yes", "no", "maybe"]
        
        while len(synthetic_data) < 500:
            specialty = specialties[question_id % len(specialties)]
            answer = answer_options[question_id % len(answer_options)]
            
            synthetic_data.append({
                "id": f"pubmedqa_{question_id:03d}",
                "question": f"Research question {question_id} about {specialty} evidence and outcomes?",
                "answer": answer,
                "context": f"Research evidence {question_id} from {specialty} studies and systematic reviews.",
                "reasoning_type": "evidence_based",
                "medical_specialty": specialty
            })
            question_id += 1
        
        logger.info(f"✅ Generated {len(synthetic_data)} comprehensive PubMedQA questions")
        return synthetic_data
    
    def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate single PubMedQA response."""
        
        expected_answer = question.get("answer", "").lower().strip()
        question_id = question.get("id", "unknown")
        
        # Handle empty responses
        if not response or not response.strip():
            return {
                "question_id": question_id,
                "score": 0.0,
                "correct": False,
                "metrics": {
                    "answer_accuracy": 0.0,
                    "evidence_quality": 0.0,
                    "overall_score": 0.0
                },
                "response": response or "",
                "expected": expected_answer,
                "error": "Empty response"
            }
        
        try:
            # Extract answer from response
            predicted_answer = self._extract_answer(response)
            
            # Calculate answer accuracy
            answer_accuracy = self._calculate_answer_accuracy(predicted_answer, expected_answer)
            
            # Assess evidence quality
            evidence_quality = self._assess_evidence_quality(response, question.get("context", ""))
            
            # Overall score
            overall_score = (answer_accuracy * 0.7) + (evidence_quality * 0.3)
            
            return {
                "question_id": question_id,
                "score": overall_score * 100,
                "correct": answer_accuracy > 0.8,  # High threshold for PubMedQA
                "metrics": {
                    "answer_accuracy": answer_accuracy,
                    "evidence_quality": evidence_quality,
                    "overall_score": overall_score
                },
                "response": response,
                "expected": expected_answer,
                "predicted_answer": predicted_answer
            }
            
        except Exception as e:
            logger.error(f"Error evaluating PubMedQA response for {question_id}: {e}")
            return {
                "question_id": question_id,
                "score": 0.0,
                "correct": False,
                "metrics": {
                    "answer_accuracy": 0.0,
                    "evidence_quality": 0.0,
                    "overall_score": 0.0
                },
                "response": response,
                "expected": expected_answer,
                "error": str(e)
            }
    
    def _extract_answer(self, response: str) -> str:
        """Extract yes/no/maybe answer from response."""
        response_lower = response.lower()
        
        # Look for explicit answers
        if "yes" in response_lower and "no" not in response_lower:
            return "yes"
        elif "no" in response_lower and "yes" not in response_lower:
            return "no"
        elif any(word in response_lower for word in ["maybe", "unclear", "mixed", "uncertain"]):
            return "maybe"
        elif "yes" in response_lower and "no" in response_lower:
            # Both present, look for context
            if "but" in response_lower or "however" in response_lower:
                return "maybe"
            # First occurrence wins
            yes_pos = response_lower.find("yes")
            no_pos = response_lower.find("no")
            return "yes" if yes_pos < no_pos else "no"
        else:
            # Default based on content
            positive_words = ["effective", "beneficial", "improves", "reduces", "prevents"]
            negative_words = ["ineffective", "harmful", "increases", "worsens"]
            
            positive_count = sum(1 for word in positive_words if word in response_lower)
            negative_count = sum(1 for word in negative_words if word in response_lower)
            
            if positive_count > negative_count:
                return "yes"
            elif negative_count > positive_count:
                return "no"
            else:
                return "maybe"
    
    def _calculate_answer_accuracy(self, predicted: str, expected: str) -> float:
        """Calculate answer accuracy."""
        if predicted == expected:
            return 1.0
        elif (predicted in ["yes", "no"] and expected == "maybe") or (expected in ["yes", "no"] and predicted == "maybe"):
            return 0.5  # Partial credit for maybe vs definitive
        else:
            return 0.0
    
    def _assess_evidence_quality(self, response: str, context: str) -> float:
        """Assess quality of evidence discussion in response."""
        if not response:
            return 0.0
        
        response_lower = response.lower()
        
        # Evidence quality indicators
        evidence_indicators = [
            "study", "trial", "research", "evidence", "meta-analysis",
            "systematic review", "randomized", "controlled", "cohort",
            "rct", "clinical trial"
        ]
        
        indicator_count = sum(1 for indicator in evidence_indicators if indicator in response_lower)
        evidence_score = min(indicator_count / 5.0, 0.6)
        
        # Context integration
        context_score = 0.0
        if context:
            context_words = set(context.lower().split())
            response_words = set(response_lower.split())
            overlap = len(context_words.intersection(response_words))
            context_score = min(overlap / 10.0, 0.4)
        
        return min(evidence_score + context_score, 1.0)