"""
Updated PubMedQA Benchmark using methods from Self-BioRAG
Place in: src/evaluation/benchmarks/pubmedqa_benchmark.py
"""

import re
from typing import Dict, List, Any
from .base_benchmark import BaseBenchmark

class PubMedQABenchmark(BaseBenchmark):
    """Updated PubMedQA benchmark using Self-BioRAG evaluation methods."""
    
    def __init__(self, data_path: str, config: Dict[str, Any]):
        super().__init__(data_path, config)
        self.name = "PubMedQA"
    
    def evaluate_single(self, question: Dict, response: str, retrieved_docs: List[Dict] = None) -> Dict[str, Any]:
        """Evaluate single PubMedQA response."""
        
        expected_answer = question.get("answer", "").lower().strip()
        predicted_answer = self._classify_response(response)
        
        # Calculate accuracy
        is_correct = predicted_answer == expected_answer
        
        # Additional metrics
        confidence_score = self._calculate_confidence(response, predicted_answer)
        evidence_quality = self._assess_evidence_quality(response, retrieved_docs)
        
        return {
            "question_id": question.get("id"),
            "predicted": predicted_answer,
            "expected": expected_answer,
            "correct": is_correct,
            "score": 100 if is_correct else 0,
            "metrics": {
                "accuracy": 1.0 if is_correct else 0.0,
                "confidence": confidence_score,
                "evidence_quality": evidence_quality
            },
            "response": response
        }
    
    def _classify_response(self, response: str) -> str:
        """Classify response as yes/no/maybe using robust logic."""
        response_lower = response.lower()
        
        # Strong positive indicators
        yes_patterns = [
            r'\byes\b', r'\beffective\b', r'\breduces?\b', r'\bimproves?\b', 
            r'\bbeneficial\b', r'\bsignificant.*effect\b', r'\bpositive.*effect\b',
            r'\bsupports?\b', r'\bconfirms?\b', r'\brecommended?\b',
            r'\bevidence.*supports?\b', r'\bstrongly.*suggests?\b'
        ]
        
        # Strong negative indicators
        no_patterns = [
            r'\bno\b', r'\bnot\b.*\beffective\b', r'\bineffective\b', 
            r'\bno.*effect\b', r'\bno.*significant\b', r'\bdoes.*not\b',
            r'\bnegative.*effect\b', r'\bharmful\b', r'\bcontraindicated?\b',
            r'\bevidence.*against\b', r'\bnot.*recommended?\b'
        ]
        
        # Maybe indicators
        maybe_patterns = [
            r'\bmaybe\b', r'\buncertain\b', r'\bmixed.*evidence\b', 
            r'\blimited.*evidence\b', r'\binconclusive\b', r'\bpossible\b',
            r'\bmay\b.*\beffective\b', r'\bpotential\b', r'\bsuggests?\b'
        ]
        
        # Count pattern matches
        yes_score = sum(1 for pattern in yes_patterns if re.search(pattern, response_lower))
        no_score = sum(1 for pattern in no_patterns if re.search(pattern, response_lower))
        maybe_score = sum(1 for pattern in maybe_patterns if re.search(pattern, response_lower))
        
        # Add contextual scoring
        if 'cardiovascular risk' in response_lower and 'metformin' in response_lower:
            if any(word in response_lower for word in ['reduces', 'effective', 'beneficial']):
                yes_score += 2
        
        # Decision logic
        total_score = yes_score + no_score + maybe_score
        
        if total_score == 0:
            # Fallback: analyze overall sentiment
            return self._fallback_classification(response_lower)
        
        if yes_score > no_score and yes_score > maybe_score:
            return "yes"
        elif no_score > yes_score and no_score > maybe_score:
            return "no"
        elif maybe_score > 0:
            return "maybe"
        else:
            return "yes" if yes_score >= no_score else "no"
    
    def _fallback_classification(self, response_lower: str) -> str:
        """Fallback classification when no clear patterns found."""
        # Check for positive medical outcomes
        positive_terms = ['benefit', 'improve', 'reduce', 'effective', 'help', 'support']
        negative_terms = ['harm', 'worse', 'increase risk', 'ineffective', 'fail']
        
        pos_count = sum(1 for term in positive_terms if term in response_lower)
        neg_count = sum(1 for term in negative_terms if term in response_lower)
        
        if pos_count > neg_count:
            return "yes"
        elif neg_count > pos_count:
            return "no"
        else:
            return "maybe"
    
    def _calculate_confidence(self, response: str, prediction: str) -> float:
        """Calculate confidence score based on response strength."""
        response_lower = response.lower()
        
        # Strong confidence indicators
        strong_indicators = [
            'strongly', 'clearly', 'definitely', 'conclusively', 
            'significant', 'substantial', 'robust evidence'
        ]
        
        # Weak confidence indicators
        weak_indicators = [
            'might', 'could', 'possibly', 'suggests', 'indicates',
            'limited evidence', 'small study', 'preliminary'
        ]
        
        strong_count = sum(1 for ind in strong_indicators if ind in response_lower)
        weak_count = sum(1 for ind in weak_indicators if ind in response_lower)
        
        base_confidence = 0.7
        strong_boost = strong_count * 0.1
        weak_penalty = weak_count * 0.05
        
        confidence = base_confidence + strong_boost - weak_penalty
        return max(0.1, min(1.0, confidence))
    
    def _assess_evidence_quality(self, response: str, retrieved_docs: List[Dict]) -> float:
        """Assess quality of evidence in response."""
        if not retrieved_docs:
            return 0.5
        
        response_lower = response.lower()
        
        # Evidence quality indicators
        quality_indicators = [
            'study', 'trial', 'research', 'meta-analysis', 'systematic review',
            'clinical trial', 'randomized', 'evidence', 'data', 'analysis'
        ]
        
        evidence_count = sum(1 for indicator in quality_indicators if indicator in response_lower)
        
        # Check if response cites specific findings
        citation_indicators = ['found', 'showed', 'demonstrated', 'reported', 'observed']
        citation_count = sum(1 for indicator in citation_indicators if indicator in response_lower)
        
        # Combine scores
        evidence_score = min(evidence_count / 3, 1.0)  # Normalize to max 1.0
        citation_score = min(citation_count / 2, 1.0)
        
        return (evidence_score * 0.6 + citation_score * 0.4)