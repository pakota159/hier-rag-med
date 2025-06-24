"""
Enhanced Generation module for Hierarchical Medical Q&A
File: src/basic_reasoning/generation.py

Optimized for medical multiple choice questions with improved answer extraction
and hierarchical reasoning for MIRAGE benchmark performance.
"""

import re
import json
from typing import Dict, List, Optional
import requests
from loguru import logger


class HierarchicalGenerator:
    """Enhanced generator for hierarchical medical Q&A responses."""

    def __init__(self, config):
        """Initialize enhanced generator with medical Q&A optimization."""
        self.config = config
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Enhanced answer extraction patterns
        self.answer_patterns = [
            r'Answer:\s*([A-E])',
            r'The answer is\s*([A-E])',
            r'Correct answer:\s*([A-E])',
            r'Final answer:\s*([A-E])',
            r'\b([A-E])\s*(?:is the|is correct|is the answer)',
            r'(?:Option|Choice)\s*([A-E])',
            r'\b([A-E])\)?\s*(?:correct|right|best)',
            r'Select\s*([A-E])',
            r'Choose\s*([A-E])'
        ]
        
        logger.info("🔧 Enhanced Medical Q&A Generator initialized")

    def generate_hierarchical_response(
        self,
        query: str,
        hierarchical_results: Dict[str, List[Dict]]
    ) -> str:
        """Generate enhanced hierarchical response for medical Q&A."""
        logger.info("🧠 Generating enhanced hierarchical medical Q&A response")
        
        # Prepare enhanced context from each tier with medical focus
        tier1_context = self._format_tier_context(
            hierarchical_results.get("tier1_patterns", []),
            "Medical Concept",
            300
        )
        
        tier2_context = self._format_tier_context(
            hierarchical_results.get("tier2_hypotheses", []),
            "Clinical Knowledge",
            400
        )
        
        tier3_context = self._format_tier_context(
            hierarchical_results.get("tier3_confirmation", []),
            "Evidence",
            400
        )

        # Build enhanced hierarchical prompt
        system_prompt = self.config.config["prompts"]["system"]
        tier1_prompt = self.config.config["prompts"]["tier1_prompt"]
        tier2_prompt = self.config.config["prompts"]["tier2_prompt"]
        tier3_prompt = self.config.config["prompts"]["tier3_prompt"]
        
        # Enhanced prompt structure for medical Q&A
        prompt = f"""{system_prompt}

=== HIERARCHICAL MEDICAL ANALYSIS ===

{tier1_prompt}
{tier1_context}

{tier2_prompt}
{tier2_context}

{tier3_prompt}
{tier3_context}

=== MEDICAL QUESTION ===
{query}

=== INSTRUCTIONS ===
1. Analyze the question using the three-tier medical knowledge above
2. Systematically evaluate each answer option
3. Select the most accurate answer based on medical evidence
4. Provide your reasoning and final answer

REQUIRED FORMAT:
Analysis: [Your systematic evaluation]
Answer: [LETTER]

Response:"""

        # Generate response with enhanced error handling
        try:
            response = self._call_ollama_enhanced(prompt)
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return self._generate_fallback_response(query)
        
        # Enhanced answer extraction and validation
        extracted_answer = self._extract_answer_enhanced(response)
        validated_response = self._validate_and_format_response(response, extracted_answer)
        
        logger.info(f"✅ Generated hierarchical response with answer: {extracted_answer}")
        return validated_response

    def _format_tier_context(self, documents: List[Dict], label: str, max_chars: int) -> str:
        """Format tier context with medical focus."""
        if not documents:
            return f"No {label.lower()} available.\n"
        
        context_parts = []
        for i, doc in enumerate(documents[:5]):  # Limit to top 5 docs per tier
            text = doc.get('text', '')
            specialty = doc.get('metadata', {}).get('medical_specialty', 'General')
            
            # Truncate and add specialty info
            truncated_text = text[:max_chars] + "..." if len(text) > max_chars else text
            context_parts.append(f"{label} {i+1} ({specialty}): {truncated_text}")
        
        return "\n".join(context_parts) + "\n"

    def _call_ollama_enhanced(self, prompt: str) -> str:
        """Enhanced Ollama API call with medical Q&A optimization."""
        payload = {
            "model": self.config.config["models"]["llm"]["name"],
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.config["models"]["llm"]["temperature"],
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 512,  # Sufficient for Q&A responses
                "stop": ["Question:", "=== ", "Human:", "Assistant:"]  # Enhanced stop tokens
            }
        }
        
        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=60  # Increased timeout for complex medical reasoning
            )
            response.raise_for_status()
            
            result = response.json()
            if "response" not in result:
                raise Exception("Invalid response format from Ollama")
            
            return result["response"].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Ollama API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Ollama response processing failed: {e}")
            raise

    def _extract_answer_enhanced(self, response: str) -> str:
        """Enhanced answer extraction with multiple pattern matching."""
        if not response:
            return "A"  # Default fallback
        
        # Try each pattern in order of specificity
        for pattern in self.answer_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                answer = matches[-1].upper()  # Take the last match (most likely final answer)
                if answer in ['A', 'B', 'C', 'D', 'E']:
                    logger.debug(f"🎯 Extracted answer '{answer}' using pattern: {pattern}")
                    return answer
        
        # Fallback: look for any single letter A-E near common keywords
        fallback_pattern = r'\b([A-E])\b'
        fallback_keywords = ['answer', 'correct', 'best', 'option', 'choice']
        
        for keyword in fallback_keywords:
            keyword_context = self._extract_context_around_keyword(response, keyword, 50)
            matches = re.findall(fallback_pattern, keyword_context, re.IGNORECASE)
            if matches:
                answer = matches[-1].upper()
                logger.debug(f"🎯 Fallback extracted answer '{answer}' near keyword '{keyword}'")
                return answer
        
        # Final fallback: first occurrence of A-E in response
        all_letters = re.findall(r'\b([A-E])\b', response)
        if all_letters:
            answer = all_letters[0].upper()
            logger.warning(f"⚠️ Using first letter found: {answer}")
            return answer
        
        # Absolute fallback
        logger.warning("⚠️ No answer found, defaulting to A")
        return "A"

    def _extract_context_around_keyword(self, text: str, keyword: str, context_chars: int) -> str:
        """Extract context around a keyword for targeted answer extraction."""
        keyword_pos = text.lower().find(keyword.lower())
        if keyword_pos == -1:
            return ""
        
        start = max(0, keyword_pos - context_chars)
        end = min(len(text), keyword_pos + len(keyword) + context_chars)
        return text[start:end]

    def _validate_and_format_response(self, response: str, extracted_answer: str) -> str:
        """Validate and format response to ensure proper answer format."""
        # Check if response already has proper format
        if re.search(r'Answer:\s*[A-E]', response, re.IGNORECASE):
            return response
        
        # Add proper answer format if missing
        if not response.strip().endswith(f"Answer: {extracted_answer}"):
            if response.strip():
                response += f"\n\nAnswer: {extracted_answer}"
            else:
                response = f"Analysis: Based on the medical knowledge provided.\n\nAnswer: {extracted_answer}"
        
        return response

    def _generate_fallback_response(self, query: str) -> str:
        """Generate fallback response when main generation fails."""
        logger.warning("⚠️ Using fallback response generation")
        
        fallback_response = """Analysis: Unable to complete full hierarchical analysis due to technical constraints. Based on general medical knowledge principles.

Answer: A"""
        
        return fallback_response

    def generate(
        self,
        query: str,
        context: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Enhanced traditional generation method (fallback)."""
        logger.info("🔄 Using enhanced traditional generation method")
        
        # Prepare enhanced context with medical specialty information
        context_text = self._format_traditional_context(context)

        # Enhanced prompt for medical Q&A
        system_prompt = system_prompt or self.config.config["prompts"]["system"]
        prompt = f"""{system_prompt}

=== MEDICAL CONTEXT ===
{context_text}

=== MEDICAL QUESTION ===
{query}

=== INSTRUCTIONS ===
Analyze the question systematically and select the correct answer.
Provide brief reasoning and end with "Answer: [LETTER]"

Response:"""

        # Generate response
        try:
            response = self._call_ollama_enhanced(prompt)
        except Exception as e:
            logger.error(f"❌ Traditional generation failed: {e}")
            return self._generate_fallback_response(query)
        
        # Extract and validate answer
        extracted_answer = self._extract_answer_enhanced(response)
        validated_response = self._validate_and_format_response(response, extracted_answer)
        
        logger.info(f"✅ Generated traditional response with answer: {extracted_answer}")
        return validated_response

    def _format_traditional_context(self, context: List[Dict[str, str]]) -> str:
        """Format traditional context with enhanced medical information."""
        if not context:
            return "No specific medical context available."
        
        context_parts = []
        for i, doc in enumerate(context[:8]):  # Increased context for better coverage
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            specialty = metadata.get('medical_specialty', 'General')
            source = metadata.get('source', 'Unknown')
            
            # Enhanced formatting with medical metadata
            context_parts.append(f"[{i+1}] {specialty} - {source}:\n{text[:400]}...")
        
        return "\n\n".join(context_parts)

    def generate_with_citations(
        self,
        query: str,
        context: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> Dict[str, any]:
        """Generate response with enhanced medical citations."""
        logger.info("📚 Generating response with enhanced medical citations")
        
        # Prepare context with enhanced citations
        context_text = "\n\n".join([
            f"[{i+1}] Medical Source - {doc.get('metadata', {}).get('medical_specialty', 'General')}:\n{doc['text']}"
            for i, doc in enumerate(context)
        ])

        # Enhanced prompt with citation requirements
        system_prompt = system_prompt or self.config.config["prompts"]["system"]
        citation_prompt = f"""{system_prompt}

CITATION REQUIREMENTS:
- Reference sources using [1], [2], etc.
- Prioritize high-quality medical sources
- Indicate specialty areas when relevant

=== MEDICAL SOURCES ===
{context_text}

=== MEDICAL QUESTION ===
{query}

Provide analysis with citations and end with "Answer: [LETTER]"

Response:"""

        # Generate response
        try:
            response = self._call_ollama_enhanced(citation_prompt)
        except Exception as e:
            logger.error(f"❌ Citation generation failed: {e}")
            response = self._generate_fallback_response(query)

        # Extract answer and validate
        extracted_answer = self._extract_answer_enhanced(response)
        validated_response = self._validate_and_format_response(response, extracted_answer)

        # Extract enhanced citations
        citations = self._extract_enhanced_citations(validated_response, context)

        logger.info(f"✅ Generated response with {len(citations)} medical citations")
        return {
            "response": validated_response,
            "citations": citations,
            "extracted_answer": extracted_answer
        }

    def _extract_enhanced_citations(self, response: str, context: List[Dict[str, str]]) -> List[Dict]:
        """Extract enhanced citations with medical metadata."""
        citations = []
        
        for i, doc in enumerate(context):
            citation_ref = f"[{i+1}]"
            if citation_ref in response:
                metadata = doc.get('metadata', {})
                citations.append({
                    "text": doc["text"][:200] + "...",  # Truncated for display
                    "source": metadata.get('source', 'Unknown'),
                    "medical_specialty": metadata.get('medical_specialty', 'General'),
                    "evidence_level": metadata.get('evidence_level', 'Unknown'),
                    "reference_number": i + 1
                })
        
        return citations