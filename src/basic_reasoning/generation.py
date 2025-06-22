"""
Generation module for Basic Reasoning system.
Implements hierarchical diagnostic reasoning generation.
"""

from typing import Dict, List, Optional
import requests
import json
from loguru import logger

from .config import Config

class HierarchicalGenerator:
    """Text generator for hierarchical diagnostic reasoning."""

    def __init__(self, config: Config):
        """Initialize hierarchical generator."""
        self.config = config
        
        llm_config = config.config["models"]["llm"]
        
        self.model_name = llm_config["name"]
        self.temperature = llm_config["temperature"]
        self.context_window = llm_config["context_window"]
        self.ollama_url = "http://localhost:11434/api/generate"

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API directly."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.context_window
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated")
            else:
                logger.error(f"Ollama API returned status {response.status_code}: {response.text}")
                return f"Error: Ollama API returned status {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API call failed: {e}")
            return f"Error: Unable to connect to Ollama: {e}"

    def generate_hierarchical_response(
        self,
        query: str,
        hierarchical_results: Dict[str, List[Dict]]
    ) -> str:
        """Generate response using hierarchical three-tier reasoning."""
        
        # Prepare context from each tier
        tier1_context = "\n".join([
            f"Pattern {i+1}: {doc['text'][:200]}..."
            for i, doc in enumerate(hierarchical_results["tier1_patterns"])
        ])
        
        tier2_context = "\n".join([
            f"Hypothesis {i+1}: {doc['text'][:300]}..."
            for i, doc in enumerate(hierarchical_results["tier2_hypotheses"])
        ])
        
        tier3_context = "\n".join([
            f"Evidence {i+1}: {doc['text'][:300]}..."
            for i, doc in enumerate(hierarchical_results["tier3_confirmation"])
        ])

        # Build hierarchical prompt
        system_prompt = self.config.config["prompts"]["system"]
        tier1_prompt = self.config.config["prompts"]["tier1_prompt"]
        tier2_prompt = self.config.config["prompts"]["tier2_prompt"]
        tier3_prompt = self.config.config["prompts"]["tier3_prompt"]
        
        prompt = f"""{system_prompt}

TIER 1 - PATTERN RECOGNITION:
{tier1_prompt}
{tier1_context}

TIER 2 - HYPOTHESIS TESTING:
{tier2_prompt}
{tier2_context}

TIER 3 - CONFIRMATION:
{tier3_prompt}
{tier3_context}

Question: {query}

Hierarchical Diagnostic Response:"""

        response = self._call_ollama(prompt)
        logger.info("Generated hierarchical diagnostic response")

        return response

    def generate(
        self,
        query: str,
        context: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response using traditional method (fallback)."""
        # Prepare context
        context_text = "\n\n".join([
            f"Document {i+1}:\n{doc['text']}"
            for i, doc in enumerate(context)
        ])

        # Prepare prompt
        system_prompt = system_prompt or self.config.config["prompts"]["system"]
        prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

        # Generate response
        response = self._call_ollama(prompt)
        logger.info("Generated response from LLM")

        return response