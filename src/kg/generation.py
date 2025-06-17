"""
Generation module for HierRAGMed.
"""

from typing import Dict, List, Optional
import requests
import json
from loguru import logger

from .config import Config

class Generator:
    """Text generator using Ollama LLM."""

    def __init__(self, config: Config):
        """Initialize generator."""
        self.config = config
        
        # Access LLM config through config.config dictionary
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

    def generate(
        self,
        query: str,
        context: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response using LLM."""
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

    def generate_with_citations(
        self,
        query: str,
        context: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> Dict[str, str]:
        """Generate response with citations."""
        # Prepare context with citations
        context_text = "\n\n".join([
            f"[{i+1}] {doc['text']}"
            for i, doc in enumerate(context)
        ])

        # Prepare prompt
        system_prompt = system_prompt or self.config.config["prompts"]["system_with_citations"]
        prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

        # Generate response
        response = self._call_ollama(prompt)
        logger.info("Generated response with citations from LLM")

        # Extract citations
        citations = []
        for i, doc in enumerate(context):
            if f"[{i+1}]" in response:
                citations.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"]
                })

        return {
            "response": response,
            "citations": citations
        }