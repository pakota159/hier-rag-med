"""
Generation module for HierRAGMed.
"""

from typing import Dict, List, Optional

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from loguru import logger

from config import Config


class Generator:
    """Text generator using Ollama LLM."""

    def __init__(self, config: Config):
        """Initialize generator."""
        self.config = config
        self.llm = Ollama(
            model=config.models["llm"]["name"],
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=config.models["llm"]["temperature"],
            num_ctx=config.models["llm"]["context_window"]
        )

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
        system_prompt = system_prompt or self.config.prompts["system"]
        prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

        # Generate response
        response = self.llm(prompt)
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
        system_prompt = system_prompt or self.config.prompts["system_with_citations"]
        prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

        # Generate response
        response = self.llm(prompt)
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