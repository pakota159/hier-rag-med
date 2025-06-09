"""
Data processing module for HierRAGMed.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm

from .config import Config


class DataProcessor:
    """Data processor for medical datasets."""

    def __init__(self, config: Config):
        """Initialize data processor."""
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.data["chunk_size"],
            chunk_overlap=config.data["chunk_overlap"],
            length_function=len,
        )

    def load_medical_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load medical dataset from HuggingFace."""
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        return pd.DataFrame(dataset["train"])

    def process_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process documents into chunks."""
        logger.info("Processing documents into chunks")
        chunks = []
        for doc in tqdm(documents):
            doc_chunks = self.text_splitter.split_text(doc["text"])
            for i, chunk in enumerate(doc_chunks):
                if i >= self.config.data["max_chunks_per_doc"]:
                    break
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source": doc.get("source", "unknown"),
                        "chunk_id": i,
                        "doc_id": doc.get("id", "unknown"),
                    }
                })
        return chunks

    def save_chunks(self, chunks: List[Dict[str, str]], output_path: Union[str, Path]) -> None:
        """Save processed chunks to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(chunks, f, indent=2)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")

    def load_chunks(self, input_path: Union[str, Path]) -> List[Dict[str, str]]:
        """Load processed chunks from file."""
        with open(input_path, "r") as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
        return chunks

    def prepare_training_data(self, dataset_name: str) -> None:
        """Prepare training data from medical dataset."""
        # Load dataset
        df = self.load_medical_dataset(dataset_name)
        
        # Convert to documents format
        documents = []
        for _, row in df.iterrows():
            doc = {
                "text": row["text"],
                "source": dataset_name,
                "id": str(row.get("id", len(documents))),
            }
            documents.append(doc)
        
        # Process documents into chunks
        chunks = self.process_documents(documents)
        
        # Save chunks
        output_path = self.config.get_data_dir("processed") / f"{dataset_name}_chunks.json"
        self.save_chunks(chunks, output_path)

    def prepare_evaluation_data(self, dataset_name: str) -> None:
        """Prepare evaluation data from medical dataset."""
        # Load dataset
        df = self.load_medical_dataset(dataset_name)
        
        # Split into train/test
        test_size = int(len(df) * self.config.evaluation["test_split"])
        test_df = df.sample(n=test_size, random_state=self.config.evaluation["random_seed"])
        
        # Save test set
        output_path = self.config.get_data_dir("processed") / f"{dataset_name}_test.json"
        test_df.to_json(output_path, orient="records", indent=2)
        logger.info(f"Saved test set to {output_path}") 