"""
Data processing module for HierRAGMed.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
import PyPDF2
import tqdm


class DocumentProcessor:
    """Document processor for medical texts."""

    def __init__(self, config: Dict):
        """Initialize document processor."""
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def process_pdf(self, pdf_path: Path, metadata: Optional[Dict] = None) -> List[Dict[str, str]]:
        """Process PDF document."""
        try:
            with open(pdf_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

            # Split text into chunks
            chunks = self.text_splitter.split_text(text)

            # Prepare documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata or {}
                doc_metadata.update({
                    "doc_id": pdf_path.stem,
                    "chunk_id": i,
                    "source": str(pdf_path),
                    "type": "pdf"
                })
                documents.append({
                    "text": chunk,
                    "metadata": doc_metadata
                })

            logger.info(f"Processed {pdf_path}: {len(documents)} chunks")
            return documents

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return []

    def process_text(self, text: str, metadata: Dict) -> List[Dict[str, str]]:
        """Process text document."""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)

            # Prepare documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata.update({
                    "chunk_id": i
                })
                documents.append({
                    "text": chunk,
                    "metadata": doc_metadata
                })

            logger.info(f"Processed text: {len(documents)} chunks")
            return documents

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return []

    def process_directory(
        self,
        directory: Path,
        metadata: Optional[Dict] = None,
        file_pattern: str = "*.pdf"
    ) -> List[Dict[str, str]]:
        """Process all documents in a directory."""
        all_documents = []
        
        for file_path in tqdm.tqdm(directory.glob(file_pattern), desc="Processing documents"):
            if file_path.suffix.lower() == ".pdf":
                documents = self.process_pdf(file_path, metadata)
                all_documents.extend(documents)
            else:
                logger.warning(f"Unsupported file type: {file_path}")

        logger.info(f"Processed directory {directory}: {len(all_documents)} total chunks")
        return all_documents

    def save_documents(self, documents: List[Dict[str, str]], output_path: Path) -> None:
        """Save processed documents to JSON file."""
        try:
            with open(output_path, "w") as f:
                json.dump(documents, f, indent=2)
            logger.info(f"Saved {len(documents)} documents to {output_path}")
        except Exception as e:
            logger.error(f"Error saving documents to {output_path}: {str(e)}")

    def load_documents(self, input_path: Path) -> List[Dict[str, str]]:
        """Load processed documents from JSON file."""
        try:
            with open(input_path, "r") as f:
                documents = json.load(f)
            logger.info(f"Loaded {len(documents)} documents from {input_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents from {input_path}: {str(e)}")
            return []
    
    # Add method to load KG datasets
    def load_kg_datasets(self, kg_data_dir: Path) -> List[Dict[str, str]]:
        """Load the fetched KG datasets."""
        all_documents = []
        
        # Load PubMed abstracts
        pubmed_file = kg_data_dir / "combined" / "all_medical_data.json"
        if pubmed_file.exists():
            with open(pubmed_file, "r") as f:
                kg_data = json.load(f)
            
            for doc in kg_data:
                # Process each document through chunking
                chunks = self.process_text(doc["text"], doc["metadata"]) 
                all_documents.extend(chunks)
        
        logger.info(f"✅ Loaded {len(all_documents)} KG documents")
        return all_documents