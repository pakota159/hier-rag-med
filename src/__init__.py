"""
HierRAGMed - Medical RAG System
"""

# Remove ALL problematic imports - these modules don't exist at root level
# from config import Config
# from generation import Generator  
# from retrieval import Retriever
# from evaluation import Evaluator
# from processing import DocumentProcessor
# from web import app, start_server

__version__ = "0.1.0"

# Only expose submodules that actually exist
__all__ = [
    "simple",
    "kg", 
    "basic_reasoning",
    "evaluation"
]