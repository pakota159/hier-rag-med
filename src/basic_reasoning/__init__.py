"""
Basic Reasoning Module - Complete Hierarchical Diagnostic Reasoning System
Includes: Data Collection + Processing + Hierarchical RAG + Web Interface
Uses foundation datasets from data/foundation/
"""

# Data Collection
from .foundation_fetchers import (
    MedReasonFetcher,
    MSDiagnosisFetcher, 
    PMCPatientsFetcher,
    DrugBankFetcher,
    fetch_foundation_datasets,
    save_foundation_datasets
)

# Core System Components
from .config import Config
from .processing import HierarchicalDocumentProcessor
from .retrieval import HierarchicalRetriever
from .generation import HierarchicalGenerator

__version__ = "0.1.0"
__all__ = [
    # Data Collection
    "MedReasonFetcher",
    "MSDiagnosisFetcher", 
    "PMCPatientsFetcher",
    "DrugBankFetcher",
    "fetch_foundation_datasets",
    "save_foundation_datasets",
    
    # Core System
    "Config",
    "HierarchicalDocumentProcessor", 
    "HierarchicalRetriever",
    "HierarchicalGenerator"
]