"""
Basic Reasoning Module - Complete Hierarchical Diagnostic Reasoning System
Includes: Data Collection + Processing + Hierarchical RAG + Web Interface
Uses foundation datasets from data/foundation/
"""

# New Critical Medical Fetchers (REAL APIs ONLY)
from .fetchers.statpearls_fetcher import StatPearlsFetcher
from .fetchers.umls_fetcher import UMLSFetcher
from .fetchers.drugbank_fetcher import DrugBankFetcher
from .fetchers.medlineplus_fetcher import MedlinePlusFetcher

# Existing Therapeutic Guidelines Fetchers
from .fetchers.who_guidelines_fetcher import WHOGuidelinesFetcher
from .fetchers.esc_guidelines_fetcher import ESCGuidelinesFetcher
from .fetchers.aha_acc_guidelines_fetcher import AHAACCGuidelinesFetcher
from .fetchers.uspstf_guidelines_fetcher import USPSTFGuidelinesFetcher
from .fetchers.uptodate_guidelines_fetcher import UpToDateGuidelinesFetcher

# Specialty Guidelines Fetchers
from .fetchers.acog_guidelines_fetcher import ACOGGuidelinesFetcher
from .fetchers.idsa_guidelines_fetcher import IDSAGuidelinesFetcher

# Foundation Data Fetcher
from .fetchers.pubmed_foundation_fetcher import PubMedFoundationFetcher

# Core System Components
from .config import Config
from .processing import HierarchicalDocumentProcessor
from .retrieval import HierarchicalRetriever
from .generation import HierarchicalGenerator

__version__ = "0.1.0"
__all__ = [
    # New Critical Medical Fetchers
    "StatPearlsFetcher",
    "UMLSFetcher", 
    "DrugBankFetcher",
    "MedlinePlusFetcher",
    
    # Existing Therapeutic Guidelines Fetchers
    "WHOGuidelinesFetcher",
    "ESCGuidelinesFetcher",
    "AHAACCGuidelinesFetcher",
    "USPSTFGuidelinesFetcher",
    "UpToDateGuidelinesFetcher",
    
    # Specialty Guidelines Fetchers
    "ACOGGuidelinesFetcher",
    "IDSAGuidelinesFetcher",
    
    # Foundation Data Fetcher
    "PubMedFoundationFetcher",
    
    # Core System
    "Config",
    "HierarchicalDocumentProcessor", 
    "HierarchicalRetriever",
    "HierarchicalGenerator"
]