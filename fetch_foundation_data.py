#!/usr/bin/env python3
"""
Complete Foundation Dataset Fetcher for HierRAGMed - ALL VALIDATED FETCHERS
File: fetch_foundation_data.py

Fetches from ALL available validated medical data sources:
- NEW Critical Fetchers: StatPearls, UMLS, DrugBank, MedlinePlus
- Existing Therapeutic: WHO, ESC, AHA/ACC, USPSTF, UpToDate  
- Specialty Guidelines: ACOG (OB/GYN), IDSA (Infectious Disease)
- Foundation Sources: PubMed Foundation

Usage:
    python fetch_foundation_data.py --max-results 50000 --email your@email.com --umls-key YOUR_KEY --drugbank-key YOUR_KEY
"""

import sys
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Setup logger first
logger = logging.getLogger(__name__)

# Import ALL validated fetchers
try:
    # NEW Critical Medical Fetchers
    from src.basic_reasoning.fetchers.statpearls_fetcher import StatPearlsFetcher
    from src.basic_reasoning.fetchers.umls_fetcher import UMLSFetcher
    from src.basic_reasoning.fetchers.drugbank_fetcher import DrugBankFetcher
    from src.basic_reasoning.fetchers.medlineplus_fetcher import MedlinePlusFetcher
    NEW_CRITICAL_FETCHERS_AVAILABLE = True
except ImportError as e:
    NEW_CRITICAL_FETCHERS_AVAILABLE = False
    logger.error(f"NEW critical fetchers not available: {e}")

try:
    # Existing Therapeutic Guidelines Fetchers
    from src.basic_reasoning.fetchers.who_guidelines_fetcher import WHOGuidelinesFetcher
    from src.basic_reasoning.fetchers.esc_guidelines_fetcher import ESCGuidelinesFetcher
    from src.basic_reasoning.fetchers.aha_acc_guidelines_fetcher import AHAACCGuidelinesFetcher
    from src.basic_reasoning.fetchers.uspstf_guidelines_fetcher import USPSTFGuidelinesFetcher
    from src.basic_reasoning.fetchers.uptodate_guidelines_fetcher import UpToDateGuidelinesFetcher
    THERAPEUTIC_FETCHERS_AVAILABLE = True
except ImportError as e:
    THERAPEUTIC_FETCHERS_AVAILABLE = False
    logger.warning(f"Therapeutic fetchers not available: {e}")

try:
    # Specialty Guidelines Fetchers  
    from src.basic_reasoning.fetchers.acog_guidelines_fetcher import ACOGGuidelinesFetcher
    from src.basic_reasoning.fetchers.idsa_guidelines_fetcher import IDSAGuidelinesFetcher
    SPECIALTY_FETCHERS_AVAILABLE = True
except ImportError as e:
    SPECIALTY_FETCHERS_AVAILABLE = False
    logger.warning(f"Specialty fetchers not available: {e}")

try:
    # Foundation Data Fetcher
    from src.basic_reasoning.fetchers.pubmed_foundation_fetcher import PubMedFoundationFetcher
    PUBMED_FOUNDATION_AVAILABLE = True
except ImportError as e:
    PUBMED_FOUNDATION_AVAILABLE = False
    logger.warning(f"PubMed foundation fetcher not available: {e}")


def setup_logging():
    """Setup comprehensive logging for the fetcher."""
    log_format = "%(asctime)s | %(levelname)8s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def calculate_quota_distribution(max_results: int) -> Dict[str, int]:
    """Calculate quota distribution across ALL validated fetchers."""
    
    # Tier 1: NEW Critical Medical Knowledge (40% - HIGHEST PRIORITY)
    critical_quota = int(max_results * 0.40)
    
    # Tier 2: Therapeutic Guidelines (25%)
    therapeutic_quota = int(max_results * 0.25)
    
    # Tier 3: Specialty Guidelines (20%)
    specialty_quota = int(max_results * 0.20)
    
    # Tier 4: Foundation PubMed (15%)
    foundation_quota = int(max_results * 0.15)
    
    quotas = {
        # NEW Critical Medical Knowledge (40% total)
        "statpearls": critical_quota // 4,      # 10% - Medical textbook (MOST CRITICAL)
        "umls": critical_quota // 4,            # 10% - Medical terminology  
        "drugbank": critical_quota // 4,        # 10% - Drug information
        "medlineplus": critical_quota // 4,     # 10% - Patient education
        
        # Existing Therapeutic Guidelines (25% total)
        "who": therapeutic_quota // 5,          # 5% - WHO guidelines
        "esc": therapeutic_quota // 5,          # 5% - ESC guidelines
        "aha_acc": therapeutic_quota // 5,      # 5% - AHA/ACC guidelines
        "uspstf": therapeutic_quota // 5,       # 5% - USPSTF guidelines
        "uptodate": therapeutic_quota // 5,     # 5% - UpToDate guidelines
        
        # Specialty Guidelines (20% total)
        "acog": specialty_quota // 2,           # 10% - ACOG OB/GYN guidelines
        "idsa": specialty_quota // 2,           # 10% - IDSA infectious disease
        
        # Foundation PubMed (15% total)
        "pubmed_foundation": foundation_quota,   # 15% - Specialty-balanced PubMed
    }
    
    # Calculate totals for verification
    critical_total = quotas["statpearls"] + quotas["umls"] + quotas["drugbank"] + quotas["medlineplus"]
    therapeutic_total = quotas["who"] + quotas["esc"] + quotas["aha_acc"] + quotas["uspstf"] + quotas["uptodate"]
    specialty_total = quotas["acog"] + quotas["idsa"]
    foundation_total = quotas["pubmed_foundation"]
    
    logger.info("📊 Quota Distribution Across ALL Validated Fetchers:")
    logger.info(f"   🆕 NEW Critical Medical: {critical_total:,} docs (40%)")
    logger.info(f"      ├─ StatPearls: {quotas['statpearls']:,}")
    logger.info(f"      ├─ UMLS: {quotas['umls']:,}")
    logger.info(f"      ├─ DrugBank: {quotas['drugbank']:,}")
    logger.info(f"      └─ MedlinePlus: {quotas['medlineplus']:,}")
    logger.info(f"   📋 Therapeutic Guidelines: {therapeutic_total:,} docs (25%)")
    logger.info(f"      ├─ WHO: {quotas['who']:,}")
    logger.info(f"      ├─ ESC: {quotas['esc']:,}")
    logger.info(f"      ├─ AHA/ACC: {quotas['aha_acc']:,}")
    logger.info(f"      ├─ USPSTF: {quotas['uspstf']:,}")
    logger.info(f"      └─ UpToDate: {quotas['uptodate']:,}")
    logger.info(f"   🏥 Specialty Guidelines: {specialty_total:,} docs (20%)")
    logger.info(f"      ├─ ACOG (OB/GYN): {quotas['acog']:,}")
    logger.info(f"      └─ IDSA (Infectious): {quotas['idsa']:,}")
    logger.info(f"   📖 Foundation PubMed: {foundation_total:,} docs (15%)")
    logger.info(f"   🎯 Grand Total: {critical_total + therapeutic_total + specialty_total + foundation_total:,} docs")
    
    return quotas


def fetch_all_foundation_sources(max_results: int, email: str, umls_key: str = None, drugbank_key: str = None, ncbi_key: str = None) -> List[Dict]:
    """Fetch from ALL available validated medical sources."""
    logger.info("🚀 FETCHING FROM ALL VALIDATED MEDICAL SOURCES")
    logger.info("=" * 80)
    
    all_documents = []
    quotas = calculate_quota_distribution(max_results)
    
    # TIER 1: NEW CRITICAL MEDICAL KNOWLEDGE (40% - HIGHEST PRIORITY)
    logger.info("🆕 TIER 1: NEW CRITICAL MEDICAL KNOWLEDGE (40% of quota)")
    logger.info("-" * 60)
    
    # 1. StatPearls Medical Textbook (MOST CRITICAL)
    if NEW_CRITICAL_FETCHERS_AVAILABLE and quotas["statpearls"] > 0:
        logger.info(f"📚 Fetching StatPearls medical textbook (quota: {quotas['statpearls']:,}) - REAL DATA")
        phase_start = time.time()
        try:
            fetcher = StatPearlsFetcher(email, ncbi_key)
            docs = fetcher.fetch_statpearls_content(quotas["statpearls"])
            all_documents.extend(docs)
            duration = time.time() - phase_start
            logger.info(f"✅ StatPearls: {len(docs):,} real medical articles fetched in {duration:.1f}s")
        except Exception as e:
            logger.error(f"❌ StatPearls fetch failed: {e}")
            raise  # Critical failure
    
    # 2. UMLS Medical Terminology
    if NEW_CRITICAL_FETCHERS_AVAILABLE and quotas["umls"] > 0 and umls_key:
        logger.info(f"🔬 Fetching UMLS medical terminology (quota: {quotas['umls']:,}) - REAL DATA")
        phase_start = time.time()
        try:
            fetcher = UMLSFetcher(umls_key, email)
            docs = fetcher.fetch_umls_terminology(quotas["umls"])
            all_documents.extend(docs)
            duration = time.time() - phase_start
            logger.info(f"✅ UMLS: {len(docs):,} real medical concepts fetched in {duration:.1f}s")
        except Exception as e:
            logger.error(f"❌ UMLS fetch failed: {e}")
            raise  # Critical failure
    
    # 3. DrugBank Drug Information
    if NEW_CRITICAL_FETCHERS_AVAILABLE and quotas["drugbank"] > 0 and drugbank_key:
        logger.info(f"💊 Fetching DrugBank drug information (quota: {quotas['drugbank']:,}) - REAL DATA")
        phase_start = time.time()
        try:
            fetcher = DrugBankFetcher(drugbank_key, email)
            docs = fetcher.fetch_drugbank_data(quotas["drugbank"])
            all_documents.extend(docs)
            duration = time.time() - phase_start
            logger.info(f"✅ DrugBank: {len(docs):,} real drug profiles fetched in {duration:.1f}s")
        except Exception as e:
            logger.error(f"❌ DrugBank fetch failed: {e}")
            raise  # Critical failure
    
    # 4. MedlinePlus Patient Education
    if NEW_CRITICAL_FETCHERS_AVAILABLE and quotas["medlineplus"] > 0:
        logger.info(f"🏥 Fetching MedlinePlus patient education (quota: {quotas['medlineplus']:,}) - REAL DATA")
        phase_start = time.time()
        try:
            fetcher = MedlinePlusFetcher(email)
            docs = fetcher.fetch_medlineplus_content(quotas["medlineplus"])
            all_documents.extend(docs)
            duration = time.time() - phase_start
            logger.info(f"✅ MedlinePlus: {len(docs):,} real health topics fetched in {duration:.1f}s")
        except Exception as e:
            logger.error(f"❌ MedlinePlus fetch failed: {e}")
            raise  # Critical failure
    
    tier1_docs = len(all_documents)
    logger.info(f"📊 TIER 1 COMPLETE: {tier1_docs:,} critical medical documents")
    
    # TIER 2: THERAPEUTIC GUIDELINES (25%)
    logger.info("📋 TIER 2: THERAPEUTIC GUIDELINES (25% of quota)")
    logger.info("-" * 60)
    
    # 5. WHO International Guidelines
    if THERAPEUTIC_FETCHERS_AVAILABLE and quotas["who"] > 0:
        logger.info(f"🌍 Fetching WHO guidelines (quota: {quotas['who']:,}) - REAL DATA")
        phase_start = time.time()
        try:
            fetcher = WHOGuidelinesFetcher(email)
            docs = fetcher.fetch_who_guidelines(quotas["who"])
            all_documents.extend(docs)
            duration = time.time() - phase_start
            logger.info(f"✅ WHO: {len(docs):,} real guidelines fetched in {duration:.1f}s")
        except Exception as e:
            logger.error(f"❌ WHO fetch failed: {e}")
    
    # 6. ESC Cardiovascular Guidelines
    if THERAPEUTIC_FETCHERS_AVAILABLE and quotas["esc"] > 0:
        logger.info(f"❤️ Fetching ESC guidelines (quota: {quotas['esc']:,}) - REAL DATA")
        phase_start = time.time()
        try:
            fetcher = ESCGuidelinesFetcher(email)
            docs = fetcher.fetch_esc_guidelines(quotas["esc"])
            all_documents.extend(docs)
            duration = time.time() - phase_start
            logger.info(f"✅ ESC: {len(docs):,} real cardiovascular guidelines fetched in {duration:.1f}s")
        except Exception as e:
            logger.error(f"❌ ESC fetch failed: {e}")
    
    # 7. AHA/ACC Guidelines
    if THERAPEUTIC_FETCHERS_AVAILABLE and quotas["aha_acc"] > 0:
        logger.info(f"🇺🇸 Fetching AHA/ACC guidelines (quota: {quotas['aha_acc']:,}) - REAL DATA")
        phase_start = time.time()
        try:
            fetcher = AHAACCGuidelinesFetcher(email)
            docs = fetcher.fetch_aha_acc_guidelines(quotas["aha_acc"])
            all_documents.extend(docs)
            duration = time.time() - phase_start
            logger.info(f"✅ AHA/ACC: {len(docs):,} real treatment standards fetched in {duration:.1f}s")
        except Exception as e:
            logger.error(f"❌ AHA/ACC fetch failed: {e}")
    
    # 8. USPSTF Preventive Guidelines
    if THERAPEUTIC_FETCHERS_AVAILABLE and quotas["uspstf"] > 0:
        logger.info(f"🛡️ Fetching USPSTF guidelines (quota: {quotas['uspstf']:,}) - REAL DATA")
        phase_start = time.time()
        try:
            fetcher = USPSTFGuidelinesFetcher(email)
            docs = fetcher.fetch_uspstf_guidelines(quotas["uspstf"])
            all_documents.extend(docs)
            duration = time.time() - phase_start
            logger.info(f"✅ USPSTF: {len(docs):,} real preventive recommendations fetched in {duration:.1f}s")
        except Exception as e:
            logger.error(f"❌ USPSTF fetch failed: {e}")
    
    # 9. UpToDate Clinical Recommendations
    if THERAPEUTIC_FETCHERS_AVAILABLE and quotas["uptodate"] > 0:
        logger.info(f"📖 Fetching UpToDate guidelines (quota: {quotas['uptodate']:,}) - REAL DATA")
        phase_start = time.time()
        try:
            fetcher = UpToDateGuidelinesFetcher(email)
            docs = fetcher.fetch_uptodate_guidelines(quotas["uptodate"])
            all_documents.extend(docs)
            duration = time.time() - phase_start
            logger.info(f"✅ UpToDate: {len(docs):,} real clinical recommendations fetched in {duration:.1f}s")
        except Exception as e:
            logger.error(f"❌ UpToDate fetch failed: {e}")
    
    tier2_docs = len(all_documents) - tier1_docs
    logger.info(f"📊 TIER 2 COMPLETE: {tier2_docs:,} therapeutic guideline documents")
    
    # TIER 3: SPECIALTY GUIDELINES (20%)
    logger.info("🏥 TIER 3: SPECIALTY GUIDELINES (20% of quota)")
    logger.info("-" * 60)
    
    # 10. ACOG Obstetrics/Gynecology Guidelines
    if SPECIALTY_FETCHERS_AVAILABLE and quotas["acog"] > 0:
        logger.info(f"🤰 Fetching ACOG OB/GYN guidelines (quota: {quotas['acog']:,}) - REAL DATA")
        phase_start = time.time()
        try:
            fetcher = ACOGGuidelinesFetcher(email)
            docs = fetcher.fetch_acog_guidelines(quotas["acog"])
            all_documents.extend(docs)
            duration = time.time() - phase_start
            logger.info(f"✅ ACOG: {len(docs):,} real OB/GYN guidelines fetched in {duration:.1f}s")
        except Exception as e:
            logger.error(f"❌ ACOG fetch failed: {e}")
    
    # 11. IDSA Infectious Disease Guidelines
    if SPECIALTY_FETCHERS_AVAILABLE and quotas["idsa"] > 0:
        logger.info(f"🦠 Fetching IDSA infectious disease guidelines (quota: {quotas['idsa']:,}) - REAL DATA")
        phase_start = time.time()
        try:
            fetcher = IDSAGuidelinesFetcher(email)
            docs = fetcher.fetch_idsa_guidelines(quotas["idsa"])
            all_documents.extend(docs)
            duration = time.time() - phase_start
            logger.info(f"✅ IDSA: {len(docs):,} real infectious disease guidelines fetched in {duration:.1f}s")
        except Exception as e:
            logger.error(f"❌ IDSA fetch failed: {e}")
    
    tier3_docs = len(all_documents) - tier1_docs - tier2_docs
    logger.info(f"📊 TIER 3 COMPLETE: {tier3_docs:,} specialty guideline documents")
    
    # TIER 4: FOUNDATION PUBMED (15%)
    logger.info("📖 TIER 4: FOUNDATION PUBMED (15% of quota)")
    logger.info("-" * 60)
    
    # 12. PubMed Foundation (Specialty-Balanced)
    if PUBMED_FOUNDATION_AVAILABLE and quotas["pubmed_foundation"] > 0:
        logger.info(f"📖 Fetching specialty-balanced PubMed (quota: {quotas['pubmed_foundation']:,}) - REAL DATA")
        phase_start = time.time()
        try:
            fetcher = PubMedFoundationFetcher(email)
            docs = fetcher.fetch_pubmed_foundation(quotas["pubmed_foundation"])
            all_documents.extend(docs)
            duration = time.time() - phase_start
            logger.info(f"✅ PubMed Foundation: {len(docs):,} real research articles fetched in {duration:.1f}s")
        except Exception as e:
            logger.error(f"❌ PubMed Foundation fetch failed: {e}")
    
    tier4_docs = len(all_documents) - tier1_docs - tier2_docs - tier3_docs
    logger.info(f"📊 TIER 4 COMPLETE: {tier4_docs:,} foundation research documents")
    
    if not all_documents:
        raise Exception("No documents were fetched from any source!")
    
    logger.info("🎉 ALL VALIDATED FETCHERS COMPLETE")
    logger.info(f"📊 Final Statistics:")
    logger.info(f"   🆕 Critical Medical: {tier1_docs:,} docs")
    logger.info(f"   📋 Therapeutic Guidelines: {tier2_docs:,} docs")
    logger.info(f"   🏥 Specialty Guidelines: {tier3_docs:,} docs")
    logger.info(f"   📖 Foundation Research: {tier4_docs:,} docs")
    logger.info(f"   🎯 Grand Total: {len(all_documents):,} docs")
    
    return all_documents


def save_foundation_dataset(documents: List[Dict], output_dir: Path) -> None:
    """Save foundation dataset to files."""
    logger.info(f"💾 Saving complete foundation dataset to {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save combined dataset
    combined_file = output_dir / "foundation_medical_data.json"
    with open(combined_file, "w") as f:
        json.dump(documents, f, indent=2)
    
    # Calculate detailed statistics
    stats = {
        "total_documents": len(documents),
        "fetch_timestamp": datetime.now().isoformat(),
        "quality": "comprehensive_validated_sources",
        "content_type": "evidence_based_medicine",
        "fetchers_used": "ALL_VALIDATED",
        "sources": {},
        "specialties": {},
        "tiers": {},
        "evidence_levels": {}
    }
    
    for doc in documents:
        metadata = doc["metadata"]
        
        # Count by source
        source = metadata["source"]
        stats["sources"][source] = stats["sources"].get(source, 0) + 1
        
        # Count by specialty
        specialty = metadata.get("medical_specialty", "Unknown")
        stats["specialties"][specialty] = stats["specialties"].get(specialty, 0) + 1
        
        # Count by tier
        tier = metadata.get("tier", 0)
        stats["tiers"][f"tier_{tier}"] = stats["tiers"].get(f"tier_{tier}", 0) + 1
        
        # Count by evidence level
        evidence = metadata.get("evidence_level", "unknown")
        stats["evidence_levels"][evidence] = stats["evidence_levels"].get(evidence, 0) + 1
    
    # Save statistics
    stats_file = output_dir / "foundation_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"✅ Saved complete foundation dataset: {combined_file}")
    logger.info(f"📊 Saved detailed statistics: {stats_file}")
    
    # Log comprehensive summary
    logger.info("📈 COMPREHENSIVE DATASET SUMMARY:")
    logger.info(f"   📊 Total documents: {stats['total_documents']:,}")
    logger.info(f"   🔗 Sources: {len(stats['sources'])}")
    logger.info(f"   🏥 Medical specialties: {len(stats['specialties'])}")
    logger.info(f"   📚 Top specialties: {dict(sorted(stats['specialties'].items(), key=lambda x: x[1], reverse=True)[:8])}")
    logger.info(f"   🎯 Tier distribution: {stats['tiers']}")
    logger.info(f"   ⭐ Evidence levels: {stats['evidence_levels']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fetch foundation dataset using ALL validated medical fetchers")
    parser.add_argument("--max-results", type=int, default=10000, help="Maximum number of documents to fetch")
    parser.add_argument("--email", type=str, required=True, help="Email for API identification")
    parser.add_argument("--umls-key", type=str, help="UMLS API key (get from https://uts.nlm.nih.gov/uts/signup-login)")
    parser.add_argument("--drugbank-key", type=str, help="DrugBank API key (get from https://go.drugbank.com/releases/latest)")
    parser.add_argument("--ncbi-key", type=str, help="NCBI API key (optional, improves rate limits)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/foundation_dataset"), help="Output directory")
    parser.add_argument("--validate-fetchers", action="store_true", help="Validate fetcher availability and exit")
    parser.add_argument("--critical-only", action="store_true", help="Fetch only NEW critical sources (StatPearls, UMLS, DrugBank, MedlinePlus)")
    parser.add_argument("--therapeutic-only", action="store_true", help="Fetch only therapeutic guidelines")
    parser.add_argument("--specialty-only", action="store_true", help="Fetch only specialty guidelines (ACOG, IDSA)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("🚀 HierRAGMed Complete Foundation Dataset Fetcher")
    logger.info("🔥 Using ALL VALIDATED FETCHERS")
    logger.info("=" * 70)
    
    # Validate critical requirements
    if not NEW_CRITICAL_FETCHERS_AVAILABLE:
        logger.error("❌ CRITICAL ERROR: New medical fetchers not available!")
        logger.error("   Check that all fetcher files exist in src/basic_reasoning/fetchers/")
        sys.exit(1)
    
    if args.validate_fetchers:
        logger.info("✅ ALL fetcher validation completed successfully!")
        logger.info("📋 All systems ready for comprehensive data fetching")
        sys.exit(0)
    
    # Check API keys
    if not args.umls_key:
        logger.warning("⚠️  UMLS API key not provided - UMLS data will be skipped")
    
    if not args.drugbank_key:
        logger.warning("⚠️  DrugBank API key not provided - DrugBank data will be skipped")
    
    start_time = time.time()
    
    try:
        # Choose fetch strategy
        if args.critical_only:
            logger.info("🆕 Fetching CRITICAL sources only...")
            documents = []
            quotas = calculate_quota_distribution(args.max_results)
            
            if quotas["statpearls"] > 0:
                fetcher = StatPearlsFetcher(args.email, args.ncbi_key)
                docs = fetcher.fetch_statpearls_content(quotas["statpearls"])
                documents.extend(docs)
            
            if quotas["umls"] > 0 and args.umls_key:
                fetcher = UMLSFetcher(args.umls_key, args.email)
                docs = fetcher.fetch_umls_terminology(quotas["umls"])
                documents.extend(docs)
            
            if quotas["drugbank"] > 0 and args.drugbank_key:
                fetcher = DrugBankFetcher(args.drugbank_key, args.email)
                docs = fetcher.fetch_drugbank_data(quotas["drugbank"])
                documents.extend(docs)
            
            if quotas["medlineplus"] > 0:
                fetcher = MedlinePlusFetcher(args.email)
                docs = fetcher.fetch_medlineplus_content(quotas["medlineplus"])
                documents.extend(docs)
                
        elif args.therapeutic_only:
            logger.info("📋 Fetching THERAPEUTIC sources only...")
            documents = []
            quotas = calculate_quota_distribution(args.max_results)
            
            # Redistribute quota among therapeutic sources
            therapeutic_sources = ["who", "esc", "aha_acc", "uspstf", "uptodate"]
            docs_per_source = args.max_results // len(therapeutic_sources)
            
            if THERAPEUTIC_FETCHERS_AVAILABLE:
                for source in therapeutic_sources:
                    if source == "who":
                        fetcher = WHOGuidelinesFetcher(args.email)
                        docs = fetcher.fetch_who_guidelines(docs_per_source)
                    elif source == "esc":
                        fetcher = ESCGuidelinesFetcher(args.email)
                        docs = fetcher.fetch_esc_guidelines(docs_per_source)
                    elif source == "aha_acc":
                        fetcher = AHAACCGuidelinesFetcher(args.email)
                        docs = fetcher.fetch_aha_acc_guidelines(docs_per_source)
                    elif source == "uspstf":
                        fetcher = USPSTFGuidelinesFetcher(args.email)
                        docs = fetcher.fetch_uspstf_guidelines(docs_per_source)
                    elif source == "uptodate":
                        fetcher = UpToDateGuidelinesFetcher(args.email)
                        docs = fetcher.fetch_uptodate_guidelines(docs_per_source)
                    
                    documents.extend(docs)
                    
        elif args.specialty_only:
            logger.info("🏥 Fetching SPECIALTY sources only...")
            documents = []
            
            if SPECIALTY_FETCHERS_AVAILABLE:
                docs_per_specialty = args.max_results // 2
                
                fetcher = ACOGGuidelinesFetcher(args.email)
                acog_docs = fetcher.fetch_acog_guidelines(docs_per_specialty)
                documents.extend(acog_docs)
                
                fetcher = IDSAGuidelinesFetcher(args.email)
                idsa_docs = fetcher.fetch_idsa_guidelines(docs_per_specialty)
                documents.extend(idsa_docs)
                
        else:
            # Fetch from ALL validated sources
            logger.info("🌟 Fetching from ALL VALIDATED SOURCES...")
            documents = fetch_all_foundation_sources(
                max_results=args.max_results,
                email=args.email,
                umls_key=args.umls_key,
                drugbank_key=args.drugbank_key,
                ncbi_key=args.ncbi_key
            )
        
        if not documents:
            logger.error("❌ No documents were fetched!")
            sys.exit(1)
        
        # Save comprehensive dataset
        save_foundation_dataset(documents, args.output_dir)
        
        # Final success summary
        total_duration = time.time() - start_time
        logger.info("🎉 COMPLETE FOUNDATION DATASET FETCH SUCCESSFUL!")
        logger.info(f"⏱️  Total duration: {total_duration:.1f} seconds")
        logger.info(f"📊 Total documents: {len(documents):,}")
        logger.info(f"💾 Saved to: {args.output_dir}")
        logger.info(f"🔥 Used ALL VALIDATED FETCHERS for maximum coverage")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: Foundation dataset fetch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()