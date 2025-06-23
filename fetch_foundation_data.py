#!/usr/bin/env python3
"""
Foundation Dataset Fetcher for HierRAGMed
File: fetch_foundation_data.py (SIMPLIFIED VERSION)

Fetches ALL foundation sources for maximum MIRAGE performance:
WHO, ESC, AHA/ACC, USPSTF, UpToDate, MedReason, MSDiagnosis, PMC, DrugBank, PubMed

Usage:
    python fetch_foundation_data.py --max-results 5000 --email your@email.com
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

# Import all fetchers
try:
    # Therapeutic fetchers
    from src.basic_reasoning.fetchers.who_guidelines_fetcher import WHOGuidelinesFetcher
    from src.basic_reasoning.fetchers.esc_guidelines_fetcher import ESCGuidelinesFetcher
    from src.basic_reasoning.fetchers.aha_acc_guidelines_fetcher import AHAACCGuidelinesFetcher
    from src.basic_reasoning.fetchers.uspstf_guidelines_fetcher import USPSTFGuidelinesFetcher
    from src.basic_reasoning.fetchers.uptodate_guidelines_fetcher import UpToDateGuidelinesFetcher
    THERAPEUTIC_FETCHERS_AVAILABLE = True
except ImportError:
    THERAPEUTIC_FETCHERS_AVAILABLE = False

try:
    # Exam-focused fetchers
    from src.basic_reasoning import fetch_foundation_datasets as fetch_old_foundation
    OLD_FETCHERS_AVAILABLE = True
except ImportError:
    OLD_FETCHERS_AVAILABLE = False

try:
    # PubMed foundation fetcher
    from src.basic_reasoning.fetchers.pubmed_foundation_fetcher import PubMedFoundationFetcher
    PUBMED_FETCHER_AVAILABLE = True
except ImportError:
    PUBMED_FETCHER_AVAILABLE = False

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging for the fetcher."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "foundation_data_fetch.log")
        ]
    )


def fetch_all_foundation_sources(max_results: int = 5000, email: str = "hierragmed@example.com") -> List[Dict]:
    """Fetch ALL foundation sources for maximum MIRAGE performance."""
    logger.info("🌟 FETCHING ALL FOUNDATION SOURCES")
    logger.info("=" * 70)
    logger.info("Sources: WHO, ESC, AHA/ACC, USPSTF, UpToDate, MedReason, MSDiagnosis, PMC, DrugBank, PubMed")
    logger.info(f"Target total: {max_results} documents")
    logger.info("Expected MIRAGE performance: 75-80%")
    logger.info("=" * 70)
    
    all_documents = []
    
    # Calculate distribution across 10 sources
    per_source = max_results // 10
    
    # 1. WHO International Guidelines
    if THERAPEUTIC_FETCHERS_AVAILABLE and per_source > 0:
        logger.info(f"🌍 Fetching WHO guidelines (max {per_source})")
        try:
            fetcher = WHOGuidelinesFetcher(email)
            docs = fetcher.fetch_who_guidelines(per_source)
            all_documents.extend(docs)
            logger.info(f"✅ WHO: {len(docs)} guidelines")
        except Exception as e:
            logger.error(f"❌ WHO fetch failed: {e}")
    
    # 2. ESC Cardiovascular Guidelines
    if THERAPEUTIC_FETCHERS_AVAILABLE and per_source > 0:
        logger.info(f"❤️ Fetching ESC guidelines (max {per_source})")
        try:
            fetcher = ESCGuidelinesFetcher(email)
            docs = fetcher.fetch_esc_guidelines(per_source)
            all_documents.extend(docs)
            logger.info(f"✅ ESC: {len(docs)} cardiovascular guidelines")
        except Exception as e:
            logger.error(f"❌ ESC fetch failed: {e}")
    
    # 3. AHA/ACC Guidelines
    if THERAPEUTIC_FETCHERS_AVAILABLE and per_source > 0:
        logger.info(f"🇺🇸 Fetching AHA/ACC guidelines (max {per_source})")
        try:
            fetcher = AHAACCGuidelinesFetcher(email)
            docs = fetcher.fetch_aha_acc_guidelines(per_source)
            all_documents.extend(docs)
            logger.info(f"✅ AHA/ACC: {len(docs)} treatment standards")
        except Exception as e:
            logger.error(f"❌ AHA/ACC fetch failed: {e}")
    
    # 4. USPSTF Preventive Guidelines
    if THERAPEUTIC_FETCHERS_AVAILABLE and per_source > 0:
        logger.info(f"🛡️ Fetching USPSTF guidelines (max {per_source})")
        try:
            fetcher = USPSTFGuidelinesFetcher(email)
            docs = fetcher.fetch_uspstf_guidelines(per_source)
            all_documents.extend(docs)
            logger.info(f"✅ USPSTF: {len(docs)} preventive recommendations")
        except Exception as e:
            logger.error(f"❌ USPSTF fetch failed: {e}")
    
    # 5. UpToDate Clinical Recommendations
    if THERAPEUTIC_FETCHERS_AVAILABLE and per_source > 0:
        logger.info(f"📚 Fetching UpToDate recommendations (max {per_source})")
        try:
            fetcher = UpToDateGuidelinesFetcher(email)
            docs = fetcher.fetch_uptodate_guidelines(per_source)
            all_documents.extend(docs)
            logger.info(f"✅ UpToDate: {len(docs)} clinical guidance")
        except Exception as e:
            logger.error(f"❌ UpToDate fetch failed: {e}")
    
    # 6-9. Exam/Reasoning Datasets (MedReason, MSDiagnosis, PMC, DrugBank)
    if OLD_FETCHERS_AVAILABLE and per_source > 0:
        logger.info(f"📚 Fetching exam & reasoning datasets (max {per_source * 4})")
        try:
            exam_docs = fetch_old_foundation(
                max_medreason=per_source,
                max_msdiagnosis=per_source,
                max_pmc=per_source,
                max_drugbank=per_source,
                email=email
            )
            all_documents.extend(exam_docs)
            logger.info(f"✅ Exam datasets: {len(exam_docs)} documents")
        except Exception as e:
            logger.error(f"❌ Exam datasets fetch failed: {e}")
    
    # 10. PubMed Research Literature
    if PUBMED_FETCHER_AVAILABLE and per_source > 0:
        logger.info(f"📖 Fetching PubMed research literature (max {per_source})")
        try:
            fetcher = PubMedFoundationFetcher(email)
            pubmed_docs = fetcher.fetch_pubmed_foundation(per_source)
            all_documents.extend(pubmed_docs)
            logger.info(f"✅ PubMed: {len(pubmed_docs)} research abstracts")
        except Exception as e:
            logger.error(f"❌ PubMed fetch failed: {e}")
    
    logger.info(f"🎉 ALL SOURCES COMPLETE: {len(all_documents)} total documents")
    logger.info(f"📊 Coverage: Therapeutic + Exam + Research datasets")
    logger.info(f"🎯 Expected MIRAGE performance: 75-80% (vs 54% current)")
    
    return all_documents


def save_foundation_dataset(documents: List[Dict], output_dir: Path) -> None:
    """Save foundation dataset with comprehensive metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_file = output_dir / f"foundation_all_data_{timestamp}.json"
    with open(main_file, "w") as f:
        json.dump(documents, f, indent=2)
    
    # Also save as latest (required by hierarchical system)
    latest_file = output_dir / "foundation_medical_data.json"
    with open(latest_file, "w") as f:
        json.dump(documents, f, indent=2)
    
    # Generate comprehensive statistics
    stats = {
        "dataset_info": {
            "type": "all_sources",
            "total_documents": len(documents),
            "generation_time": datetime.now().isoformat(),
            "sources_included": ["WHO", "ESC", "AHA/ACC", "USPSTF", "UpToDate", "MedReason", "MSDiagnosis", "PMC", "DrugBank", "PubMed"],
            "expected_mirage_performance": "75-80%"
        },
        "sources": {},
        "tiers": {},
        "organizations": {},
        "medical_specialties": {},
        "evidence_levels": {},
        "therapeutic_areas": {}
    }
    
    # Analyze documents
    for doc in documents:
        metadata = doc.get("metadata", {})
        
        # Source distribution
        source = metadata.get("source", "unknown")
        stats["sources"][source] = stats["sources"].get(source, 0) + 1
        
        # Tier distribution
        tier = metadata.get("tier", 0)
        stats["tiers"][f"tier_{tier}"] = stats["tiers"].get(f"tier_{tier}", 0) + 1
        
        # Organization distribution
        org = metadata.get("organization", "Unknown")
        stats["organizations"][org] = stats["organizations"].get(org, 0) + 1
        
        # Specialty distribution
        specialty = metadata.get("medical_specialty", "General")
        stats["medical_specialties"][specialty] = stats["medical_specialties"].get(specialty, 0) + 1
        
        # Evidence level distribution
        evidence = metadata.get("evidence_level", "unknown")
        stats["evidence_levels"][evidence] = stats["evidence_levels"].get(evidence, 0) + 1
        
        # Therapeutic area distribution
        area = metadata.get("therapeutic_area", "General")
        stats["therapeutic_areas"][area] = stats["therapeutic_areas"].get(area, 0) + 1
    
    # Save statistics
    stats_file = output_dir / "foundation_all_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    # Save tier distribution for hierarchical system
    tier_file = output_dir / "tier_distribution.json"
    tier_data = {
        "tier_1_pattern_recognition": stats["tiers"].get("tier_1", 0),
        "tier_2_hypothesis_testing": stats["tiers"].get("tier_2", 0),
        "tier_3_confirmation": stats["tiers"].get("tier_3", 0),
        "total": len(documents)
    }
    with open(tier_file, "w") as f:
        json.dump(tier_data, f, indent=2)
    
    logger.info(f"💾 Saved foundation dataset: {main_file}")
    logger.info(f"💾 Saved as latest: {latest_file}")
    logger.info(f"📊 Saved statistics: {stats_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Foundation Dataset Fetcher for HierRAGMed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Fetches ALL foundation sources for maximum MIRAGE performance:
- WHO International Guidelines
- ESC Cardiovascular Guidelines  
- AHA/ACC Treatment Standards
- USPSTF Preventive Guidelines
- UpToDate Clinical Recommendations
- MedReason Knowledge Graph Reasoning
- MSDiagnosis Multi-step Diagnostics
- PMC Patient Cases
- DrugBank Pharmacology
- PubMed Research Literature

Expected MIRAGE Performance: 75-80% (vs 54% current)

Example:
  python fetch_foundation_data.py --max-results 5000 --email your@email.com
        """
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        default=5000,
        help="Maximum total documents to fetch across all sources (default: 5000)"
    )
    parser.add_argument(
        "--email",
        type=str,
        default="hierragmed@example.com",
        help="Email for API requests (default: hierragmed@example.com)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/foundation_dataset"),
        help="Output directory (default: data/foundation_dataset)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with reduced document counts (max 1000)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Adjust for quick mode
    if args.quick:
        max_results = min(args.max_results, 1000)
        logger.info(f"🏃 Quick mode: reduced to {max_results} documents")
    else:
        max_results = args.max_results
    
    logger.info("🚀 HierRAGMed Foundation Dataset Fetcher (ALL SOURCES)")
    logger.info(f"📄 Max documents: {max_results}")
    logger.info(f"📧 Email: {args.email}")
    logger.info(f"📁 Output: {args.output_dir}")
    
    try:
        start_time = time.time()
        
        # Fetch all foundation sources
        documents = fetch_all_foundation_sources(
            max_results=max_results,
            email=args.email
        )
        
        if not documents:
            logger.error("❌ No documents were fetched!")
            logger.error("Check that fetcher modules are available:")
            logger.error(f"  - Therapeutic fetchers: {'✅' if THERAPEUTIC_FETCHERS_AVAILABLE else '❌'}")
            logger.error(f"  - Exam fetchers: {'✅' if OLD_FETCHERS_AVAILABLE else '❌'}")
            logger.error(f"  - PubMed fetcher: {'✅' if PUBMED_FETCHER_AVAILABLE else '❌'}")
            return 1
        
        # Save the dataset
        save_foundation_dataset(documents, args.output_dir)
        
        # Final summary
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️ Total time: {elapsed_time:.1f} seconds")
        logger.info(f"📊 Final count: {len(documents)} documents")
        logger.info(f"📈 Coverage: {len(set(doc['metadata'].get('source', 'unknown') for doc in documents))} different sources")
        
        # Next steps
        print("\n🎯 Next Steps:")
        print("1. Run hierarchical system setup:")
        print("   python setup_hierarchical_system.py")
        print("2. Test with MIRAGE evaluation:")
        print("   python src/evaluation/run_evaluation.py")
        print("3. Expected MIRAGE performance: 75-80%")
        
        logger.info("✅ Foundation dataset creation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Foundation data fetching interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"❌ Foundation data fetching failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())