#!/usr/bin/env python3
"""
UNIFIED Foundation Dataset Fetcher for HierRAGMed
File: fetch_foundation_data.py (REPLACES EXISTING FILE)

Combines therapeutic guidelines with selective clinical cases.
Provides options for exam-focused (old) vs therapeutic-focused (new) datasets.

Usage:
    # NEW: Therapeutic-focused foundation (recommended)
    python fetch_foundation_data.py --therapeutic --max-results 3000
    
    # OLD: Exam-focused foundation (current behavior)
    python fetch_foundation_data.py --exam-focused --max-results 3000
    
    # HYBRID: Mix therapeutic + clinical cases
    python fetch_foundation_data.py --hybrid --max-results 3000
"""

import sys
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import both old and new fetchers
try:
    # New therapeutic fetchers
    from src.basic_reasoning.fetchers.who_guidelines_fetcher import WHOGuidelinesFetcher
    from src.basic_reasoning.fetchers.esc_guidelines_fetcher import ESCGuidelinesFetcher
    from src.basic_reasoning.fetchers.aha_acc_guidelines_fetcher import AHAACCGuidelinesFetcher
    from src.basic_reasoning.fetchers.uspstf_guidelines_fetcher import USPSTFGuidelinesFetcher
    from src.basic_reasoning.fetchers.uptodate_guidelines_fetcher import UpToDateGuidelinesFetcher
    THERAPEUTIC_FETCHERS_AVAILABLE = True
except ImportError:
    THERAPEUTIC_FETCHERS_AVAILABLE = False
    
try:
    # Old exam-focused fetchers (if they exist)
    from src.basic_reasoning import fetch_foundation_datasets as fetch_old_foundation
    OLD_FETCHERS_AVAILABLE = True
except ImportError:
    OLD_FETCHERS_AVAILABLE = False

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging for the fetcher."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "foundation_data_fetch.log")
        ]
    )


def fetch_therapeutic_foundation(
    max_who: int = 600,
    max_esc: int = 600,
    max_aha_acc: int = 600,
    max_uspstf: int = 400,
    max_uptodate: int = 800,
    email: str = "hierragmed@example.com"
) -> List[Dict]:
    """Fetch NEW therapeutic-focused foundation datasets."""
    logger.info("🎯 FETCHING THERAPEUTIC FOUNDATION (NEW APPROACH)")
    logger.info("=" * 60)
    logger.info("Focus: Clinical benefits, evidence-based treatments")
    logger.info("Expected performance: 70-75% MIRAGE (vs 54% with exam data)")
    logger.info("=" * 60)
    
    if not THERAPEUTIC_FETCHERS_AVAILABLE:
        logger.error("❌ Therapeutic fetchers not available!")
        logger.error("Please create the therapeutic fetchers first:")
        logger.error("   1. Create src/basic_reasoning/fetchers/ directory")
        logger.error("   2. Add the 5 therapeutic fetcher files")
        logger.error("   3. Re-run this script")
        return []
    
    all_documents = []
    
    # WHO International Guidelines
    if max_who > 0:
        logger.info(f"🌍 Fetching WHO guidelines (max {max_who})")
        fetcher = WHOGuidelinesFetcher(email)
        docs = fetcher.fetch_who_guidelines(max_who)
        all_documents.extend(docs)
        logger.info(f"✅ WHO: {len(docs)} therapeutic guidelines")
    
    # ESC Cardiovascular Guidelines
    if max_esc > 0:
        logger.info(f"❤️ Fetching ESC guidelines (max {max_esc})")
        fetcher = ESCGuidelinesFetcher(email)
        docs = fetcher.fetch_esc_guidelines(max_esc)
        all_documents.extend(docs)
        logger.info(f"✅ ESC: {len(docs)} cardiovascular therapeutics")
    
    # AHA/ACC Guidelines
    if max_aha_acc > 0:
        logger.info(f"🇺🇸 Fetching AHA/ACC guidelines (max {max_aha_acc})")
        fetcher = AHAACCGuidelinesFetcher(email)
        docs = fetcher.fetch_aha_acc_guidelines(max_aha_acc)
        all_documents.extend(docs)
        logger.info(f"✅ AHA/ACC: {len(docs)} treatment standards")
    
    # USPSTF Preventive Guidelines
    if max_uspstf > 0:
        logger.info(f"🛡️ Fetching USPSTF guidelines (max {max_uspstf})")
        fetcher = USPSTFGuidelinesFetcher(email)
        docs = fetcher.fetch_uspstf_guidelines(max_uspstf)
        all_documents.extend(docs)
        logger.info(f"✅ USPSTF: {len(docs)} preventive recommendations")
    
    # UpToDate Clinical Recommendations
    if max_uptodate > 0:
        logger.info(f"📚 Fetching UpToDate recommendations (max {max_uptodate})")
        fetcher = UpToDateGuidelinesFetcher(email)
        docs = fetcher.fetch_uptodate_guidelines(max_uptodate)
        all_documents.extend(docs)
        logger.info(f"✅ UpToDate: {len(docs)} clinical guidance")
    
    logger.info(f"🎉 THERAPEUTIC FOUNDATION COMPLETE: {len(all_documents)} documents")
    return all_documents


def fetch_exam_foundation(
    max_medreason: int = 1000,
    max_msdiagnosis: int = 1000,
    max_pmc: int = 500,
    max_drugbank: int = 1000,
    email: str = "hierragmed@example.com"
) -> List[Dict]:
    """Fetch OLD exam-focused foundation datasets."""
    logger.info("📚 FETCHING EXAM FOUNDATION (OLD APPROACH)")
    logger.info("=" * 60)
    logger.info("Focus: Medical exam questions, contraindications")
    logger.info("Current performance: ~54% MIRAGE")
    logger.info("⚠️  WARNING: Exam-focused, not therapeutic benefits")
    logger.info("=" * 60)
    
    if not OLD_FETCHERS_AVAILABLE:
        logger.error("❌ Old fetchers not available!")
        logger.error("This suggests the old foundation system may not be properly set up.")
        return []
    
    try:
        documents = fetch_old_foundation(
            max_medreason=max_medreason,
            max_msdiagnosis=max_msdiagnosis,
            max_pmc=max_pmc,
            max_drugbank=max_drugbank,
            email=email
        )
        logger.info(f"📚 EXAM FOUNDATION COMPLETE: {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"❌ Failed to fetch exam foundation: {e}")
        return []


def fetch_hybrid_foundation(
    # Therapeutic components (70% of total)
    max_who: int = 500,
    max_esc: int = 400,
    max_aha_acc: int = 400,
    max_uspstf: int = 300,
    max_uptodate: int = 400,
    # Clinical cases (30% of total - keep useful PMC cases)
    max_pmc: int = 500,
    email: str = "hierragmed@example.com"
) -> List[Dict]:
    """Fetch HYBRID foundation: therapeutic guidelines + clinical cases."""
    logger.info("🔄 FETCHING HYBRID FOUNDATION (MIXED APPROACH)")
    logger.info("=" * 60)
    logger.info("Mix: 70% therapeutic guidelines + 30% clinical cases")
    logger.info("Expected performance: ~65-70% MIRAGE")
    logger.info("=" * 60)
    
    all_documents = []
    
    # Fetch therapeutic guidelines (70%)
    therapeutic_docs = fetch_therapeutic_foundation(
        max_who=max_who,
        max_esc=max_esc,
        max_aha_acc=max_aha_acc,
        max_uspstf=max_uspstf,
        max_uptodate=max_uptodate,
        email=email
    )
    all_documents.extend(therapeutic_docs)
    
    # Add clinical cases for diversity (30%)
    if max_pmc > 0 and OLD_FETCHERS_AVAILABLE:
        logger.info(f"📋 Adding PMC clinical cases (max {max_pmc})")
        try:
            clinical_docs = fetch_old_foundation(
                max_medreason=0,  # Skip exam questions
                max_msdiagnosis=0,  # Skip synthetic scenarios
                max_pmc=max_pmc,  # Keep clinical cases
                max_drugbank=0,  # Skip template drugs
                email=email
            )
            all_documents.extend(clinical_docs)
            logger.info(f"✅ Added {len(clinical_docs)} clinical cases")
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch clinical cases: {e}")
    
    logger.info(f"🎉 HYBRID FOUNDATION COMPLETE: {len(all_documents)} documents")
    return all_documents


def save_foundation_datasets(documents: List[Dict], output_dir: Path, dataset_type: str) -> None:
    """Save foundation datasets with comprehensive metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_file = output_dir / f"foundation_{dataset_type}_data_{timestamp}.json"
    with open(main_file, "w") as f:
        json.dump(documents, f, indent=2)
    
    # Also save as latest
    latest_file = output_dir / "foundation_medical_data.json"
    with open(latest_file, "w") as f:
        json.dump(documents, f, indent=2)
    
    # Generate comprehensive statistics
    stats = {
        "dataset_info": {
            "type": dataset_type,
            "total_documents": len(documents),
            "generation_time": datetime.now().isoformat(),
            "therapeutic_focus": dataset_type in ["therapeutic", "hybrid"],
            "exam_focus": dataset_type == "exam"
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
        condition = metadata.get("condition", "General")
        stats["therapeutic_areas"][condition] = stats["therapeutic_areas"].get(condition, 0) + 1
    
    # Save statistics
    stats_file = output_dir / f"foundation_{dataset_type}_statistics.json"
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
    """Main execution function with unified options."""
    parser = argparse.ArgumentParser(
        description="Unified Foundation Dataset Fetcher for HierRAGMed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset Types:
  --therapeutic    NEW: Evidence-based therapeutic guidelines (70-75% MIRAGE expected)
  --exam-focused   OLD: Medical exam questions & contraindications (54% MIRAGE current)
  --hybrid         MIX: Therapeutic guidelines + clinical cases (65-70% MIRAGE expected)

Examples:
  # Recommended: New therapeutic approach
  python fetch_foundation_data.py --therapeutic --max-results 3000
  
  # Current behavior: Exam-focused
  python fetch_foundation_data.py --exam-focused --max-results 3000
  
  # Balanced: Mix therapeutic + cases
  python fetch_foundation_data.py --hybrid --max-results 2500
        """
    )
    
    # Dataset type selection (mutually exclusive)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--therapeutic",
        action="store_true",
        help="NEW: Fetch therapeutic guidelines (WHO, ESC, AHA/ACC, USPSTF, UpToDate)"
    )
    dataset_group.add_argument(
        "--exam-focused",
        action="store_true",
        help="OLD: Fetch exam-focused datasets (MedReason, MSDiagnosis, PMC, DrugBank)"
    )
    dataset_group.add_argument(
        "--hybrid",
        action="store_true",
        help="MIX: Therapeutic guidelines + clinical cases"
    )
    
    # Common arguments
    parser.add_argument(
        "--max-results",
        type=int,
        default=3000,
        help="Maximum total documents to fetch (default: 3000)"
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
        help="Quick test with reduced document counts"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Determine dataset type
    if args.therapeutic:
        dataset_type = "therapeutic"
    elif args.exam_focused:
        dataset_type = "exam"
    else:  # hybrid
        dataset_type = "hybrid"
    
    logger.info("🚀 HierRAGMed Unified Foundation Dataset Fetcher")
    logger.info(f"📊 Dataset type: {dataset_type.upper()}")
    logger.info(f"📄 Max documents: {args.max_results}")
    logger.info(f"📧 Email: {args.email}")
    logger.info(f"📁 Output: {args.output_dir}")
    
    try:
        start_time = time.time()
        
        # Adjust counts for quick mode
        if args.quick:
            max_results = min(args.max_results, 500)
            logger.info(f"🏃 Quick mode: reduced to {max_results} documents")
        else:
            max_results = args.max_results
        
        # Fetch appropriate dataset
        if dataset_type == "therapeutic":
            # Distribute across therapeutic sources
            per_source = max_results // 5
            documents = fetch_therapeutic_foundation(
                max_who=per_source,
                max_esc=per_source,
                max_aha_acc=per_source,
                max_uspstf=per_source,
                max_uptodate=per_source,
                email=args.email
            )
        elif dataset_type == "exam":
            # Distribute across exam sources
            per_source = max_results // 4
            documents = fetch_exam_foundation(
                max_medreason=per_source,
                max_msdiagnosis=per_source,
                max_pmc=per_source // 2,  # PMC is slower
                max_drugbank=per_source,
                email=args.email
            )
        else:  # hybrid
            # 70% therapeutic, 30% clinical cases
            therapeutic_count = int(max_results * 0.7)
            clinical_count = int(max_results * 0.3)
            documents = fetch_hybrid_foundation(
                max_who=therapeutic_count // 5,
                max_esc=therapeutic_count // 5,
                max_aha_acc=therapeutic_count // 5,
                max_uspstf=therapeutic_count // 5,
                max_uptodate=therapeutic_count // 5,
                max_pmc=clinical_count,
                email=args.email
            )
        
        # Save datasets
        save_foundation_datasets(documents, args.output_dir, dataset_type)
        
        # Print summary
        elapsed_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("✅ Foundation dataset fetching completed successfully!")
        logger.info(f"📊 Total documents: {len(documents)}")
        logger.info(f"⏱️  Time elapsed: {elapsed_time:.1f} seconds")
        logger.info(f"📁 Saved to: {args.output_dir}")
        
        # Performance expectations
        if dataset_type == "therapeutic":
            logger.info("🎯 Expected MIRAGE performance: 70-75% (+15-20 points)")
        elif dataset_type == "exam":
            logger.info("📚 Expected MIRAGE performance: ~54% (current)")
        else:  # hybrid
            logger.info("🔄 Expected MIRAGE performance: 65-70% (+10-15 points)")
        
        logger.info("🔄 Next steps:")
        logger.info("   1. python setup_hierarchical_system.py")
        logger.info("   2. python src/evaluation/run_evaluation.py --models hierarchical_system")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⏹️ Fetch interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())