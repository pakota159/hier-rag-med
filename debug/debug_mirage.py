#!/usr/bin/env python3
"""
MIRAGE Benchmark Analysis Script
Analyzes the mirage/benchmark.json file to check ground truth answers and data quality.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional

def load_mirage_data(file_path: str) -> Dict:
    """Load MIRAGE benchmark data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded MIRAGE data from {file_path}")
        return data
    except Exception as e:
        print(f"❌ Error loading MIRAGE data: {e}")
        return {}

def analyze_questions(data: Dict) -> Dict[str, Any]:
    """Analyze questions for missing ground truth and data quality."""
    
    stats = {
        "total_questions": 0,
        "questions_with_answers": 0,
        "questions_without_answers": 0,
        "questions_with_options": 0,
        "questions_without_options": 0,
        "datasets": {},
        "missing_answers_by_dataset": defaultdict(list),
        "answer_distribution": Counter(),
        "option_count_distribution": Counter(),
        "sample_questions": {
            "with_answers": [],
            "without_answers": [],
            "malformed": []
        }
    }
    
    print("\n" + "="*80)
    print("🔍 ANALYZING MIRAGE BENCHMARK DATA")
    print("="*80)
    
    for dataset_name, dataset_questions in data.items():
        print(f"\n📊 Analyzing dataset: {dataset_name}")
        
        dataset_stats = {
            "total": 0,
            "with_answers": 0,
            "without_answers": 0,
            "with_options": 0,
            "answer_distribution": Counter()
        }
        
        if not isinstance(dataset_questions, dict):
            print(f"⚠️ Dataset {dataset_name} is not a dictionary")
            continue
        
        for question_id, question_data in dataset_questions.items():
            stats["total_questions"] += 1
            dataset_stats["total"] += 1
            
            if not isinstance(question_data, dict):
                print(f"⚠️ Question {question_id} is not a dictionary")
                stats["sample_questions"]["malformed"].append({
                    "dataset": dataset_name,
                    "id": question_id,
                    "data": str(question_data)[:100]
                })
                continue
            
            # Check for answer
            answer = question_data.get("answer", "")
            if answer and str(answer).strip():
                stats["questions_with_answers"] += 1
                dataset_stats["with_answers"] += 1
                
                # Normalize answer
                answer_normalized = str(answer).strip().upper()
                stats["answer_distribution"][answer_normalized] += 1
                dataset_stats["answer_distribution"][answer_normalized] += 1
                
                # Sample question with answer
                if len(stats["sample_questions"]["with_answers"]) < 3:
                    stats["sample_questions"]["with_answers"].append({
                        "dataset": dataset_name,
                        "id": question_id,
                        "question": question_data.get("question", "")[:100] + "...",
                        "answer": answer_normalized,
                        "options": question_data.get("options", {})
                    })
            else:
                stats["questions_without_answers"] += 1
                dataset_stats["without_answers"] += 1
                stats["missing_answers_by_dataset"][dataset_name].append(question_id)
                
                # Sample question without answer
                if len(stats["sample_questions"]["without_answers"]) < 5:
                    stats["sample_questions"]["without_answers"].append({
                        "dataset": dataset_name,
                        "id": question_id,
                        "question": question_data.get("question", "")[:100] + "...",
                        "answer": f"'{answer}'",
                        "options": question_data.get("options", {})
                    })
            
            # Check for options
            options = question_data.get("options", {})
            if options:
                stats["questions_with_options"] += 1
                dataset_stats["with_options"] += 1
                
                # Count number of options
                if isinstance(options, dict):
                    option_count = len(options)
                elif isinstance(options, list):
                    option_count = len(options)
                else:
                    option_count = 0
                
                stats["option_count_distribution"][option_count] += 1
            else:
                stats["questions_without_options"] += 1
        
        # Save dataset stats
        stats["datasets"][dataset_name] = dataset_stats
        
        print(f"   📝 Total questions: {dataset_stats['total']:,}")
        print(f"   ✅ With answers: {dataset_stats['with_answers']:,}")
        print(f"   ❌ Without answers: {dataset_stats['without_answers']:,}")
        print(f"   📋 With options: {dataset_stats['with_options']:,}")
        
        if dataset_stats["without_answers"] > 0:
            missing_percentage = (dataset_stats["without_answers"] / dataset_stats["total"]) * 100
            print(f"   ⚠️ Missing answers: {missing_percentage:.1f}%")
    
    return stats

def print_detailed_analysis(stats: Dict[str, Any]):
    """Print detailed analysis results."""
    
    print("\n" + "="*80)
    print("📊 DETAILED ANALYSIS RESULTS")
    print("="*80)
    
    # Overall statistics
    print(f"\n🎯 OVERALL STATISTICS:")
    print(f"   📝 Total questions: {stats['total_questions']:,}")
    print(f"   ✅ Questions with answers: {stats['questions_with_answers']:,}")
    print(f"   ❌ Questions without answers: {stats['questions_without_answers']:,}")
    print(f"   📋 Questions with options: {stats['questions_with_options']:,}")
    print(f"   📄 Questions without options: {stats['questions_without_options']:,}")
    
    # Missing answer percentage
    if stats['total_questions'] > 0:
        missing_percentage = (stats['questions_without_answers'] / stats['total_questions']) * 100
        print(f"   ⚠️ Missing answer rate: {missing_percentage:.2f}%")
    
    # Answer distribution
    print(f"\n📊 ANSWER DISTRIBUTION:")
    for answer, count in stats['answer_distribution'].most_common():
        percentage = (count / stats['questions_with_answers']) * 100 if stats['questions_with_answers'] > 0 else 0
        print(f"   {answer}: {count:,} ({percentage:.1f}%)")
    
    # Option count distribution
    print(f"\n📋 OPTION COUNT DISTRIBUTION:")
    for option_count, count in sorted(stats['option_count_distribution'].items()):
        print(f"   {option_count} options: {count:,} questions")
    
    # Dataset breakdown
    print(f"\n📂 DATASET BREAKDOWN:")
    for dataset_name, dataset_stats in stats['datasets'].items():
        missing_rate = (dataset_stats['without_answers'] / dataset_stats['total']) * 100 if dataset_stats['total'] > 0 else 0
        print(f"   {dataset_name}:")
        print(f"     Total: {dataset_stats['total']:,}")
        print(f"     Missing answers: {dataset_stats['without_answers']:,} ({missing_rate:.1f}%)")
        print(f"     Answer distribution: {dict(dataset_stats['answer_distribution'])}")
    
    # Sample questions without answers
    print(f"\n❌ SAMPLE QUESTIONS WITHOUT ANSWERS:")
    for i, sample in enumerate(stats['sample_questions']['without_answers'][:5]):
        print(f"   {i+1}. Dataset: {sample['dataset']}, ID: {sample['id']}")
        print(f"      Question: {sample['question']}")
        print(f"      Answer field: {sample['answer']}")
        print(f"      Has options: {bool(sample['options'])}")
        print()
    
    # Sample questions with answers
    print(f"\n✅ SAMPLE QUESTIONS WITH ANSWERS:")
    for i, sample in enumerate(stats['sample_questions']['with_answers'][:3]):
        print(f"   {i+1}. Dataset: {sample['dataset']}, ID: {sample['id']}")
        print(f"      Question: {sample['question']}")
        print(f"      Answer: {sample['answer']}")
        print(f"      Options: {sample['options']}")
        print()

def save_analysis_report(stats: Dict[str, Any], output_file: str):
    """Save analysis report to file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"📁 Analysis report saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving report: {e}")

def main():
    """Main analysis function."""
    
    # Find MIRAGE benchmark file
    possible_paths = [
        "mirage/benchmark.json",
        "/workspace/hierragmed/mirage/benchmark.json",
        "../mirage/benchmark.json",
        "../../mirage/benchmark.json"
    ]
    
    benchmark_file = None
    for path in possible_paths:
        if Path(path).exists():
            benchmark_file = path
            break
    
    if not benchmark_file:
        print("❌ Could not find mirage/benchmark.json")
        print("📂 Searched paths:")
        for path in possible_paths:
            print(f"   - {path}")
        print("\n💡 Please run this script from the project root directory")
        sys.exit(1)
    
    print(f"📁 Found MIRAGE benchmark: {benchmark_file}")
    
    # Load and analyze data
    data = load_mirage_data(benchmark_file)
    if not data:
        sys.exit(1)
    
    stats = analyze_questions(data)
    print_detailed_analysis(stats)
    
    # Save report
    output_file = "mirage_analysis_report.json"
    save_analysis_report(stats, output_file)
    
    # Summary recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    missing_rate = (stats['questions_without_answers'] / stats['total_questions']) * 100
    
    if missing_rate > 5:
        print(f"⚠️ HIGH missing answer rate ({missing_rate:.1f}%)")
        print("   - Consider filtering out questions without ground truth")
        print("   - Check if answers are stored in different fields")
        print("   - Validate data source integrity")
    elif missing_rate > 1:
        print(f"⚠️ MODERATE missing answer rate ({missing_rate:.1f}%)")
        print("   - Review questions without answers")
        print("   - Consider excluding problematic datasets")
    else:
        print(f"✅ LOW missing answer rate ({missing_rate:.1f}%)")
        print("   - Data quality looks good")
    
    print(f"\n📊 Total usable questions: {stats['questions_with_answers']:,}")
    print(f"🎯 Evaluation should work with {stats['questions_with_answers']:,} questions")

if __name__ == "__main__":
    main()