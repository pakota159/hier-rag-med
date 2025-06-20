#!/usr/bin/env python3
"""
Test script for all benchmarks to check if the download and processing is correct
"""

import sys
import json
from pathlib import Path

# Add your project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_benchmark(benchmark_name, benchmark_class):
    """Generic test function for a benchmark."""
    print(f"🧪 Testing {benchmark_name} benchmark...")
    
    try:
        # Test configuration
        config = {
            "name": benchmark_name,
            "sample_size": 10,  # Limit for testing
            "random_seed": 42
        }
        
        # Initialize benchmark
        print(f"   🔧 Initializing {benchmark_name} benchmark...")
        benchmark = benchmark_class(config)
        
        # Load dataset
        print("   📥 Loading dataset...")
        dataset = benchmark.get_questions()
        
        print(f"✅ Successfully loaded benchmark")
        print(f"   📊 Dataset size: {len(dataset)} questions")
        
        if dataset:
            # Analyze first question
            sample_q = dataset[0]
            print(f"   📋 Sample question fields: {list(sample_q.keys())}")
            print(f"   🏥 Medical specialty: {sample_q.get('medical_specialty', 'N/A')}")
            print(f"   🧠 Reasoning type: {sample_q.get('reasoning_type', 'N/A')}")
            
            # Test evaluation
            print("   🔍 Testing evaluation...")
            test_response = "The answer is A"
            eval_result = benchmark.evaluate_response(sample_q, test_response, [])
            print(f"   📈 Evaluation result: {eval_result}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   Make sure the benchmark class is in the correct location")
        return False
    except Exception as e:
        print(f"❌ Benchmark test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing All Benchmark Integrations")
    print("=" * 50)
    
    from src.evaluation.benchmarks.mirage_benchmark import MIRAGEBenchmark
    from src.evaluation.benchmarks.medreason_benchmark import MedReasonBenchmark
    from src.evaluation.benchmarks.pubmedqa_benchmark import PubMedQABenchmark
    from src.evaluation.benchmarks.msmarco_benchmark import MSMARCOBenchmark

    benchmarks_to_test = [
        ("MIRAGE", MIRAGEBenchmark),
        ("MedReason", MedReasonBenchmark),
        ("PubMedQA", PubMedQABenchmark),
        ("MSMARCO", MSMARCOBenchmark),
    ]
    
    results = []
    
    for benchmark_name, benchmark_class in benchmarks_to_test:
        print(f"\n📋 Testing {benchmark_name}")
        print("-" * 30)
        
        try:
            success = test_benchmark(benchmark_name, benchmark_class)
            results.append((benchmark_name, success))
        except Exception as e:
            print(f"❌ {benchmark_name} crashed: {e}")
            results.append((benchmark_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Result: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! All benchmarks are working correctly.")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
        
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)