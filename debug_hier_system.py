#!/usr/bin/env python3
"""
Complete Hierarchical System Debug Script
File: debug_hierarchical_system.py

Tests every component to identify performance issues.
Run this to find exactly where the problem is.
"""

import sys
import json
import traceback
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_1_foundation_data():
    """Test 1: Check foundation data quality and sources."""
    print("=" * 60)
    print("🔍 TEST 1: Foundation Data Analysis")
    print("=" * 60)
    
    try:
        foundation_files = [
            Path('data/foundation_dataset/foundation_medical_data.json'),
            Path('data/foundation/foundation_medical_data.json')
        ]
        
        for f in foundation_files:
            if f.exists():
                print(f"📁 Found foundation data: {f}")
                with open(f) as file:
                    data = json.load(file)
                
                print(f"📊 Total documents: {len(data)}")
                
                # Analyze sources
                sources = {}
                therapeutic_count = 0
                for doc in data:
                    source = doc['metadata'].get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
                    
                    # Check for therapeutic content
                    text = doc.get('text', '').lower()
                    if any(word in text for word in ['benefit', 'reduce', 'improve', 'prevent', 'effective']):
                        therapeutic_count += 1
                
                print("🔬 Data sources:")
                for source, count in sources.items():
                    print(f"   {source}: {count} docs")
                
                therapeutic_ratio = therapeutic_count / len(data)
                print(f"📈 Therapeutic content: {therapeutic_count}/{len(data)} ({therapeutic_ratio*100:.1f}%)")
                
                if therapeutic_ratio > 0.7:
                    print("✅ STRONG therapeutic focus detected")
                elif therapeutic_ratio > 0.3:
                    print("🔄 MODERATE therapeutic focus detected")
                else:
                    print("❌ LOW therapeutic focus - mostly exam/danger focused")
                
                return True, len(data), therapeutic_ratio
        
        print("❌ No foundation data found")
        return False, 0, 0
        
    except Exception as e:
        print(f"❌ Foundation data test failed: {e}")
        traceback.print_exc()
        return False, 0, 0


def test_2_collections():
    """Test 2: Check if hierarchical collections are working."""
    print("\n" + "=" * 60)
    print("🔍 TEST 2: Hierarchical Collections")
    print("=" * 60)
    
    try:
        from basic_reasoning.config import Config
        from basic_reasoning.retrieval import HierarchicalRetriever
        
        config = Config()
        retriever = HierarchicalRetriever(config)
        
        # Load collections
        retriever.load_hierarchical_collections()
        print("✅ Collections loaded successfully")
        
        # Test search
        test_queries = [
            "diabetes treatment metformin",
            "medical ethics disclosure",
            "research methodology",
            "clinical procedure guidelines"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            results = retriever.hierarchical_search(query)
            
            total_results = sum(len(docs) for docs in results.values())
            print(f"   Total retrieved: {total_results} documents")
            
            for tier_name, docs in results.items():
                print(f"   {tier_name}: {len(docs)} docs")
                if docs:
                    # Analyze retrieved content
                    sample_doc = docs[0]
                    content = sample_doc.get('content', sample_doc.get('text', ''))[:150]
                    source = sample_doc.get('metadata', {}).get('source', 'unknown')
                    print(f"      Sample: {content}...")
                    print(f"      Source: {source}")
        
        return True
        
    except Exception as e:
        print(f"❌ Collections test failed: {e}")
        traceback.print_exc()
        return False


def test_3_generation():
    """Test 3: Check generation with different methods."""
    print("\n" + "=" * 60)
    print("🔍 TEST 3: Generation Testing")
    print("=" * 60)
    
    try:
        from basic_reasoning.config import Config
        from basic_reasoning.generation import HierarchicalGenerator
        import inspect
        
        config = Config()
        generator = HierarchicalGenerator(config)
        
        # Check available methods
        methods = [m for m in dir(generator) if not m.startswith('_') and callable(getattr(generator, m))]
        print(f"📋 Available methods: {methods}")
        
        # Check method signatures
        for method_name in ['generate', 'generate_hierarchical_response']:
            if hasattr(generator, method_name):
                method = getattr(generator, method_name)
                sig = inspect.signature(method)
                print(f"🔍 {method_name} signature: {sig}")
        
        # Test direct Ollama call
        print("\n🤖 Testing direct Ollama call:")
        test_prompt = "Answer this multiple choice question. Choose A, B, C, or D.\nQuestion: What is the best first-line treatment for type 2 diabetes?\nA) Insulin\nB) Metformin\nC) Sulfonylureas\nD) Diet only\nAnswer:"
        
        try:
            response = generator._call_ollama(test_prompt)
            print(f"   Raw response: '{response}'")
            print(f"   Response length: {len(response)} chars")
            
            # Check if it's a valid letter answer
            clean_response = response.strip().upper()
            if clean_response in ['A', 'B', 'C', 'D']:
                print("✅ Gives single letter answer")
            elif any(letter in clean_response for letter in ['A', 'B', 'C', 'D']):
                print("🔄 Contains letter but with extra text")
            else:
                print("❌ No letter answer found")
            
            return True, response
            
        except Exception as e:
            print(f"❌ Direct Ollama call failed: {e}")
            return False, str(e)
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        traceback.print_exc()
        return False, str(e)


def test_4_mirage_questions():
    """Test 4: Analyze MIRAGE questions and test system on them."""
    print("\n" + "=" * 60)
    print("🔍 TEST 4: MIRAGE Question Analysis")
    print("=" * 60)
    
    try:
        # Load MIRAGE data
        mirage_file = Path('mirage/benchmark.json')
        if not mirage_file.exists():
            print(f"❌ MIRAGE data not found at {mirage_file}")
            return False
        
        with open(mirage_file) as f:
            data = json.load(f)
        
        # Analyze question types
        question_types = {}
        total_questions = 0
        
        for dataset_name, questions in data.items():
            question_count = len(questions)
            question_types[dataset_name] = question_count
            total_questions += question_count
        
        print(f"📊 MIRAGE composition:")
        for dataset, count in question_types.items():
            percentage = (count / total_questions) * 100
            print(f"   {dataset}: {count} questions ({percentage:.1f}%)")
        
        # Test on different question types
        print("\n🧪 Testing system on sample questions:")
        
        try:
            from basic_reasoning.config import Config
            from basic_reasoning.retrieval import HierarchicalRetriever
            from basic_reasoning.generation import HierarchicalGenerator
            
            config = Config()
            retriever = HierarchicalRetriever(config)
            generator = HierarchicalGenerator(config)
            retriever.load_hierarchical_collections()
            
            # Test on one question from each type
            for dataset_name, questions in data.items():
                if not questions:
                    continue
                    
                first_question_id = list(questions.keys())[0]
                sample_question = questions[first_question_id]
                
                question_text = sample_question.get('question', '')
                expected_answer = sample_question.get('answer', '')
                options = sample_question.get('options', {})
                
                print(f"\n📋 {dataset_name.upper()} question test:")
                print(f"   Question: {question_text[:100]}...")
                print(f"   Expected: {expected_answer}")
                
                # Test retrieval
                hierarchical_results = retriever.hierarchical_search(question_text)
                total_retrieved = sum(len(docs) for docs in hierarchical_results.values())
                print(f"   Retrieved: {total_retrieved} docs")
                
                # Check relevance of retrieved content
                if total_retrieved > 0:
                    sample_content = ""
                    for docs in hierarchical_results.values():
                        if docs:
                            sample_content = docs[0].get('content', docs[0].get('text', ''))
                            break
                    
                    # Check if retrieved content matches question type
                    content_lower = sample_content.lower()
                    if 'ethics' in question_text.lower() and 'benefit' in content_lower:
                        print("   ⚠️ Ethics question but therapeutic content retrieved")
                    elif 'treatment' in question_text.lower() and 'benefit' in content_lower:
                        print("   ✅ Treatment question with therapeutic content")
                    else:
                        print("   🔄 Mixed relevance")
                
                # Test generation (try different methods)
                try:
                    # Method 1: Try with hierarchical results
                    if hasattr(generator, 'generate_hierarchical_response'):
                        sig = str(inspect.signature(generator.generate_hierarchical_response))
                        if 'hierarchical_results' in sig:
                            model_response = generator.generate_hierarchical_response(hierarchical_results, question_text)
                        else:
                            model_response = generator.generate_hierarchical_response(question_text)
                    else:
                        # Method 2: Try regular generate
                        context = []
                        for docs in hierarchical_results.values():
                            context.extend(docs)
                        model_response = generator.generate(question_text, context)
                    
                    print(f"   Model output: '{model_response.strip()}'")
                    
                    # Check accuracy
                    model_answer = model_response.strip().upper()
                    if model_answer == expected_answer:
                        print("   ✅ CORRECT")
                    elif expected_answer in model_answer:
                        print("   🔄 CONTAINS CORRECT")
                    else:
                        print("   ❌ WRONG")
                        
                except Exception as e:
                    print(f"   ❌ Generation failed: {e}")
                
                # Only test first few to avoid too much output
                break
            
            return True
            
        except Exception as e:
            print(f"❌ System test on MIRAGE failed: {e}")
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ MIRAGE analysis failed: {e}")
        traceback.print_exc()
        return False


def test_5_comparison():
    """Test 5: Compare with baseline/random performance."""
    print("\n" + "=" * 60)
    print("🔍 TEST 5: Performance Comparison")
    print("=" * 60)
    
    try:
        # Calculate expected random performance
        mirage_file = Path('mirage/benchmark.json')
        if mirage_file.exists():
            with open(mirage_file) as f:
                data = json.load(f)
            
            # Analyze option counts
            option_counts = {}
            for dataset_name, questions in data.items():
                for question_id, question in questions.items():
                    options = question.get('options', {})
                    num_options = len(options)
                    option_counts[num_options] = option_counts.get(num_options, 0) + 1
            
            print("📊 MIRAGE option analysis:")
            total_questions = sum(option_counts.values())
            weighted_random = 0
            
            for num_options, count in option_counts.items():
                percentage = (count / total_questions) * 100
                random_chance = (1 / num_options) * 100 if num_options > 0 else 0
                weighted_random += random_chance * (count / total_questions)
                print(f"   {num_options} options: {count} questions ({percentage:.1f}%) - Random: {random_chance:.1f}%")
            
            print(f"\n📈 Expected random performance: {weighted_random:.1f}%")
            print(f"📈 Your system performance: 40.7%")
            print(f"📈 Performance vs random: {40.7 - weighted_random:.1f} percentage points")
            
            if 40.7 > weighted_random + 10:
                print("✅ Significantly better than random")
            elif 40.7 > weighted_random:
                print("🔄 Better than random but not by much")
            else:
                print("❌ Close to or worse than random")
            
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False


def main():
    """Run all diagnostic tests."""
    print("🚀 Hierarchical System Complete Diagnostic")
    print("🎯 Goal: Find why MIRAGE performance is 40.7% instead of 70-75%")
    print("=" * 80)
    
    results = {}
    
    # Run all tests
    results['foundation'] = test_1_foundation_data()
    results['collections'] = test_2_collections()
    results['generation'] = test_3_generation()
    results['mirage'] = test_4_mirage_questions()
    results['comparison'] = test_5_comparison()
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    foundation_ok, doc_count, therapeutic_ratio = results['foundation'] if isinstance(results['foundation'], tuple) else (results['foundation'], 0, 0)
    generation_ok, sample_response = results['generation'] if isinstance(results['generation'], tuple) else (results['generation'], "")
    
    print(f"📊 Foundation Data: {'✅ OK' if foundation_ok else '❌ ISSUE'} ({doc_count} docs, {therapeutic_ratio*100:.1f}% therapeutic)")
    print(f"🔍 Collections: {'✅ OK' if results['collections'] else '❌ ISSUE'}")
    print(f"🤖 Generation: {'✅ OK' if generation_ok else '❌ ISSUE'}")
    print(f"📋 MIRAGE Compat: {'✅ OK' if results['mirage'] else '❌ ISSUE'}")
    print(f"📈 Comparison: {'✅ OK' if results['comparison'] else '❌ ISSUE'}")
    
    # Diagnosis
    print("\n🔍 LIKELY ISSUES:")
    
    if not foundation_ok:
        print("❌ Foundation data missing or corrupted")
    elif therapeutic_ratio < 0.3:
        print("❌ Foundation data not therapeutic enough for good performance")
    
    if not results['collections']:
        print("❌ Hierarchical retrieval not working properly")
    
    if not generation_ok:
        print("❌ Generation system has errors")
    elif "A" not in str(sample_response) and "B" not in str(sample_response):
        print("❌ Model not giving letter answers as expected by MIRAGE")
    
    if not results['mirage']:
        print("❌ System not compatible with MIRAGE question format")
    
    # Recommendations
    print("\n🚀 RECOMMENDATIONS:")
    
    if therapeutic_ratio > 0.7 and results['collections'] and generation_ok:
        print("💡 System components work - issue likely in prompt engineering or MIRAGE format mismatch")
        print("🔧 Try: Improve prompts to handle diverse MIRAGE question types (ethics, research, procedures)")
    elif not foundation_ok:
        print("🔧 Fix: Recreate foundation data with proper therapeutic guidelines")
    elif therapeutic_ratio < 0.5:
        print("🔧 Fix: Add more diverse medical guidelines (ethics, procedures, research)")
    else:
        print("🔧 Fix: Debug individual component issues found above")
    
    print("\n" + "=" * 80)
    print("✅ Diagnostic complete! Check issues and recommendations above.")


if __name__ == "__main__":
    main()