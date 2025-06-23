#!/usr/bin/env python3
"""
Debug script for testing the Hierarchical Reasoning System
debug/debug_gen.py

Tests if the hierarchy system (basic_reasoning folder) works correctly
by evaluating random MIRAGE questions and showing the complete pipeline
"""

import sys
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Configure logger - minimal logging
logger.remove()
logger.add(
    sys.stdout,
    level="WARNING",
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
)

def load_mirage_questions() -> List[Dict[str, Any]]:
    """Load MIRAGE questions from local mirage folder."""
    print("🔍 Loading MIRAGE questions...")
    
    # Try to find MIRAGE benchmark.json
    mirage_paths = [
        project_root / "mirage" / "benchmark.json",
        Path("mirage") / "benchmark.json",
        Path("../mirage") / "benchmark.json",
    ]
    
    benchmark_file = None
    for path in mirage_paths:
        if path.exists():
            benchmark_file = path
            break
    
    if not benchmark_file:
        print("❌ MIRAGE benchmark.json not found!")
        print("💡 Expected locations:")
        for path in mirage_paths:
            print(f"   - {path}")
        return []
    
    try:
        print(f"📁 Loading from: {benchmark_file}")
        
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process MIRAGE data structure
        all_questions = []
        
        if isinstance(data, dict):
            for dataset_name, questions in data.items():
                if isinstance(questions, list):
                    for i, item in enumerate(questions):
                        formatted_item = format_mirage_question(item, f"{dataset_name}_{i}", dataset_name)
                        if formatted_item:
                            all_questions.append(formatted_item)
                            
                elif isinstance(questions, dict):
                    for sub_key, question_data in questions.items():
                        if isinstance(question_data, dict):
                            formatted_item = format_mirage_question(question_data, f"{dataset_name}_{sub_key}", dataset_name)
                            if formatted_item:
                                all_questions.append(formatted_item)
                                
        elif isinstance(data, list):
            for i, item in enumerate(data):
                formatted_item = format_mirage_question(item, f"mirage_{i}", "mirage")
                if formatted_item:
                    all_questions.append(formatted_item)
        
        print(f"✅ Loaded {len(all_questions)} MIRAGE questions")
        return all_questions
        
    except Exception as e:
        print(f"❌ Failed to load MIRAGE data: {e}")
        return []

def format_mirage_question(item: Dict, question_id: str, dataset_name: str) -> Optional[Dict]:
    """Format a single MIRAGE question to standard format."""
    try:
        if isinstance(item, str):
            return {
                'id': question_id,
                'question': item,
                'options': [],
                'correct_answer': '',
                'explanation': '',
                'dataset': dataset_name
            }
        
        if not isinstance(item, dict):
            return None
        
        # Extract question components with flexible field names
        question_text = (item.get('question') or 
                        item.get('query') or 
                        item.get('text') or 
                        item.get('prompt') or 
                        item.get('input') or '')
        
        # Handle options in various formats
        options = []
        if 'options' in item:
            options = item['options']
        elif 'choices' in item:
            options = item['choices']
        elif 'answers' in item:
            options = item['answers']
        else:
            # Look for A, B, C, D options
            for key in ['A', 'B', 'C', 'D', 'E']:
                if key in item:
                    options.append(item[key])
        
        # Convert options to list if it's a dict
        if isinstance(options, dict):
            options = list(options.values())
        
        correct_answer = (item.get('answer') or 
                         item.get('correct_answer') or 
                         item.get('label') or 
                         item.get('target') or 
                         item.get('output') or '')
        
        explanation = (item.get('explanation') or 
                      item.get('rationale') or 
                      item.get('reasoning') or 
                      item.get('solution') or '')
        
        # Skip items without essential fields
        if not question_text:
            return None
        
        return {
            'id': question_id,
            'question': question_text,
            'options': options,
            'correct_answer': correct_answer,
            'explanation': explanation,
            'dataset': dataset_name
        }
        
    except Exception as e:
        logger.warning(f"❌ Failed to format question {question_id}: {e}")
        return None

def initialize_hierarchical_system():
    """Initialize the hierarchical reasoning system."""
    print("🚀 Initializing Hierarchical System...")
    
    try:
        # Import hierarchical system components
        from basic_reasoning.config import Config
        from basic_reasoning.retrieval import HierarchicalRetriever
        from basic_reasoning.generation import HierarchicalGenerator
        
        # Load configuration
        config = Config()
        
        # Initialize components
        retriever = HierarchicalRetriever(config)
        generator = HierarchicalGenerator(config)
        
        # Load hierarchical collections
        retriever.load_hierarchical_collections()
        print("✅ Hierarchical system initialized")
        
        return retriever, generator, config
        
    except Exception as e:
        print(f"❌ Failed to initialize hierarchical system: {e}")
        raise

def test_hierarchical_retrieval(retriever, question: str) -> Dict[str, Any]:
    """Test hierarchical retrieval for a question."""
    try:
        start_time = time.time()
        
        # Perform hierarchical search
        hierarchical_results = retriever.hierarchical_search(question)
        
        retrieval_time = time.time() - start_time
        
        # Analyze results
        tier1_count = len(hierarchical_results.get("tier1_patterns", []))
        tier2_count = len(hierarchical_results.get("tier2_hypotheses", []))
        tier3_count = len(hierarchical_results.get("tier3_confirmation", []))
        
        return {
            'results': hierarchical_results,
            'time': retrieval_time,
            'tier_counts': {
                'tier1': tier1_count,
                'tier2': tier2_count,
                'tier3': tier3_count
            }
        }
        
    except Exception as e:
        print(f"❌ Hierarchical retrieval failed: {e}")
        return {
            'results': {},
            'time': 0,
            'tier_counts': {'tier1': 0, 'tier2': 0, 'tier3': 0},
            'error': str(e)
        }

def test_hierarchical_generation(generator, question: str, hierarchical_results: Dict) -> Dict[str, Any]:
    """Test hierarchical generation for a question."""
    try:
        start_time = time.time()
        
        # Generate hierarchical response
        response = generator.generate_hierarchical_response(question, hierarchical_results)
        
        generation_time = time.time() - start_time
        
        return {
            'response': response,
            'time': generation_time
        }
        
    except Exception as e:
        print(f"❌ Hierarchical generation failed: {e}")
        return {
            'response': f"Error: {str(e)}",
            'time': 0,
            'error': str(e)
        }

def display_question_details(question: Dict[str, Any], index: int):
    """Display detailed information about a question."""
    print("\n" + "="*80)
    print(f"🔬 QUESTION {index + 1}")
    print("="*80)
    print(f"📝 ID: {question.get('id', 'Unknown')}")
    print(f"📚 Dataset: {question.get('dataset', 'Unknown')}")
    print(f"❓ Question: {question['question']}")
    
    if question['options']:
        print(f"\n📋 Options:")
        for i, option in enumerate(question['options']):
            print(f"   {chr(65 + i)}. {option}")
    
    if question['correct_answer']:
        print(f"\n✅ Correct Answer: {question['correct_answer']}")
    
    if question['explanation']:
        print(f"\n💡 Explanation: {question['explanation']}")

def display_retrieval_results(retrieval_data: Dict[str, Any]):
    """Display hierarchical retrieval results."""
    print(f"\n🔍 HIERARCHICAL RETRIEVAL RESULTS")
    print(f"⏱️ Time: {retrieval_data['time']:.2f} seconds")
    
    if 'error' in retrieval_data:
        print(f"❌ Error: {retrieval_data['error']}")
        return
    
    results = retrieval_data['results']
    tier_counts = retrieval_data['tier_counts']
    
    print(f"\n📊 Retrieved Documents by Tier:")
    print(f"   🧩 Tier 1 (Pattern Recognition): {tier_counts['tier1']} documents")
    print(f"   🔬 Tier 2 (Hypothesis Testing): {tier_counts['tier2']} documents")
    print(f"   ✅ Tier 3 (Confirmation): {tier_counts['tier3']} documents")
    
    # Show sample documents from each tier
    for tier_name, docs in results.items():
        if docs and len(docs) > 0:
            print(f"\n📄 Sample from {tier_name}:")
            sample_doc = docs[0]
            text_preview = sample_doc.get('text', '')[:200] + "..."
            print(f"   {text_preview}")

def display_generation_results(generation_data: Dict[str, Any]):
    """Display hierarchical generation results."""
    print(f"\n🤖 HIERARCHICAL GENERATION RESULTS")
    print(f"⏱️ Time: {generation_data['time']:.2f} seconds")
    
    if 'error' in generation_data:
        print(f"❌ Error: {generation_data['error']}")
        return
    
    response = generation_data['response']
    print(f"\n🧠 Generated Response:")
    print(f"   Length: {len(response)} characters")
    print(f"   Content: {response}")

def analyze_answer_accuracy(question: Dict[str, Any], generated_response: str) -> Dict[str, Any]:
    """Analyze if the generated response matches the correct answer."""
    correct_answer = question.get('correct_answer', '').strip()
    response = generated_response.strip()
    
    # Extract potential answers from response
    extracted_answers = []
    
    # Look for option letters (A, B, C, D, E)
    import re
    option_matches = re.findall(r'\b([A-E])\b', response.upper())
    if option_matches:
        extracted_answers.extend(option_matches)
    
    # Look for "answer is" patterns
    answer_patterns = [
        r'answer is\s+([A-E])',
        r'correct answer is\s+([A-E])',
        r'the answer is\s+([A-E])',
        r'answer:\s*([A-E])',
        r'option\s+([A-E])',
        r'choice\s+([A-E])'
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, response.upper())
        extracted_answers.extend(matches)
    
    # Check if response contains the correct answer text
    answer_text_match = False
    if correct_answer and question.get('options'):
        try:
            correct_idx = ord(correct_answer.upper()) - ord('A')
            if 0 <= correct_idx < len(question['options']):
                correct_text = question['options'][correct_idx].lower()
                answer_text_match = correct_text in response.lower()
        except:
            pass
    
    # Determine if correct
    is_correct = False
    if correct_answer.upper() in [ans.upper() for ans in extracted_answers]:
        is_correct = True
    elif answer_text_match:
        is_correct = True
    
    return {
        'is_correct': is_correct,
        'correct_answer': correct_answer,
        'extracted_answers': list(set(extracted_answers)),
        'answer_text_match': answer_text_match,
        'response_length': len(response),
        'has_option_format': bool(question.get('options')),
        'analysis': {
            'found_option_letters': bool(option_matches),
            'found_answer_patterns': bool(any(re.search(p, response.upper()) for p in answer_patterns)),
            'contains_correct_text': answer_text_match
        }
    }

def display_accuracy_analysis(accuracy_data: Dict[str, Any]):
    """Display accuracy analysis results."""
    print(f"\n🎯 ACCURACY ANALYSIS")
    
    if accuracy_data.get('error'):
        print("❌ Cannot analyze due to generation error")
        return
    
    is_correct = accuracy_data['is_correct']
    correct_answer = accuracy_data['correct_answer']
    extracted_answers = accuracy_data['extracted_answers']
    
    print(f"✅ Correct: {'YES' if is_correct else 'NO'}")
    print(f"📝 Expected Answer: {correct_answer}")
    print(f"🔍 Extracted Answers: {extracted_answers if extracted_answers else 'None found'}")
    
    analysis = accuracy_data['analysis']
    print(f"\n🔬 Response Analysis:")
    print(f"   Found option letters (A,B,C,D,E): {'✅' if analysis['found_option_letters'] else '❌'}")
    print(f"   Found answer patterns ('answer is'): {'✅' if analysis['found_answer_patterns'] else '❌'}")
    print(f"   Contains correct answer text: {'✅' if analysis['contains_correct_text'] else '❌'}")
    
    if not is_correct:
        print(f"\n⚠️ ACCURACY ISSUES DETECTED:")
        if not analysis['found_option_letters'] and not analysis['found_answer_patterns']:
            print("   - Response doesn't contain clear answer format (A, B, C, D, E)")
            print("   - Model may not be following multiple choice format")
        if not extracted_answers:
            print("   - No recognizable answers extracted from response")
            print("   - Model may be generating explanatory text without clear choice")
        if extracted_answers and correct_answer not in extracted_answers:
            print(f"   - Model chose wrong answer: {extracted_answers} vs correct {correct_answer}")

def diagnose_system_issues(results: List[Dict[str, Any]]):
    """Diagnose potential system issues based on results."""
    print(f"\n🔧 SYSTEM DIAGNOSIS")
    print("="*60)
    
    total_questions = len(results)
    correct_answers = sum(1 for r in results if r.get('accuracy', {}).get('is_correct', False))
    accuracy_rate = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    
    print(f"📊 Overall Accuracy: {accuracy_rate:.1f}% ({correct_answers}/{total_questions})")
    
    # Analyze common issues
    issues = {
        'no_option_letters': 0,
        'no_answer_patterns': 0,
        'wrong_answers': 0,
        'generation_errors': 0,
        'retrieval_errors': 0,
        'empty_responses': 0,
        'no_extracted_answers': 0
    }
    
    for result in results:
        if 'error' in result.get('generation', {}):
            issues['generation_errors'] += 1
        elif 'error' in result.get('retrieval', {}):
            issues['retrieval_errors'] += 1
        elif result.get('accuracy'):
            acc = result['accuracy']
            if not acc.get('is_correct', False):
                if not acc['analysis']['found_option_letters']:
                    issues['no_option_letters'] += 1
                if not acc['analysis']['found_answer_patterns']:
                    issues['no_answer_patterns'] += 1
                if acc['extracted_answers'] and acc['correct_answer'] not in acc['extracted_answers']:
                    issues['wrong_answers'] += 1
                if not acc['extracted_answers']:
                    issues['no_extracted_answers'] += 1
                if acc['response_length'] < 10:
                    issues['empty_responses'] += 1
    
    print(f"\n🔍 Issue Breakdown:")
    for issue, count in issues.items():
        if count > 0:
            percentage = (count / total_questions) * 100
            print(f"   {issue.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    # Provide specific recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if issues['generation_errors'] > 0:
        print("🔴 CRITICAL: Generation errors detected")
        print("   - Check if Ollama is running: ollama list")
        print("   - Verify model is available: ollama pull mistral:7b-instruct")
        print("   - Check Ollama service: curl http://localhost:11434/api/tags")
    
    if issues['retrieval_errors'] > 0:
        print("🔴 CRITICAL: Retrieval errors detected")
        print("   - Verify hierarchical collections exist")
        print("   - Re-run setup: python setup_hierarchical_system.py")
        print("   - Check ChromaDB installation")
    
    if issues['no_option_letters'] > total_questions * 0.5:
        print("🟡 FORMAT ISSUE: Model not generating option letters (A,B,C,D,E)")
        print("   - Update system prompt to emphasize multiple choice format")
        print("   - Add explicit instruction: 'Answer with only the letter (A, B, C, D, or E)'")
        print("   - Check if prompts in config.yaml are properly formatted")
    
    if issues['no_answer_patterns'] > total_questions * 0.5:
        print("🟡 FORMAT ISSUE: Model not using clear answer patterns")
        print("   - Modify prompts to include: 'The answer is: [LETTER]'")
        print("   - Add response format examples in system prompt")
    
    if issues['wrong_answers'] > 0 and issues['no_option_letters'] == 0:
        print("🟠 ACCURACY ISSUE: Model generating wrong answers but in correct format")
        print("   - Check retrieved document quality and relevance")
        print("   - Verify tier distribution is balanced")
        print("   - Consider improving retrieval similarity threshold")
        print("   - Check if foundation dataset contains relevant medical knowledge")
    
    if issues['no_extracted_answers'] > total_questions * 0.3:
        print("🟡 EXTRACTION ISSUE: Cannot extract clear answers from responses")
        print("   - Model may be generating explanations without clear choices")
        print("   - Update prompt to require explicit answer selection")
        print("   - Add answer extraction logic improvements")
    
    if accuracy_rate < 30:
        print("🔴 CRITICAL: Very low accuracy - major system issues")
        print("   PRIORITY FIXES:")
        print("   1. Verify all system components are working")
        print("   2. Check foundation dataset quality and size")
        print("   3. Validate prompt engineering for medical Q&A")
        print("   4. Test with simpler questions first")
    elif accuracy_rate < 50:
        print("🟠 MODERATE: Below expected accuracy")
        print("   IMPROVEMENT AREAS:")
        print("   1. Optimize prompts for multiple choice format")
        print("   2. Improve retrieval document relevance")
        print("   3. Balance tier distribution")
    else:
        print("🟢 GOOD: Accuracy within reasonable range")
        print("   OPTIMIZATION OPPORTUNITIES:")
        print("   1. Fine-tune retrieval parameters")
        print("   2. Enhance answer extraction logic")
    """Main debug function."""
    logger.info("🧪 Starting Hierarchical System Debug")
    logger.info("="*70)
    
    # Load MIRAGE questions
    questions = load_mirage_questions()
    if not questions:
        logger.error("❌ No MIRAGE questions loaded. Cannot proceed.")
        return
    
    # Select 5 random questions
    num_questions = min(5, len(questions))
    selected_questions = random.sample(questions, num_questions)
    
    logger.info(f"🎲 Selected {num_questions} random questions for testing")
    
    # Initialize hierarchical system
    try:
        retriever, generator, config = initialize_hierarchical_system()
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        return
    
    # Test each question
    results = []
    
    for i, question in enumerate(selected_questions):
        logger.info(f"\n🧪 Testing question {i + 1}/{num_questions}")
        
        # Display question details
        display_question_details(question, i)
        
        # Test retrieval
        retrieval_data = test_hierarchical_retrieval(retriever, question['question'])
        display_retrieval_results(retrieval_data)
        
        # Test generation
        generation_data = test_hierarchical_generation(
            generator, 
            question['question'], 
            retrieval_data['results']
        )
        display_generation_results(generation_data)
        
        # Analyze answer accuracy
        if 'error' not in generation_data:
            accuracy_analysis = analyze_answer_accuracy(question, generation_data['response'])
            display_accuracy_analysis(accuracy_analysis)
        else:
            accuracy_analysis = {'is_correct': False, 'error': True}
        
        # Store results
        results.append({
            'question': question,
            'retrieval': retrieval_data,
            'generation': generation_data,
            'accuracy': accuracy_analysis
        })
        
        print("\n" + "-"*80)
    
    # Summary
    print("\n" + "="*80)
    print("📊 DEBUG SUMMARY")
    print("="*80)
    
    total_retrieval_time = sum(r['retrieval']['time'] for r in results)
    total_generation_time = sum(r['generation']['time'] for r in results)
    
    retrieval_errors = sum(1 for r in results if 'error' in r['retrieval'])
    generation_errors = sum(1 for r in results if 'error' in r['generation'])
    correct_answers = sum(1 for r in results if r.get('accuracy', {}).get('is_correct', False))
    
    print(f"📈 Performance Metrics:")
    print(f"   Questions tested: {len(results)}")
    print(f"   Total retrieval time: {total_retrieval_time:.2f}s")
    print(f"   Total generation time: {total_generation_time:.2f}s")
    print(f"   Average time per question: {(total_retrieval_time + total_generation_time) / len(results):.2f}s")
    
    print(f"\n🔍 Error Analysis:")
    print(f"   Retrieval errors: {retrieval_errors}/{len(results)}")
    print(f"   Generation errors: {generation_errors}/{len(results)}")
    print(f"   Correct answers: {correct_answers}/{len(results)}")
    
    success_rate = ((len(results) - retrieval_errors - generation_errors) / len(results)) * 100
    accuracy_rate = (correct_answers / len(results)) * 100
    print(f"   Technical success rate: {success_rate:.1f}%")
    print(f"   Answer accuracy rate: {accuracy_rate:.1f}%")
    
    # Tier analysis
    successful_retrievals = [r for r in results if 'error' not in r['retrieval']]
    if successful_retrievals:
        avg_tier1 = sum(r['retrieval']['tier_counts']['tier1'] for r in successful_retrievals) / len(successful_retrievals)
        avg_tier2 = sum(r['retrieval']['tier_counts']['tier2'] for r in successful_retrievals) / len(successful_retrievals)
        avg_tier3 = sum(r['retrieval']['tier_counts']['tier3'] for r in successful_retrievals) / len(successful_retrievals)
        
        print(f"\n🏗️ Tier Distribution (avg docs per question):")
        print(f"   Tier 1 (Pattern Recognition): {avg_tier1:.1f}")
        print(f"   Tier 2 (Hypothesis Testing): {avg_tier2:.1f}")
        print(f"   Tier 3 (Confirmation): {avg_tier3:.1f}")
    
    # Detailed system diagnosis
    diagnose_system_issues(results)
    
    print("\n" + "="*80)
    print("🎯 CONCLUSION")
    print("="*80)
    
    if accuracy_rate < 30:
        print("🔴 CRITICAL ISSUES DETECTED")
        print(f"   Accuracy: {accuracy_rate:.1f}% (Expected: >70%)")
        print("   This matches your evaluation results of 23% accuracy")
        print("   Focus on the CRITICAL recommendations above")
    elif accuracy_rate < 50:
        print("🟠 SIGNIFICANT ISSUES DETECTED")
        print(f"   Accuracy: {accuracy_rate:.1f}% (Expected: >70%)")
        print("   System partially working but needs optimization")
    else:
        print("🟢 SYSTEM WORKING WELL")
        print(f"   Accuracy: {accuracy_rate:.1f}%")
        print("   Minor optimizations may help")
    
    print("="*80)

def main():
    """Main debug function."""
    print("🧪 Starting Hierarchical System Debug")
    print("="*70)
    
    # Load MIRAGE questions
    questions = load_mirage_questions()
    if not questions:
        print("❌ No MIRAGE questions loaded. Cannot proceed.")
        return
    
    # Select exactly 5 random questions
    num_questions = 5
    if len(questions) < 5:
        print(f"⚠️ Only {len(questions)} questions available, using all")
        selected_questions = questions
        num_questions = len(questions)
    else:
        selected_questions = random.sample(questions, num_questions)
    
    print(f"🎲 Selected {num_questions} random questions for testing")
    
    # Initialize hierarchical system
    try:
        retriever, generator, config = initialize_hierarchical_system()
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Test each question (exactly 5 times, no infinite loop)
    results = []
    
    for i, question in enumerate(selected_questions):
        print(f"\n🧪 Testing question {i + 1}/{num_questions}")
        
        # Display question details
        display_question_details(question, i)
        
        # Test retrieval
        retrieval_data = test_hierarchical_retrieval(retriever, question['question'])
        display_retrieval_results(retrieval_data)
        
        # Test generation
        generation_data = test_hierarchical_generation(
            generator, 
            question['question'], 
            retrieval_data['results']
        )
        display_generation_results(generation_data)
        
        # Analyze answer accuracy
        if 'error' not in generation_data:
            accuracy_analysis = analyze_answer_accuracy(question, generation_data['response'])
            display_accuracy_analysis(accuracy_analysis)
        else:
            accuracy_analysis = {'is_correct': False, 'error': True}
        
        # Store results
        results.append({
            'question': question,
            'retrieval': retrieval_data,
            'generation': generation_data,
            'accuracy': accuracy_analysis
        })
        
        print("\n" + "-"*80)
        
        # Safety check: ensure we don't exceed 5 questions
        if i + 1 >= 5:
            break
    
    # Summary
    print("\n" + "="*80)
    print("📊 DEBUG SUMMARY")
    print("="*80)
    
    total_retrieval_time = sum(r['retrieval']['time'] for r in results)
    total_generation_time = sum(r['generation']['time'] for r in results)
    
    retrieval_errors = sum(1 for r in results if 'error' in r['retrieval'])
    generation_errors = sum(1 for r in results if 'error' in r['generation'])
    correct_answers = sum(1 for r in results if r.get('accuracy', {}).get('is_correct', False))
    
    print(f"📈 Performance Metrics:")
    print(f"   Questions tested: {len(results)}")
    print(f"   Total retrieval time: {total_retrieval_time:.2f}s")
    print(f"   Total generation time: {total_generation_time:.2f}s")
    print(f"   Average time per question: {(total_retrieval_time + total_generation_time) / len(results):.2f}s")
    
    print(f"\n🔍 Error Analysis:")
    print(f"   Retrieval errors: {retrieval_errors}/{len(results)}")
    print(f"   Generation errors: {generation_errors}/{len(results)}")
    print(f"   Correct answers: {correct_answers}/{len(results)}")
    
    success_rate = ((len(results) - retrieval_errors - generation_errors) / len(results)) * 100
    accuracy_rate = (correct_answers / len(results)) * 100
    print(f"   Technical success rate: {success_rate:.1f}%")
    print(f"   Answer accuracy rate: {accuracy_rate:.1f}%")
    
    # Tier analysis
    successful_retrievals = [r for r in results if 'error' not in r['retrieval']]
    if successful_retrievals:
        avg_tier1 = sum(r['retrieval']['tier_counts']['tier1'] for r in successful_retrievals) / len(successful_retrievals)
        avg_tier2 = sum(r['retrieval']['tier_counts']['tier2'] for r in successful_retrievals) / len(successful_retrievals)
        avg_tier3 = sum(r['retrieval']['tier_counts']['tier3'] for r in successful_retrievals) / len(successful_retrievals)
        
        print(f"\n🏗️ Tier Distribution (avg docs per question):")
        print(f"   Tier 1 (Pattern Recognition): {avg_tier1:.1f}")
        print(f"   Tier 2 (Hypothesis Testing): {avg_tier2:.1f}")
        print(f"   Tier 3 (Confirmation): {avg_tier3:.1f}")
    
    # Detailed system diagnosis
    diagnose_system_issues(results)
    
    print("\n" + "="*80)
    print("🎯 CONCLUSION")
    print("="*80)
    
    if accuracy_rate < 30:
        print("🔴 CRITICAL ISSUES DETECTED")
        print(f"   Accuracy: {accuracy_rate:.1f}% (Expected: >70%)")
        print("   This matches your evaluation results of 23% accuracy")
        print("   Focus on the CRITICAL recommendations above")
    elif accuracy_rate < 50:
        print("🟠 SIGNIFICANT ISSUES DETECTED")
        print(f"   Accuracy: {accuracy_rate:.1f}% (Expected: >70%)")
        print("   System partially working but needs optimization")
    else:
        print("🟢 SYSTEM WORKING WELL")
        print(f"   Accuracy: {accuracy_rate:.1f}%")
        print("   Minor optimizations may help")
    
    print("="*80)


if __name__ == "__main__":
    main()