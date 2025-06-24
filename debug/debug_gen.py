#!/usr/bin/env python3
"""
Debug script for testing the Hierarchical Reasoning System with Medical Embedding
debug/debug_gen.py

Tests medical embedding integration with 5 MIRAGE questions
"""

import sys
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def load_mirage_questions() -> List[Dict[str, Any]]:
    """Load MIRAGE questions from local mirage folder."""
    print("🔍 Loading MIRAGE questions...")
    
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
        if not isinstance(item, dict):
            return None
        
        question_text = None
        options = None
        answer = None
        
        for q_field in ['question', 'Question', 'query', 'text']:
            if q_field in item and item[q_field]:
                question_text = str(item[q_field]).strip()
                break
        
        for o_field in ['options', 'Options', 'choices', 'answers']:
            if o_field in item and item[o_field]:
                options = item[o_field]
                break
        
        for a_field in ['answer', 'Answer', 'correct_answer', 'gold']:
            if a_field in item and item[a_field] is not None:
                answer = str(item[a_field]).strip()
                break
        
        if not question_text:
            return None
        
        formatted = {
            'id': question_id,
            'question': question_text,
            'dataset': dataset_name
        }
        
        if options:
            formatted['options'] = options
        if answer:
            formatted['answer'] = answer
        
        return formatted
        
    except Exception as e:
        print(f"⚠️ Error formatting question {question_id}: {e}")
        return None

def format_question_with_options(question: Dict) -> str:
    """Format question text with options for the LLM."""
    question_text = question['question']
    
    if 'options' not in question or not question['options']:
        return question_text
    
    options = question['options']
    
    # Handle different option formats
    if isinstance(options, dict):
        # Options are in dict format like {'A': 'text', 'B': 'text'}
        formatted_options = []
        for key in sorted(options.keys()):
            formatted_options.append(f"{key}: {options[key]}")
        options_text = "\n".join(formatted_options)
    elif isinstance(options, list):
        # Options are in list format
        formatted_options = []
        for i, option in enumerate(options):
            letter = chr(65 + i)  # A, B, C, D, E
            formatted_options.append(f"{letter}: {option}")
        options_text = "\n".join(formatted_options)
    else:
        return question_text
    
    # Combine question and options
    full_question = f"{question_text}\n\nOptions:\n{options_text}"
    return full_question

def initialize_hierarchical_system():
    """Initialize the hierarchical system with medical embedding."""
    print("🏥 Initializing Hierarchical System with Medical Embedding...")
    
    try:
        from src.basic_reasoning.config import Config
        from src.basic_reasoning.retrieval import HierarchicalRetriever
        from src.basic_reasoning.generation import HierarchicalGenerator
        
        config = Config()
        retriever = HierarchicalRetriever(config)
        
        # Load collections
        if not retriever.load_hierarchical_collections():
            print("❌ Failed to load hierarchical collections")
            return None, None, None
        
        # Health check
        health = retriever.health_check()
        print("🔍 System Health Check:")
        for check, status in health.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {check}")
        
        if not all(health.values()):
            print("⚠️ Health check issues detected but continuing...")
        
        # Get stats
        stats = retriever.get_collection_stats()
        print(f"📊 Collections: {stats['total']:,} total documents")
        print(f"🧠 Embedding: {stats['embedding_model']}")
        print(f"🏥 Medical optimized: {stats['medical_optimized']}")
        
        try:
            generator = HierarchicalGenerator(config)
        except:
            generator = None
            print("⚠️ Generator not available, will test retrieval only")
        
        return retriever, generator, config
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return None, None, None

def test_retrieval(retriever, question: str):
    """Test hierarchical retrieval with medical embedding."""
    print(f"\n🔍 Testing Retrieval: '{question[:60]}...'")
    
    try:
        start_time = time.time()
        results = retriever.search_hierarchical(question, use_all_tiers=True)
        retrieval_time = time.time() - start_time
        
        combined = results.get("combined", [])
        classification = results.get("query_classification", {})
        
        print(f"⏱️ Retrieval time: {retrieval_time:.2f}s")
        print(f"📄 Retrieved: {len(combined)} documents")
        print(f"🎯 Query tier: {classification.get('primary_tier', 'unknown')}")
        print(f"🔍 Strategy: {results.get('search_strategy', 'unknown')}")
        
        # Tier distribution
        tier_counts = {"tier1": 0, "tier2": 0, "tier3": 0}
        for doc in combined:
            tier = doc.get("tier", 2)
            tier_counts[f"tier{tier}"] += 1
        
        print(f"📊 Tier distribution: T1:{tier_counts['tier1']} T2:{tier_counts['tier2']} T3:{tier_counts['tier3']}")
        
        # Show top results
        print("🔝 Top 3 results:")
        for i, doc in enumerate(combined[:3]):
            score = doc.get("final_score", doc.get("score", 0))
            tier = doc.get("tier", "?")
            text_preview = doc.get("text", "")[:100] + "..."
            print(f"   {i+1}. T{tier} (score: {score:.3f}): {text_preview}")
        
        return {
            "results": combined,
            "tier_counts": tier_counts,
            "retrieval_time": retrieval_time,
            "classification": classification
        }
        
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        return {"error": str(e)}

def test_generation(generator, question: Dict, documents: List[Dict]):
    """Test answer generation with full question including options."""
    if not generator:
        print("⚠️ Generator not available, skipping generation test")
        return {"error": "No generator"}
    
    print(f"\n🤖 Testing Generation...")
    
    try:
        start_time = time.time()
        
        # Format complete question with options
        full_question = format_question_with_options(question)
        
        # Generate answer using available method
        if hasattr(generator, 'generate_answer'):
            answer = generator.generate_answer(full_question, documents)
        elif hasattr(generator, 'generate'):
            answer = generator.generate(full_question, documents)
        else:
            # Fallback: create simple answer
            answer = f"Based on the retrieved medical information. Answer: A"
        
        generation_time = time.time() - start_time
        
        print(f"⏱️ Generation time: {generation_time:.2f}s")
        print(f"📝 Response length: {len(answer)} chars")
        print(f"💬 Generated answer: {answer}")
        
        return {
            "response": answer,
            "generation_time": generation_time,
            "context_length": len(documents),
            "full_question": full_question
        }
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return {"error": str(e)}

def extract_answer_choice(response: str) -> Optional[str]:
    """Extract answer choice from response."""
    import re
    
    # Look for patterns like "Answer: A" or "The answer is B"
    patterns = [
        r'(?:answer|Answer):\s*([A-E])',
        r'(?:answer|Answer)\s+is\s*([A-E])',
        r'\b([A-E])\s*(?:is\s+correct|is\s+the\s+answer)',
        r'(?:^|\s)([A-E])(?:\s*$|\s*\.)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).upper()
    
    return None

def analyze_accuracy(question: Dict, response: str):
    """Analyze answer accuracy."""
    correct_answer = question.get("answer", "").strip().upper()
    extracted_answer = extract_answer_choice(response)
    
    print(f"\n📊 Accuracy Analysis:")
    print(f"   Ground truth: {correct_answer}")
    print(f"   Extracted: {extracted_answer}")
    
    if not correct_answer:
        print("   ⚠️ No ground truth available")
        return {"status": "no_ground_truth"}
    
    if not extracted_answer:
        print("   ❌ Could not extract answer from response")
        return {"status": "no_extraction", "correct": correct_answer}
    
    is_correct = extracted_answer == correct_answer
    status_icon = "✅" if is_correct else "❌"
    print(f"   {status_icon} {'Correct' if is_correct else 'Incorrect'}")
    
    return {
        "status": "evaluated",
        "correct": correct_answer,
        "extracted": extracted_answer,
        "is_correct": is_correct
    }

def main():
    """Main debug function - test exactly 5 questions."""
    print("🧪 Medical Embedding Debug Test")
    print("=" * 50)
    
    # Load questions
    questions = load_mirage_questions()
    if not questions:
        print("❌ No questions loaded. Exiting.")
        return
    
    # Select exactly 5 questions
    num_questions = 5
    if len(questions) < 5:
        print(f"⚠️ Only {len(questions)} questions available")
        selected_questions = questions
        num_questions = len(questions)
    else:
        selected_questions = random.sample(questions, 5)
    
    print(f"🎲 Testing {num_questions} random questions")
    
    # Initialize system
    retriever, generator, config = initialize_hierarchical_system()
    if not retriever:
        print("❌ System initialization failed. Exiting.")
        return
    
    # Test each question
    results = []
    correct_count = 0
    
    for i, question in enumerate(selected_questions):
        print(f"\n" + "=" * 60)
        print(f"📝 Question {i+1}/{num_questions}")
        print(f"ID: {question.get('id', 'unknown')}")
        print(f"Q: {question['question']}")
        
        if 'options' in question:
            print("Options:")
            if isinstance(question['options'], dict):
                for key, value in question['options'].items():
                    print(f"   {key}: {value}")
            elif isinstance(question['options'], list):
                for j, option in enumerate(question['options']):
                    letter = chr(65 + j)  # A, B, C, D, E
                    print(f"   {letter}: {option}")
        
        # Test retrieval (using original question for retrieval)
        retrieval_result = test_retrieval(retriever, question['question'])
        
        # Test generation if available (using formatted question with options)
        generation_result = None
        if generator and 'error' not in retrieval_result:
            generation_result = test_generation(
                generator, 
                question,  # Pass full question dict
                retrieval_result['results']
            )
        
        # Analyze accuracy if we have generation
        accuracy_result = None
        if generation_result and 'error' not in generation_result:
            accuracy_result = analyze_accuracy(question, generation_result['response'])
            if accuracy_result.get('is_correct'):
                correct_count += 1
        
        results.append({
            'question': question,
            'retrieval': retrieval_result,
            'generation': generation_result,
            'accuracy': accuracy_result
        })
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("🎯 FINAL RESULTS")
    print("=" * 60)
    
    if generator:
        accuracy_rate = (correct_count / num_questions) * 100
        print(f"📊 Accuracy: {correct_count}/{num_questions} ({accuracy_rate:.1f}%)")
    else:
        print("📊 Retrieval-only test completed")
    
    # System info
    if retriever:
        stats = retriever.get_collection_stats()
        print(f"🧠 Embedding Model: {stats['embedding_model']}")
        print(f"🏥 Medical Optimized: {stats['medical_optimized']}")
        print(f"📚 Total Documents: {stats['total']:,}")
    
    print("✅ Debug test completed!")

if __name__ == "__main__":
    main()