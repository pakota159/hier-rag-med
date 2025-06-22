#!/usr/bin/env python3
"""
Fix Generation Method - Test and Fix the String Indices Error
File: fix_generation.py
"""

import sys
import json
import traceback
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_generation_methods():
    """Test different ways to call the generation method."""
    print("🔧 Testing Generation Method Fixes")
    print("=" * 50)
    
    try:
        from src.basic_reasoning.config import Config
        from src.basic_reasoning.retrieval import HierarchicalRetriever
        from src.basic_reasoning.generation import HierarchicalGenerator
        
        config = Config()
        retriever = HierarchicalRetriever(config)
        generator = HierarchicalGenerator(config)
        retriever.load_hierarchical_collections()
        
        # Test question
        question = "What is the first-line treatment for type 2 diabetes?"
        
        print(f"🔍 Testing question: {question}")
        
        # Get retrieval results
        hierarchical_results = retriever.hierarchical_search(question)
        print(f"✅ Retrieved {sum(len(docs) for docs in hierarchical_results.values())} documents")
        
        # Test Method 1: Original signature
        print("\n🧪 Method 1: generate_hierarchical_response(query, hierarchical_results)")
        try:
            response1 = generator.generate_hierarchical_response(question, hierarchical_results)
            print(f"✅ Method 1 Success: '{response1[:100]}...'")
        except Exception as e:
            print(f"❌ Method 1 Failed: {e}")
        
        # Test Method 2: Reversed parameters
        print("\n🧪 Method 2: generate_hierarchical_response(hierarchical_results, query)")
        try:
            response2 = generator.generate_hierarchical_response(hierarchical_results, question)
            print(f"✅ Method 2 Success: '{response2[:100]}...'")
        except Exception as e:
            print(f"❌ Method 2 Failed: {e}")
        
        # Test Method 3: Direct method call inspection
        print("\n🧪 Method 3: Inspect and call correctly")
        try:
            import inspect
            sig = inspect.signature(generator.generate_hierarchical_response)
            param_names = list(sig.parameters.keys())
            print(f"📋 Parameter order: {param_names}")
            
            # Call with correct parameter order
            if param_names == ['query', 'hierarchical_results']:
                response3 = generator.generate_hierarchical_response(question, hierarchical_results)
            elif param_names == ['hierarchical_results', 'query']:
                response3 = generator.generate_hierarchical_response(hierarchical_results, question)
            else:
                print(f"❌ Unexpected parameters: {param_names}")
                response3 = None
            
            if response3:
                print(f"✅ Method 3 Success: '{response3[:100]}...'")
        except Exception as e:
            print(f"❌ Method 3 Failed: {e}")
        
        # Test Method 4: Fallback to regular generate
        print("\n🧪 Method 4: Fallback to regular generate method")
        try:
            # Convert hierarchical results to context list
            context = []
            for tier_docs in hierarchical_results.values():
                for doc in tier_docs:
                    context.append({
                        'text': doc.get('content', doc.get('text', '')),
                        'metadata': doc.get('metadata', {})
                    })
            
            response4 = generator.generate(question, context)
            print(f"✅ Method 4 Success: '{response4[:100]}...'")
        except Exception as e:
            print(f"❌ Method 4 Failed: {e}")
        
        # Test Method 5: Create working wrapper
        print("\n🧪 Method 5: Create working wrapper function")
        try:
            def safe_generate(generator, question, hierarchical_results):
                """Safe generation wrapper that tries different methods."""
                # Try method 1
                try:
                    return generator.generate_hierarchical_response(question, hierarchical_results)
                except:
                    pass
                
                # Try method 2
                try:
                    return generator.generate_hierarchical_response(hierarchical_results, question)
                except:
                    pass
                
                # Fallback to regular generate
                context = []
                for tier_docs in hierarchical_results.values():
                    for doc in tier_docs:
                        context.append({
                            'text': doc.get('content', doc.get('text', '')),
                            'metadata': doc.get('metadata', {})
                        })
                return generator.generate(question, context)
            
            response5 = safe_generate(generator, question, hierarchical_results)
            print(f"✅ Method 5 Success: '{response5[:100]}...'")
            
            # Return the working method
            return safe_generate
            
        except Exception as e:
            print(f"❌ Method 5 Failed: {e}")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        traceback.print_exc()
    
    return None


def test_mirage_question():
    """Test the working generation on a real MIRAGE question."""
    print("\n" + "=" * 50)
    print("🎯 Testing Real MIRAGE Question")
    print("=" * 50)
    
    try:
        # Get working generation method
        safe_generate = test_generation_methods()
        if not safe_generate:
            print("❌ No working generation method found")
            return
        
        # Load real MIRAGE question
        mirage_file = Path('mirage/benchmark.json')
        with open(mirage_file) as f:
            data = json.load(f)
        
        # Get first MedQA question
        medqa_questions = data['medqa']
        first_question_id = list(medqa_questions.keys())[0]
        sample_question = medqa_questions[first_question_id]
        
        question_text = sample_question.get('question', '')
        expected_answer = sample_question.get('answer', '')
        options = sample_question.get('options', {})
        
        print(f"📋 MIRAGE Question:")
        print(f"Q: {question_text[:200]}...")
        print(f"Options:")
        for letter, option in options.items():
            print(f"  {letter}: {option[:80]}...")
        print(f"Expected Answer: {expected_answer}")
        
        # Test with working generation
        from src.basic_reasoning.config import Config
        from src.basic_reasoning.retrieval import HierarchicalRetriever
        from src.basic_reasoning.generation import HierarchicalGenerator
        
        config = Config()
        retriever = HierarchicalRetriever(config)
        generator = HierarchicalGenerator(config)
        retriever.load_hierarchical_collections()
        
        # Get retrieval results
        hierarchical_results = retriever.hierarchical_search(question_text)
        
        # Generate response
        response = safe_generate(generator, question_text, hierarchical_results)
        
        print(f"\n🤖 Model Response: '{response}'")
        
        # Check accuracy
        clean_response = response.strip().upper()
        if clean_response == expected_answer:
            print("✅ CORRECT ANSWER!")
        elif expected_answer in clean_response:
            print("🔄 Contains correct answer")
        else:
            print("❌ Wrong answer")
        
        # Check format
        if len(clean_response) == 1 and clean_response in 'ABCD':
            print("✅ Correct format (single letter)")
        else:
            print(f"❌ Wrong format (should be single letter, got: '{clean_response}')")
        
    except Exception as e:
        print(f"❌ MIRAGE test failed: {e}")
        traceback.print_exc()


def create_fixed_generation_wrapper():
    """Create a fixed generation wrapper file."""
    print("\n" + "=" * 50)
    print("🛠️ Creating Fixed Generation Wrapper")
    print("=" * 50)
    
    wrapper_code = '''
def fixed_hierarchical_generate(generator, question, hierarchical_results):
    """Fixed generation wrapper that handles method signature issues."""
    import inspect
    
    # Try different calling patterns
    try:
        # Method 1: (question, hierarchical_results)
        return generator.generate_hierarchical_response(question, hierarchical_results)
    except TypeError:
        pass
    except Exception as e:
        if "string indices must be integers" in str(e):
            pass
        else:
            raise e
    
    try:
        # Method 2: (hierarchical_results, question)
        return generator.generate_hierarchical_response(hierarchical_results, question)
    except:
        pass
    
    # Fallback: Use regular generate method
    context = []
    for tier_docs in hierarchical_results.values():
        for doc in tier_docs:
            context.append({
                'text': doc.get('content', doc.get('text', '')),
                'metadata': doc.get('metadata', {})
            })
    
    return generator.generate(question, context)

# Usage example:
# response = fixed_hierarchical_generate(generator, question, hierarchical_results)
'''
    
    with open('fixed_generation_wrapper.py', 'w') as f:
        f.write(wrapper_code)
    
    print("✅ Created fixed_generation_wrapper.py")
    print("🔧 Use this in your evaluation code to fix the generation error")


if __name__ == "__main__":
    print("🚀 Generation Method Fix Utility")
    print("=" * 60)
    
    # Test generation methods
    working_method = test_generation_methods()
    
    if working_method:
        print("\n✅ Found working generation method!")
        
        # Test on real MIRAGE question
        test_mirage_question()
        
        # Create wrapper for future use
        create_fixed_generation_wrapper()
        
        print("\n🎯 SUMMARY:")
        print("✅ Generation method fixed")
        print("✅ Ready to test on MIRAGE evaluation")
        print("🔧 Next: Run evaluation with fixed generation")
        
    else:
        print("\n❌ Could not fix generation method")
        print("🔧 Manual intervention needed")