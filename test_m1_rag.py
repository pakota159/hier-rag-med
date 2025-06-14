#!/usr/bin/env python3
"""
M1 MacBook Pro test script for basic RAG functionality.
"""

import sys
import os
from pathlib import Path
import requests
import time

# Get the project root directory
project_root = Path(__file__).parent.parent if "tests" in str(Path(__file__).parent) else Path(__file__).parent
src_path = project_root / "src"

# Add src to Python path
sys.path.insert(0, str(src_path))
print(f"Added to path: {src_path}")

# Change to project root directory
os.chdir(project_root)
print(f"Working directory: {os.getcwd()}")

def check_ollama():
    """Check if Ollama is running and model is available."""
    print("🔍 Checking Ollama setup...")
    
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = response.json().get("models", [])
        
        model_names = [model["name"] for model in models]
        print(f"   Available models: {model_names}")
        
        if "mistral:7b-instruct" in model_names:
            print("✅ Ollama is running with mistral:7b-instruct")
            return True
        else:
            print("❌ mistral:7b-instruct not found")
            print("💡 Run: ollama pull mistral:7b-instruct")
            return False
            
    except requests.exceptions.RequestException:
        print("❌ Ollama not running")
        print("💡 Run: ollama serve &")
        return False

def create_test_data():
    """Create test medical documents."""
    print("📄 Creating test medical documents...")
    
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Diabetes document
    diabetes_content = """
Type 2 Diabetes Mellitus

Type 2 diabetes is a chronic condition that affects how your body processes blood sugar (glucose).

Symptoms:
- Increased thirst and frequent urination
- Increased hunger
- Weight loss
- Fatigue
- Blurred vision
- Slow-healing sores
- Frequent infections

Risk Factors:
- Weight (obesity increases risk)
- Fat distribution (abdominal fat)
- Inactivity
- Family history
- Age (risk increases after age 45)
- Race and ethnicity
- Blood lipid levels
- High blood pressure

Treatment:
- Healthy eating
- Regular exercise
- Weight loss
- Blood sugar monitoring
- Diabetes medications (metformin)
- Insulin therapy
"""
    
    # Hypertension document
    hypertension_content = """
Hypertension (High Blood Pressure)

Hypertension is a common condition where blood pressure in your arteries is consistently too high.

Blood Pressure Categories:
- Normal: Less than 120/80 mmHg
- Elevated: 120-129 systolic and less than 80 diastolic
- Stage 1: 130-139 systolic or 80-89 diastolic
- Stage 2: 140/90 mmHg or higher
- Hypertensive crisis: Higher than 180/120 mmHg

Symptoms:
- Most people have no symptoms
- Severe cases may cause headaches, shortness of breath, nosebleeds

Risk Factors:
- Age
- Race
- Family history
- Being overweight or obese
- Not being physically active
- Using tobacco
- Too much salt in diet
- Drinking too much alcohol
- Stress

Treatment:
- Lifestyle changes (diet, exercise, weight loss)
- ACE inhibitors
- Calcium channel blockers
- Diuretics
- Beta blockers
"""
    
    # Write files
    (data_dir / "diabetes.txt").write_text(diabetes_content)
    (data_dir / "hypertension.txt").write_text(hypertension_content)
    
    print("✅ Created test documents")

def test_m1_rag():
    """Test the RAG pipeline on M1 Mac."""
    print("🚀 Testing HierRAGMed on M1 MacBook Pro\n")
    
    # Check Ollama first
    if not check_ollama():
        return False
    
    # Create test data
    create_test_data()
    
    try:
        # Import after path setup
        from config import Config
        from processing import DocumentProcessor  
        from retrieval import Retriever
        from generation import Generator
        
        print("\n1. Loading M1-optimized configuration...")
        config = Config(Path("config.yaml"))
        print("✅ Configuration loaded")
        
        print("\n2. Processing documents...")
        processor = DocumentProcessor(config.config["processing"])
        
        # Process text files
        documents = []
        for txt_file in Path("data/raw").glob("*.txt"):
            with open(txt_file, 'r') as f:
                text = f.read()
            
            docs = processor.process_text(text, {
                "source": str(txt_file),
                "doc_id": txt_file.stem
            })
            documents.extend(docs)
        
        print(f"✅ Processed {len(documents)} chunks")
        
        print("\n3. Setting up vector store...")
        retriever = Retriever(config)
        
        # Check MPS availability
        print(f"   Using device: {config.config['models']['embedding']['device']}")
        
        # List existing collections
        existing_collections = retriever.list_collections()
        print(f"   Existing collections: {existing_collections}")
        
        # Create collection (will delete if exists)
        collection_name = "test_medical_m1"
        retriever.create_collection(collection_name)
        retriever.add_documents(documents)
        print("✅ Vector store created")
        
        print("\n4. Testing search...")
        test_queries = [
            "What are diabetes symptoms?",
            "How to treat high blood pressure?",
            "What causes diabetes?"
        ]
        
        for query in test_queries:
            print(f"\n   📋 {query}")
            results = retriever.search(query, n_results=2)
            if results:
                print(f"      ✅ Found {len(results)} results")
                print(f"      📄 Top match: {results[0]['text'][:100]}...")
            else:
                print("      ❌ No results found")
        
        print("\n5. Testing generation...")
        generator = Generator(config)
        
        query = "What are the main symptoms of diabetes?"
        results = retriever.search(query, n_results=3)
        
        print(f"   📋 Generating answer for: {query}")
        print("   🔄 Calling Ollama (may take 10-30 seconds on M1)...")
        
        start_time = time.time()
        answer = generator.generate(query, results)
        end_time = time.time()
        
        print(f"   ⏱️  Generation took {end_time - start_time:.1f} seconds")
        print(f"   ✅ Answer generated!")
        print(f"   📝 {answer[:200]}...")
        
        print("\n6. Testing full pipeline...")
        questions = [
            "What medications treat diabetes?",
            "What are normal blood pressure values?",
        ]
        
        for question in questions:
            print(f"\n   ❓ {question}")
            docs = retriever.search(question, n_results=2)
            answer = generator.generate(question, docs)
            print(f"   💬 {answer[:100]}...")
        
        print("\n🎉 M1 MacBook Pro RAG test successful!")
        print("\n📋 Performance Notes:")
        print("   - Using MPS (Metal Performance Shaders) for embeddings")
        print("   - Reduced batch size for M1 memory constraints")
        print("   - Ollama running natively on Apple Silicon")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_m1_rag()
    if success:
        print("\n✅ Ready for development!")
        print("   Next: python -m src.main (for web interface)")
    else:
        print("\n❌ Setup needs attention. Check the errors above.")