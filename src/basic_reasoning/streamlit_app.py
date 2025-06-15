"""
Streamlit app for Basic Reasoning - Hierarchical Diagnostic RAG System
Uses foundation datasets from data/foundation/
"""

import streamlit as st
import sys
from pathlib import Path
import time
import os
import json

# Fix torch/streamlit compatibility issue
os.environ["STREAMLIT_DISABLE_AUTOINDEX"] = "true"

# Add project root and src to path
project_root = Path(__file__).parent.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    # Use absolute imports instead of relative imports
    from basic_reasoning.config import Config
    from basic_reasoning.processing import HierarchicalDocumentProcessor
    from basic_reasoning.retrieval import HierarchicalRetriever
    from basic_reasoning.generation import HierarchicalGenerator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure you're running from the project root directory.")
    st.error("Try: streamlit run src/basic_reasoning/streamlit_app.py --server.port 8503")
    st.stop()

# Page config
st.set_page_config(
    page_title="Ních gà - Hierarchical Diagnostic RAG",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .question-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .user-question {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    
    .bot-response {
        background: white;
        border: 2px solid #e3f2fd;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        line-height: 1.6;
    }
    
    .tier-section {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 4px solid #ff9800;
        box-shadow: 0 2px 8px rgba(255, 152, 0, 0.2);
    }
    
    .metrics-container {
        background: #e8f5e8;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        border-left: 4px solid #4caf50;
    }
    
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .disclaimer {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0;
        border-left: 5px solid #e17055;
        box-shadow: 0 4px 12px rgba(225, 112, 85, 0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def setup_hierarchical_rag_system():
    """Setup and cache the hierarchical RAG system."""
    try:
        with st.spinner("🔧 Setting up Hierarchical Diagnostic RAG System..."):
            # Load config
            config = Config()
            
            # Initialize components
            processor = HierarchicalDocumentProcessor(config.config["processing"])
            retriever = HierarchicalRetriever(config)
            generator = HierarchicalGenerator(config)
            
            # Check for foundation datasets
            foundation_dir = Path("data/foundation")
            
            try:
                retriever.load_hierarchical_collections()
                st.success("✅ Loaded existing hierarchical collections")
            except ValueError:
                st.info("🔧 Creating hierarchical collections...")
                
                # Load foundation dataset
                if not foundation_dir.exists() or not (foundation_dir / "foundation_medical_data.json").exists():
                    st.error("❌ Foundation dataset not found")
                    with st.expander("🔧 Setup Foundation Dataset"):
                        st.code("""
# Create foundation dataset
python fetch_foundation_data.py --max-results 1000

# Quick test  
python fetch_foundation_data.py --quick

# MedReason only
python fetch_foundation_data.py --medreason-only --max-results 500
""")
                    return None, None
                
                # Load and process foundation data
                all_docs = processor.load_foundation_dataset(foundation_dir)
                organized_docs = processor.organize_by_reasoning_type(all_docs)
                
                # Create hierarchical collections
                retriever.create_hierarchical_collections()
                retriever.add_documents_to_tiers(organized_docs)
                
                st.success(f"✅ Hierarchical system created with {len(all_docs)} chunks across 3 tiers")
            
            return retriever, generator
            
    except Exception as e:
        st.error(f"❌ Setup failed: {str(e)}")
        st.error("Make sure Ollama is running and the mistral:7b-instruct model is available.")
        with st.expander("🔧 Setup Instructions"):
            st.code("""
# Start Ollama
ollama serve &

# Pull the model
ollama pull mistral:7b-instruct

# Create foundation dataset
python fetch_foundation_data.py --quick
""")
        return None, None

def ask_hierarchical_question(question, retriever, generator):
    """Process a medical question using hierarchical reasoning."""
    if not question.strip():
        return None
        
    try:
        start_time = time.time()
        
        # Perform hierarchical search
        with st.spinner("🧠 Performing hierarchical diagnostic reasoning..."):
            hierarchical_results = retriever.hierarchical_search(question)
        
        # Generate hierarchical response
        with st.spinner("🤖 Generating hierarchical diagnostic response..."):
            response = generator.generate_hierarchical_response(question, hierarchical_results)
        
        processing_time = time.time() - start_time
        
        return {
            'answer': response,
            'hierarchical_results': hierarchical_results,
            'time': processing_time
        }
        
    except Exception as e:
        return {
            'answer': f"❌ Error processing question: {str(e)}",
            'hierarchical_results': {},
            'time': 0
        }

def main():
    """Main application."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Ních gà - Hierarchical Diagnostic RAG</h1>
        <p style="margin: 0; font-size: 1.1em;">Basic Reasoning System with Three-Tier Clinical Decision Making</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9em; opacity: 0.9;">Pattern Recognition → Hypothesis Testing → Confirmation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Setup system
    retriever, generator = setup_hierarchical_rag_system()
    
    if not retriever or not generator:
        st.error("❌ Failed to initialize the hierarchical system.")
        st.stop()
    
    # Initialize session state
    if 'hierarchical_current_question' not in st.session_state:
        st.session_state.hierarchical_current_question = ""
    if 'hierarchical_process_question' not in st.session_state:
        st.session_state.hierarchical_process_question = False
    if 'hierarchical_last_processed_question' not in st.session_state:
        st.session_state.hierarchical_last_processed_question = ""
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🧠 Hierarchical System Status")
        st.success("✅ Three-Tier Diagnostic RAG Ready")
        
        # Check foundation dataset
        foundation_file = Path("data/foundation/foundation_medical_data.json")
        if foundation_file.exists():
            try:
                with open(foundation_file, "r") as f:
                    foundation_data = json.load(f)
                st.markdown("### 📊 Foundation Dataset Loaded")
                st.success(f"📚 **{len(foundation_data):,}** total documents")
                
                # Show breakdown by source
                source_counts = {}
                for doc in foundation_data:
                    source = doc["metadata"]["source"]
                    source_counts[source] = source_counts.get(source, 0) + 1
                
                st.markdown("**🔬 Dataset Sources:**")
                for source, count in source_counts.items():
                    icon = {"medreason": "🧠", "msdiagnosis": "🏥", "pmc_patients": "📚", "drugbank": "💊"}.get(source, "📄")
                    st.write(f"{icon} **{source}**: {count:,} docs")
                    
            except Exception as e:
                st.warning(f"Could not read foundation stats: {e}")
        else:
            st.error("❌ Foundation dataset not found")
        
        st.markdown("### 🎯 Three-Tier Architecture")
        st.info("""
        **🔍 Tier 1 - Pattern Recognition**  
        Fast symptom-disease associations  
        Drug information and quick lookups
        
        **🧠 Tier 2 - Hypothesis Testing**  
        Knowledge graph reasoning chains  
        Multi-step diagnostic scenarios
        
        **✅ Tier 3 - Confirmation**  
        Patient case studies and evidence  
        Clinical documentation validation
        """)
        
        st.markdown("### 💡 Hierarchical Examples")
        example_questions = [
            "What is the three-tier reasoning for diagnosing chest pain?",
            "How do you systematically approach shortness of breath?", 
            "What are the reasoning chains for diabetes diagnosis?",
            "Explain the hierarchical approach to abdominal pain",
            "How does pattern recognition work in medical diagnosis?",
            "What evidence confirms a diagnosis of heart failure?",
            "How do drug interactions affect diagnostic reasoning?",
            "What are the steps in hypothesis testing for stroke?",
            "How do you confirm a diagnosis using clinical evidence?",
            "Explain knowledge graph guided medical reasoning"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(f"🧠 {question}", key=f"hierarchical_example_{i}", use_container_width=True):
                st.session_state.hierarchical_current_question = question
                st.session_state.hierarchical_process_question = True
                st.rerun()
        
        st.markdown("### 🔄 System Comparison")
        st.caption("""
        **Simple**: 3 test documents  
        **KG**: 5K medical documents  
        **Basic Reasoning**: 95K+ with hierarchical 3-tier reasoning
        """)
    
    # Question input section
    st.markdown("""
    <div class="question-container">
        <h3 style="margin-top: 0; color: white;">🔍 Ask Your Hierarchical Diagnostic Question</h3>
        <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9); font-size: 0.9em;">💡 Experience three-tier clinical reasoning: Pattern Recognition → Hypothesis Testing → Confirmation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create form for question input
    with st.form(key="hierarchical_question_form", clear_on_submit=True):
        user_question = st.text_input(
            "Enter your diagnostic question:",
            value="",
            placeholder="e.g., What is the three-tier reasoning for diagnosing chest pain?",
            key="hierarchical_question_input",
            label_visibility="collapsed"
        )
        
        submitted = st.form_submit_button("🧠 Start Hierarchical Reasoning")
    
    # Hide submit button with CSS
    st.markdown("""
    <style>
    .stForm button[kind="primaryFormSubmit"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Process question
    if submitted and user_question.strip():
        st.session_state.hierarchical_current_question = user_question.strip()
        st.session_state.hierarchical_process_question = True
    
    if st.session_state.hierarchical_process_question and st.session_state.hierarchical_current_question.strip():
        question_to_process = st.session_state.hierarchical_current_question
        
        # Reset process flag
        st.session_state.hierarchical_process_question = False
        
        if question_to_process != st.session_state.hierarchical_last_processed_question:
            st.session_state.hierarchical_last_processed_question = question_to_process
            
            # Display user question
            st.markdown(f"""
            <div class="user-question">
                <h4 style="margin: 0 0 0.5rem 0;">🧑‍⚕️ Your Diagnostic Question</h4>
                <p style="margin: 0; font-size: 1.1em;">{question_to_process}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Process and display hierarchical response
            result = ask_hierarchical_question(question_to_process, retriever, generator)
            
            if result:
                # Display hierarchical answer
                st.markdown(f"""
                <div class="bot-response">
                    <h4 style="margin: 0 0 1rem 0; color: #2c3e50;">🧠 Hierarchical Diagnostic Response</h4>
                    <div style="font-size: 1.05em; line-height: 1.7;">
                        {result['answer']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display processing time
                st.markdown(f"""
                <div class="metrics-container">
                    <strong>⏱️ Hierarchical Processing Time: {result['time']:.1f} seconds</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Display hierarchical breakdown
                if result['hierarchical_results']:
                    with st.expander("🔍 View Three-Tier Reasoning Breakdown", expanded=True):
                        hierarchical_results = result['hierarchical_results']
                        
                        # Tier 1: Pattern Recognition
                        if hierarchical_results.get("tier1_patterns"):
                            st.markdown("""
                            <div class="tier-section">
                                <h5 style="margin: 0 0 0.5rem 0; color: #d35400;">🔍 Tier 1: Pattern Recognition</h5>
                                <p style="margin: 0 0 0.5rem 0; font-size: 0.9em;"><em>Fast initial screening and symptom-disease associations</em></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for i, result_item in enumerate(hierarchical_results["tier1_patterns"]):
                                source = result_item['metadata'].get('source', 'unknown')
                                st.write(f"**Pattern {i+1}** ({source}) - Score: {result_item['score']:.2f}")
                                st.write(f"📄 {result_item['text'][:200]}...")
                                st.write("---")
                        
                        # Tier 2: Hypothesis Testing
                        if hierarchical_results.get("tier2_hypotheses"):
                            st.markdown("""
                            <div class="tier-section">
                                <h5 style="margin: 0 0 0.5rem 0; color: #d35400;">🧠 Tier 2: Hypothesis Testing</h5>
                                <p style="margin: 0 0 0.5rem 0; font-size: 0.9em;"><em>Knowledge graph reasoning and systematic diagnostic chains</em></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for i, result_item in enumerate(hierarchical_results["tier2_hypotheses"]):
                                source = result_item['metadata'].get('source', 'unknown')
                                reasoning_type = result_item['metadata'].get('reasoning_type', 'reasoning')
                                st.write(f"**Hypothesis {i+1}** ({source}, {reasoning_type}) - Score: {result_item['score']:.2f}")
                                st.write(f"🧠 {result_item['text'][:250]}...")
                                st.write("---")
                        
                        # Tier 3: Confirmation
                        if hierarchical_results.get("tier3_confirmation"):
                            st.markdown("""
                            <div class="tier-section">
                                <h5 style="margin: 0 0 0.5rem 0; color: #d35400;">✅ Tier 3: Confirmation</h5>
                                <p style="margin: 0 0 0.5rem 0; font-size: 0.9em;"><em>Clinical evidence and case study validation</em></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for i, result_item in enumerate(hierarchical_results["tier3_confirmation"]):
                                source = result_item['metadata'].get('source', 'unknown')
                                st.write(f"**Evidence {i+1}** ({source}) - Score: {result_item['score']:.2f}")
                                st.write(f"✅ {result_item['text'][:250]}...")
                                st.write("---")
            
            # Clear current question
            st.session_state.hierarchical_current_question = ""
    
    # Footer with disclaimer
    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
        <h4 style="margin: 0 0 1rem 0; color: #d35400;">⚠️ Important Medical Disclaimer</h4>
        <p style="color: #2c3e50; margin: 0; font-size: 1em; line-height: 1.6;">
            <strong>This is a research and educational tool for studying hierarchical diagnostic reasoning.</strong> 
            The three-tier reasoning system is designed to demonstrate clinical decision-making patterns but should not be used as a substitute 
            for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare 
            professionals for medical decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()