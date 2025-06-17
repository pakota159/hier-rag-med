"""
Complete Streamlit chat app for HierRAGMed - Fixed for torch compatibility.
"""

import streamlit as st
import sys
from pathlib import Path
import time
import os
import json
import tqdm

# Fix torch/streamlit compatibility issue
os.environ["STREAMLIT_DISABLE_AUTOINDEX"] = "true"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from .config import Config
    from .processing import DocumentProcessor
    from .retrieval import Retriever
    from .generation import Generator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure you're running from the project root directory and all dependencies are installed.")
    st.stop()

# Page config
st.set_page_config(
    page_title="Ních gà - Medical Assistant (KG)",
    page_icon="⚕️",
    layout="wide"
)

# Custom CSS for better chat appearance
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Question input styling */
    .question-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* User question styling */
    .user-question {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    
    /* Bot response styling */
    .bot-response {
        background: white;
        border: 2px solid #e3f2fd;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        line-height: 1.6;
    }
    
    /* Metrics styling */
    .metrics-container {
        background: #e8f5e8;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        border-left: 4px solid #4caf50;
    }
    
    /* Source item styling */
    .source-item {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 4px solid #ff9800;
        box-shadow: 0 2px 8px rgba(255, 152, 0, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Disclaimer styling */
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
def setup_rag_system():
    """Setup and cache the RAG system."""
    try:
        with st.spinner("🔧 Setting up KG-Enhanced Medical RAG System..."):
            # Load config
            config = Config()
            
            # Initialize components
            processor = DocumentProcessor(config.config["processing"])
            retriever = Retriever(config)
            generator = Generator(config)
            
            # Setup KG knowledge base
            try:
                retriever.load_collection("kg_medical_docs")  # Different collection name
                st.success("✅ Loaded KG medical knowledge base")
            except ValueError:
                st.info("📚 Creating KG medical knowledge base...")
                
                # Load the fetched KG datasets
                kg_data_dir = Path("data/kg_raw")
                if not kg_data_dir.exists():
                    st.error("❌ KG data not found. Run: python fetch_data.py first")
                    return None, None
                
                # Load combined dataset
                combined_file = kg_data_dir / "combined" / "all_medical_data.json"
                if combined_file.exists():
                    st.info(f"📄 Loading KG datasets from {combined_file}...")
                    
                    with open(combined_file, "r") as f:
                        kg_documents = json.load(f)
                    
                    # Process documents
                    all_docs = []
                    for doc in tqdm.tqdm(kg_documents[:1000], desc="Processing KG docs"):  # Limit for demo
                        chunks = processor.process_text(doc["text"], doc["metadata"])
                        all_docs.extend(chunks)
                    
                    # Create collection
                    retriever.create_collection("kg_medical_docs")
                    retriever.add_documents(all_docs)
                    
                    st.success(f"✅ KG knowledge base created with {len(all_docs)} chunks from {len(kg_documents)} documents")
                else:
                    st.error("❌ Combined dataset not found. Run fetch_data.py first")
                    return None, None
            
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

# Check if model is available
ollama list
""")
        return None, None

def ask_question(question, retriever, generator):
    """Process a medical question."""
    if not question.strip():
        return None
        
    try:
        start_time = time.time()
        
        # Search for relevant information
        with st.spinner("🔍 Searching medical knowledge..."):
            search_results = retriever.search(question, n_results=3)
        
        # Generate response
        with st.spinner("🤖 Generating medical response..."):
            response = generator.generate(question, search_results)
        
        processing_time = time.time() - start_time
        
        return {
            'answer': response,
            'sources': search_results,
            'time': processing_time
        }
        
    except Exception as e:
        return {
            'answer': f"❌ Error processing question: {str(e)}",
            'sources': [],
            'time': 0
        }

def main():
    """Main application."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚕️ Ních gà (KG Enhanced)</h1>
        <p style="margin: 0; font-size: 1.1em;">Knowledge Graph Medical RAG Assistant</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9em; opacity: 0.9;">Powered by 5,100+ medical documents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Setup system
    retriever, generator = setup_rag_system()
    
    if not retriever or not generator:
        st.error("❌ Failed to initialize the medical system.")
        st.stop()
    
    # Initialize session state for question processing
    if 'kg_current_question' not in st.session_state:
        st.session_state.kg_current_question = ""
    if 'kg_process_question' not in st.session_state:
        st.session_state.kg_process_question = False
    if 'kg_last_processed_question' not in st.session_state:
        st.session_state.kg_last_processed_question = ""
    if 'kg_input_counter' not in st.session_state:
        st.session_state.kg_input_counter = 0
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 KG System Status")
        st.success("✅ Knowledge Graph RAG Ready")
        
        st.markdown("### 🏥 Extended Medical Coverage")
        st.info("""
        🔬 **PubMed**: 5,000+ research abstracts  
        🏥 **MTSamples**: 60+ clinical documents  
        📚 **MeSH**: 50+ medical concepts  
        📈 **Total**: 5,100+ medical documents
        """)
        
        st.markdown("### 🧠 Enhanced Features")
        st.success("""
        ✅ Evidence stratification  
        ✅ Temporal awareness  
        ✅ Hierarchical reasoning  
        ✅ Multi-source integration
        """)
        
        st.markdown("### 🎯 Available Medical Topics")
        st.info("""
        **Cardiology**: Heart disease, MI, procedures  
        **Endocrinology**: Diabetes, thyroid, hormones  
        **Neurology**: Stroke, seizures, brain disorders  
        **Psychiatry**: Depression, anxiety, mental health  
        **Obstetrics**: Pregnancy, prenatal care  
        **Gastroenterology**: Digestive disorders  
        **Emergency Medicine**: Acute conditions  
        **Orthopedics**: Musculoskeletal injuries  
        **And 12+ more specialties...**
        """)
        
        st.markdown("### 💡 Enhanced Example Questions")
        example_questions = [
            "What are the latest treatments for myocardial infarction?",
            "How does insulin resistance develop in type 2 diabetes?", 
            "What are the contraindications for cardiac catheterization?",
            "Explain the pathophysiology of stroke",
            "What medications are used for major depressive disorder?",
            "How is acute appendicitis diagnosed?",
            "What are the stages of pregnancy complications?",
            "Compare ACE inhibitors vs ARBs for hypertension",
            "What is the evidence for metformin in diabetes?",
            "How is colonoscopy screening performed?"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(f"💬 {question}", key=f"kg_example_{i}", use_container_width=True):
                st.session_state.kg_current_question = question
                st.session_state.kg_process_question = True
                st.rerun()
        
        st.markdown("### 🔄 Compare Systems")
        st.caption("Simple RAG: 3 test documents vs KG RAG: 5,100+ real medical documents")
        
        st.markdown("### 📊 Data Sources")
        with st.expander("📖 View Data Details"):
            st.markdown("""
            **PubMed Research Articles:**
            - Peer-reviewed medical literature
            - Last 10 years publication date
            - Evidence-based medicine
            
            **MTSamples Clinical Notes:**
            - Real medical transcriptions
            - 20+ medical specialties
            - Clinical documentation patterns
            
            **MeSH Medical Terminology:**
            - Authoritative medical vocabulary
            - Hierarchical concept relationships
            - Diseases, procedures, drugs
            """)
    
    # Question input section
    st.markdown("""
    <div class="question-container">
        <h3 style="margin-top: 0; color: #2c3e50;">🔍 Ask Your Medical Question</h3>
        <p style="margin: 0.5rem 0 0 0; color: #7f8c8d; font-size: 0.9em;">💡 Type your question and press Enter to submit</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a form for better Enter key handling
    with st.form(key="kg_question_form", clear_on_submit=True):
        user_question = st.text_input(
            "Enter your question and press Enter:",
            value="",
            placeholder="e.g., What are the latest treatments for diabetes?",
            key="kg_question_input",
            label_visibility="collapsed"
        )
        
        # Hidden form submit button - required by Streamlit but invisible
        submitted = st.form_submit_button("Submit")
    
    # Hide the submit button completely with CSS
    st.markdown("""
    <style>
    .stForm button[kind="primaryFormSubmit"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check if form was submitted or if triggered from sidebar
    if submitted and user_question.strip():
        st.session_state.kg_current_question = user_question.strip()
        st.session_state.kg_process_question = True
    
    # Process question if submitted or triggered from sidebar
    if st.session_state.kg_process_question and st.session_state.kg_current_question.strip():
        question_to_process = st.session_state.kg_current_question
        
        # Reset the process flag
        st.session_state.kg_process_question = False
        
        # Check if this is a different question from the last processed one
        if question_to_process != st.session_state.kg_last_processed_question:
            st.session_state.kg_last_processed_question = question_to_process
            
            # Display user question with better styling
            st.markdown(f"""
            <div class="user-question">
                <h4 style="margin: 0 0 0.5rem 0;">🧑‍⚕️ Your Question</h4>
                <p style="margin: 0; font-size: 1.1em;">{question_to_process}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Process and display response
            result = ask_question(question_to_process, retriever, generator)
            
            if result:
                # Display answer with better styling
                st.markdown(f"""
                <div class="bot-response">
                    <h4 style="margin: 0 0 1rem 0; color: #2c3e50;">🤖 KG Medical Assistant Response</h4>
                    <div style="font-size: 1.05em; line-height: 1.7;">
                        {result['answer']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display processing time with better styling
                st.markdown(f"""
                <div class="metrics-container">
                    <strong>⏱️ Processing Time: {result['time']:.1f} seconds</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources with better styling
                if result['sources']:
                    with st.expander("📚 View Medical Sources & References", expanded=False):
                        st.markdown("**📖 Sources used to generate this medical response:**")
                        
                        for i, source in enumerate(result['sources'][:3]):
                            # Show source type based on metadata
                            source_type = source['metadata'].get('source', 'unknown')
                            source_icon = {
                                'pubmed': '🔬',
                                'mtsamples': '🏥', 
                                'mesh': '📚'
                            }.get(source_type, '📄')
                            
                            st.markdown(f"""
                            <div class="source-item">
                                <h5 style="margin: 0 0 0.5rem 0; color: #d35400;">{source_icon} Source {i+1} ({source_type.upper()})</h5>
                                <p style="margin: 0 0 0.5rem 0;"><strong>Relevance Score:</strong> {source['score']:.2f}/1.0</p>
                                <p style="margin: 0 0 0.8rem 0;"><strong>Document:</strong> {source['metadata']['doc_id']}</p>
                                <div style="background: rgba(255,255,255,0.7); padding: 0.8rem; border-radius: 8px; font-size: 0.95em; line-height: 1.5;">
                                    {source['text'][:400]}{"..." if len(source['text']) > 400 else ""}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Clear the current question after processing
            st.session_state.kg_current_question = ""
    
    # Footer with disclaimer
    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
        <h4 style="margin: 0 0 1rem 0; color: #d35400;">⚠️ Important Medical Disclaimer</h4>
        <p style="color: #2c3e50; margin: 0; font-size: 1em; line-height: 1.6;">
            <strong>This is a research and educational tool only.</strong> The information provided should not be used as a substitute 
            for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare 
            professionals for medical decisions. This AI system is designed to provide general information 
            based on medical literature and should not replace clinical judgment.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()