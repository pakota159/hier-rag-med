"""
Complete Streamlit chat app for HierRAGMed - Fixed for torch compatibility.
"""

import streamlit as st
import sys
from pathlib import Path
import time
import os

# Fix torch/streamlit compatibility issue
os.environ["STREAMLIT_DISABLE_AUTOINDEX"] = "true"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config import Config
    from processing import DocumentProcessor
    from retrieval import Retriever
    from generation import Generator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure you're running from the project root directory and all dependencies are installed.")
    st.stop()

# Page config
st.set_page_config(
    page_title="Ních gà - Medical Assistant",
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
        with st.spinner("🔧 Setting up Medical RAG System..."):
            # Load config
            config_path = Path("config.yaml")
            if not config_path.exists():
                st.error("❌ config.yaml not found. Please make sure you're running from the project root directory.")
                return None, None
                
            config = Config(config_path)
            
            # Initialize components
            processor = DocumentProcessor(config.config["processing"])
            retriever = Retriever(config)
            generator = Generator(config)
            
            # Setup knowledge base
            try:
                retriever.load_collection("medical_docs")
                st.success("✅ Loaded existing medical knowledge base")
            except ValueError:
                st.info("📚 Creating medical knowledge base...")
                
                # Check for documents in data/raw directory
                raw_data_path = Path("data/raw")
                documents_found = False
                all_docs = []
                
                if raw_data_path.exists():
                    for txt_file in raw_data_path.glob("*.txt"):
                        st.info(f"📄 Loading {txt_file.name}...")
                        try:
                            with open(txt_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            docs = processor.process_text(content, {
                                "source": "medical_documents",
                                "doc_id": txt_file.stem
                            })
                            all_docs.extend(docs)
                            documents_found = True
                        except Exception as e:
                            st.warning(f"⚠️ Could not read {txt_file.name}: {e}")
                
                # If no documents found, create sample data
                if not documents_found:
                    st.info("📝 No documents found in data/raw/, creating sample data...")
                    
                    # Sample medical data
                    medical_data = {
                        "diabetes": """
Type 2 Diabetes Information:

Symptoms:
• Increased thirst (polydipsia)
• Frequent urination (polyuria)
• Increased hunger (polyphagia)
• Unexplained weight loss
• Fatigue and weakness
• Blurred vision
• Slow-healing cuts and wounds
• Frequent infections

Risk Factors:
• Family history of diabetes
• Overweight or obesity
• Age 45 or older
• Physical inactivity
• High blood pressure
• Abnormal cholesterol levels

Treatment Options:
• Lifestyle modifications (diet and exercise)
• Metformin (first-line medication)
• Other diabetes medications as needed
• Regular blood glucose monitoring
• Regular medical check-ups
""",
                        "hypertension": """
Hypertension (High Blood Pressure) Information:

Blood Pressure Categories:
• Normal: Less than 120/80 mmHg
• Elevated: 120-129 systolic, less than 80 diastolic
• Stage 1: 130-139 systolic or 80-89 diastolic
• Stage 2: 140/90 mmHg or higher
• Hypertensive Crisis: Higher than 180/120 mmHg

Symptoms:
• Often called "silent killer" - usually no symptoms
• Severe hypertension may cause:
  - Headaches
  - Shortness of breath
  - Chest pain
  - Dizziness

Treatment:
• Lifestyle changes:
  - Reduce sodium intake
  - Regular physical activity
  - Maintain healthy weight
  - Limit alcohol consumption
• Medications:
  - ACE inhibitors
  - ARBs (Angiotensin Receptor Blockers)
  - Diuretics
  - Beta-blockers
  - Calcium channel blockers
""",
                        "pregnancy": """
Pregnancy Information:

Early Signs and Symptoms:
• Missed menstrual period
• Nausea and vomiting (morning sickness)
• Breast tenderness and enlargement
• Fatigue and increased sleepiness
• Frequent urination
• Food aversions or cravings
• Mood changes
• Light spotting (implantation bleeding)

Pregnancy Trimesters:
• First Trimester (Weeks 1-12):
  - Organ development
  - Morning sickness common
  - Important prenatal vitamin intake
• Second Trimester (Weeks 13-27):
  - Often called "golden period"
  - Energy levels improve
  - Baby movements felt
• Third Trimester (Weeks 28-40):
  - Rapid baby growth
  - Preparation for delivery
  - More frequent doctor visits

Prenatal Care:
• Regular doctor visits
• Prenatal vitamins with folic acid
• Healthy diet and exercise
• Avoid alcohol, smoking, and certain medications
• Monitor weight gain
• Stay hydrated
• Get adequate rest
"""
                    }
                    
                    # Process sample documents
                    for doc_id, content in medical_data.items():
                        docs = processor.process_text(content, {
                            "source": "sample_medical_data",
                            "doc_id": doc_id
                        })
                        all_docs.extend(docs)
                
                # Create collection and add documents
                retriever.create_collection("medical_docs")
                retriever.add_documents(all_docs)
                
                if documents_found:
                    st.success(f"✅ Medical knowledge base created with {len(all_docs)} chunks from data/raw/")
                else:
                    st.success("✅ Medical knowledge base created with sample data")
            
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
        <h1>⚕️ Ních gà</h1>
        <p style="margin: 0; font-size: 1.1em;">Medical RAG Chat Assistant</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9em; opacity: 0.9;">Ask medical questions and receive evidence-based answers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Setup system
    retriever, generator = setup_rag_system()
    
    if not retriever or not generator:
        st.error("❌ Failed to initialize the medical system.")
        st.stop()
    
    # Initialize session state for question processing
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    if 'process_question' not in st.session_state:
        st.session_state.process_question = False
    if 'last_processed_question' not in st.session_state:
        st.session_state.last_processed_question = ""
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 System Status")
        st.success("✅ Medical RAG System Ready")
        
        st.markdown("### 🏥 Available Topics")
        st.info("""
        🩺 **Diabetes** - symptoms, treatment, risk factors  
        🫀 **Hypertension** - blood pressure, medications  
        🤱 **Pregnancy** - prenatal care, symptoms, stages  
        """)
        
        st.markdown("### ➕ Add More Topics")
        st.caption("Add your medical documents to `data/raw/` to expand the knowledge base")
        
        st.markdown("### 💡 Example Questions")
        example_questions = [
            "What are the symptoms of diabetes?",
            "How is high blood pressure treated?",
            "What are normal blood pressure values?",
            "What are early signs of pregnancy?",
            "What should I eat during pregnancy?",
            "What are the pregnancy trimesters?"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(f"💬 {question}", key=f"example_{i}", use_container_width=True):
                st.session_state.current_question = question
                st.session_state.process_question = True
    
    # Question input section
    st.markdown("""
    <div class="question-container">
        <h3 style="margin-top: 0; color: #2c3e50;">🔍 Ask Your Medical Question</h3>
        <p style="margin: 0.5rem 0 0 0; color: #7f8c8d; font-size: 0.9em;">💡 Type your question and press Enter to submit</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input form - Enter to submit
    user_question = st.text_input(
        "Enter your question and press Enter:",
        value=st.session_state.current_question,
        placeholder="e.g., What are the symptoms of diabetes?",
        key="question_input",
        label_visibility="collapsed",
        on_change=None
    )
    
    # Check if question was submitted (Enter pressed) or triggered from sidebar
    question_submitted = False
    
    # If user typed something new and it's different from stored question, they pressed Enter
    if user_question and user_question != st.session_state.get('last_processed_question', ''):
        question_submitted = True
        st.session_state.current_question = user_question
    
    # Process question if Enter was pressed or if triggered from sidebar/quick buttons
    if question_submitted or st.session_state.process_question:
        question_to_process = st.session_state.current_question
        
        if question_to_process.strip():
            # Reset the process flag and store the processed question
            st.session_state.process_question = False
            st.session_state.last_processed_question = question_to_process
            
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
                    <h4 style="margin: 0 0 1rem 0; color: #2c3e50;">🤖 Medical Assistant Response</h4>
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
                            st.markdown(f"""
                            <div class="source-item">
                                <h5 style="margin: 0 0 0.5rem 0; color: #d35400;">📄 Source {i+1}</h5>
                                <p style="margin: 0 0 0.5rem 0;"><strong>Relevance Score:</strong> {source['score']:.2f}/1.0</p>
                                <p style="margin: 0 0 0.8rem 0;"><strong>Document:</strong> {source['metadata']['doc_id']}</p>
                                <div style="background: rgba(255,255,255,0.7); padding: 0.8rem; border-radius: 8px; font-size: 0.95em; line-height: 1.5;">
                                    {source['text'][:400]}{"..." if len(source['text']) > 400 else ""}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Clear the current question after processing
            st.session_state.current_question = ""
        
        elif question_submitted:
            st.warning("⚠️ Please enter a medical question first.")
    
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