# HierRAGMed

A medical RAG (Retrieval-Augmented Generation) system for local development, built with Python, Ollama, LangChain, ChromaDB, and sentence-transformers.

## Features

- Document processing and chunking for medical texts
- Semantic and hybrid search using ChromaDB
- Local LLM integration with Ollama
- FastAPI web interface
- Comprehensive evaluation metrics
- Configurable system prompts

## Prerequisites

- **Python 3.10+**
- **Conda** (Miniconda or Anaconda)
- **Ollama** for local LLM
- **Cross-platform**: Windows, macOS, Linux

## 🚀 Quick Setup

### 1. Install Prerequisites

**Ollama Installation:**
```bash
# macOS
brew install ollama

# Windows
winget install Ollama.Ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Setup Project

```bash
# Clone repository
git clone https://github.com/pakota159/hier-rag-med.git
cd hier-rag-med

# Create conda environment
conda create -n rag-med python=3.10 -y

# Activate environment
conda activate rag-med

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/{raw,processed,vector_db,logs}
```

### 3. Setup Ollama Model

```bash
# Start Ollama service
ollama serve &

# Pull the required model
ollama pull mistral:7b-instruct
```

### 4. Run Test

```bash
streamlit run src/simple/streamlit_app.py --server.port 8501
```

## 📊 Dataset Expansion for Hierarchical Diagnostic Reasoning

HierRAGMed includes two expansion phases to scale from basic RAG to production-ready hierarchical diagnostic reasoning:

### **Phase 1: Foundation Dataset (Target: ~95K documents)**
**Timeline**: 2-3 weeks | **Status**: Ready for implementation

| Dataset | Size | Type | Purpose | Implementation |
|---------|------|------|---------|----------------|
| **Current Base** | 5.1K | PubMed + MTSamples + MeSH | Medical knowledge foundation | ✅ Complete |
| **MedReason** | 32K | Structured reasoning chains | Diagnostic thinking patterns | 📋 Priority 1 |
| **MSDiagnosis** | Variable | Multi-step diagnosis | Three-tier reasoning (Primary→Differential→Final) | 📋 Priority 1 |
| **PMC-Patients** | 50K | Patient case studies | Clinical reasoning examples | 📋 Priority 2 |
| **Drug Database** | 5K | Medication information | Treatment knowledge | 📋 Priority 3 |

**Expected Performance**: 68-72% medical QA accuracy, enables basic hierarchical reasoning training

#### **Key Datasets Details**

**🧠 MedReason Dataset (2025)**
- **Innovation**: Knowledge graph-guided reasoning chains
- **Quality**: Validated by medical professionals across specialties
- **Format**: Question-answer pairs with step-by-step explanations
- **Clinical Value**: Provides structured "thinking paths" for medical problem-solving
- **Access**: Open source - https://github.com/UCSC-VLAA/MedReason

**🏥 MSDiagnosis Dataset (2024)**
- **Innovation**: EMR-based multi-step diagnostic scenarios
- **Quality**: Professional medical team annotation with three-round validation
- **Format**: Primary diagnosis → Differential diagnosis → Final diagnosis chains
- **Clinical Value**: Directly matches real clinical diagnostic workflows
- **Access**: Research dataset (requires academic access)

**📚 PMC-Patients Dataset**
- **Innovation**: Comprehensive patient cases from medical literature
- **Quality**: Published medical case studies with peer review
- **Format**: Structured patient summaries with outcomes
- **Clinical Value**: Covers rare diseases and complex diagnostic scenarios
- **Access**: Open source with 167K total patient cases

### **Phase 2: Production Dataset (Target: ~250K+ documents)**
**Timeline**: 3-4 months | **Status**: Future expansion

| Dataset | Size | Type | Purpose | Access |
|---------|------|------|---------|--------|
| **MIMIC-IV** | 100K+ | Real clinical notes | Authentic clinical reasoning | Requires training |
| **Clinical Guidelines** | 10K | WHO/CDC/AHA protocols | Evidence-based medicine | Public APIs |
| **Medical Textbooks** | 20K+ | Harrison's/UpToDate excerpts | Comprehensive medical knowledge | License required |
| **Drug Interactions** | 15K | DrugBank + RxNorm | Pharmacology reasoning | Open research |
| **International Data** | 25K+ | Global medical standards | Diverse clinical practices | Various sources |

**Expected Performance**: 75-78% medical QA accuracy, professional-grade diagnostic reasoning

### **Three-Tier Architecture Data Mapping**

#### **Tier 1: Pattern Recognition (Fast Hypothesis Generation)**
- **MeSH concepts**: Quick symptom-disease associations
- **ICD-10 codes**: Rapid disease classification  
- **Drug databases**: Medication identification
- **Clinical keywords**: Symptom pattern matching

#### **Tier 2: Hypothesis Testing (Systematic Evidence Collection)**
- **MedReason chains**: Structured diagnostic reasoning
- **MSDiagnosis**: Multi-step diagnostic validation
- **Clinical guidelines**: Evidence-based protocols
- **PMC-Patients**: Differential diagnosis examples

#### **Tier 3: Confirmation (Comprehensive Verification)**
- **MIMIC-IV discharge summaries**: Final diagnosis confirmation
- **Medical textbooks**: Authoritative knowledge verification
- **Drug interaction databases**: Treatment safety confirmation
- **International guidelines**: Cross-validation of standards

## 🌍 Platform-Specific Notes

### Windows
```powershell
# Activate environment
conda activate rag-med

# Start Ollama service (runs as Windows Service)
ollama serve

# In new PowerShell window
ollama pull mistral:7b-instruct

# Run application
python -m src.main
```

### macOS/Linux
```bash
# Start Ollama in background
ollama serve &

# Pull model
ollama pull mistral:7b-instruct

# Run application
python -m src.main
```

## 📁 Project Structure

```
hierragmed/
├── src/
│   ├── simple/           # Basic RAG implementation
│   ├── kg/              # Knowledge Graph enhanced version
│   ├── __init__.py
│   ├── config.py
│   ├── processing.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── evaluation.py
│   ├── web.py
│   └── main.py
├── data/
│   ├── raw/             # Original documents (Phase 1: 5K)
│   ├── kg_raw/          # KG enhanced data (Phase 1: 95K)
│   ├── processed/       # Processed documents
│   ├── vector_db/       # ChromaDB storage
│   └── logs/            # Application logs
├── fetch_data.py        # Data fetching utilities
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
├── environment.yml      # Conda environment (optional)
└── README.md
```

## ⚙️ Configuration

The system is configured through `config.yaml`. Key configurations include:

- Model settings (embedding and LLM)
- Retrieval parameters
- Document processing options
- System prompts
- Web interface settings

## 📚 Usage Examples

### 1. Basic Setup (Current: 5K documents)
```bash
# Run simple version
streamlit run src/simple/streamlit_app.py --server.port 8501
```

### 2. Fetch Phase 1 Datasets (Target: 95K documents)
```bash
# Fetch comprehensive medical datasets
python fetch_data.py --source all --max-results 1000

# Run KG-enhanced version
streamlit run src/kg/streamlit_app.py --server.port 8501
```

### 3. Process Documents
```python
from src import Config, DocumentProcessor

config = Config()
processor = DocumentProcessor(config["processing"])

# Process a single PDF
documents = processor.process_pdf("path/to/document.pdf")

# Process a directory of PDFs
documents = processor.process_directory("path/to/documents/")

# Save processed documents
processor.save_documents(documents, "data/processed/documents.json")
```

### 4. Create Vector Store
```python
from src import Retriever

retriever = Retriever(config)
retriever.create_collection("medical_docs")
retriever.add_documents(documents)
```

### 5. Query the System
```python
from src import Generator

generator = Generator(config)

# Simple query
results = retriever.search("What are the symptoms of diabetes?")
answer = generator.generate("What are the symptoms of diabetes?", results)

# Query with citations
results = retriever.hybrid_search("What are the symptoms of diabetes?")
response = generator.generate_with_citations("What are the symptoms of diabetes?", results)
```

### 6. Web Interface
```bash
# Start the web interface
python -m src.main

# Access API documentation
# http://localhost:8000/docs

# Health check
# http://localhost:8000/health
```

## 🔌 API Endpoints

- **POST /query**: Submit a query to the RAG system
- **GET /collections**: List available document collections
- **GET /health**: Health check endpoint
- **GET /docs**: Interactive API documentation

## 📊 Evaluation

The system includes comprehensive evaluation metrics:

- **Retrieval metrics**: precision, recall, F1, semantic similarity
- **Generation metrics**: ROUGE scores, semantic similarity

```python
from src import Evaluator

evaluator = Evaluator(config)
metrics = evaluator.evaluate_rag(
    query="What are the symptoms of diabetes?",
    retrieved_docs=results,
    generated_text=answer,
    reference_docs=reference_docs,
    reference_text=reference_answer
)
```

## 🔧 Environment Management

### Activate Environment
```bash
conda activate rag-med
```

### Update Dependencies
```bash
conda activate rag-med
pip install -r requirements.txt --upgrade
```

### Deactivate Environment
```bash
conda deactivate
```

### Remove Environment
```bash
conda env remove -n rag-med
```

## 🚨 Troubleshooting

### Common Issues

**1. Ollama Connection Error**
```bash
# Make sure Ollama is running
ollama serve

# Check if model is available
ollama list

# Pull model if missing
ollama pull mistral:7b-instruct
```

**2. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**3. ChromaDB Permission Issues**
```bash
# Fix directory permissions
chmod -R 755 data/vector_db/
```

**4. Port Already in Use**
```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill process or use different port
python -m src.main --port 8001
```

## ✅ Verification

Test your setup:

```bash
# Activate environment
conda activate rag-med

# Test imports
python -c "import langchain, chromadb, sentence_transformers, ollama; print('✅ All imports successful')"

# Test Ollama connection
python -c "import ollama; print(ollama.list())"

# Run health check
curl http://localhost:8000/health
```

## 🚀 Roadmap

### **Current Status**
- ✅ Basic RAG implementation (5K documents)
- ✅ Knowledge Graph enhanced version infrastructure
- ✅ Multi-source data fetching capabilities

### **Phase 1 Goals (Next 2-3 weeks)**
- 📋 Integrate MedReason reasoning chains (32K)
- 📋 Add MSDiagnosis multi-step diagnostics
- 📋 Include PMC-Patients case studies (50K)
- 📋 Implement basic hierarchical retrieval architecture

### **Phase 2 Goals (3-4 months)**
- 📋 Full MIMIC-IV integration (100K+ clinical notes)
- 📋 Medical textbook knowledge base
- 📋 Complete three-tier diagnostic reasoning implementation
- 📋 Production-ready deployment optimization

### **Innovation Target**
Transform from traditional RAG to **Hierarchical Diagnostic Reasoning RAG** that mirrors clinical decision-making patterns:
- **Pattern Recognition** → **Hypothesis Testing** → **Confirmation**
- Evidence-stratified retrieval with temporal awareness
- Clinical workflow alignment with established medical frameworks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 Citation

If you use HierRAGMed in your research, please cite:

```bibtex
@software{hierragmed2024,
  title={HierRAGMed: Hierarchical Diagnostic Reasoning for Medical Question Answering},
  author={Your Name},
  year={2024},
  url={https://github.com/pakota159/hier-rag-med}
}
```

---

**Need help?** Create an issue or check the troubleshooting section above.