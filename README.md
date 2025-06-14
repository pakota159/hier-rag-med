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
│   ├── __init__.py
│   ├── config.py
│   ├── processing.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── evaluation.py
│   ├── web.py
│   └── main.py
├── data/
│   ├── raw/          # Original documents
│   ├── processed/    # Processed documents
│   ├── vector_db/    # ChromaDB storage
│   └── logs/         # Application logs
├── config.yaml       # Configuration file
├── requirements.txt  # Python dependencies
├── environment.yml   # Conda environment (optional)
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

### 1. Process Documents
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

### 2. Create Vector Store
```python
from src import Retriever

retriever = Retriever(config)
retriever.create_collection("medical_docs")
retriever.add_documents(documents)
```

### 3. Query the System
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

### 4. Web Interface
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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Need help?** Create an issue or check the troubleshooting section above.