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

- Python 3.9+
- Ollama installed and running locally
- M1 MacBook (or compatible system)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hierragmed.git
cd hierragmed
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Pull the required Ollama model:
```bash
ollama pull mistral
```

## Project Structure

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
│   ├── raw/
│   ├── processed/
│   ├── vector_db/
│   └── logs/
├── config.yaml
├── requirements.txt
└── README.md
```

## Configuration

The system is configured through `config.yaml`. Key configurations include:

- Model settings (embedding and LLM)
- Retrieval parameters
- Document processing options
- System prompts
- Web interface settings

## Usage

1. Process documents:
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

2. Create and populate vector store:
```python
from src import Retriever

retriever = Retriever(config)
retriever.create_collection("medical_docs")
retriever.add_documents(documents)
```

3. Query the system:
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

4. Start the web interface:
```bash
python -m src.main
```

The API will be available at `http://localhost:8000`. You can access the interactive API documentation at `http://localhost:8000/docs`.

## API Endpoints

- `POST /query`: Submit a query to the RAG system
- `GET /collections`: List available document collections

## Evaluation

The system includes comprehensive evaluation metrics:

- Retrieval metrics: precision, recall, F1, semantic similarity
- Generation metrics: ROUGE scores, semantic similarity

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 