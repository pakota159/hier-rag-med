"""
Web interface module for HierRAGMed.
"""

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import Config
from generation import Generator
from retrieval import Retriever


class Query(BaseModel):
    """Query model for API requests."""
    text: str
    collection_name: str
    n_results: Optional[int] = None
    filter_metadata: Optional[Dict] = None
    with_citations: Optional[bool] = False


class Response(BaseModel):
    """Response model for API responses."""
    answer: str
    citations: Optional[List[Dict[str, str]]] = None
    retrieved_docs: List[Dict[str, str]]


app = FastAPI(
    title="HierRAGMed API",
    description="API for medical document retrieval and question answering",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config = Config()
retriever = Retriever(config)
generator = Generator(config)


@app.post("/query", response_model=Response)
async def query_endpoint(query: Query):
    """Query endpoint for RAG pipeline."""
    try:
        # Load collection
        retriever.load_collection(query.collection_name)

        # Retrieve documents
        retrieved_docs = retriever.hybrid_search(
            query.text,
            n_results=query.n_results,
            filter_metadata=query.filter_metadata
        )

        # Generate response
        if query.with_citations:
            result = generator.generate_with_citations(
                query.text,
                retrieved_docs
            )
            return Response(
                answer=result["response"],
                citations=result["citations"],
                retrieved_docs=retrieved_docs
            )
        else:
            answer = generator.generate(
                query.text,
                retrieved_docs
            )
            return Response(
                answer=answer,
                retrieved_docs=retrieved_docs
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections")
async def list_collections():
    """List available collections."""
    try:
        collections = retriever.client.list_collections()
        return {"collections": [col.name for col in collections]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server."""
    uvicorn.run(app, host=host, port=port) 