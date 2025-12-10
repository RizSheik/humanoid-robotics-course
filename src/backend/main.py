# RAG Chatbot Backend for Physical AI & Humanoid Robotics

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
from pathlib import Path
import sys
import os

# Add the project root to the Python path to import modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

app = FastAPI(
    title="Physical AI & Humanoid Robotics RAG Chatbot",
    description="Retrieval-Augmented Generation chatbot for the Physical AI & Humanoid Robotics textbook",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    query: str
    max_contexts: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.7

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    confidence: float

class Document(BaseModel):
    id: str
    content: str
    title: str
    module: str
    chapter: str
    url: str

# Mock data for demonstration - in real implementation, this would connect to your vector store
MOCK_DOCUMENTS = [
    Document(
        id="m1c1-001",
        content="ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.",
        title="ROS 2 Architecture & Nodes",
        module="Module 1",
        chapter="Chapter 1",
        url="/docs/module-1-the-robotic-nervous-system/module-1-overview"
    ),
    Document(
        id="m1c2-001", 
        content="In ROS 2, topics enable asynchronous message-passing communication between nodes. Services provide synchronous request/response communication, and Actions offer a more advanced form of communication for long-running tasks with feedback.",
        title="Topics, Services & Actions",
        module="Module 1", 
        chapter="Chapter 2",
        url="/docs/module-1-the-robotic-nervous-system/module-1-deep-dive"
    )
]

@app.get("/")
def read_root():
    return {"message": "Physical AI & Humanoid Robotics RAG Chatbot API"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Main endpoint for the RAG chatbot. Takes a user query and returns a response
    with citations to the textbook content.
    """
    try:
        # In a real implementation, this would:
        # 1. Embed the user query using an embedding model
        # 2. Query the vector store (Qdrant) for relevant documents
        # 3. Format the retrieved documents as context
        # 4. Pass the query + context to an LLM (e.g., OpenAI GPT)
        # 5. Return the response with source citations
        
        # For now, using mock implementation
        logger.info(f"Received query: {chat_request.query}")
        
        # Mock response - in real implementation, this would come from LLM
        response = f"I found information related to your query '{chat_request.query}' in the Physical AI & Humanoid Robotics textbook. Based on the content, here's a comprehensive answer..."
        sources = [doc.url for doc in MOCK_DOCUMENTS[:chat_request.max_contexts]]
        confidence = 0.85  # Mock confidence score
        
        return ChatResponse(
            response=response,
            sources=sources,
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/documents", response_model=List[Document])
async def get_documents(
    module: Optional[str] = Query(None, description="Filter by module"),
    chapter: Optional[str] = Query(None, description="Filter by chapter")
):
    """
    Retrieve documents from the textbook content.
    In a real implementation, this would query the vector store.
    """
    try:
        # Filter documents based on query parameters
        filtered_docs = MOCK_DOCUMENTS
        
        if module:
            filtered_docs = [doc for doc in filtered_docs if module.lower() in doc.module.lower()]
        
        if chapter:
            filtered_docs = [doc for doc in filtered_docs if chapter.lower() in doc.chapter.lower()]
        
        return filtered_docs
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy", "service": "RAG Chatbot API"}

if __name__ == "__main__":
    # This allows running the API directly with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)