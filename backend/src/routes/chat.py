from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from backend.src.database.connection import get_db
from backend.src.services.chat_service import ChatService, EmbeddingService
from backend.src.models.chat import ChatQuery, UserSession

router = APIRouter()


@router.post("/chat/query", tags=["Chat"])
async def submit_query_to_rag_chatbot(
    query: str,
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Submit a query to the RAG chatbot and receive a contextual response based on educational content
    """
    service = ChatService(db)
    response = await service.query_content(query, session_id)
    
    # For now, return a simple response structure
    # In a full implementation, this would return the saved query object
    return {
        "query": query,
        "session_id": session_id,
        "response": response,
        "timestamp": "2022-01-01T00:00:00Z"  # Placeholder timestamp
    }


@router.get("/chat/history", tags=["Chat"])
async def get_chat_history(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve chat history for a specific session
    """
    service = ChatService(db)
    history = await service.get_session_history(session_id)
    
    # Convert to response format
    queries = []
    for h in history:
        queries.append({
            "id": h.id,
            "query": h.query,
            "response": h.response,
            "timestamp": h.timestamp.isoformat() if h.timestamp else None
        })
    
    return {
        "sessionId": session_id,
        "queries": queries
    }


@router.post("/session", tags=["User"])
async def create_or_get_session(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Creates a new user session or retrieves an existing one
    """
    service = ChatService(db)
    session = await service.create_or_get_session(session_id, user_id)
    
    return {
        "sessionId": session.session_id,
        "userId": session.user_id,
        "createdAt": session.created_at.isoformat() if session.created_at else None,
        "lastAccessedAt": session.last_accessed_at.isoformat() if session.last_accessed_at else None,
        "currentModule": session.current_module,
        "currentChapterId": session.current_chapter_id
    }


@router.put("/session/{sessionId}/progress", tags=["User"])
async def update_session_progress(
    session_id: str,
    current_module: Optional[int] = None,
    current_chapter_id: Optional[str] = None,
    progress_data: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Updates the user's learning progress
    """
    service = ChatService(db)
    success = await service.update_session_progress(
        session_id, current_module, current_chapter_id, progress_data
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with id {session_id} not found"
        )
    
    return {
        "sessionId": session_id,
        "currentModule": current_module,
        "currentChapterId": current_chapter_id,
        "progressData": progress_data,
        "updatedAt": "2022-01-01T00:00:00Z"  # Placeholder
    }