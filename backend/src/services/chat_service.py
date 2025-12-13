import asyncio
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from datetime import datetime
import uuid
import logging

from backend.src.database.models import ChatQuery as ChatQueryDB, UserSession as UserSessionDB, ContentEmbedding as ContentEmbeddingDB, Chapter as ChapterDB
from backend.src.models.chat import ChatQuery, UserSession, ContentEmbedding
from backend.src.config import settings
from backend.src.services.openai_service import openai_service

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def create_or_get_session(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> UserSessionDB:
        """Create a new session or retrieve an existing one."""
        # Use provided session_id or generate a new one
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Check if session exists
        query = select(UserSessionDB).where(UserSessionDB.session_id == session_id)
        result = await self.db.execute(query)
        session = result.scalar_one_or_none()
        
        if session:
            # Update last accessed time
            session.last_accessed_at = datetime.utcnow()
            await self.db.commit()
            return session
        
        # Create new session
        db_session = UserSessionDB(
            session_id=session_id,
            user_id=user_id
        )
        self.db.add(db_session)
        await self.db.commit()
        await self.db.refresh(db_session)
        return db_session

    async def get_session_history(self, session_id: str) -> List[ChatQueryDB]:
        """Retrieve chat history for a specific session."""
        query = select(ChatQueryDB).where(
            ChatQueryDB.session_id == session_id
        ).order_by(ChatQueryDB.timestamp.desc())
        result = await self.db.execute(query)
        queries = result.scalars().all()
        return queries

    async def query_content(self, query_text: str, session_id: str) -> str:
        """Process a user query and return an AI-generated response based on educational content."""
        try:
            # Find the session
            session = await self.create_or_get_session(session_id)
            
            # Search for relevant content using embeddings
            relevant_chunks = await self.search_relevant_content(query_text)
            
            if not relevant_chunks:
                # If no relevant content is found, return a default response
                response = "I couldn't find specific information about this topic in the educational content. Please check the relevant chapters."
                
                # Save the query and response
                db_query = ChatQueryDB(
                    session_id=session_id,
                    query=query_text,
                    response=response,
                    sources="[]"
                )
                self.db.add(db_query)
                await self.db.commit()
                
                return response

            # Prepare the context for the LLM by combining relevant chunks
            context = "\n\n".join([chunk.text for chunk in relevant_chunks])
            
            # Prepare messages for the chat model
            messages = [
                {
                    "role": "system",
                    "content": f"You are an educational assistant helping students learn about humanoid robotics. Use the following context from the course material to answer questions: {context}"
                },
                {
                    "role": "user",
                    "content": query_text
                }
            ]
            
            # Generate response using OpenAI
            response = await openai_service.generate_chat_response(messages)
            
            # Save the query and response to the database
            sources = [{"chapterId": chunk.chapter_id, "text": chunk.text[:200]} for chunk in relevant_chunks]
            
            db_query = ChatQueryDB(
                session_id=session_id,
                query=query_text,
                response=response,
                sources=str(sources),  # Store as JSON string for simplicity
                timestamp=datetime.utcnow()
            )
            self.db.add(db_query)
            await self.db.commit()
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing chat query: {str(e)}")
            # In case of error, return a helpful message to the user
            error_response = "Sorry, I encountered an issue processing your request. Please try again."
            
            # Save the query and error response
            db_query = ChatQueryDB(
                session_id=session_id,
                query=query_text,
                response=error_response,
                sources="[]",
                timestamp=datetime.utcnow()
            )
            self.db.add(db_query)
            await self.db.commit()
            
            return error_response

    async def search_relevant_content(self, query_text: str) -> List[ContentEmbeddingDB]:
        """Search for content relevant to the query using vector similarity."""
        try:
            # Get embedding for the query
            query_embeddings = await openai_service.get_embeddings([query_text])
            if not query_embeddings:
                return []
            
            query_embedding = query_embeddings[0]
            
            # This is a simplified approach - in a real implementation, you would:
            # 1. Compute similarity between the query embedding and stored embeddings
            # 2. Return the most similar content chunks
            
            # For now, we'll return all content embeddings as a placeholder
            # In a real implementation, this would involve actual vector similarity computation
            query = select(ContentEmbeddingDB).order_by(ContentEmbeddingDB.created_at.desc()).limit(settings.max_search_results)
            result = await self.db.execute(query)
            all_embeddings = result.scalars().all()
            
            # In a full implementation, we would compute cosine similarity between
            # query_embedding and each stored embedding, then return the most similar ones
            # For now, returning the latest embeddings
            return all_embeddings[:settings.max_search_results]
            
        except Exception as e:
            logger.error(f"Error searching for relevant content: {str(e)}")
            return []

    async def update_session_progress(self, session_id: str, current_module: Optional[int] = None, 
                                    current_chapter_id: Optional[str] = None, progress_data: Optional[str] = None) -> bool:
        """Update user session progress."""
        # Find the session
        query = select(UserSessionDB).where(UserSessionDB.session_id == session_id)
        result = await self.db.execute(query)
        session = result.scalar_one_or_none()
        
        if not session:
            return False
        
        # Update session progress
        if current_module is not None:
            session.current_module = current_module
        if current_chapter_id is not None:
            session.current_chapter_id = current_chapter_id
        if progress_data is not None:
            session.progress = progress_data
        
        session.last_accessed_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(session)
        return True

    async def get_all_sessions(self) -> List[UserSessionDB]:
        """Get all user sessions."""
        query = select(UserSessionDB).order_by(UserSessionDB.last_accessed_at.desc())
        result = await self.db.execute(query)
        sessions = result.scalars().all()
        return sessions

class EmbeddingService:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def create_embeddings_for_content(self, content_id: str, text: str, content_type: str = "chapter") -> bool:
        """Create vector embeddings for educational content."""
        try:
            # Generate embedding for the text
            embeddings = await openai_service.get_embeddings([text])
            if not embeddings:
                logger.warning(f"Failed to generate embeddings for content {content_id}")
                return False
            
            # Create embedding record
            db_embedding = ContentEmbeddingDB(
                content_id=content_id,
                embedding=str(embeddings[0]),  # Store as string representation
                text=text,
                type=content_type
            )
            self.db.add(db_embedding)
            await self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Error creating embeddings for content {content_id}: {str(e)}")
            return False

    async def batch_create_embeddings(self, content_items: List[tuple]) -> int:
        """Create embeddings for multiple content items."""
        success_count = 0
        for content_id, text, content_type in content_items:
            if await self.create_embeddings_for_content(content_id, text, content_type):
                success_count += 1
        return success_count

    async def search_similar_content(self, query: str, threshold: float = 0.7, limit: int = 5) -> List[dict]:
        """Search for content similar to the query using vector embeddings."""
        # Generate embedding for the query
        query_embeddings = await openai_service.get_embeddings([query])
        if not query_embeddings:
            return []
        
        query_embedding = query_embeddings[0]
        
        # In a real implementation, this would compute cosine similarity
        # against all stored embeddings and return the most similar ones.
        # For now, we'll return a placeholder.
        
        # This would require vector similarity search which might need
        # a dedicated vector database or pgvector extension
        query = select(ContentEmbeddingDB).limit(limit)
        result = await self.db.execute(query)
        embeddings = result.scalars().all()
        
        results = []
        for emb in embeddings:
            results.append({
                "id": emb.id,
                "content_id": emb.content_id,
                "text": emb.text,
                "type": emb.type,
                "relevance_score": 0.8  # Placeholder
            })
        
        return results