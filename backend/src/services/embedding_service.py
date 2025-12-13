import logging
from typing import List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.src.database.models import ContentEmbedding as ContentEmbeddingDB, Chapter as ChapterDB
from backend.src.services.openai_service import openai_service

logger = logging.getLogger(__name__)

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
                type=content_type,
                created_at=datetime.utcnow()
            )
            self.db.add(db_embedding)
            await self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Error creating embeddings for content {content_id}: {str(e)}")
            return False

    async def batch_create_embeddings(self, content_items: List[Tuple[str, str, str]]) -> int:
        """Create embeddings for multiple content items."""
        success_count = 0
        for content_id, text, content_type in content_items:
            if await self.create_embeddings_for_content(content_id, text, content_type):
                success_count += 1
        return success_count

    async def search_similar_content(self, query: str, limit: int = 5) -> List[dict]:
        """Search for content similar to the query using vector embeddings."""
        try:
            # Generate embedding for the query
            query_embeddings = await openai_service.get_embeddings([query])
            if not query_embeddings:
                return []
            
            query_embedding = query_embeddings[0]
            
            # Retrieve all embeddings from the database
            query = select(ContentEmbeddingDB)
            result = await self.db.execute(query)
            all_embeddings = result.scalars().all()
            
            # Calculate similarities
            similarities = []
            for emb in all_embeddings:
                # Convert string embedding back to array
                stored_embedding = eval(emb.embedding)  # Note: In a production system, you'd store embeddings differently
                similarity = await self.calculate_similarity(query_embedding, stored_embedding)
                similarities.append((emb, similarity))
            
            # Sort by similarity score
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top results
            results = []
            for emb, score in similarities[:limit]:
                if score >= 0.5:  # Threshold for relevance
                    results.append({
                        "id": emb.id,
                        "content_id": emb.content_id,
                        "text": emb.text,
                        "type": emb.type,
                        "relevance_score": score,
                        "chapterTitle": await self.get_chapter_title(emb.content_id)  # Try to get chapter info
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []

    async def calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Convert to numpy arrays
            arr1 = np.array(emb1).reshape(1, -1)
            arr2 = np.array(emb2).reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(arr1, arr2)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    async def get_chapter_title(self, content_id: str) -> str:
        """Get chapter title based on content ID."""
        try:
            # This would typically join with the chapters table
            # For now, returning a placeholder
            return "Chapter Title"
        except Exception as e:
            logger.error(f"Error getting chapter title: {str(e)}")
            return "Unknown Chapter"

    async def get_content_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split content into chunks for embedding generation."""
        # Simple chunking by character count
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        return chunks

    async def create_embeddings_from_chapters(self) -> bool:
        """Generate embeddings for all chapters in the database."""
        try:
            # Get all chapters
            query = select(ChapterDB)
            result = await self.db.execute(query)
            chapters = result.scalars().all()
            
            for chapter in chapters:
                # Create chunks from chapter content
                chunks = await self.get_content_chunks(chapter.content)
                for i, chunk in enumerate(chunks):
                    content_id = f"{chapter.id}_chunk_{i}"
                    await self.create_embeddings_for_content(
                        content_id=content_id,
                        text=chunk,
                        content_type="chapter_section"
                    )
            
            return True
        except Exception as e:
            logger.error(f"Error creating embeddings from chapters: {str(e)}")
            return False