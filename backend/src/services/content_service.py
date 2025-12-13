from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from backend.src.database.models import Module as ModuleDB, Chapter as ChapterDB, LearningContent as LearningContentDB
from backend.src.models.content import ModuleCreate, ModuleUpdate, ChapterCreate, ChapterUpdate
from backend.src.database.connection import get_db
from backend.src.config import settings
import logging

logger = logging.getLogger(__name__)

class ContentService:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def get_all_modules(self) -> List[ModuleDB]:
        """Get all modules with their chapters."""
        query = select(ModuleDB).order_by(ModuleDB.order)
        result = await self.db.execute(query)
        modules = result.scalars().all()
        return modules

    async def get_module_by_id(self, module_id: str) -> Optional[ModuleDB]:
        """Get a specific module by its ID."""
        query = select(ModuleDB).where(ModuleDB.id == module_id)
        result = await self.db.execute(query)
        module = result.scalar_one_or_none()
        return module

    async def create_module(self, module_data: ModuleCreate) -> ModuleDB:
        """Create a new module."""
        db_module = ModuleDB(
            title=module_data.title,
            description=module_data.description,
            order=module_data.order
        )
        self.db.add(db_module)
        await self.db.commit()
        await self.db.refresh(db_module)
        return db_module

    async def update_module(self, module_id: str, module_data: ModuleUpdate) -> Optional[ModuleDB]:
        """Update an existing module."""
        query = select(ModuleDB).where(ModuleDB.id == module_id)
        result = await self.db.execute(query)
        module = result.scalar_one_or_none()
        
        if not module:
            return None
        
        update_data = module_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(module, field, value)
        
        await self.db.commit()
        await self.db.refresh(module)
        return module

    async def delete_module(self, module_id: str) -> bool:
        """Delete a module."""
        query = select(ModuleDB).where(ModuleDB.id == module_id)
        result = await self.db.execute(query)
        module = result.scalar_one_or_none()
        
        if not module:
            return False
        
        await self.db.delete(module)
        await self.db.commit()
        return True

    async def get_chapter_by_id(self, chapter_id: str) -> Optional[ChapterDB]:
        """Get a specific chapter by its ID."""
        query = select(ChapterDB).where(ChapterDB.id == chapter_id)
        result = await self.db.execute(query)
        chapter = result.scalar_one_or_none()
        return chapter

    async def get_chapters_by_module(self, module_id: str) -> List[ChapterDB]:
        """Get all chapters for a specific module."""
        query = select(ChapterDB).where(ChapterDB.module_id == module_id).order_by(ChapterDB.order)
        result = await self.db.execute(query)
        chapters = result.scalars().all()
        return chapters

    async def create_chapter(self, chapter_data: ChapterCreate) -> ChapterDB:
        """Create a new chapter."""
        # Verify the module exists
        module = await self.get_module_by_id(chapter_data.module_id)
        if not module:
            raise ValueError(f"Module with ID {chapter_data.module_id} does not exist")
        
        db_chapter = ChapterDB(
            module_id=chapter_data.module_id,
            title=chapter_data.title,
            content=chapter_data.content,
            order=chapter_data.order,
            learning_objectives=chapter_data.learning_objectives,
            prerequisites=chapter_data.prerequisites
        )
        self.db.add(db_chapter)
        await self.db.commit()
        await self.db.refresh(db_chapter)
        return db_chapter

    async def update_chapter(self, chapter_id: str, chapter_data: ChapterUpdate) -> Optional[ChapterDB]:
        """Update an existing chapter."""
        query = select(ChapterDB).where(ChapterDB.id == chapter_id)
        result = await self.db.execute(query)
        chapter = result.scalar_one_or_none()
        
        if not chapter:
            return None
        
        update_data = chapter_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(chapter, field, value)
        
        await self.db.commit()
        await self.db.refresh(chapter)
        return chapter

    async def delete_chapter(self, chapter_id: str) -> bool:
        """Delete a chapter."""
        query = select(ChapterDB).where(ChapterDB.id == chapter_id)
        result = await self.db.execute(query)
        chapter = result.scalar_one_or_none()
        
        if not chapter:
            return False
        
        await self.db.delete(chapter)
        await self.db.commit()
        return True

    async def search_content(self, query_text: str, limit: Optional[int] = 10) -> List:
        """
        Search content across all modules and chapters.
        This is a basic text search implementation.
        For full RAG functionality, see the embedding service.
        """
        # This would normally use vector similarity search against embeddings
        # For now, we'll do a simple text search in the database
        # In a production implementation, this would search against pre-generated embeddings
        
        # Search in modules
        module_search = select(ModuleDB).where(
            ModuleDB.title.contains(query_text) | 
            ModuleDB.description.contains(query_text)
        )
        module_result = await self.db.execute(module_search)
        modules = module_result.scalars().all()
        
        # Search in chapters
        chapter_search = select(ChapterDB).where(
            ChapterDB.title.contains(query_text) | 
            ChapterDB.content.contains(query_text)
        )
        chapter_result = await self.db.execute(chapter_search)
        chapters = chapter_result.scalars().all()
        
        # Combine and limit results
        results = []
        for module in modules[:limit//2]:  # Limit modules to half
            results.append({
                "type": "module",
                "id": module.id,
                "title": module.title,
                "contentPreview": module.description or "",
                "moduleId": module.id,
                "moduleName": module.title,
                "relevanceScore": 0.8  # Placeholder score
            })
        
        remaining_slots = limit - len(results)
        for chapter in chapters[:remaining_slots]:  # Fill remaining slots with chapters
            preview = chapter.content[:200] if chapter.content else ""
            results.append({
                "type": "chapter",
                "id": chapter.id,
                "title": chapter.title,
                "contentPreview": preview,
                "moduleId": chapter.module_id,
                "moduleName": chapter.module.title if chapter.module else "Unknown Module",
                "relevanceScore": 0.7  # Placeholder score
            })
        
        return results