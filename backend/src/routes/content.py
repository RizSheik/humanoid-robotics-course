from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from backend.src.database.connection import get_db
from backend.src.services.content_service import ContentService
from backend.src.models.content import Module, Chapter

router = APIRouter()

@router.get("/modules", response_model=List[Module], tags=["Content"])
async def get_all_modules(db: AsyncSession = Depends(get_db)):
    """
    Retrieve a list of all educational modules in the book
    """
    service = ContentService(db)
    modules = await service.get_all_modules()

    # Convert SQLAlchemy objects to Pydantic models
    module_list = []
    for module in modules:
        # Get chapters for this module
        chapters = await service.get_chapters_by_module(module.id)
        chapter_list = [
            {
                "id": chapter.id,
                "title": chapter.title,
                "order": chapter.order
            }
            for chapter in chapters
        ]

        module_dict = {
            "id": module.id,
            "title": module.title,
            "description": module.description,
            "order": module.order,
            "created_at": module.created_at,
            "updated_at": module.updated_at
        }

        # Add chapters to the module
        module_dict["chapters"] = chapter_list
        module_list.append(Module(**module_dict))

    return module_list


@router.get("/modules/{module_id}", response_model=Module, tags=["Content"])
async def get_module_by_id(module_id: str, db: AsyncSession = Depends(get_db)):
    """
    Retrieve details of a specific module and its chapters
    """
    service = ContentService(db)
    module = await service.get_module_by_id(module_id)

    if not module:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Module with id {module_id} not found"
        )

    # Get chapters for this module
    chapters = await service.get_chapters_by_module(module_id)
    chapter_list = [
        {
            "id": chapter.id,
            "title": chapter.title,
            "order": chapter.order
        }
        for chapter in chapters
    ]

    module_dict = {
        "id": module.id,
        "title": module.title,
        "description": module.description,
        "order": module.order,
        "created_at": module.created_at,
        "updated_at": module.updated_at
    }

    # Add chapters to the module
    module_dict["chapters"] = chapter_list

    return Module(**module_dict)


@router.get("/chapters/{chapter_id}", response_model=Chapter, tags=["Content"])
async def get_chapter_by_id(chapter_id: str, db: AsyncSession = Depends(get_db)):
    """
    Retrieve the content of a specific chapter
    """
    service = ContentService(db)
    chapter = await service.get_chapter_by_id(chapter_id)

    if not chapter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chapter with id {chapter_id} not found"
        )

    # Create the response model with chapter details
    chapter_response = {
        "id": chapter.id,
        "module_id": chapter.module_id,
        "title": chapter.title,
        "content": chapter.content,
        "order": chapter.order,
        "learning_objectives": chapter.learning_objectives,
        "prerequisites": chapter.prerequisites,
        "created_at": chapter.created_at,
        "updated_at": chapter.updated_at
    }

    return Chapter(**chapter_response)