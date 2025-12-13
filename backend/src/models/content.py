from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid


# Module models
class ModuleBase(BaseModel):
    title: str
    description: Optional[str] = None
    order: int  # 1-4


class ModuleCreate(ModuleBase):
    pass


class ModuleUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    order: Optional[int] = None


class Module(ModuleBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Chapter models
class ChapterBase(BaseModel):
    module_id: str
    title: str
    content: str  # Markdown format
    order: int  # 1-4 within module
    learning_objectives: Optional[List[str]] = []
    prerequisites: Optional[List[str]] = []


class ChapterCreate(ChapterBase):
    pass


class ChapterUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    order: Optional[int] = None
    learning_objectives: Optional[List[str]] = None
    prerequisites: Optional[List[str]] = None


class Chapter(ChapterBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# LearningContent models
class LearningContentBase(BaseModel):
    chapter_id: str
    type: str  # 'text', 'image', 'diagram', 'video', 'code', 'exercise', 'quiz'
    content: Optional[str] = None  # For text/code, path for media
    alt_text: Optional[str] = None  # For accessibility
    caption: Optional[str] = None
    order: int  # Order within chapter


class LearningContentCreate(LearningContentBase):
    pass


class LearningContentUpdate(BaseModel):
    chapter_id: Optional[str] = None
    type: Optional[str] = None
    content: Optional[str] = None
    alt_text: Optional[str] = None
    caption: Optional[str] = None
    order: Optional[int] = None


class LearningContent(LearningContentBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Image models
class ImageBase(BaseModel):
    filename: str
    path: str  # Relative to src/static/img/
    alt_text: str  # For accessibility
    caption: Optional[str] = None
    associated_chapter_id: Optional[str] = None


class ImageCreate(ImageBase):
    pass


class ImageUpdate(BaseModel):
    filename: Optional[str] = None
    path: Optional[str] = None
    alt_text: Optional[str] = None
    caption: Optional[str] = None
    associated_chapter_id: Optional[str] = None


class Image(ImageBase):
    id: str
    upload_date: datetime

    class Config:
        from_attributes = True