from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


# ChatQuery models
class ChatQueryBase(BaseModel):
    session_id: str
    query: str
    response: str
    sources: Optional[str] = None  # JSON string containing references to educational content
    is_helpful: Optional[bool] = None  # Optional feedback


class ChatQueryCreate(ChatQueryBase):
    pass


class ChatQueryUpdate(BaseModel):
    response: Optional[str] = None
    sources: Optional[str] = None
    is_helpful: Optional[bool] = None


class ChatQuery(ChatQueryBase):
    id: str
    timestamp: datetime
    created_at: datetime

    class Config:
        from_attributes = True


# UserSession models
class UserSessionBase(BaseModel):
    session_id: str
    current_module: Optional[int] = None
    current_chapter_id: Optional[str] = None
    progress: Optional[str] = None  # JSON string tracking user's progress


class UserSessionCreate(UserSessionBase):
    pass


class UserSessionUpdate(BaseModel):
    session_id: Optional[str] = None
    current_module: Optional[int] = None
    current_chapter_id: Optional[str] = None
    progress: Optional[str] = None


class UserSession(UserSessionBase):
    id: str
    last_accessed_at: datetime
    created_at: datetime

    class Config:
        from_attributes = True


# ContentEmbedding models
class ContentEmbeddingBase(BaseModel):
    content_id: str  # Can reference LearningContent or Chapter
    embedding: str  # Vector as string representation
    text: str  # Original text that was embedded
    type: str  # 'chapter', 'section', 'paragraph', 'image_caption'


class ContentEmbeddingCreate(ContentEmbeddingBase):
    pass


class ContentEmbeddingUpdate(BaseModel):
    content_id: Optional[str] = None
    embedding: Optional[str] = None
    text: Optional[str] = None
    type: Optional[str] = None


class ContentEmbedding(ContentEmbeddingBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True