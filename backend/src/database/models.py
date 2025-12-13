from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, Float, ForeignKey, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class Module(Base):
    __tablename__ = "modules"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    description = Column(Text)
    order = Column(Integer, nullable=False)  # 1-4
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to chapters
    chapters = relationship("Chapter", back_populates="module", order_by="Chapter.order")


class Chapter(Base):
    __tablename__ = "chapters"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    module_id = Column(String, ForeignKey("modules.id"), nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)  # Markdown format
    order = Column(Integer, nullable=False)  # 1-4 within module
    learning_objectives = Column(ARRAY(String))
    prerequisites = Column(ARRAY(String))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    module = relationship("Module", back_populates="chapters")
    learning_content = relationship("LearningContent", back_populates="chapter")
    content_embeddings = relationship("ContentEmbedding", back_populates="chapter")


class LearningContent(Base):
    __tablename__ = "learning_content"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chapter_id = Column(String, ForeignKey("chapters.id"), nullable=False)
    type = Column(String, nullable=False)  # 'text', 'image', 'diagram', 'video', 'code', 'exercise', 'quiz'
    content = Column(Text)  # For text/code, path for media
    alt_text = Column(String)  # For accessibility
    caption = Column(String)
    order = Column(Integer, nullable=False)  # Order within chapter
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    chapter = relationship("Chapter", back_populates="learning_content")


class ChatQuery(Base):
    __tablename__ = "chat_queries"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String)  # Optional for logged-in users
    session_id = Column(String, nullable=False)  # Required for anonymous users
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    sources = Column(Text)  # JSON string containing references to educational content
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_helpful = Column(Boolean)  # Optional feedback
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user_session = relationship("UserSession", back_populates="queries")


class ContentEmbedding(Base):
    __tablename__ = "content_embeddings"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content_id = Column(String)  # Can reference LearningContent or Chapter
    embedding = Column(String)  # Vector as string representation
    text = Column(Text, nullable=False)  # Original text that was embedded
    type = Column(String, nullable=False)  # 'chapter', 'section', 'paragraph', 'image_caption'
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    chapter = relationship("Chapter", back_populates="content_embeddings")


class UserSession(Base):
    __tablename__ = "user_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False)  # For anonymous users
    user_id = Column(String)  # Optional for logged-in users
    current_module = Column(Integer)  # Module user is currently viewing
    current_chapter_id = Column(String, ForeignKey("chapters.id"))  # Chapter user is currently viewing
    progress = Column(Text)  # JSON string tracking user's progress
    last_accessed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    current_chapter = relationship("Chapter")
    queries = relationship("ChatQuery", back_populates="user_session")


class Image(Base):
    __tablename__ = "images"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    path = Column(String, nullable=False)  # Relative to src/static/img/
    alt_text = Column(String, nullable=False)  # For accessibility
    caption = Column(String)
    associated_chapter_id = Column(String, ForeignKey("chapters.id"))
    upload_date = Column(DateTime, default=datetime.utcnow)

    # Relationship
    chapter = relationship("Chapter")