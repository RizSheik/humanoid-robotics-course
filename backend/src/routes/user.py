from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from backend.src.database.connection import get_db
from backend.src.services.chat_service import ChatService

router = APIRouter()