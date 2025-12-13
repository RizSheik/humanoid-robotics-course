from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# User models (if authentication is needed in the future)
class UserBase(BaseModel):
    email: str
    is_active: bool = True
    is_superuser: bool = False


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    email: Optional[str] = None
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None


class User(UserBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True