from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── 요청 스키마 ─────────────────────────────────────────

class PostCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="제목")
    content: str = Field(..., min_length=1, description="본문")
    author: str = Field(..., min_length=1, max_length=50, description="작성자")


class PostUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1)


# ── 응답 스키마 ─────────────────────────────────────────

class PostResponse(BaseModel):
    id: int
    title: str
    content: str
    author: str
    views: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PostSummary(BaseModel):
    """목록 조회용 (본문 제외)"""
    id: int
    title: str
    author: str
    views: int
    created_at: datetime

    model_config = {"from_attributes": True}


class PostListResponse(BaseModel):
    total: int
    page: int
    size: int
    posts: list[PostSummary]
