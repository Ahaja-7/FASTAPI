from datetime import datetime

from pydantic import BaseModel, Field


class ApiResponseLogItem(BaseModel):
    id: int
    response_time_ms: int
    timeout_yn: bool
    category: str
    error_message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ApiResponseLogListResponse(BaseModel):
    total: int
    items: list[ApiResponseLogItem]


class ApiLogIndexResponse(BaseModel):
    collection_name: str
    indexed_count: int


class ApiLogQnaRequest(BaseModel):
    question: str = Field(..., min_length=1, description="질문")
    top_k: int | None = Field(None, ge=1, le=10, description="검색할 문서 수")


class ApiLogReference(BaseModel):
    id: int
    score: float
    category: str
    response_time_ms: int
    timeout_yn: bool
    created_at: datetime
    error_message: str | None
    content: str


class ApiLogQnaResponse(BaseModel):
    answer: str
    references: list[ApiLogReference]
