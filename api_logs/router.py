from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api_logs.schemas import (
    ApiLogIndexResponse,
    ApiLogQnaRequest,
    ApiLogQnaResponse,
    ApiResponseLogListResponse,
)
from api_logs.service import ApiLogService, get_api_log_service
from database import get_db

router = APIRouter(prefix="/api-response-logs", tags=["api-response-logs"])


@router.get("", response_model=ApiResponseLogListResponse, summary="API 응답 로그 조회")
def list_api_response_logs(
    db: Session = Depends(get_db),
    service: ApiLogService = Depends(get_api_log_service),
):
    return service.get_logs(db)


@router.post("/index", response_model=ApiLogIndexResponse, summary="API 응답 로그 Qdrant 적재")
def index_api_response_logs(
    db: Session = Depends(get_db),
    service: ApiLogService = Depends(get_api_log_service),
):
    return service.index_logs(db)


@router.post("/qna", response_model=ApiLogQnaResponse, summary="API 응답 로그 QnA")
def api_response_log_qna(
    data: ApiLogQnaRequest,
    service: ApiLogService = Depends(get_api_log_service),
):
    return service.answer_question(data)
