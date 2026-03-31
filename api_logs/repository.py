from sqlalchemy import select
from sqlalchemy.orm import Session

from models import ApiResponseLog


class ApiLogRepository:
    def find_all(self, db: Session) -> list[ApiResponseLog]:
        return db.scalars(
            select(ApiResponseLog).order_by(ApiResponseLog.created_at.desc())
        ).all()


def get_api_log_repository() -> ApiLogRepository:
    return ApiLogRepository()
