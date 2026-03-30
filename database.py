from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from config import settings


engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,      # 연결 끊김 자동 감지
    pool_recycle=3600,       # 1시간마다 커넥션 재사용
    echo=False,              # SQL 로그 출력 (개발 시 True)
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
