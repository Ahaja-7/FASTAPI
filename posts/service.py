"""
Service Layer
- 비즈니스 규칙 및 흐름 제어
- 예외 처리 (존재 여부 검증 등)
- Repository를 통해서만 DB 접근
"""
from sqlalchemy.orm import Session

from models import Post
from common.exceptions import PostNotFoundException
from posts.repository import PostRepository, get_post_repository
from posts.schemas import PostCreate, PostUpdate, PostListResponse
from fastapi import Depends


class PostService:

    def __init__(self, repo: PostRepository):
        self.repo = repo

    # ── 목록 조회 ──────────────────────────────────────────────────────────

    def get_posts(self, db: Session, page: int, size: int) -> PostListResponse:
        offset = (page - 1) * size
        total  = self.repo.count(db)
        posts  = self.repo.find_all(db, offset=offset, limit=size)
        return PostListResponse(total=total, page=page, size=size, posts=posts)

    # ── 단건 조회 ──────────────────────────────────────────────────────────

    def get_post(self, db: Session, post_id: int) -> Post:
        post = self.repo.find_by_id(db, post_id)
        if not post:
            raise PostNotFoundException()
        self.repo.increment_views(db, post)
        return post

    # ── 게시글 생성 ────────────────────────────────────────────────────────

    def create_post(self, db: Session, data: PostCreate) -> Post:
        return self.repo.save(db, data)

    # ── 게시글 수정 ────────────────────────────────────────────────────────

    def update_post(self, db: Session, post_id: int, data: PostUpdate) -> Post:
        post = self.repo.find_by_id(db, post_id)
        if not post:
            raise PostNotFoundException()
        return self.repo.update(db, post, data)

    # ── 게시글 삭제 ────────────────────────────────────────────────────────

    def delete_post(self, db: Session, post_id: int) -> None:
        post = self.repo.find_by_id(db, post_id)
        if not post:
            raise PostNotFoundException()
        self.repo.delete(db, post)


# Depends() 주입용 팩토리
def get_post_service(
    repo: PostRepository = Depends(get_post_repository),
) -> PostService:
    return PostService(repo)