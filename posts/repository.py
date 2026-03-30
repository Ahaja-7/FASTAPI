"""
Repository Layer
- DB 접근(ORM 쿼리)만 담당
- 비즈니스 로직 없음
- 예외 처리 없음
"""
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from models import Post
from posts.schemas import PostCreate, PostUpdate


class PostRepository:

    def find_all(self, db: Session, offset: int, limit: int) -> list[Post]:
        return db.scalars(
            select(Post).order_by(Post.created_at.desc()).offset(offset).limit(limit)
        ).all()

    def count(self, db: Session) -> int:
        return db.scalar(select(func.count()).select_from(Post))

    def find_by_id(self, db: Session, post_id: int) -> Post | None:
        return db.get(Post, post_id)

    def save(self, db: Session, data: PostCreate) -> Post:
        post = Post(**data.model_dump())
        db.add(post)
        db.commit()
        db.refresh(post)
        return post

    def update(self, db: Session, post: Post, data: PostUpdate) -> Post:
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(post, field, value)
        db.commit()
        db.refresh(post)
        return post

    def delete(self, db: Session, post: Post) -> None:
        db.delete(post)
        db.commit()

    def increment_views(self, db: Session, post: Post) -> Post:
        post.views += 1
        db.commit()
        db.refresh(post)
        return post


# Depends() 주입용 팩토리
def get_post_repository() -> PostRepository:
    return PostRepository()