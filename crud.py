from sqlalchemy.orm import Session
from sqlalchemy import select, func

from models import Post
from schemas import PostCreate, PostUpdate


def get_posts(db: Session, page: int = 1, size: int = 20) -> tuple[int, list[Post]]:
    """게시글 목록 조회 (페이지네이션)"""
    offset = (page - 1) * size
    total = db.scalar(select(func.count()).select_from(Post))
    posts = db.scalars(
        select(Post).order_by(Post.created_at.desc()).offset(offset).limit(size)
    ).all()
    return total, posts


def get_post(db: Session, post_id: int) -> Post | None:
    """게시글 단건 조회"""
    return db.get(Post, post_id)


def create_post(db: Session, data: PostCreate) -> Post:
    """게시글 생성"""
    post = Post(**data.model_dump())
    db.add(post)
    db.commit()
    db.refresh(post)
    return post


def update_post(db: Session, post: Post, data: PostUpdate) -> Post:
    """게시글 수정 (변경된 필드만 업데이트)"""
    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(post, field, value)
    db.commit()
    db.refresh(post)
    return post


def delete_post(db: Session, post: Post) -> None:
    """게시글 삭제"""
    db.delete(post)
    db.commit()


def increment_views(db: Session, post: Post) -> Post:
    """조회수 증가"""
    post.views += 1
    db.commit()
    db.refresh(post)
    return post
