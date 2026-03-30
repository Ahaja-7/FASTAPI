from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

import crud
from database import get_db
from schemas import PostCreate, PostUpdate, PostResponse, PostListResponse

router = APIRouter(prefix="/posts", tags=["게시판"])


@router.get("", response_model=PostListResponse, summary="게시글 목록 조회")
def list_posts(
    page: int = Query(1 , ge=1, description="페이지 번호"),
    size: int = Query(20, ge=1, le=100, description="페이지당 항목 수"),
    db: Session = Depends(get_db),
):
    total, posts = crud.get_posts(db, page=page, size=size)
    return PostListResponse(total=total, page=page, size=size, posts=posts)


@router.get("/{post_id}", response_model=PostResponse, summary="게시글 상세 조회")
def get_post(post_id: int, db: Session = Depends(get_db)):
    post = crud.get_post(db, post_id)
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="게시글을 찾을 수 없습니다.")
    crud.increment_views(db, post)
    return post


@router.post("", response_model=PostResponse, status_code=status.HTTP_201_CREATED, summary="게시글 작성")
def create_post(data: PostCreate, db: Session = Depends(get_db)):
    return crud.create_post(db, data)


@router.patch("/{post_id}", response_model=PostResponse, summary="게시글 수정")
def update_post(post_id: int, data: PostUpdate, db: Session = Depends(get_db)):
    post = crud.get_post(db, post_id)
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="게시글을 찾을 수 없습니다.")
    return crud.update_post(db, post, data)


@router.delete("/{post_id}", status_code=status.HTTP_204_NO_CONTENT, summary="게시글 삭제")
def delete_post(post_id: int, db: Session = Depends(get_db)):
    post = crud.get_post(db, post_id)
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="게시글을 찾을 수 없습니다.")
    crud.delete_post(db, post)
