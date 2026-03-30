-- 데이터베이스 생성
CREATE DATABASE IF NOT EXISTS board_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE board_db;

-- 게시글 테이블 (SQLAlchemy가 자동 생성하지만 참고용으로 포함)
CREATE TABLE IF NOT EXISTS posts (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    title       VARCHAR(200)  NOT NULL COMMENT '제목',
    content     TEXT          NOT NULL COMMENT '본문',
    author      VARCHAR(50)   NOT NULL COMMENT '작성자',
    views       INT           NOT NULL DEFAULT 0 COMMENT '조회수',
    created_at  DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '작성일',
    updated_at  DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP
                              ON UPDATE CURRENT_TIMESTAMP COMMENT '수정일',
    INDEX idx_created_at (created_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
