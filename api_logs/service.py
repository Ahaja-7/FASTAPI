from fastapi import Depends, HTTPException, status
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sqlalchemy.orm import Session

from api_logs.repository import ApiLogRepository, get_api_log_repository
from api_logs.schemas import (
    ApiLogIndexResponse,
    ApiLogQnaRequest,
    ApiLogQnaResponse,
    ApiLogReference,
    ApiResponseLogListResponse,
)
from config import settings
from models import ApiResponseLog


class ApiLogService:
    def __init__(self, repo: ApiLogRepository):
        self.repo = repo

    def get_logs(self, db: Session) -> ApiResponseLogListResponse:
        logs = self.repo.find_all(db)
        return ApiResponseLogListResponse(total=len(logs), items=logs)



    def index_logs(self, db: Session) -> ApiLogIndexResponse:
        logs = self.repo.find_all(db)
        documents = [self._to_document(log) for log in logs]

        embeddings = self._get_embeddings()
        client = self._get_qdrant_client()
        sample_text = documents[0].page_content if documents else "api response log"
        vector_size = len(embeddings.embed_query(sample_text))

        client.recreate_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=qdrant_models.VectorParams(
                size=vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.QDRANT_COLLECTION,
            embedding=embeddings,
        )

        if documents:
            vector_store.add_documents(documents)

        return ApiLogIndexResponse(
            collection_name=settings.QDRANT_COLLECTION,
            indexed_count=len(documents),
        )




    def answer_question(self, request: ApiLogQnaRequest) -> ApiLogQnaResponse:
        client = self._get_qdrant_client()
        if not client.collection_exists(settings.QDRANT_COLLECTION):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Qdrant 컬렉션이 없습니다. 먼저 /api/v1/api-response-logs/index 를 호출하세요.",
            )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.QDRANT_COLLECTION,
            embedding=self._get_embeddings(),
        )

        top_k = request.top_k or settings.QNA_TOP_K
        matched_documents = vector_store.similarity_search_with_score(
            request.question,
            k=top_k,
        )
        if not matched_documents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="질문과 관련된 로그를 찾지 못했습니다.",
            )

        context = "\n\n".join(
            f"[문서 {index}]\n{document.page_content}"
            for index, (document, _) in enumerate(matched_documents, start=1)
        )

        chat_model = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
        )
        response = chat_model.invoke(
            [
                SystemMessage(
                    content=(
                        "너는 API 응답 로그 분석 도우미다. "
                        "반드시 한국어로만 답변해라. "
                        "반드시 제공된 로그 문맥만 근거로 답변하고, "
                        "문맥에 없는 내용은 모른다고 답해라. "
                        "답변은 간결하고 실무적으로 작성해라."
                    )
                ),
                HumanMessage(
                    content=(
                        f"질문:\n{request.question}\n\n"
                        f"로그 문맥:\n{context if context else '검색된 로그가 없습니다.'}"
                    )
                ),
            ]
        )

        references = [
            ApiLogReference(
                id=document.metadata["id"],
                score=float(score),
                category=document.metadata["category"],
                response_time_ms=document.metadata["response_time_ms"],
                timeout_yn=document.metadata["timeout_yn"],
                created_at=document.metadata["created_at"],
                error_message=document.metadata["error_message"],
                content=document.page_content,
            )
            for document, score in matched_documents
        ]

        return ApiLogQnaResponse(answer=str(response.content), references=references)

    @staticmethod
    def _to_document(log: ApiResponseLog) -> Document:
        error_message = log.error_message or "없음"
        return Document(
            page_content=(
                f"로그 ID: {log.id}\n"
                f"응답 시간(ms): {log.response_time_ms}\n"
                f"타임아웃 여부: {'Y' if log.timeout_yn else 'N'}\n"
                f"카테고리: {log.category}\n"
                f"오류 메시지: {error_message}\n"
                f"생성 시각: {log.created_at.isoformat()}"
            ),
            metadata={
                "id": log.id,
                "response_time_ms": log.response_time_ms,
                "timeout_yn": log.timeout_yn,
                "category": log.category,
                "error_message": log.error_message,
                "created_at": log.created_at,
            },
        )

    @staticmethod
    def _get_embeddings() -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model=settings.OLLAMA_EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )

    @staticmethod
    def _get_qdrant_client() -> QdrantClient:
        if settings.QDRANT_URL:
            return QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )

        return QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
            https=False,
        )


def get_api_log_service(
    repo: ApiLogRepository = Depends(get_api_log_repository),
) -> ApiLogService:
    return ApiLogService(repo)
