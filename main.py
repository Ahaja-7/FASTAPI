from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import models  # noqa: F401
from api_logs.router import router as api_log_router
from database import Base, engine
from posts.router import router as post_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FastAPI Board and Log QnA API",
    description="FastAPI + MySQL + Qdrant + Ollama example",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(post_router, prefix="/api/v1")
app.include_router(api_log_router, prefix="/api/v1")


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "message": "API server is running."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
