import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from app.api import router
from app.clients.gemini import GeminiClient
from app.clients.memory import ChatMemory
from app.clients.vectors import VectorStore
from app.config import Settings
from app.db import create_database
from app.errors import AppError
from app.middleware import RequestLimits
from app.services.chat import ChatService
from app.services.documents import DocumentService

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        engine, database = create_database(config.database_url)
        redis = Redis.from_url(config.redis_url, decode_responses=True, socket_timeout=5)
        gemini = GeminiClient(config)
        vectors = VectorStore(config)
        memory = ChatMemory(redis, config)
        application.state.documents = DocumentService(config, database, gemini, vectors, memory)
        application.state.chat = ChatService(config, database, gemini, vectors, memory)
        try:
            yield
        finally:
            await gemini.close()
            await vectors.client.close()
            await redis.aclose()
            await engine.dispose()

    application = FastAPI(
        title="Palm Mind",
        version="0.1.0",
        description="Document RAG and interview booking",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    application.add_middleware(RequestLimits, max_upload_bytes=config.max_upload_bytes)
    application.include_router(router)

    @application.exception_handler(AppError)
    async def application_error(request: Request, exc: AppError) -> JSONResponse:
        headers = {"Retry-After": "30"} if exc.status_code == 429 else None
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
            headers=headers,
        )

    async def dependency_error(request: Request, exc: Exception) -> JSONResponse:
        logger.warning("Dependency failure: %s", type(exc).__name__)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "dependency_unavailable",
                    "message": "A backend service is unavailable. Try again.",
                }
            },
        )

    for exception in (
        RedisError,
        SQLAlchemyError,
        httpx.HTTPError,
        UnexpectedResponse,
        ResponseHandlingException,
    ):
        application.add_exception_handler(exception, dependency_error)

    @application.exception_handler(TimeoutError)
    async def timeout_error(request: Request, exc: TimeoutError) -> JSONResponse:
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "code": "request_timeout",
                    "message": "Processing timed out. Retry the request.",
                }
            },
        )

    return application


app = create_app()
