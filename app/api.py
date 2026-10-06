from typing import Annotated, cast

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.schemas import ChatRequest, ChatResponse, ChunkStrategy, DocumentResponse
from app.services.chat import ChatService
from app.services.documents import DocumentService

router = APIRouter()


def document_service(request: Request) -> DocumentService:
    return cast(DocumentService, request.app.state.documents)


def chat_service(request: Request) -> ChatService:
    return cast(ChatService, request.app.state.chat)


@router.post("/documents", response_model=DocumentResponse, tags=["Documents"])
async def ingest_document(
    service: Annotated[DocumentService, Depends(document_service)],
    file: Annotated[UploadFile, File()],
    strategy: Annotated[ChunkStrategy, Form()] = ChunkStrategy.RECURSIVE,
    chunk_size: Annotated[int, Form(ge=200, le=2000)] = 1000,
    chunk_overlap: Annotated[int, Form(ge=0, le=500)] = 150,
) -> DocumentResponse:
    try:
        return await service.ingest(file, strategy, chunk_size, chunk_overlap)
    finally:
        await file.close()


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    body: ChatRequest, service: Annotated[ChatService, Depends(chat_service)]
) -> ChatResponse:
    return await service.respond(body)
