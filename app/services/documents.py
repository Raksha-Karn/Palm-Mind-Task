import asyncio
import hashlib
import logging
from pathlib import PurePath
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.concurrency import run_in_threadpool

from app.clients.gemini import GeminiClient
from app.clients.memory import ChatMemory
from app.clients.vectors import VectorStore
from app.config import Settings
from app.db import Document
from app.errors import AppError
from app.schemas import ChunkStrategy, DocumentResponse
from app.services.chunking import chunk_text, extract_text

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        settings: Settings,
        database: async_sessionmaker[AsyncSession],
        gemini: GeminiClient,
        vectors: VectorStore,
        memory: ChatMemory,
    ) -> None:
        self.settings = settings
        self.database = database
        self.gemini = gemini
        self.vectors = vectors
        self.memory = memory

    async def ingest(
        self, file: UploadFile, strategy: ChunkStrategy, chunk_size: int, overlap: int
    ) -> DocumentResponse:
        self.gemini.require_key()
        filename = PurePath((file.filename or "document").replace("\\", "/")).name[:255]
        extension = PurePath(filename).suffix.lower()
        if extension not in {".pdf", ".txt"}:
            raise AppError(415, "unsupported_file", "Upload a .pdf or .txt file.")
        if overlap >= chunk_size:
            raise AppError(422, "invalid_chunk_options", "Overlap must be smaller than chunk size.")
        data = await file.read(self.settings.max_upload_bytes + 1)
        if len(data) > self.settings.max_upload_bytes:
            raise AppError(413, "file_too_large", "File exceeds the upload limit.")
        pages = await run_in_threadpool(
            extract_text, data, extension, self.settings.max_document_characters
        )
        chunks = chunk_text(pages, strategy, chunk_size, overlap, self.settings.max_chunks)
        checksum = hashlib.sha256(data).hexdigest()
        key = hashlib.sha256(
            f"{checksum}:{extension}:{strategy}:{chunk_size}:{overlap}:"
            f"{self.settings.gemini_embedding_model}:{self.settings.embedding_dimensions}".encode()
        ).hexdigest()
        async with self.memory.lock(f"document:{key}", lease_seconds=330), self.database() as db:
            document = await db.scalar(select(Document).where(Document.ingestion_key == key))
            if document is not None and document.status == "ready":
                return self.response(document, duplicate=True)
            if document is None:
                document = Document(
                    id=uuid4(),
                    ingestion_key=key,
                    filename=filename,
                    checksum=checksum,
                    strategy=strategy.value,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    embedding_model=self.settings.gemini_embedding_model,
                    embedding_dimensions=self.settings.embedding_dimensions,
                    status="processing",
                )
                db.add(document)
            else:
                document.status = "processing"
            await db.commit()
            document_id = document.id
            try:
                async with asyncio.timeout(300):
                    await self.vectors.prepare()
                    embeddings = await self.gemini.embed(
                        [chunk.text for chunk in chunks], "RETRIEVAL_DOCUMENT"
                    )
                    await self.vectors.upsert(document.id, document.filename, chunks, embeddings)
                    document.chunk_count = len(chunks)
                    document.status = "ready"
                    await db.commit()
            except Exception:
                await db.rollback()
                document = await db.get(Document, document_id)
                if document is not None:
                    document.status = "failed"
                    await db.commit()
                    try:
                        await self.vectors.delete_document(document.id)
                    except Exception:
                        logger.warning("Vector cleanup failed for document %s", document.id)
                raise
            return self.response(document)

    @staticmethod
    def response(document: Document, duplicate: bool = False) -> DocumentResponse:
        return DocumentResponse(
            document_id=document.id,
            filename=document.filename,
            strategy=ChunkStrategy(document.strategy),
            chunk_count=document.chunk_count,
            duplicate=duplicate,
        )
