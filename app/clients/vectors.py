from uuid import UUID, uuid5

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import Settings
from app.errors import AppError
from app.schemas import Source
from app.services.chunking import Chunk


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            timeout=15,
        )

    async def prepare(self) -> None:
        collection = self.settings.qdrant_collection
        if not await self.client.collection_exists(collection):
            try:
                await self.client.create_collection(
                    collection,
                    vectors_config=models.VectorParams(
                        size=self.settings.embedding_dimensions, distance=models.Distance.COSINE
                    ),
                )
            except UnexpectedResponse as exc:
                if exc.status_code != 409:
                    raise
            await self.client.create_payload_index(
                collection, "document_id", field_schema=models.PayloadSchemaType.KEYWORD
            )
            await self.client.create_payload_index(
                collection, "embedding_model", field_schema=models.PayloadSchemaType.KEYWORD
            )
        info = await self.client.get_collection(collection)
        params = info.config.params.vectors
        if (
            not isinstance(params, models.VectorParams)
            or params.size != self.settings.embedding_dimensions
            or params.distance != models.Distance.COSINE
        ):
            raise AppError(
                503, "vector_config_mismatch", "Use a new Qdrant collection for these dimensions."
            )

    async def upsert(
        self, document_id: UUID, filename: str, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        for start in range(0, len(chunks), 64):
            points = [
                models.PointStruct(
                    id=str(uuid5(document_id, str(chunk.index))),
                    vector=embedding,
                    payload={
                        "document_id": str(document_id),
                        "filename": filename,
                        "page": chunk.page,
                        "chunk_index": chunk.index,
                        "text": chunk.text,
                        "embedding_model": self.settings.gemini_embedding_model,
                    },
                )
                for chunk, embedding in zip(
                    chunks[start : start + 64], embeddings[start : start + 64], strict=True
                )
            ]
            await self.client.upsert(self.settings.qdrant_collection, points=points, wait=True)

    async def delete_document(self, document_id: UUID) -> None:
        await self.client.delete(
            self.settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=str(document_id))
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def search(self, embedding: list[float], document_ids: list[UUID]) -> list[Source]:
        must = [
            models.FieldCondition(
                key="embedding_model",
                match=models.MatchValue(value=self.settings.gemini_embedding_model),
            )
        ]
        if document_ids:
            must.append(
                models.FieldCondition(
                    key="document_id", match=models.MatchAny(any=[str(id_) for id_ in document_ids])
                )
            )
        result = await self.client.query_points(
            self.settings.qdrant_collection,
            query=embedding,
            query_filter=models.Filter(must=must),
            limit=self.settings.retrieval_limit * 4,
            score_threshold=self.settings.minimum_score,
            with_payload=True,
        )
        sources: list[Source] = []
        for point in result.points:
            payload = point.payload or {}
            sources.append(
                Source(
                    citation=len(sources) + 1,
                    document_id=UUID(str(payload["document_id"])),
                    filename=str(payload["filename"]),
                    page=payload.get("page"),
                    chunk_index=int(payload["chunk_index"]),
                    text=str(payload["text"]),
                    score=point.score,
                )
            )
        return sources
