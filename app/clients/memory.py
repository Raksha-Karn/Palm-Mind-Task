from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from uuid import UUID

from redis.asyncio import Redis
from redis.exceptions import LockError

from app.config import Settings
from app.errors import AppError
from app.schemas import ChatResponse, Message, Session

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class ChatMemory:
    def __init__(self, redis: Redis, settings: Settings) -> None:
        self.redis = redis
        self.settings = settings

    @asynccontextmanager
    async def lock(self, key: str, lease_seconds: int = 120) -> "AsyncIterator[None]":
        lock = self.redis.lock(f"raksha:lock:{key}", timeout=lease_seconds, blocking=False)
        if not await lock.acquire():
            raise AppError(409, "request_in_progress", "Another request is processing. Try again.")
        try:
            yield
        finally:
            try:
                await lock.release()
            except LockError:
                pass

    async def load(self, session_id: UUID) -> Session:
        raw = await self.redis.get(f"raksha:session:{session_id}")
        return Session.model_validate_json(raw) if raw else Session()

    async def cached(self, session_id: UUID, request_id: UUID) -> ChatResponse | None:
        raw = await self.redis.get(f"raksha:response:{session_id}:{request_id}")
        return ChatResponse.model_validate_json(raw) if raw else None

    async def save(
        self, session_id: UUID, session: Session, message: str, response: ChatResponse
    ) -> None:
        session.history.extend(
            [
                Message(role="user", content=message),
                Message(role="assistant", content=response.answer),
            ]
        )
        session.history = session.history[-self.settings.history_turns * 2 :]
        async with self.redis.pipeline(transaction=True) as pipeline:
            pipeline.set(
                f"raksha:session:{session_id}",
                session.model_dump_json(),
                ex=self.settings.session_ttl_seconds,
            )
            pipeline.set(
                f"raksha:response:{session_id}:{response.request_id}",
                response.model_dump_json(),
                ex=self.settings.session_ttl_seconds,
            )
            await pipeline.execute()
