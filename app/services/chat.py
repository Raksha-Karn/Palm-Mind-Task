import asyncio
from datetime import UTC, datetime
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.clients.gemini import GeminiClient
from app.clients.memory import ChatMemory
from app.clients.vectors import VectorStore
from app.config import Settings
from app.db import Booking, Document
from app.errors import AppError
from app.schemas import (
    BookingDraft,
    BookingResponse,
    ChatRequest,
    ChatResponse,
    GroundedAnswer,
    QueryPlan,
    Session,
    Source,
)


class ChatService:
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

    async def respond(self, request: ChatRequest) -> ChatResponse:
        session_id = request.session_id or uuid4()
        async with self.memory.lock(f"chat:{session_id}"), asyncio.timeout(90):
            cached = await self.memory.cached(session_id, request.request_id)
            if cached is not None:
                return cached
            session = await self.memory.load(session_id)
            response = ChatResponse(session_id=session_id, request_id=request.request_id, answer="")
            if request.confirm_booking:
                response.booking = await self.confirm(session_id, session, request.draft_id)
                response.booking_status = "booked"
                local = response.booking.starts_at.astimezone(ZoneInfo(response.booking.timezone))
                response.answer = (
                    f"Interview booking saved for {response.booking.name} on "
                    f"{local:%Y-%m-%d at %H:%M} ({response.booking.timezone})."
                )
                if session.draft is not None and session.draft.draft_id == request.draft_id:
                    session.draft = None
                response.draft = session.draft
            else:
                plan = await self.gemini.plan(request.message, session)
                match plan.intent:
                    case "booking":
                        self.collect(session, plan, response)
                    case "cancel_booking":
                        had_draft = session.draft is not None
                        session.draft = None
                        response.booking_status = "cancelled" if had_draft else "none"
                        response.answer = (
                            "Pending booking cancelled."
                            if had_draft
                            else "No pending booking. Saved bookings cannot be cancelled here."
                        )
                    case "question":
                        response.answer, response.sources = await self.retrieve(
                            plan.standalone_query, request.document_ids
                        )
                        response.draft = session.draft
                        if session.draft:
                            response.booking_status = (
                                "collecting"
                                if session.draft.missing_fields()
                                else "awaiting_confirmation"
                            )
            await self.memory.save(session_id, session, request.message, response)
            return response

    def collect(self, session: Session, plan: QueryPlan, response: ChatResponse) -> None:
        fields = plan.booking_fields.model_dump(exclude_none=True)
        if session.draft is None:
            session.draft = BookingDraft(
                draft_id=uuid4(), timezone=self.settings.booking_timezone, **fields
            )
        elif fields:
            updated = session.draft.model_dump() | fields
            candidate = BookingDraft.model_validate(updated)
            if candidate != session.draft:
                candidate.draft_id = uuid4()
                session.draft = candidate
        draft = session.draft
        if plan.clarification:
            draft.date = None
            draft.time = None
            draft.draft_id = uuid4()
            response.answer = plan.clarification
            response.booking_status = "collecting"
        elif draft.missing_fields():
            response.answer = "Please provide your " + ", ".join(draft.missing_fields()) + "."
            response.booking_status = "collecting"
        elif self.starts_at(draft) <= datetime.now(UTC):
            draft.date = None
            draft.time = None
            draft.draft_id = uuid4()
            response.answer = "Please choose a future date and time for the interview."
            response.booking_status = "collecting"
        else:
            response.answer = (
                f"Please confirm: {draft.name}, {draft.email}, "
                f"{draft.date} at {draft.time:%H:%M} ({draft.timezone}). "
                "Use Confirm booking to save these details."
            )
            response.booking_status = "awaiting_confirmation"
        response.draft = draft

    @staticmethod
    def starts_at(draft: BookingDraft) -> datetime:
        if draft.date is None or draft.time is None:
            raise AppError(422, "incomplete_booking", "Complete all booking fields first.")
        return datetime.combine(draft.date, draft.time, ZoneInfo(draft.timezone)).astimezone(UTC)

    async def confirm(
        self, session_id: UUID, session: Session, draft_id: UUID | None
    ) -> BookingResponse:
        async with self.database() as db:
            existing = await db.scalar(
                select(Booking).where(
                    Booking.draft_id == draft_id, Booking.session_id == session_id
                )
            )
            if existing:
                return self.booking_response(existing)
            draft = session.draft
            if draft is None or draft.draft_id != draft_id:
                raise AppError(
                    409, "stale_booking", "Booking details changed or expired. Review again."
                )
            if draft.missing_fields():
                raise AppError(422, "incomplete_booking", "Complete all booking fields first.")
            starts_at = self.starts_at(draft)
            if starts_at <= datetime.now(UTC):
                raise AppError(422, "past_booking", "Choose a future date and time.")
            assert draft.name is not None and draft.email is not None
            booking = Booking(
                id=uuid4(),
                draft_id=draft.draft_id,
                session_id=session_id,
                name=draft.name,
                email=str(draft.email),
                starts_at=starts_at,
                timezone=draft.timezone,
            )
            db.add(booking)
            await db.commit()
            return self.booking_response(booking)

    @staticmethod
    def booking_response(booking: Booking) -> BookingResponse:
        return BookingResponse(
            booking_id=booking.id,
            name=booking.name,
            email=booking.email,
            starts_at=booking.starts_at,
            timezone=booking.timezone,
        )

    async def retrieve(self, query: str, document_ids: list[UUID]) -> tuple[str, list[Source]]:
        async with self.database() as db:
            if document_ids:
                ready = set(
                    await db.scalars(
                        select(Document.id).where(
                            Document.id.in_(document_ids),
                            Document.status == "ready",
                            Document.embedding_model == self.settings.gemini_embedding_model,
                            Document.embedding_dimensions == self.settings.embedding_dimensions,
                        )
                    )
                )
                if set(document_ids) != ready:
                    raise AppError(404, "document_not_ready", "A selected document is not ready.")
            has_documents = await db.scalar(
                select(Document.id)
                .where(
                    Document.status == "ready",
                    Document.embedding_model == self.settings.gemini_embedding_model,
                    Document.embedding_dimensions == self.settings.embedding_dimensions,
                )
                .limit(1)
            )
            if has_documents is None:
                return "Upload a document first so I can answer questions about it.", []
        embedding = (await self.gemini.embed([query], "RETRIEVAL_QUERY"))[0]
        candidates = await self.vectors.search(embedding, document_ids)
        async with self.database() as db:
            ready_ids = set(
                await db.scalars(
                    select(Document.id).where(
                        Document.id.in_([source.document_id for source in candidates]),
                        Document.status == "ready",
                    )
                )
            )
        sources = [source for source in candidates if source.document_id in ready_ids][
            : self.settings.retrieval_limit
        ]
        for citation, source in enumerate(sources, 1):
            source.citation = citation
        if not sources:
            return "I could not find relevant information in the uploaded documents.", []
        result = await self.gemini.answer(query, sources)
        return self.format_answer(result, sources)

    @staticmethod
    def format_answer(result: GroundedAnswer, sources: list[Source]) -> tuple[str, list[Source]]:
        citations = set(result.citations)
        valid = {source.citation for source in sources}
        if not citations <= valid:
            raise AppError(
                502, "invalid_citations", "The model returned invalid source references."
            )
        references = [source for source in sources if source.citation in citations]
        markers = " ".join(f"[{source.citation}]" for source in references)
        answer = f"{result.answer.rstrip()} {markers}".strip()
        return answer, references
