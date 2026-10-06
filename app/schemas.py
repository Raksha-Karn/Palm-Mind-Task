import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    WithJsonSchema,
    field_validator,
    model_validator,
)


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChunkStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"


class DocumentResponse(Schema):
    document_id: UUID
    filename: str
    strategy: ChunkStrategy
    chunk_count: int
    status: Literal["ready"] = "ready"
    duplicate: bool = False


class ChatRequest(Schema):
    message: str = Field(min_length=1, max_length=4000)
    session_id: UUID | None = None
    request_id: UUID
    document_ids: list[UUID] = Field(default_factory=list, max_length=20)
    confirm_booking: bool = False
    draft_id: UUID | None = None

    @field_validator("message")
    @classmethod
    def nonblank_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message must not be blank")
        return value

    @model_validator(mode="after")
    def confirmation_requires_draft(self) -> "ChatRequest":
        if self.confirm_booking and (self.draft_id is None or self.session_id is None):
            raise ValueError("Confirmation requires session_id and draft_id")
        return self


class BookingFields(Schema):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    email: EmailStr | None = Field(default=None, max_length=254)
    date: datetime.date | None = None
    time: (
        Annotated[
            datetime.time,
            WithJsonSchema({"type": "string", "pattern": r"^([01]\d|2[0-3]):[0-5]\d$"}),
        ]
        | None
    ) = None

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str | None) -> str | None:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("Name must not be blank")
        return value

    @field_validator("time")
    @classmethod
    def local_time(cls, value: datetime.time | None) -> datetime.time | None:
        if value is not None and (value.tzinfo is not None or value.second or value.microsecond):
            raise ValueError("Use a local time in HH:MM format")
        return value


class BookingDraft(BookingFields):
    draft_id: UUID
    timezone: str

    def missing_fields(self) -> list[str]:
        return [key for key in ("name", "email", "date", "time") if getattr(self, key) is None]


class BookingResponse(Schema):
    booking_id: UUID
    name: str
    email: EmailStr
    starts_at: datetime.datetime
    timezone: str


class Source(Schema):
    citation: int
    document_id: UUID
    filename: str
    page: int | None
    chunk_index: int
    text: str
    score: float


class ChatResponse(Schema):
    session_id: UUID
    request_id: UUID
    answer: str
    sources: list[Source] = Field(default_factory=list)
    booking_status: Literal[
        "none", "collecting", "awaiting_confirmation", "booked", "cancelled"
    ] = "none"
    draft: BookingDraft | None = None
    booking: BookingResponse | None = None


class Message(Schema):
    role: Literal["user", "assistant"]
    content: str


class Session(Schema):
    history: list[Message] = Field(default_factory=list)
    draft: BookingDraft | None = None


class QueryPlan(Schema):
    intent: Literal["question", "booking", "cancel_booking"]
    standalone_query: str = Field(min_length=1, max_length=4000)
    booking_fields: BookingFields = Field(default_factory=BookingFields)
    clarification: str | None = Field(default=None, max_length=500)


class GroundedAnswer(Schema):
    answer: str = Field(min_length=1, max_length=8000)
    citations: list[int] = Field(default_factory=list, max_length=10)
