from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), extra="ignore")

    gemini_api_key: SecretStr
    gemini_chat_model: str
    gemini_embedding_model: str
    embedding_dimensions: int = Field(ge=128, le=3072)
    database_url: str
    redis_url: str
    qdrant_url: str
    qdrant_collection: str
    booking_timezone: str
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, ge=1024)
    max_document_characters: int = Field(default=500_000, ge=1000)
    max_chunks: int = Field(default=1000, ge=1)
    session_ttl_seconds: int = Field(default=86400, ge=60)
    history_turns: int = Field(default=12, ge=1, le=50)
    retrieval_limit: int = Field(default=5, ge=1, le=10)
    minimum_score: float = Field(default=0.3, ge=-1, le=1)

    @field_validator("booking_timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        ZoneInfo(value)
        return value
