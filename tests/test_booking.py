from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas import BookingDraft, ChatRequest
from app.services.chat import ChatService


def test_booking_time_converts_to_utc() -> None:
    date = (datetime.now(UTC) + timedelta(days=2)).date()
    draft = BookingDraft(
        draft_id=uuid4(),
        name="Raksha",
        email="raksha@example.com",
        date=date,
        time="10:30",
        timezone="Asia/Kathmandu",
    )
    assert ChatService.starts_at(draft) == datetime.combine(date, datetime.min.time(), UTC).replace(
        hour=4, minute=45
    )


def test_confirmation_requires_a_session_and_draft() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="Confirm", request_id=uuid4(), confirm_booking=True)
