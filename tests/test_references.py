from uuid import uuid4

import pytest

from app.errors import AppError
from app.schemas import GroundedAnswer, Source
from app.services.chat import ChatService


def source() -> Source:
    return Source(
        citation=1,
        document_id=uuid4(),
        filename="sample.txt",
        page=None,
        chunk_index=0,
        text="The interview lasts 45 minutes.",
        score=0.9,
    )


def test_formats_references_from_valid_source_ids() -> None:
    reference = source()
    result = GroundedAnswer(answer="The interview lasts 45 minutes.", citations=[1])
    answer, references = ChatService.format_answer(result, [reference])
    assert answer == "The interview lasts 45 minutes. [1]"
    assert references == [reference]


def test_rejects_unknown_source_ids() -> None:
    result = GroundedAnswer(answer="The interview lasts 45 minutes.", citations=[99])
    with pytest.raises(AppError) as caught:
        ChatService.format_answer(result, [source()])
    assert caught.value.code == "invalid_citations"
