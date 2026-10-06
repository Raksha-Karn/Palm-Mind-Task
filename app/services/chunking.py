from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader

from app.errors import AppError
from app.schemas import ChunkStrategy


@dataclass(frozen=True)
class TextPage:
    text: str
    page: int | None


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int | None
    index: int


def extract_text(data: bytes, extension: str, max_characters: int) -> list[TextPage]:
    pages: list[TextPage] = []
    try:
        if extension == ".txt":
            pages = [TextPage(data.decode("utf-8-sig"), None)]
        else:
            if not data.startswith(b"%PDF-"):
                raise AppError(422, "invalid_pdf", "The file is not a valid PDF.")
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise AppError(422, "encrypted_pdf", "Upload an unencrypted PDF.")
            total = 0
            for number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                total += len(text)
                if total > max_characters:
                    raise AppError(413, "document_too_large", "Extracted text exceeds the limit.")
                pages.append(TextPage(text, number))
    except AppError:
        raise
    except UnicodeDecodeError as exc:
        raise AppError(422, "invalid_encoding", "TXT files must use UTF-8 encoding.") from exc
    except Exception as exc:
        raise AppError(422, "extraction_failed", "Could not extract text from this file.") from exc

    if sum(len(page.text) for page in pages) > max_characters:
        raise AppError(413, "document_too_large", "Extracted text exceeds the limit.")
    pages = [TextPage(page.text.replace("\x00", "").strip(), page.page) for page in pages]
    pages = [page for page in pages if page.text]
    if not pages:
        raise AppError(422, "empty_document", "No text found. Scanned PDFs require OCR.")
    return pages


def chunk_text(
    pages: list[TextPage], strategy: ChunkStrategy, size: int, overlap: int, max_chunks: int
) -> list[Chunk]:
    if size < 1 or overlap < 0 or overlap >= size:
        raise AppError(422, "invalid_chunk_options", "Overlap must be smaller than chunk size.")
    chunks: list[Chunk] = []
    for page in pages:
        start = 0
        while start < len(page.text):
            end = min(start + size, len(page.text))
            if strategy == ChunkStrategy.RECURSIVE and end < len(page.text):
                lower_bound = start + max(size // 2, overlap + 1)
                for separator in ("\n\n", ". ", "\n", " "):
                    boundary = page.text.rfind(separator, lower_bound, end)
                    if boundary >= 0:
                        end = boundary + len(separator)
                        break
            text = page.text[start:end].strip()
            if text:
                chunks.append(Chunk(text, page.page, len(chunks)))
                if len(chunks) > max_chunks:
                    raise AppError(413, "too_many_chunks", "Document produces too many chunks.")
            if end == len(page.text):
                break
            start = end - overlap
    return chunks
