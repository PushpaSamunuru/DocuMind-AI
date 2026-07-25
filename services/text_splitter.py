from dataclasses import dataclass

from services.pdf_processor import DocumentPage


@dataclass
class TextChunk:
    chunk_id: str
    document_name: str
    page_number: int
    text: str


def split_text(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[str]:
    """Split text into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    cleaned_text = " ".join(text.split())

    if not cleaned_text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(cleaned_text):
        end = min(start + chunk_size, len(cleaned_text))
        chunk = cleaned_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(cleaned_text):
            break

        start = end - chunk_overlap

    return chunks


def create_document_chunks(
    pages: list[DocumentPage],
) -> list[TextChunk]:
    """Convert extracted PDF pages into searchable text chunks."""

    document_chunks: list[TextChunk] = []

    for page in pages:
        page_chunks = split_text(page.text)

        for chunk_index, chunk_text in enumerate(page_chunks):
            safe_name = page.document_name.replace(" ", "_")

            document_chunks.append(
                TextChunk(
                    chunk_id=(
                        f"{safe_name}-page-{page.page_number}-"
                        f"chunk-{chunk_index + 1}"
                    ),
                    document_name=page.document_name,
                    page_number=page.page_number,
                    text=chunk_text,
                )
            )

    return document_chunks