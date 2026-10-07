from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    index: int
    text: str
    page_number: int | None = None


def chunk_text(
    text: str, *, chunk_size: int, chunk_overlap: int, page_number: int | None = None
) -> list[DocumentChunk]:
    """Split text into overlapping character windows, preferring word boundaries."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            boundary = cleaned.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(DocumentChunk(index=len(chunks), text=piece, page_number=page_number))
        if end >= len(cleaned):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks
