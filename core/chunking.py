"""
TeachRAG - Chunking

Splits each extracted page/section into overlapping, roughly fixed-size
chunks so retrieval can find a tightly relevant piece of text rather than
a whole page. Metadata (source, page) is carried onto every chunk so
citations stay accurate down to the chunk level.
"""

from typing import List, Dict

from core.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple character-based sliding-window chunker with sentence-ish breaks."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        # try to break on a sentence/paragraph boundary near the end
        if end < n:
            break_point = text.rfind(". ", start, end)
            if break_point == -1 or break_point < start + int(chunk_size * 0.5):
                break_point = text.rfind("\n", start, end)
            if break_point != -1 and break_point > start + int(chunk_size * 0.5):
                end = break_point + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return chunks


def chunk_records(records: List[Dict]) -> List[Dict]:
    """
    Turn extracted {source, page, text} records into chunk records:
    {source, page, chunk_id, text}.
    """
    chunked = []
    for rec in records:
        pieces = chunk_text(rec["text"])
        for i, piece in enumerate(pieces):
            chunked.append({
                "source": rec["source"],
                "page": rec["page"],
                "chunk_id": f"{rec['source']}::{rec['page']}::{i}",
                "text": piece,
            })
    return chunked
