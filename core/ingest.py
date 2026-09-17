"""
TeachRAG - Document Ingestion

Extracts plain text from the two formats you said you'll actually use:
PDF and Word (.docx). Each extracted unit keeps metadata (source filename,
page/section number) so later answers can cite exactly where they came from.
"""

import os
from typing import List, Dict

from pypdf import PdfReader
from docx import Document as DocxDocument


def extract_pdf(file_path: str) -> List[Dict]:
    """Return one record per page: {"source", "page", "text"}."""
    filename = os.path.basename(file_path)
    records = []

    reader = PdfReader(file_path)
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            records.append({"source": filename, "page": i, "text": text})

    return records


def extract_docx(file_path: str) -> List[Dict]:
    """
    Return one record per section: {"source", "page", "text"}.
    Word docs don't have fixed pages, so we group paragraphs into
    pseudo-sections (~1500 chars each) and label them "section N" instead
    of a real page number.
    """
    filename = os.path.basename(file_path)
    doc = DocxDocument(file_path)

    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    records = []
    buffer = []
    buffer_len = 0
    section_num = 1
    SECTION_TARGET = 1500

    def flush():
        nonlocal buffer, buffer_len, section_num
        if buffer:
            records.append({
                "source": filename,
                "page": f"section {section_num}",
                "text": "\n".join(buffer),
            })
            section_num += 1
            buffer = []
            buffer_len = 0

    for para in paragraphs:
        buffer.append(para)
        buffer_len += len(para)
        if buffer_len >= SECTION_TARGET:
            flush()
    flush()

    return records


def extract_file(file_path: str) -> List[Dict]:
    """Dispatch on extension. Raises ValueError for unsupported types."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_pdf(file_path)
    elif ext == ".docx":
        return extract_docx(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. TeachRAG currently supports .pdf and .docx."
        )


def extract_files(file_paths: List[str]) -> List[Dict]:
    """Extract text records from multiple files, skipping ones that fail."""
    all_records = []
    errors = []

    for fp in file_paths:
        try:
            all_records.extend(extract_file(fp))
        except Exception as e:
            errors.append(f"{os.path.basename(fp)}: {e}")

    return all_records, errors
