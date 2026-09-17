"""
TeachRAG - RAG (Retrieval-Augmented Generation)

100% LOCAL - vector index lives on disk, embeddings run on your machine,
generation goes to a local Ollama server. No external API calls, so
course material never leaves the environment this runs on.

How it works (mirrors the shape of ChemALLM's core/rag.py, swapped from
SQL lookups against a compound DB to vector search over document chunks):

1. Instructor uploads PDFs/Word docs -> ingest_documents() extracts,
   chunks, embeds, and indexes them.
2. Student asks a question -> retrieve() embeds the question and finds
   the most similar chunks in the index.
3. Chunks below MIN_SIMILARITY are dropped - if nothing clears the bar,
   the bot says the material doesn't cover it, instead of guessing.
4. Remaining chunks are formatted as labelled, cited context and injected
   into the LLM prompt, alongside the strict "answer only from this"
   instruction in core/llm.py's GROUNDED_SYSTEM_PROMPT.
"""

import os
from typing import List, Dict, Tuple

from core.config import TOP_K, MIN_SIMILARITY, UPLOADS_DIR
from core.ingest import extract_files
from core.chunking import chunk_records
from core.embeddings import embed_texts, embed_query
from core.vectorstore import VectorStore


class TeachRAG:
    """Local Retrieval-Augmented Generation over instructor-uploaded course material."""

    def __init__(self, top_k: int = TOP_K, min_similarity: float = MIN_SIMILARITY):
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.store = VectorStore()
        self.store.load()  # no-op if nothing has been indexed yet

    # --------------------------------------------------
    # Ingestion
    # --------------------------------------------------
    def ingest_documents(self, file_paths: List[str], replace: bool = False) -> Dict:
        """
        Extract + chunk + embed + index the given files.
        replace=True rebuilds the index from scratch (use when the
        instructor wants to start over); replace=False adds to what's
        already indexed.
        """
        records, errors = extract_files(file_paths)
        chunks = chunk_records(records)

        if not chunks:
            return {
                "status": "error" if errors else "empty",
                "message": "No text could be extracted from the uploaded file(s).",
                "errors": errors,
            }

        texts = [c["text"] for c in chunks]
        vectors = embed_texts(texts)

        if replace or self.store.is_empty():
            self.store.build(chunks, vectors)
        else:
            self.store.add(chunks, vectors)

        self.store.save()

        return {
            "status": "ok",
            "chunks_added": len(chunks),
            "errors": errors,
            "stats": self.store.stats(),
        }

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------
    def retrieve(self, query: str) -> List[Tuple[Dict, float]]:
        """Return [(chunk, similarity_score), ...] above the relevance floor."""
        if not query or not query.strip():
            return []
        if self.store.is_empty():
            return []

        q_vector = embed_query(query)
        results = self.store.search(q_vector, self.top_k)
        return [(chunk, score) for chunk, score in results if score >= self.min_similarity]

    # --------------------------------------------------
    # Context formatting
    # --------------------------------------------------
    def format_context(self, results: List[Tuple[Dict, float]]) -> str:
        if not results:
            return ("[COURSE MATERIAL]: Nothing sufficiently relevant was found in the uploaded "
                    "material for this question. Tell the student this plainly - do not guess.")

        # NOTE: deliberately NOT labelled "Passage 1 / Passage 2 / ..." here. Numbering
        # them invites the model to write "Passage 3 says..." in its answer, which is
        # meaningless to the student. Each excerpt carries its own citation instead, and
        # the system prompt tells the model to cite that way, not by position in this list.
        lines = ["[COURSE MATERIAL - relevant excerpts below, unordered, each with its own citation]\n"]

        for chunk, score in results:
            lines.append(f"(Source: {chunk['source']}, {chunk['page']})")
            lines.append(chunk["text"])
            lines.append("")

        lines.append("[END OF COURSE MATERIAL - write one synthesized explanation from the excerpts "
                      "above; cite using (Source: file, page) inline, never by position/number.]")

        return "\n".join(lines)

    # --------------------------------------------------
    # Prompt builder
    # --------------------------------------------------
    def build_rag_prompt(self, question: str) -> Tuple[str, List[Tuple[Dict, float]]]:
        retrieved = self.retrieve(question)
        context = self.format_context(retrieved)

        prompt = f"{context}\n\n[STUDENT QUESTION]\n{question}"
        return prompt, retrieved

    # --------------------------------------------------
    # Stats
    # --------------------------------------------------
    def get_stats(self) -> Dict:
        return self.store.stats()


# --------------------------------------------------
# Singleton (same pattern as ChemALLM)
# --------------------------------------------------
_rag_instance: TeachRAG = None


def get_rag() -> TeachRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = TeachRAG()
    return _rag_instance
