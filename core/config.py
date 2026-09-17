"""
TeachRAG - Configuration

Central place for every tunable knob. Override any of these with
environment variables of the same name (e.g. `export OLLAMA_MODEL=llama3.1`).
"""

import os

# --------------------------------------------------
# LLM (Ollama) - "general" open models, no chemical/domain fine-tuning
# --------------------------------------------------
# Point this at any general-purpose model you've pulled into Ollama,
# e.g. "llama3.1", "llama3.1:8b", "mistral", "qwen2.5", "gemma2".
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# --------------------------------------------------
# Embeddings (local, no external API calls)
# --------------------------------------------------
# Small, fast, good-enough sentence embedding model. Runs on CPU.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --------------------------------------------------
# Chunking
# --------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))       # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # characters of overlap between chunks

# --------------------------------------------------
# Retrieval
# --------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))
# Minimum cosine similarity for a chunk to be considered relevant enough
# to answer from. Below this, the bot says the material doesn't cover it.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.30"))

# --------------------------------------------------
# Storage
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.getenv("STORAGE_DIR", os.path.join(BASE_DIR, "storage"))
UPLOADS_DIR = os.path.join(STORAGE_DIR, "uploads")
INDEX_PATH = os.path.join(STORAGE_DIR, "index.faiss")
METADATA_PATH = os.path.join(STORAGE_DIR, "chunks.json")

os.makedirs(UPLOADS_DIR, exist_ok=True)
