# Course Q&A Assistant (TeachRAG)

A Streamlit chatbot that answers student questions **only** from course
material you upload (PDF and Word `.docx` files) — no outside knowledge,
no hallucinated citations, and a clear "not covered" answer when the
material doesn't address the question.

It runs on a local Ollama server serving a general open-source LLM
(Llama, Mistral, Qwen, Gemma, ...) for generation, so no student data or
course content leaves the machine this runs on. Retrieval is a proper
document RAG pipeline built for unstructured course material — extract
text, chunk it, embed it locally, and search it with a vector index.

## How it works

1. **Ingest** — PDFs are read page-by-page, Word docs are split into
   ~1500-character sections. (`core/ingest.py`)
2. **Chunk** — each page/section is split into overlapping ~900-character
   chunks so retrieval can zero in on the relevant part, not a whole page.
   (`core/chunking.py`)
3. **Embed** — chunks are embedded locally with
   `sentence-transformers/all-MiniLM-L6-v2` (no API calls). (`core/embeddings.py`)
4. **Index** — embeddings go into a FAISS index, persisted to
   `storage/index.faiss` + `storage/chunks.json` so you don't need to
   re-embed on every restart. (`core/vectorstore.py`)
5. **Retrieve + guardrail** — a question is embedded and matched against
   the index; chunks below a relevance threshold are dropped, and if
   nothing clears the bar the bot says so instead of guessing.
   (`core/rag.py`)
6. **Generate** — the retrieved, cited passages are injected into a
   strict "answer ONLY from this context" prompt sent to your local
   Ollama model. (`core/llm.py`)

## Setup

### 1. Install Ollama and pull a general-purpose model

```bash
# https://ollama.com
ollama serve &
ollama pull llama3.1        # or mistral, qwen2.5, gemma2, etc.
```

### 2. Install Python dependencies

```bash
cd teach-rag
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Open the sidebar, upload your course PDFs/Word docs, click
**"Build / Update knowledge base"**, and start asking questions in the
chat box.

## Configuration

All the knobs live in `core/config.py` and can be overridden with
environment variables of the same name:

| Setting | Default | What it does |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1` | Which local model to use for answers |
| `OLLAMA_HOST` | `http://localhost:11434` | Where your Ollama server runs |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | Chunking granularity |
| `TOP_K` | `5` | How many passages to retrieve per question |
| `MIN_SIMILARITY` | `0.30` | Relevance floor — below this, the bot says "not covered" |

## Keeping answers "in-bounds"

Two layers enforce the "only answer from uploaded material" requirement:

1. **Retrieval floor** (`MIN_SIMILARITY` in `core/rag.py`) — if nothing
   retrieved is actually relevant to the question, no context is sent to
   the LLM as usable material, and the prompt makes that explicit.
2. **Prompt guardrail** (`GROUNDED_SYSTEM_PROMPT` in `core/llm.py`) — the
   model is instructed, repeatedly and explicitly, to answer only from
   the provided passages, cite sources, and admit when something isn't
   covered — rather than to fall back on its own training knowledge.

No approach with an LLM is 100% leak-proof (a capable-enough student can
usually find prompt-injection-style workarounds), but this combination
covers the normal case well. If you need harder guarantees for exams or
graded work, consider also restricting the deployment (e.g. no internet
access for the app's host) and reviewing chat transcripts periodically.

## Project layout

```
teach-rag/
├── app.py                 # Streamlit UI
├── requirements.txt
├── core/
│   ├── config.py           # all settings
│   ├── ingest.py           # PDF/DOCX text extraction
│   ├── chunking.py         # text -> overlapping chunks
│   ├── embeddings.py       # local sentence-transformers embeddings
│   ├── vectorstore.py      # FAISS index + persistence
│   ├── rag.py              # retrieval + prompt building + guardrail
│   └── llm.py              # Ollama client + grounded system prompt
└── storage/                # created at runtime: uploads/, index.faiss, chunks.json
```

## Adding more file types later

`core/ingest.py` dispatches by file extension in `extract_file()`. To add
PowerPoint (`.pptx`) or plain text later, add an `extract_pptx()` /
`extract_txt()` function that returns the same
`[{"source", "page", "text"}, ...]` shape, and register the extension in
`extract_file()`.
