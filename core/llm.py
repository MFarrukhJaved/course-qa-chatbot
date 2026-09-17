"""
TeachRAG - LLM Client

Adapted from the ChemALLM project's core/llm.py. Same idea: talk to a
LOCAL Ollama server running a general-purpose open model (Llama, Mistral,
Qwen, Gemma, ...) - no domain fine-tuning, no external API calls, no
student data leaving the machine it runs on.

What changed vs. ChemALLM's llm.py:
- Dropped the chemical/business-report-specific prompt methods
  (generate_cleaning_summary, generate_ml_summary, etc.) - not relevant here.
- Added `ask_grounded()`, which is the one method the teaching chatbot
  actually calls: it enforces "answer ONLY from the supplied course
  material, and say so plainly when the material doesn't cover it."
"""

import json
import requests
from typing import Optional, Dict, List, Generator

from core.config import OLLAMA_MODEL, OLLAMA_HOST


# This is the guardrail. It is deliberately blunt and repeated in the
# user-turn prompt (see core/rag.py) because small/medium open models
# follow instructions more reliably when they're reinforced, not just
# stated once in a system message.
GROUNDED_SYSTEM_PROMPT = """You are a teaching assistant chatbot. You answer questions using
ONLY the course material excerpts provided to you in the context below - nothing else.

Rules you must always follow:
- Base your entire answer strictly on the provided context. Do not use outside knowledge,
  even if you are confident it is correct.
- If the context does not contain enough information to answer, say so plainly, e.g.
  "The course material I have doesn't cover that." Do not guess or fill gaps.
- Do not fabricate citations, page numbers, or quotes.

How to write the answer (this matters as much as the content):
- WRITE ONE SYNTHESIZED, FLOWING EXPLANATION - as if you had read and understood the
  material yourself, not as a list of what each excerpt says. Never write things like
  "Excerpt 1 says...", "Passage 3 explains...", or "According to excerpt 2...". The
  student never sees how the material was split up internally, so referring to it that
  way is confusing and meaningless to them.
- Cite sources naturally and sparingly, inline, using only the filename and page/section,
  e.g. "the least squares fit minimizes the residual sum of squares (ISLP.pdf, p.62)".
  Never say "Excerpt" or "Passage" - just the source citation in parentheses.
- Use LaTeX for ANY mathematical notation, formulas, or equations: wrap inline math in
  single dollar signs, e.g. $\\beta_0 + \\beta_1 X$, and standalone equations in double
  dollar signs on their own line, e.g. $$Y = \\beta_0 + \\beta_1 X + \\epsilon$$.
  Never write math in plain text (no "beta_0", no "x^2" without LaTeX).
- Use light markdown structure to make the answer easy to scan: short paragraphs,
  **bold** for key terms the first time you introduce them, and bullet points only when
  listing multiple distinct items (not for a single continuous explanation).
- Match the level of detail to the question - a quick factual question gets a concise
  answer; a "explain this concept" question gets a fuller, well-organized one. Don't pad
  with information the student didn't ask for.
- If the student asks something unrelated to the course material entirely (e.g. general
  chit-chat, or a different subject), politely explain you can only help with the
  uploaded course material."""


class LLMClient:
    """Client for interacting with a local Ollama LLM."""

    def __init__(self, model: str = None, host: str = None):
        self.model = model or OLLAMA_MODEL
        self.host = (host or OLLAMA_HOST).rstrip("/")
        self.api_url = f"{self.host}/api"

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                return any(self.model in m or m.startswith(self.model) for m in models)
            return False
        except Exception:
            return False

    def get_available_models(self) -> List[str]:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
            return []
        except Exception:
            return []

    # --------------------------------------------------
    # Core chat
    # --------------------------------------------------
    def chat(self, prompt: str, system_prompt: str = None,
             temperature: float = 0.2, max_tokens: int = 2048) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        try:
            resp = requests.post(f"{self.api_url}/chat", json=payload, timeout=180)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.exceptions.ConnectionError:
            return ("[ERROR] Cannot connect to Ollama. Start it with `ollama serve`, "
                    f"and make sure the model is pulled: `ollama pull {self.model}`.")
        except Exception as e:
            return f"[ERROR] LLM request failed: {e}"

    def chat_stream(self, prompt: str, system_prompt: str = None,
                     temperature: float = 0.2) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }

        try:
            resp = requests.post(f"{self.api_url}/chat", json=payload, stream=True, timeout=180)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
        except requests.exceptions.ConnectionError:
            yield ("[ERROR] Cannot connect to Ollama. Start it with `ollama serve`, "
                   f"and make sure the model is pulled: `ollama pull {self.model}`.")
        except Exception as e:
            yield f"[ERROR] {e}"

    # --------------------------------------------------
    # Grounded Q&A - the method the teaching chatbot uses
    # --------------------------------------------------
    def ask_grounded(self, rag_prompt: str, temperature: float = 0.2) -> str:
        """Answer a student question using ONLY the RAG-built context."""
        return self.chat(rag_prompt, system_prompt=GROUNDED_SYSTEM_PROMPT, temperature=temperature)

    def ask_grounded_stream(self, rag_prompt: str, temperature: float = 0.2) -> Generator[str, None, None]:
        yield from self.chat_stream(rag_prompt, system_prompt=GROUNDED_SYSTEM_PROMPT, temperature=temperature)


# --------------------------------------------------
# Singleton (same pattern as ChemALLM)
# --------------------------------------------------
_client: Optional[LLMClient] = None


def get_llm_client(model: str = None) -> LLMClient:
    global _client
    if _client is None or (model and _client.model != model):
        _client = LLMClient(model=model)
    return _client
