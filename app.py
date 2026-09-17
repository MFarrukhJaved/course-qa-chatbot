"""
TeachRAG - Streamlit chat interface

Instructor uploads course material (PDF/Word). Students chat with a bot
that answers ONLY from that material, citing sources, and says plainly
when something isn't covered. Same Streamlit-based approach as ChemALLM,
adapted for course documents instead of a chemical compound database.

Run with:
    streamlit run app.py
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import UPLOADS_DIR, TOP_K, MIN_SIMILARITY
from core.rag import get_rag
from core.llm import get_llm_client

st.set_page_config(page_title="Course Q&A Assistant", page_icon="🎓", layout="wide")

rag = get_rag()

# --------------------------------------------------
# Sidebar - instructor controls
# --------------------------------------------------
with st.sidebar:
    st.header("📚 Course Material")

    llm_probe = get_llm_client()
    available_models = llm_probe.get_available_models()

    if available_models:
        default_idx = 0
        model_choice = st.selectbox("LLM model (from your local Ollama)", available_models, index=default_idx)
    else:
        st.warning("Can't reach Ollama, or no models are pulled yet.")
        st.code("ollama serve\nollama pull llama3.1", language="bash")
        model_choice = None

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDFs / Word docs",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Only what you upload here is used to answer questions.",
    )

    replace_index = st.checkbox(
        "Replace existing material (rebuild from scratch)",
        value=False,
        help="Leave unchecked to add these files to what's already indexed.",
    )

    if st.button("Build / Update knowledge base", type="primary", disabled=not uploaded_files):
        saved_paths = []
        for f in uploaded_files:
            dest = os.path.join(UPLOADS_DIR, f.name)
            with open(dest, "wb") as out:
                out.write(f.getbuffer())
            saved_paths.append(dest)

        with st.spinner(f"Reading and indexing {len(saved_paths)} file(s)..."):
            result = rag.ingest_documents(saved_paths, replace=replace_index)

        if result["status"] == "ok":
            st.success(f"Indexed {result['chunks_added']} passages from {len(saved_paths)} file(s).")
            if result["errors"]:
                st.warning("Some files had issues:\n" + "\n".join(result["errors"]))
        else:
            st.error(result.get("message", "Ingestion failed."))
            if result.get("errors"):
                st.error("\n".join(result["errors"]))

    st.divider()

    stats = rag.get_stats()
    st.subheader("Current knowledge base")
    st.metric("Documents indexed", stats["num_documents"])
    st.metric("Passages indexed", stats["num_chunks"])
    if stats["documents"]:
        with st.expander("Show indexed files"):
            for doc in stats["documents"]:
                st.write(f"- {doc}")

    with st.expander("Retrieval settings"):
        st.caption(f"Top-K passages per question: {TOP_K}")
        st.caption(f"Minimum relevance threshold: {MIN_SIMILARITY}")
        st.caption("Edit core/config.py to change these.")

# --------------------------------------------------
# Main - chat interface
# --------------------------------------------------
st.title("🎓 Course Q&A Assistant")
st.caption(
    "This chatbot answers only from the course material your instructor has uploaded. "
    "It will not use outside knowledge, and it will tell you when a question isn't covered."
)

if rag.get_stats()["num_chunks"] == 0:
    st.info("No course material has been indexed yet. Ask your instructor to upload it in the sidebar.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for src, page, score in msg["sources"]:
                    st.caption(f"{src} — {page} (relevance {score:.2f})")

question = st.chat_input("Ask a question about the course material...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if rag.get_stats()["num_chunks"] == 0:
            answer = "No course material has been uploaded yet, so I have nothing to answer from."
            sources = []
        else:
            with st.spinner("Searching course material..."):
                prompt, retrieved = rag.build_rag_prompt(question)

            with st.spinner("Thinking..."):
                llm = get_llm_client(model=model_choice)
                answer = llm.ask_grounded(prompt)

            sources = [(c["source"], c["page"], score) for c, score in retrieved]

        st.markdown(answer)
        if sources:
            with st.expander("Sources used"):
                for src, page, score in sources:
                    st.caption(f"{src} — {page} (relevance {score:.2f})")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
