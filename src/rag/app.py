"""
Streamlit chat interface for the Superstore RAG pipeline.

Usage:
    streamlit run src/rag/app.py

Optional env vars:
    OLLAMA_MODEL   : Ollama model to use (default: llama3.2:3b)
    CHROMA_DB_DIR  : Path to ChromaDB directory (default: src/vector_db/chroma_db)
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_HERE)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import streamlit as st
from rag.pipeline import RAGPipeline

_DEFAULT_DB = os.path.join(_SRC_DIR, "vector_db", "chroma_db")
_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
_DB_DIR = os.environ.get("CHROMA_DB_DIR", _DEFAULT_DB)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_sources(summary_hits: list[dict], txn_hits: list[dict]) -> None:
    total = len(summary_hits) + len(txn_hits)
    with st.expander(f"Sources ({total} retrieved)", expanded=False):
        if summary_hits:
            st.markdown("**Aggregated summaries**")
            for src in summary_hits:
                dist_label = "HIGH" if src["distance"] < 0.5 else "MED" if src["distance"] < 0.9 else "LOW"
                st.markdown(f"[{dist_label}] `{src['id']}` dist={src['distance']:.4f}")
                st.caption(src["text"][:300])
        if txn_hits:
            st.markdown("**Transaction-level examples**")
            for src in txn_hits:
                dist_label = "HIGH" if src["distance"] < 0.5 else "MED" if src["distance"] < 0.9 else "LOW"
                st.markdown(f"[{dist_label}] `{src['id']}` dist={src['distance']:.4f}")
                st.caption(src["text"][:300])


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def load_pipeline(mode: str) -> RAGPipeline:
    return RAGPipeline(persist_dir=_DB_DIR, model=_MODEL, mode=mode)


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mode" not in st.session_state:
        st.session_state.mode = "direct"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar() -> tuple[dict | None, bool, str]:
    with st.sidebar:
        st.title("Superstore RAG")
        st.caption(f"Model: `{_MODEL}`")
        st.divider()

        st.subheader("Retrieval mode")
        mode = st.radio(
            "Mode",
            ["direct", "agent"],
            index=0,
            help=(
                "**direct** — pipeline calls both tools, builds context, LLM answers once.\n\n"
                "**agent** — LLM decides which tools to call and with what queries/filters."
            ),
        )
        if mode != st.session_state.mode:
            st.session_state.mode = mode
            st.session_state.messages = []
            load_pipeline(mode).reset_memory()

        st.divider()

        # Metadata filter is only meaningful in direct mode
        where: dict | None = None
        if mode == "direct":
            st.subheader("Metadata Filter (optional)")
            filter_type = st.selectbox("Filter by", ["None", "Category", "Region", "Year"])
            if filter_type == "Category":
                cat = st.selectbox("Category", ["Technology", "Furniture", "Office Supplies"])
                where = {"category": cat}
            elif filter_type == "Region":
                reg = st.selectbox("Region", ["West", "East", "Central", "South"])
                where = {"region": reg}
            elif filter_type == "Year":
                yr = st.selectbox("Year", ["2014", "2015", "2016", "2017"])
                where = {"year": yr}
            st.divider()
        else:
            st.caption("In agent mode the LLM decides its own filters.")
            st.divider()

        show_sources = st.toggle("Show retrieved sources", value=True)

        if st.button("Clear conversation"):
            st.session_state.messages = []
            load_pipeline(mode).reset_memory()
            st.rerun()

        st.divider()
        st.subheader("Example questions")
        examples = [
            "What is the sales trend from 2014 to 2017?",
            "Which season has the highest sales?",
            "Which category generates the most revenue?",
            "Which sub-categories have the highest profit margins?",
            "Which region performs best in terms of profit?",
            "Compare Technology and Furniture sales trends.",
            "How do West and East regions compare in profit?",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state["prefill"] = ex
                st.rerun()

    return where, show_sources, mode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Superstore Sales RAG",
        page_icon="📊",
        layout="wide",
    )
    _init_state()

    where, show_sources, mode = _sidebar()

    st.title("📊 Superstore Sales Analysis")
    st.caption(f"Ask questions about the Superstore dataset (2014–2017) · mode: `{mode}`")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if show_sources and (msg.get("summary_hits") or msg.get("txn_hits")):
                _render_sources(msg.get("summary_hits", []), msg.get("txn_hits", []))

    prefill = st.session_state.pop("prefill", None)
    question = st.chat_input("Ask a question about Superstore sales...") or prefill
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    rag = load_pipeline(mode)
    with st.chat_message("assistant"):
        spinner_msg = "Retrieving and generating..." if mode == "direct" else "Agent thinking and retrieving..."
        with st.spinner(spinner_msg):
            result = rag.ask(question, summary_where=where, use_memory=True)

        st.markdown(result["answer"])

        if show_sources:
            _render_sources(result["summary_hits"], result["txn_hits"])

    st.session_state.messages.append({
        "role":         "assistant",
        "content":      result["answer"],
        "summary_hits": result["summary_hits"],
        "txn_hits":     result["txn_hits"],
    })


if __name__ == "__main__":
    main()