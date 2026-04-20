"""
RAG pipeline: query → retrieval → prompt engineering → LLM → response.

Two operating modes
-------------------
direct  (default)
    The pipeline calls both retrieval tools unconditionally, builds a
    two-section context (summaries then transactions), and calls the LLM once.
    Predictable, fast, easy to debug.

agent
    The pipeline passes tool schemas to the LLM and runs an agentic loop:
    the model decides which tools to call, with what queries and filters,
    until it has enough context to answer.  Requires an Ollama model with
    function/tool-calling support (llama3.2, mistral, …).

Switching modes is a single constructor parameter.  The retrieval logic lives
entirely in tools.py — the pipeline never touches ChromaDB directly.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_HERE)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import json
import ollama
from vector_db.store import get_client, get_embedding_function, get_collection
from rag.tools import RetrievalTool, make_tools

_DEFAULT_DB = os.path.join(_SRC_DIR, "vector_db", "chroma_db")

_SYSTEM_DIRECT = """\
You are a business intelligence analyst specialized in retail sales data.
You answer questions about the Superstore dataset (2014–2017, ~10,000 transactions).
Base your answers strictly on the provided context snippets.
Be concise, cite specific numbers, and clearly label years, categories, or regions.
If the context lacks sufficient data to answer, say so explicitly."""

_SYSTEM_AGENT = """\
You are a business intelligence analyst with access to a Superstore sales database (2014–2017).
Use the available tools to retrieve the data you need before answering.
Guidelines:
- Call search_summaries for aggregate statistics, trends, rankings, and comparisons.
- Call search_transactions for concrete order examples or to verify individual-level claims.
- You may call tools multiple times with different queries or filters.
- Once you have sufficient data, give a concise, number-backed answer."""


# ---------------------------------------------------------------------------
# Context formatting (used in direct mode)
# ---------------------------------------------------------------------------

def _parse_where(where) -> dict | None:
    """
    Normalise the 'where' argument coming from an LLM tool call.

    The LLM sometimes serialises the filter as a JSON string instead of a
    nested dict.  Also guards against logically impossible filters such as
    {"$and": [{"region": "West"}, {"region": "East"}]} — matching a single
    field against two distinct values with AND is always empty; drop the
    filter and let the tool return unfiltered results instead.
    """
    if where is None:
        return None
    if isinstance(where, str):
        try:
            where = json.loads(where)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(where, dict):
        return None

    # Detect contradictory $and: same key appears with two different values
    if "$and" in where:
        clauses = where["$and"]
        if isinstance(clauses, list):
            seen: dict[str, set] = {}
            for clause in clauses:
                if isinstance(clause, dict):
                    for k, v in clause.items():
                        seen.setdefault(k, set()).add(str(v))
            if any(len(vals) > 1 for vals in seen.values()):
                return None  # contradictory — drop filter entirely

    return where


def _fmt_hits(hits: list[dict]) -> str:
    return "\n\n".join(f"[{h['id']}] {h['text']}" for h in hits)


def _build_direct_context(summary_hits: list[dict], txn_hits: list[dict]) -> str:
    parts: list[str] = []
    if summary_hits:
        parts.append("### Aggregated summaries\n" + _fmt_hits(summary_hits))
    if txn_hits:
        parts.append("### Transaction-level examples\n" + _fmt_hits(txn_hits))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG pipeline for Superstore sales analysis.

    Parameters
    ----------
    persist_dir : path to the ChromaDB directory built by build_index.py
    model       : Ollama model name (e.g. 'llama3.2:3b', 'mistral')
    mode        : 'direct' or 'agent' — see module docstring
    n_summary   : default top-k for search_summaries  (direct mode)
    n_txn       : default top-k for search_transactions (direct mode)
    """

    def __init__(
        self,
        persist_dir: str = _DEFAULT_DB,
        model: str = "llama3.2:3b",
        mode: str = "direct",
        n_summary: int = 5,
        n_txn: int = 2,
    ):
        if mode not in ("direct", "agent"):
            raise ValueError(f"mode must be 'direct' or 'agent', got {mode!r}")

        self.model = model
        self.mode = mode
        self.n_summary = n_summary
        self.n_txn = n_txn

        client = get_client(persist_dir)
        ef = get_embedding_function()
        self.tools: dict[str, RetrievalTool] = make_tools(
            get_collection(client, "summaries",    ef),
            get_collection(client, "transactions", ef),
        )

        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Public retrieval helpers (delegate to tool objects)
    # ------------------------------------------------------------------

    def retrieve_summaries(
        self,
        query: str,
        where: dict | None = None,
        n_results: int | None = None,
    ) -> list[dict]:
        return self.tools["search_summaries"](
            query=query, where=where,
            n_results=n_results or self.n_summary,
        )

    def retrieve_transactions(
        self,
        query: str,
        where: dict | None = None,
        n_results: int | None = None,
    ) -> list[dict]:
        return self.tools["search_transactions"](
            query=query, where=where,
            n_results=n_results or self.n_txn,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        summary_where: dict | None = None,
        txn_where: dict | None = None,
        include_transactions: bool = True,
        use_memory: bool = True,
    ) -> dict:
        """
        Answer a question using the configured retrieval mode.

        Parameters
        ----------
        question             : natural-language question
        summary_where        : metadata filter passed to search_summaries (direct mode)
        txn_where            : metadata filter passed to search_transactions (direct mode)
        include_transactions : if False, skip transaction retrieval (direct mode only)
        use_memory           : prepend recent conversation history to the prompt

        Returns
        -------
        dict:
          answer        : LLM response text
          summary_hits  : chunks from 'summaries' collection
          txn_hits      : chunks from 'transactions' collection
          mode          : 'direct' or 'agent'
        """
        if self.mode == "direct":
            answer, summary_hits, txn_hits = self._run_direct(
                question, summary_where, txn_where, include_transactions, use_memory,
            )
        else:
            answer, summary_hits, txn_hits = self._run_agent(question, use_memory)

        return {
            "answer":       answer,
            "summary_hits": summary_hits,
            "txn_hits":     txn_hits,
            "mode":         self.mode,
        }

    # ------------------------------------------------------------------
    # Direct mode
    # ------------------------------------------------------------------

    def _run_direct(
        self,
        question: str,
        summary_where: dict | None,
        txn_where: dict | None,
        include_transactions: bool,
        use_memory: bool,
    ) -> tuple[str, list[dict], list[dict]]:
        summary_hits = self.retrieve_summaries(question, where=summary_where)
        txn_hits = (
            self.retrieve_transactions(question, where=txn_where)
            if include_transactions else []
        )
        context = _build_direct_context(summary_hits, txn_hits)

        messages = [{"role": "system", "content": _SYSTEM_DIRECT}]
        if use_memory and self.history:
            messages.extend(self.history[-6:])
        messages.append({
            "role":    "user",
            "content": f"Context (from Superstore sales database):\n{context}\n\nQuestion: {question}",
        })

        answer = ollama.chat(model=self.model, messages=messages).message.content
        self._update_history(question, answer, use_memory)
        return answer, summary_hits, txn_hits

    # ------------------------------------------------------------------
    # Agent mode — agentic tool-call loop
    # ------------------------------------------------------------------

    def _run_agent(
        self,
        question: str,
        use_memory: bool,
    ) -> tuple[str, list[dict], list[dict]]:
        tool_schemas = [t.to_ollama_schema() for t in self.tools.values()]
        summary_hits: list[dict] = []
        txn_hits:     list[dict] = []

        messages: list = [{"role": "system", "content": _SYSTEM_AGENT}]
        if use_memory and self.history:
            messages.extend(self.history[-6:])
        messages.append({"role": "user", "content": question})

        while True:
            response = ollama.chat(model=self.model, messages=messages, tools=tool_schemas)
            msg = response.message

            if not msg.tool_calls:
                answer = msg.content
                self._update_history(question, answer, use_memory)
                return answer, summary_hits, txn_hits

            # Append assistant turn (with tool_calls) so the model sees its own decisions
            messages.append(msg)

            for tc in msg.tool_calls:
                name = tc.function.name
                args = tc.function.arguments  # dict from ollama SDK

                if name not in self.tools:
                    result_text = f"Error: unknown tool '{name}'."
                else:
                    hits = self.tools[name](
                        query=args.get("query", ""),
                        where=_parse_where(args.get("where")),
                        n_results=args.get("n_results"),
                    )
                    result_text = _fmt_hits(hits) if hits else "No results found."
                    if name == "search_summaries":
                        summary_hits.extend(hits)
                    else:
                        txn_hits.extend(hits)

                messages.append({"role": "tool", "content": result_text})

    # ------------------------------------------------------------------
    # Conversation memory
    # ------------------------------------------------------------------

    def _update_history(self, question: str, answer: str, use_memory: bool) -> None:
        if use_memory:
            self.history.append({"role": "user",      "content": question})
            self.history.append({"role": "assistant",  "content": answer})

    def reset_memory(self) -> None:
        self.history.clear()