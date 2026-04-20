"""
Retrieval tool definitions for the RAG pipeline.

Each RetrievalTool bundles three things:
  - a JSON Schema describing its parameters (for the LLM to reason about)
  - a human-readable description (used in the system prompt / tool list)
  - a callable implementation that talks to ChromaDB

This means the same tool objects can be:
  - called directly by the pipeline  (direct mode)
  - handed to the LLM as function-call schemas  (agent mode)

Adding a new retrieval strategy only requires adding a new RetrievalTool here;
the pipeline and app need no changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from vector_db.store import query as vdb_query


@dataclass
class RetrievalTool:
    name: str
    description: str
    parameters: dict        # JSON Schema for the function's arguments
    _fn: Callable = field(repr=False)

    def __call__(self, query: str, where: dict | None = None, n_results: int | None = None) -> list[dict]:
        kwargs: dict = {"query_text": query}
        if where is not None:
            kwargs["where"] = where
        if n_results is not None:
            kwargs["n_results"] = n_results
        return self._fn(**kwargs)

    def to_ollama_schema(self) -> dict:
        """Return the tool definition dict expected by ollama.chat(tools=[...])."""
        return {
            "type": "function",
            "function": {
                "name":        self.name,
                "description": self.description,
                "parameters":  self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Parameter schemas
# ---------------------------------------------------------------------------

_WHERE_PROP = {
    "type":        "object",
    "description": (
        "Optional ChromaDB metadata filter (must be a JSON object, not a string). "
        "Single key: {\"category\": \"Technology\"}. "
        "Multi-key AND across DIFFERENT keys: {\"$and\": [{\"region\": \"West\"}, {\"year\": \"2017\"}]}. "
        "IMPORTANT: do NOT use $and to match the same key against two values "
        "(e.g. region=West AND region=East) — that is always empty. "
        "For comparisons (West vs East, Tech vs Furniture), make two separate tool calls instead."
    ),
}

_SUMMARIES_SCHEMA = {
    "type": "object",
    "properties": {
        "query":     {"type": "string",  "description": "Natural language search query"},
        "where":     _WHERE_PROP,
        "n_results": {"type": "integer", "description": "Number of results (default 5)"},
    },
    "required": ["query"],
}

_TRANSACTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "query":     {"type": "string",  "description": "Natural language search query"},
        "where":     _WHERE_PROP,
        "n_results": {"type": "integer", "description": "Number of results (default 3)"},
    },
    "required": ["query"],
}

_SUMMARIES_DESCRIPTION = (
    "Search pre-aggregated analytical documents: monthly/annual/seasonal sales summaries, "
    "category and sub-category revenue, regional and state performance, comparative statistics "
    "(Technology vs Furniture, West vs East), overall trends, and profit/discount analyses. "
    "Use this for most analytical and trend questions. "
    "Valid 'where' keys: type, year, month, quarter, season, region, category, sub_category, segment, state."
)

_TRANSACTIONS_DESCRIPTION = (
    "Search individual transaction rows (one document per sales order). "
    "Use this when you need concrete examples, specific customer/product details, "
    "or to ground claims about discounts and losses at the order level. "
    "Valid 'where' keys: year, month, region, category, sub_category, segment, state."
)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_tools(col_summaries, col_transactions) -> dict[str, RetrievalTool]:
    """
    Bind retrieval tools to live ChromaDB collection objects.

    Calling this once during RAGPipeline.__init__ keeps collections open
    and reused across queries.

    Returns a dict keyed by tool name, in call-priority order.
    """

    def _search_summaries(query_text: str, where: dict | None = None, n_results: int = 5) -> list[dict]:
        return vdb_query(col_summaries, query_text, n_results=n_results, where=where)

    def _search_transactions(query_text: str, where: dict | None = None, n_results: int = 3) -> list[dict]:
        return vdb_query(col_transactions, query_text, n_results=n_results, where=where)

    tools = [
        RetrievalTool(
            name="search_summaries",
            description=_SUMMARIES_DESCRIPTION,
            parameters=_SUMMARIES_SCHEMA,
            _fn=_search_summaries,
        ),
        RetrievalTool(
            name="search_transactions",
            description=_TRANSACTIONS_DESCRIPTION,
            parameters=_TRANSACTIONS_SCHEMA,
            _fn=_search_transactions,
        ),
    ]
    return {t.name: t for t in tools}