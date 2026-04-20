"""
RAG demo — runs 5+ analysis queries covering all graded categories.

Usage:
    python src/rag/demo.py [--model llama3.2:3b] [--persist-dir <path>]

Categories covered:
  Trend    — annual sales trend, seasonal patterns, profit margin changes
  Category — top revenue category, highest-margin sub-categories, discount patterns
  Regional — best-performing region/state/city
  Comparative — Technology vs Furniture, West vs East profit
"""

import argparse
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_HERE)
_DEFAULT_DB = os.path.join(_SRC_DIR, "vector_db", "chroma_db")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from rag.pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Query definitions: (label, category, question, where_filter)
# ---------------------------------------------------------------------------
DEMO_QUERIES = [
    # ── Trend ───────────────────────────────────────────────────────────────
    (
        "Annual Sales Trend (2014–2017)",
        "Trend",
        "What is the overall sales trend from 2014 to 2017? "
        "How did total sales and profit margin change each year?",
        None,
    ),
    (
        "Seasonal Patterns",
        "Trend",
        "Which season or quarter consistently has the highest sales? "
        "Is there a clear seasonal pattern?",
        None,
    ),
    (
        "Profit Margin Changes Over Years",
        "Trend",
        "How did profit margins change year over year from 2014 to 2017? "
        "In which year was the margin highest?",
        None,
    ),

    # ── Category ────────────────────────────────────────────────────────────
    (
        "Top Revenue Category",
        "Category",
        "Which product category generates the most total revenue? "
        "Provide revenue figures for all categories.",
        None,
    ),
    (
        "Highest-Margin Sub-Categories",
        "Category",
        "Which sub-categories have the highest profit margins? "
        "Which sub-categories are the least profitable or have negative margins?",
        None,
    ),
    (
        "Discount Patterns in Technology",
        "Category",
        "What is the discount pattern in the Technology category? "
        "How does the average discount compare to other categories?",
        {"category": "Technology"},
    ),

    # ── Regional ────────────────────────────────────────────────────────────
    (
        "Best-Performing Region",
        "Regional",
        "Which region has the best sales and profit performance? "
        "Rank all four regions by total sales and profit margin.",
        None,
    ),
    (
        "Top States by Sales",
        "Regional",
        "Which states have the highest total sales revenue? "
        "List the top 5 states.",
        None,
    ),

    # ── Comparative ─────────────────────────────────────────────────────────
    (
        "Technology vs Furniture Trends",
        "Comparative",
        "Compare the sales trends of Technology and Furniture categories "
        "from 2014 to 2017. Which grew faster and which is more profitable?",
        None,
    ),
    (
        "West vs East Profit Comparison",
        "Comparative",
        "Compare the West and East regions in terms of total profit and "
        "profit margin. Which region consistently outperforms the other?",
        None,
    ),
]


def run_demo(model: str, persist_dir: str) -> None:
    print(f"Initializing RAG pipeline (model={model})...")
    rag = RAGPipeline(persist_dir=persist_dir, model=model)

    category_counts: dict[str, int] = {}

    for i, (label, category, question, where) in enumerate(DEMO_QUERIES, 1):
        category_counts[category] = category_counts.get(category, 0) + 1
        print(f"\n{'=' * 72}")
        print(f"[{i:02d}] [{category}] {label}")
        print(f"Q: {question}")
        if where:
            print(f"Filter: {where}")
        print("-" * 72)

        result = rag.ask(question, summary_where=where, use_memory=False)

        print(f"\nAnswer:\n{result['answer']}")

        print(f"\nSummary sources ({len(result['summary_hits'])} retrieved):")
        for src in result["summary_hits"][:3]:
            dist_label = "HIGH" if src["distance"] < 0.5 else "MED"
            snippet = src["text"][:120].replace("\n", " ")
            print(f"  [{dist_label} dist={src['distance']:.4f}] {src['id']}: {snippet}...")

        if result["txn_hits"]:
            print(f"\nTransaction examples ({len(result['txn_hits'])} retrieved):")
            for src in result["txn_hits"]:
                dist_label = "HIGH" if src["distance"] < 0.5 else "MED"
                snippet = src["text"][:120].replace("\n", " ")
                print(f"  [{dist_label} dist={src['distance']:.4f}] {src['id']}: {snippet}...")

    print(f"\n{'=' * 72}")
    print(f"Demo complete — {len(DEMO_QUERIES)} queries across categories: "
          + ", ".join(f"{k}×{v}" for k, v in category_counts.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="llama3.2:3b",
                        help="Ollama model name (default: llama3.2:3b)")
    parser.add_argument("--persist-dir", default=_DEFAULT_DB,
                        help="Path to ChromaDB directory")
    args = parser.parse_args()

    run_demo(args.model, args.persist_dir)