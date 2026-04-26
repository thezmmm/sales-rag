"""
Retrieval accuracy test — evaluates ChromaDB retrieval quality with structured metrics.

Metrics computed per query:
  - Hit@1   : top result contains all expected keywords
  - Hit@3   : any of the top-3 results contains all expected keywords
  - MRR     : mean reciprocal rank of first keyword-matching result (within top-k)
  - dist@1  : cosine distance of the top result (lower = more similar)

Usage:
    python src/vector_db/retrieval_test.py [--n 5]

Default persist directory: src/vector_db/chroma_db
"""

import argparse
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_HERE)
_DEFAULT_DB = os.path.join(_HERE, "chroma_db")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from vector_db.store import get_client, get_embedding_function, get_collection, query

# ---------------------------------------------------------------------------
# Test cases
# Fields:
#   label            – human-readable name
#   collection       – "transactions" or "summaries"
#   query            – the natural-language query string
#   where            – optional ChromaDB metadata filter dict
#   expected_keywords – list of lowercase strings; ALL must appear in text for a hit
# ---------------------------------------------------------------------------
TEST_CASES = [
    # ── Trend ────────────────────────────────────────────────────────────────
    {
        "label": "Annual sales trend",
        "collection": "summaries",
        "query": "sales trend over the years",
        "where": None,
        "expected_keywords": ["sales", "2017"],
    },
    {
        "label": "Seasonal / quarterly pattern",
        "collection": "summaries",
        "query": "which quarter or season has the highest sales",
        "where": None,
        "expected_keywords": ["quarter", "sales"],
    },
    {
        "label": "Profit margin change by year",
        "collection": "summaries",
        "query": "profit margin change over years",
        "where": None,
        "expected_keywords": ["profit", "margin"],
    },

    # ── Category ─────────────────────────────────────────────────────────────
    {
        "label": "Top revenue category",
        "collection": "summaries",
        "query": "which category generates the most revenue",
        "where": None,
        "expected_keywords": ["technology", "sales"],
    },
    {
        "label": "Highest-margin sub-category",
        "collection": "summaries",
        "query": "sub-category with highest profit margin",
        "where": None,
        "expected_keywords": ["profit", "margin"],
    },
    {
        "label": "Discount pattern — Technology",
        "collection": "summaries",
        "query": "average discount rate in Technology category",
        "where": {"category": "Technology"},
        "expected_keywords": ["technology", "discount"],
    },
    {
        "label": "Furniture sales summary",
        "collection": "summaries",
        "query": "furniture total sales and profit",
        "where": {"category": "Furniture"},
        "expected_keywords": ["furniture", "sales"],
    },

    # ── Regional ─────────────────────────────────────────────────────────────
    {
        "label": "Best performing region",
        "collection": "summaries",
        "query": "which region has the best sales and profit",
        "where": None,
        "expected_keywords": ["west", "sales"],
    },
    {
        "label": "West region performance",
        "collection": "summaries",
        "query": "West region total sales and profit margin",
        "where": {"region": "West"},
        "expected_keywords": ["west", "sales"],
    },
    {
        "label": "Top states by sales",
        "collection": "summaries",
        "query": "top states ranked by total sales",
        "where": None,
        "expected_keywords": ["california", "sales"],
    },
    {
        "label": "South region summary",
        "collection": "summaries",
        "query": "South region sales and profit",
        "where": {"region": "South"},
        "expected_keywords": ["south", "sales"],
    },

    # ── Comparative ───────────────────────────────────────────────────────────
    {
        "label": "Technology vs Furniture",
        "collection": "summaries",
        "query": "compare Technology and Furniture sales trends",
        "where": None,
        "expected_keywords": ["technology", "furniture"],
    },
    {
        "label": "West vs East profit",
        "collection": "summaries",
        "query": "West region vs East region profit comparison",
        "where": None,
        "expected_keywords": ["west", "east"],
    },

    # ── Transaction-level ────────────────────────────────────────────────────
    {
        "label": "High-discount loss transaction",
        "collection": "transactions",
        "query": "large discount resulted in a loss negative profit",
        "where": None,
        "expected_keywords": ["discount", "loss"],
    },
    {
        "label": "High-value Technology order",
        "collection": "transactions",
        "query": "expensive Technology product high sales order",
        "where": {"category": "Technology"},
        "expected_keywords": ["technology", "sales"],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _keywords_match(text: str, keywords: list[str]) -> bool:
    """Return True if ALL keywords appear (case-insensitive) in text."""
    lower = text.lower()
    return all(kw.lower() in lower for kw in keywords)


def _reciprocal_rank(results: list[dict], keywords: list[str]) -> float:
    """Return 1/rank of the first result whose text matches all keywords, else 0."""
    for rank, r in enumerate(results, 1):
        if _keywords_match(r["text"], keywords):
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_tests(persist_dir: str, n_results: int) -> None:
    client = get_client(persist_dir)
    ef = get_embedding_function()
    collections = {
        "transactions": get_collection(client, "transactions", ef),
        "summaries":    get_collection(client, "summaries",    ef),
    }

    hit1_count   = 0   # keyword match in top-1
    hit3_count   = 0   # keyword match in top-3
    rr_sum       = 0.0 # reciprocal rank accumulator
    dist1_sum    = 0.0 # sum of top-1 distances
    dist_pass    = 0   # top-1 distance < 0.5 (high similarity)

    per_case = []

    for i, tc in enumerate(TEST_CASES, 1):
        label    = tc["label"]
        col_name = tc["collection"]
        q        = tc["query"]
        where    = tc["where"]
        keywords = tc["expected_keywords"]

        col     = collections[col_name]
        results = query(col, q, n_results=n_results, where=where)

        hit1  = bool(results) and _keywords_match(results[0]["text"], keywords)
        hit3  = any(_keywords_match(r["text"], keywords) for r in results[:3])
        rr    = _reciprocal_rank(results, keywords)
        dist1 = results[0]["distance"] if results else 1.0

        hit1_count  += int(hit1)
        hit3_count  += int(hit3)
        rr_sum      += rr
        dist1_sum   += dist1
        dist_pass   += int(dist1 < 0.5)

        per_case.append((i, label, hit1, hit3, rr, dist1, results, q, where, keywords))

    n = len(TEST_CASES)
    mrr        = rr_sum / n
    avg_dist1  = dist1_sum / n

    # ── Per-case detail ──────────────────────────────────────────────────────
    for (i, label, hit1, hit3, rr, dist1, results, q, where, keywords) in per_case:
        filter_str = f"  filter : {where}" if where else ""
        kw_str = ", ".join(f'"{k}"' for k in keywords)

        h1 = "PASS" if hit1 else "FAIL"
        h3 = "PASS" if hit3 else "FAIL"

        print(f"\n{'='*70}")
        print(f"[{i:02d}] {label}")
        print(f"  query   : \"{q}\"")
        print(f"  keywords: [{kw_str}]")
        print(f"  source  : {results[0]['metadata'].get('source', results[0]['metadata']) if results else 'n/a'}{filter_str}")
        print(f"  Hit@1={h1}  Hit@3={h3}  RR={rr:.3f}  dist@1={dist1:.4f}")

        if results:
            print()
        for rank, r in enumerate(results, 1):
            dist = r["distance"]
            level = "HIGH" if dist < 0.5 else "MED" if dist < 0.9 else "LOW"
            match = "MATCH" if _keywords_match(r["text"], keywords) else "-----"
            snippet = r["text"][:180].strip().encode("ascii", errors="replace").decode("ascii")
            print(f"  #{rank} [{level}][{match}] dist={dist:.4f}  id={r['id']}")
            print(f"      {snippet}")
            if rank < len(results):
                print()

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ACCURACY SUMMARY")
    print(f"{'='*70}")
    print(f"  Total test cases : {n}")
    print(f"  Hit@1 (keyword)  : {hit1_count}/{n}  ({hit1_count/n*100:.1f}%)")
    print(f"  Hit@3 (keyword)  : {hit3_count}/{n}  ({hit3_count/n*100:.1f}%)")
    print(f"  MRR              : {mrr:.4f}")
    print(f"  Avg dist@1       : {avg_dist1:.4f}")
    print(f"  High-sim (d<0.5) : {dist_pass}/{n}  ({dist_pass/n*100:.1f}%)")
    print(f"{'='*70}")

    # ── Per-category breakdown ───────────────────────────────────────────────
    categories = {
        "Trend":       [0, 1, 2],
        "Category":    [3, 4, 5, 6],
        "Regional":    [7, 8, 9, 10],
        "Comparative": [11, 12],
        "Transaction": [13, 14],
    }
    print("\nPer-category breakdown:")
    for cat_name, indices in categories.items():
        cases = [per_case[i] for i in indices]
        c_hit1 = sum(int(c[2]) for c in cases)
        c_hit3 = sum(int(c[3]) for c in cases)
        c_mrr  = sum(c[4] for c in cases) / len(cases)
        c_dist = sum(c[5] for c in cases) / len(cases)
        m = len(cases)
        print(f"  {cat_name:<12} Hit@1={c_hit1}/{m} ({c_hit1/m*100:.0f}%)  "
              f"Hit@3={c_hit3}/{m} ({c_hit3/m*100:.0f}%)  "
              f"MRR={c_mrr:.3f}  avg_dist={c_dist:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", default=_DEFAULT_DB)
    parser.add_argument("--n", type=int, default=5, help="Results per query (default: 5)")
    args = parser.parse_args()

    run_tests(args.persist_dir, args.n)