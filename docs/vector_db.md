# Vector Database Module

This document covers the three modules under `src/vector_db/` that build, persist, and query the ChromaDB vector index.

```
src/vector_db/
├── store.py           # ChromaDB client, collection management, upsert, query
├── build_index.py     # One-shot script: embed all documents and persist the index
└── retrieval_test.py  # Manual test runner: 13 representative queries with relevance scoring
```

---

## Design overview

Two separate ChromaDB collections are maintained:

| Collection | Contents | Typical size |
|---|---|---|
| `transactions` | One document per sales order row | ~9,994 |
| `summaries` | All aggregated analytical documents | ~350 |

Keeping them separate allows the pipeline to search either collection independently or apply different top-k values to each, and prevents the much larger transaction set from drowning out aggregated summaries during similarity search.

**Embedding model:** `BAAI/bge-small-en-v1.5`
- 33M parameters, 384-dimensional embeddings
- Chosen over the `all-MiniLM-L6-v2` baseline based on a 13-query retrieval comparison: average top-1 cosine distance **0.1796** vs **0.2701**
- Distance metric: cosine (configured via `hnsw:space: cosine`)

---

## store.py

Low-level wrapper around the ChromaDB Python SDK. All other modules import from here; none of them import `chromadb` directly.

### `get_client(persist_dir="./chroma_db") → PersistentClient`

Opens (or creates) a persistent ChromaDB database at the given directory path.  
All data is written to disk automatically — no explicit save call is needed.

---

### `get_embedding_function() → SentenceTransformerEmbeddingFunction`

Returns the shared embedding function wrapping `BAAI/bge-small-en-v1.5`.  
ChromaDB calls this function automatically during both `upsert` (indexing) and `query` (retrieval), ensuring the same model is used for both operations.

---

### `get_collection(client, name, ef) → Collection`

Opens an existing collection or creates it if it does not exist (`get_or_create_collection`).

| Parameter | Description |
|---|---|
| `client` | A `PersistentClient` from `get_client()` |
| `name` | Must be `"transactions"` or `"summaries"` — validated against `_COLLECTIONS` |
| `ef` | Embedding function from `get_embedding_function()` |

The collection is created with `{"hnsw:space": "cosine"}` so all distance scores are in the range `[0, 2]` where 0 = identical and 2 = opposite. In practice, relevant results have distances below 0.5.

Raises `ValueError` if `name` is not one of the two known collections.

---

### `upsert_chunks(collection, chunks) → None`

Writes a list of `Chunk` objects (from `chunker.py`) into a collection using ChromaDB's `upsert` semantics — existing documents with the same ID are updated rather than duplicated.

Internally batches writes at 2,000 documents per call to stay within ChromaDB's recommended limit.

| Chunk field | Mapped to |
|---|---|
| `id` | ChromaDB document ID |
| `text` | The stored document (used for retrieval) |
| `metadata` | Key-value metadata (used for filtering) |

---

### `query(collection, query_text, n_results=5, where=None) → list[dict]`

Runs a similarity search against a collection and returns the top-`n_results` matches.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `query_text` | `str` | Natural-language question or phrase |
| `n_results` | `int` | Number of results to return (default 5) |
| `where` | `dict \| None` | ChromaDB metadata filter (see below) |

**Filter syntax:**

```python
# Single field
{"region": "West"}

# AND across different keys
{"$and": [{"category": "Technology"}, {"year": "2017"}]}
```

> **Important:** `$and` with the same key appearing twice (e.g. `region=West AND region=East`) is logically impossible and always returns zero results. The pipeline's `_parse_where()` detects and drops such filters automatically.

**Returns:** A list of dicts, one per result:

```python
{
    "id":       str,    # document ID
    "text":     str,    # original document content
    "metadata": dict,   # stored key-value metadata
    "distance": float,  # cosine distance (lower = more similar)
}
```

---

## build_index.py

One-shot CLI script that processes the full dataset and writes it into ChromaDB. Must be run once before the RAG pipeline can answer questions.

### Usage

```bash
python src/vector_db/build_index.py [--chunk-size 1000] [--persist-dir <path>] [--smoke-test]
```

| Flag | Default | Description |
|---|---|---|
| `--chunk-size` | `1000` | Character chunk size passed to `chunk_documents()` |
| `--persist-dir` | `src/vector_db/chroma_db` | Directory where ChromaDB writes its files |
| `--smoke-test` | off | Run 4 sample queries after building and print results |

### `build_index(chunk_size=1000, persist_dir="./chroma_db") → None`

Executes the full indexing pipeline in six steps:

```
1. load_data()               → DataFrame
2. build_all_texts()         → (transaction_docs, summary_docs)
3. chunk_documents()         → transaction_chunks, summary_chunks
4. get_client() + get_embedding_function()
5. upsert_chunks()           → "transactions" collection
6. upsert_chunks()           → "summaries" collection
```

Progress and timing are printed to stdout for each step. The transaction upsert is the most time-consuming step (~2–5 minutes depending on hardware, as embeddings are computed locally).

### `_smoke_test(persist_dir) → None`

Runs 4 hardcoded queries after indexing to verify the collections are queryable:

| Collection | Query | Filter |
|---|---|---|
| summaries | `"total sales by year"` | none |
| summaries | `"West region profit margin"` | `{"region": "West"}` |
| summaries | `"Technology category revenue"` | `{"category": "Technology"}` |
| transactions | `"high discount office supplies"` | none |

---

## retrieval_test.py

Manual test runner covering 13 representative queries across all four analytical question types required by the project.

### Usage

```bash
python src/vector_db/retrieval_test.py [--persist-dir <path>] [--n 3]
```

| Flag | Default | Description |
|---|---|---|
| `--persist-dir` | `src/vector_db/chroma_db` | ChromaDB directory to query |
| `--n` | `3` | Number of results to return per query |

### Test cases

| # | Label | Collection | Filter |
|---|---|---|---|
| 01 | Annual sales trend | summaries | — |
| 02 | Seasonal pattern | summaries | — |
| 03 | Profit margin change by year | summaries | — |
| 04 | Top revenue category | summaries | — |
| 05 | Highest margin sub-category | summaries | — |
| 06 | Discount pattern — Technology | summaries | `category=Technology` |
| 07 | Best performing region | summaries | — |
| 08 | West region performance | summaries | `region=West` |
| 09 | Top states by sales | summaries | — |
| 10 | Technology vs Furniture | summaries | — |
| 11 | West vs East profit | summaries | — |
| 12 | High-discount transactions | transactions | — |
| 13 | High-value Technology orders | transactions | `category=Technology` |

### Relevance scoring

Each result is assigned a label based on cosine distance:

| Label | Distance threshold | Meaning |
|---|---|---|
| `HIGH` | < 0.5 | Strong semantic match |
| `MED` | 0.5 – 0.9 | Reasonable match |
| `LOW` | ≥ 0.9 | Weak or unrelated |

A query is counted as **passed** if the top-1 result has distance < 0.9 (HIGH or MED). The final line prints `X/13 queries returned a high/medium relevance top result`.

---

## Metadata filter reference

Valid `where` keys differ by collection:

**`summaries` collection:**

| Key | Example values |
|---|---|
| `type` | `"monthly_summary"`, `"annual_summary"`, `"category_summary"`, `"regional_summary"`, `"seasonal_summary"`, `"statistical_summary"`, `"trend_summary"`, `"comparative_summary"` |
| `year` | `"2014"` – `"2017"` |
| `month` | `"1"` – `"12"` |
| `quarter` | `"Q1"` – `"Q4"` |
| `season` | `"Winter"`, `"Spring"`, `"Summer"`, `"Fall"` |
| `region` | `"West"`, `"East"`, `"Central"`, `"South"` |
| `category` | `"Technology"`, `"Furniture"`, `"Office Supplies"` |
| `sub_category` | e.g. `"Copiers"`, `"Chairs"`, `"Binders"` |
| `segment` | `"Consumer"`, `"Corporate"`, `"Home Office"` |
| `state` | e.g. `"California"`, `"New York"` |

**`transactions` collection:**

| Key | Example values |
|---|---|
| `year` | `"2014"` – `"2017"` |
| `month` | `"1"` – `"12"` |
| `region` | `"West"`, `"East"`, `"Central"`, `"South"` |
| `category` | `"Technology"`, `"Furniture"`, `"Office Supplies"` |
| `sub_category` | e.g. `"Machines"`, `"Tables"` |
| `segment` | `"Consumer"`, `"Corporate"`, `"Home Office"` |
| `state` | e.g. `"California"`, `"Texas"` |

> All metadata values are stored as strings, including numeric fields like `year` and `month`.
