# Superstore Sales RAG

A Retrieval-Augmented Generation (RAG) system for analysing the Superstore Sales Dataset (9,994 transactions, 2014–2017). Natural-language questions about sales trends, category performance, regional comparisons, and profit patterns are answered by retrieving relevant data from a ChromaDB vector database and passing it to a local or cloud LLM.

**Course:** Data Warehousing and Business Intelligence — University of Helsinki

---

## Features

- **Two retrieval modes**
  - *Direct* — retrieves top-k documents from both collections, builds a context block, and calls the LLM once
  - *Agent* — agentic tool-call loop: the LLM decides which tools to call, with what queries and filters, until it has enough data to answer
- **Streaming output** — tokens stream to the UI in real time (direct mode)
- **Live agent trace** — each tool call and result is displayed as it happens (agent mode)
- **Two LLM backends** — Ollama (local) and any OpenAI-compatible API (OpenAI, Groq, Together AI, LM Studio, …)
- **Metadata filtering** — filter retrieval by category, region, year, and more from the sidebar
- **Conversation memory** — last 6 turns kept in context automatically

---

## Project structure

```
sales-rag/
├── data/
│   └── Superstore.csv          # raw dataset (not committed)
├── src/
│   ├── data_processing/
│   │   ├── loader.py           # load and inspect the CSV
│   │   ├── text_converter.py   # convert rows/aggregations to text documents
│   │   └── chunker.py          # split documents into fixed-size chunks
│   ├── vector_db/
│   │   ├── store.py            # ChromaDB wrapper (client, upsert, query)
│   │   ├── build_index.py      # one-shot indexing script
│   │   └── retrieval_test.py   # 13-query retrieval test suite
│   ├── llm/
│   │   ├── base.py             # LLMProvider abstract base class
│   │   ├── ollama.py           # Ollama provider
│   │   ├── openai.py           # OpenAI-compatible provider
│   │   └── __init__.py         # make_llm() factory
│   └── rag/
│       ├── tools.py            # RetrievalTool definitions and JSON schemas
│       ├── pipeline.py         # RAGPipeline (direct + agent modes)
│       └── app.py              # Streamlit chat interface
├── docs/
│   ├── data_processing.md
│   ├── vector_db.md
│   ├── pipeline.md
│   └── contribution.md
├── .env.example                # configuration template
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Clone and install dependencies

```bash
git clone https://github.com/thezmmm/sales-rag.git
cd sales-rag
pip install -r requirements.txt
```

### 2. Place the dataset

Download the Superstore Sales Dataset from Kaggle (`vivek468/superstore-dataset-final`) and save it as:

```
data/Superstore.csv
```

### 3. Configure the environment

Copy `.env.example` to `.env` and fill in the values for your chosen LLM provider:

```bash
cp .env.example .env
```

**Using Ollama (local, free):**
```ini
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
```
Pull the model first: `ollama pull llama3.2:3b`

**Using an OpenAI-compatible API:**
```ini
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.groq.com/openai/v1   # omit for official OpenAI
```

### 4. Build the vector index

This step embeds all documents and persists them to ChromaDB. It only needs to be run once.

```bash
python src/vector_db/build_index.py
```

Options:
```
--chunk-size  INT   Character chunk size (default: 1000)
--persist-dir PATH  ChromaDB directory (default: src/vector_db/chroma_db)
--smoke-test        Run 4 sample queries after building
```

### 5. Launch the app

```bash
streamlit run src/rag/app.py
```

The app will be available at `http://localhost:8501`.

---

## Configuration reference

All runtime settings are read from `.env`. Credentials are never exposed in the UI.

| Key | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | `ollama` or `openai` | `ollama` |
| `OLLAMA_MODEL` | Ollama model tag | `llama3.2:3b` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OPENAI_MODEL` | Model name for OpenAI-compatible API | `gpt-4o-mini` |
| `OPENAI_API_KEY` | API key | — |
| `OPENAI_BASE_URL` | Custom endpoint base URL | `https://api.openai.com/v1` |
| `CHROMA_DB_PATH` | ChromaDB persistence directory | `./src/vector_db/chroma_db` |
| `EMBEDDING_MODEL` | Sentence-transformers model name | `BAAI/bge-small-en-v1.5` |
| `DATA_PATH` | Path to the Superstore CSV | `./data/superstore.csv` |

---

## Using the pipeline programmatically

```python
from src.rag.pipeline import RAGPipeline

# Reads provider/model/credentials from .env automatically
rag = RAGPipeline(mode="direct")

result = rag.ask("Which region has the highest profit margin?")
print(result["answer"])

# Streaming (direct mode)
for chunk in rag.stream("What is the sales trend from 2014 to 2017?"):
    print(chunk, end="", flush=True)

# Agent mode with live events
rag_agent = RAGPipeline(mode="agent")
for event in rag_agent.stream_agent("Compare Technology and Furniture profitability."):
    if event["type"] == "tool_call":
        print(f"Calling {event['name']}: {event['query']}")
    elif event["type"] == "answer":
        print(event["text"])
```

---

## Tech stack

| Component | Tool |
|---|---|
| Language | Python 3.9+ |
| Vector database | ChromaDB |
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim, cosine) |
| LLM — local | Ollama (llama3.2, mistral, …) |
| LLM — cloud | OpenAI API / Groq / Together AI / any compatible endpoint |
| Frontend | Streamlit |
| Data processing | Pandas, NumPy |

---

## Documentation

| Document | Description |
|---|---|
| [`docs/data_processing.md`](docs/data_processing.md) | Data loading, text conversion, chunking |
| [`docs/vector_db.md`](docs/vector_db.md) | ChromaDB setup, indexing, retrieval, metadata filters |
| [`docs/pipeline.md`](docs/pipeline.md) | LLM abstraction, RAG pipeline, direct and agent modes |
| [`docs/contribution.md`](docs/contribution.md) | Team division of work |
