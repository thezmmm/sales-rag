# AI Tool Usage Record

**Project:** RAG-Based Sales Data Analysis
**Course:** Data Warehousing and Business Intelligence — University of Helsinki

---

## Tools Used

| Tool | Access | Purpose |
|---|---|---|
| Claude Sonnet 4.6 | Claude Code CLI | Code generation, debugging, documentation |
| ChatGPT (gpt-3.5-turbo) | OpenAI-compatible API | LLM backend for RAG query answering |

---

## Usage Log

### Entry 001 — Project initialisation

- **Tool**: Claude Sonnet 4.6
- **AI did**: Generated `CLAUDE.md`, `.gitignore`, and `AI_TOOL_USAGE.md` from the project PDF
- **Student contribution**: Verified content against the original PDF; confirmed gitignore rules fit the tech stack

---

### Entry 002 — Data loading

- **Tool**: Claude Sonnet 4.6
- **AI did**: Implemented `loader.py` — CSV loading with encoding handling, date column parsing, and a dataset overview utility
- **Student contribution**: Specified which columns needed type conversion and what format downstream code required

---

### Entry 003 — Data processing

- **Tool**: Claude Sonnet 4.6
- **AI did**: Implemented `text_converter.py` (13 builder functions: transaction sentences, monthly/annual/seasonal summaries, cross-dimension aggregations, statistical overviews, trend and comparative documents) and `chunker.py` (character-level chunking with overlap, three sizes: 500 / 1000 / 2000)
- **Student contribution**: Designed the document taxonomy and dimension combinations needed for the analysis questions; decided to separate transaction docs from summary docs; chose chunk sizes and overlap strategy

---

### Entry 004 — Vector database setup

- **Tool**: Claude Sonnet 4.6
- **AI did**: Implemented `store.py` (ChromaDB client, two-collection design, batched upsert, similarity search with metadata filters) and `build_index.py` (end-to-end indexing script with progress output and smoke test)
- **Student contribution**: Decided on the two-collection architecture (transactions vs summaries); learned ChromaDB filter syntax through the conversation; determined which metadata fields to store

---

### Entry 005 — Retrieval testing

- **Tool**: Claude Sonnet 4.6
- **AI did**: Implemented `retrieval_test.py` with 13 test queries across all four required analysis categories; outputs ranked results with distance scores and HIGH / MED / LOW relevance labels
- **Student contribution**: Defined the test cases based on the assignment's required analysis questions; evaluated whether retrieved documents were semantically correct

---

### Entry 006 — Embedding model comparison

- **Tool**: Claude Sonnet 4.6
- **AI did**: Implemented `compare_embeddings.py` to benchmark four candidate models by building a full ChromaDB index for each and running all 13 retrieval test cases
- **Problems encountered**: `all-mpnet-base-v2` took 574s to index on CPU (5× slower than baseline)
- **Running results**: All 4 models achieved 13/13 relevant results. `BAAI/bge-small-en-v1.5` achieved the best avg top-1 cosine distance (0.1796 vs baseline 0.2701) and was selected as the production model
- **Student contribution**: Chose which models to benchmark; interpreted per-query distance results; made the final selection based on the quality/speed tradeoff

---

### Entry 007 — RAG pipeline

- **Tool**: Claude Sonnet 4.6
- **AI did**: Implemented `tools.py` (two retrieval tools as first-class objects with JSON Schema for LLM function-calling), `pipeline.py` (direct mode and agent mode with agentic tool-call loop, streaming, conversation memory), `demo.py`, and `app.py` (Streamlit chat UI with provider/model selector, metadata filters, live agent trace)
- **Student contribution**: Designed the overall pipeline architecture — two operating modes, two-collection retrieval strategy, tool-as-object design; directed iterative refinements to context structure, prompt engineering, and agent loop behaviour

---

### Entry 008 — Documentation

- **Tool**: Claude Sonnet 4.6
- **AI did**: Wrote `docs/data_processing.md`, `docs/vector_db.md`, `docs/pipeline.md`, `docs/contribution.md`, and `README.md` by reading the full source code of each module
- **Student contribution**: Reviewed all documents for accuracy; updated member names in `contribution.md`; simplified detail level; verified README setup steps against the actual environment

---

### Entry 009 — RAG query answering (runtime)

- **Tool**: ChatGPT (gpt-3.5-turbo) via OpenAI-compatible API
- **AI did**: Answered analytical questions about the Superstore dataset at runtime — served as the LLM backend for the RAG pipeline in both direct and agent modes
- **Student contribution**: Configured the API endpoint and model in `.env`; evaluated response quality across multiple question types (trend, category, regional, comparative); identified cases where the model produced incorrect or incomplete answers

---

## Overall Student Contribution

- Architectural decisions: chunking strategy, embedding model selection, two-collection vector DB design, RAG pipeline modes
- Debugging and integration of AI-generated components into a working end-to-end system
- Retrieval quality optimisation: identified semantic gaps between query language and document text; added targeted summary documents; validated retrieved results against intended analytical questions
- Testing and evaluation of end-to-end RAG output quality across both pipeline modes

---

> This section does not count toward the technical report page limit (per course policy).
