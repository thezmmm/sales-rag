# AI Tool Usage Record

Project: RAG-Based Sales Data Analysis

Course: Data Warehousing and Business Intelligence — University of Helsinki

---

## 1. Tools Used

| Tool | Version / Access | Purpose |
|------|-----------------|---------|
| Claude (Anthropic) | Claude Sonnet 4.6 via Claude Code CLI | Project setup, code generation, debugging |
| | | |
| | | |

---

## 2. Usage Log

---

### Entry 001 — Initialize project

- **Tool**: Claude Sonnet 4.6
- **Description**: Build project intro docs — `CLAUDE.md` (project overview from PDF), `.gitignore` (Python/ML/ChromaDB), `AI_TOOL_USAGE.md` (this file)
- **Student Contribution**: Verified CLAUDE.md content against the original PDF; confirmed gitignore rules fit the tech stack

---

### Entry 002 — Data loading (`loader.py`)

- **Tool**: Claude Sonnet 4.6
- **Description**: Generated `loader.py` to load and clean the Superstore CSV — date parsing, column standardization, dtype enforcement
- **Student Contribution**: Specified which columns needed parsing and what clean output format downstream code required

---

### Entry 003 — Data processing (`text_converter.py`, `chunker.py`)

- **Tool**: Claude Sonnet 4.6
- **Description**: AI implemented text conversion (transaction sentences, monthly/annual/seasonal summaries, cross-dimension aggregations) and chunking (character-level split with overlap, three sizes: 500 / 1000 / 2000 chars)
- **Student Contribution**: Designed the document taxonomy and dimension combinations needed for the analysis questions; decided to separate transaction docs from summary docs; chose chunk sizes and overlap strategy

---

### Entry 004 — Vector database setup (`store.py`, `build_index.py`)

- **Tool**: Claude Sonnet 4.6
- **Description**: AI explained ChromaDB concepts (collections, embedding functions, upsert, metadata filtering, distance metrics) and helped set up `store.py` (client initialization, two-collection design, batch upsert, similarity search with `where` filters) and `build_index.py` (end-to-end index building script with progress output and smoke test)
- **Student Contribution**: Learned how vector databases work through the conversation; decided on the two-collection architecture (transactions vs summaries); understood the ChromaDB filter syntax and how metadata fields enable filtered retrieval

---

### Entry 005 — Retrieval test (`retrieval_test.py`)

- **Tool**: Claude Sonnet 4.6
- **Description**: AI wrote `retrieval_test.py` with 13 test queries covering all four required analysis categories (trend, category, regional, comparative) plus transaction-level retrieval; outputs ranked results with distance scores and relevance labels
- **Student Contribution**: Defined the test cases based on the assignment's required analysis questions; evaluated whether retrieved results were semantically correct

---

### Entry 006 — Embedding model comparison (`compare_embeddings.py`, `docs/embedding_comparison.md`)

- **Tool**: Claude Sonnet 4.6
- **Description**: AI wrote `compare_embeddings.py` to benchmark four candidate embedding models by building a full ChromaDB index for each, running all 13 retrieval test cases, and generating a Markdown report with per-query distance tables and divergence analysis. Models tested: `all-MiniLM-L6-v2` (baseline), `multi-qa-MiniLM-L6-cos-v1`, `all-mpnet-base-v2`, `BAAI/bge-small-en-v1.5`.
- **Problems encountered**: `all-mpnet-base-v2` took 574s to index on CPU (5× slower than baseline); `bge-small-en-v1.5` required a first-time model download from HuggingFace.
- **Running results**: All 4 models achieved 13/13 HIGH relevance. `bge-small-en-v1.5` achieved the best avg top-1 cosine distance (0.1796), outperforming even the 3.3× larger `mpnet-base-v2` (0.2656). Production model updated accordingly.
- **Student Contribution**: Decided which models to include; interpreted per-query divergence results; made the final model selection based on the quality/speed tradeoff

---

### Entry 007 — RAG pipeline (`src/rag/tools.py`, `src/rag/pipeline.py`, `src/rag/demo.py`, `src/rag/app.py`)

- **Tool**: Claude Sonnet 4.6
- **Description**: AI assisted in implementing the full RAG pipeline. `tools.py` defines two retrieval tools (`search_summaries`, `search_transactions`) as first-class objects carrying both a callable implementation and a JSON Schema, so they can be invoked either by the pipeline directly or handed to the LLM as function-calling schemas. `pipeline.py` implements two operating modes: **direct** (pipeline calls both tools, builds a two-section context, calls LLM once) and **agent** (LLM receives tool schemas and drives retrieval via an agentic tool-call loop). `demo.py` runs 10 predefined analysis queries across the four required categories. `app.py` provides a Streamlit chat UI with sidebar mode selector and metadata filters.
- **Student Contribution**: Designed the overall RAG pipeline flow — query routing logic, the two-collection retrieval strategy, and the decision to structure tools as first-class objects to support future LLM-driven retrieval. Specified that the pipeline should support both a deterministic direct mode and an agentic mode where the model controls tool invocation. Directed iterative refinements to the context structure and tool schema descriptions.

---

<!-- Copy the entry block above to add more entries -->


## 3. Overall Student Contribution

Beyond AI-generated code, the following work was done independently:

- [ ] Dataset download and inspection
- [ ] Architectural decisions (chunking strategy, embedding model choice, LLM selection)
- [ ] Debugging and fixing AI-generated code
- [ ] Integration of components into a working pipeline
- [ ] Prompt engineering for analytical queries
- [ ] Testing and evaluation of RAG output quality
- [ ] Interpretation of analysis results
- [ ] Writing the technical report
- [x] Retrieval quality optimization — identified semantic gaps between query language and document text (e.g. "revenue" vs "sales", "trend" vs "annual summary"); designed and added targeted summary documents; rewrote annual summary text to surface year-over-year trend intent; validated that retrieved documents actually answered the intended analytical questions

---

> This section does not count toward the technical report page limit (per course policy).