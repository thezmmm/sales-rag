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

---

> This section does not count toward the technical report page limit (per course policy).