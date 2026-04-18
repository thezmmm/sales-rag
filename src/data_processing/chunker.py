"""
Chunking strategies for RAG document preparation.

Rationale:
- Transactions are short (~200 chars) so they stay as single chunks regardless of size.
- Aggregated summaries (~150-400 chars) also fit within any chunk size.
- Chunk size mainly affects how many transaction records are grouped together
  when batch-chunking is applied, or how summary text is split if it grows long.

Three sizes are tested:
  500  — fine-grained, each chunk carries narrow context, higher retrieval precision.
  1000 — balanced, recommended default for this dataset.
  2000 — coarse-grained, more context per chunk, may dilute relevance for specific queries.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict
    chunk_size: int


def _split_text(text: str, chunk_size: int, overlap: int = 50) -> list[str]:
    """Split a long text into overlapping character-level chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_documents(docs: list[dict], chunk_size: int, overlap: int = 50) -> list[Chunk]:
    """
    Convert raw documents into Chunk objects.
    Documents shorter than chunk_size are kept intact.
    Longer documents (e.g. large summaries) are split with overlap.
    """
    chunks = []
    for doc in docs:
        parts = _split_text(doc["text"], chunk_size, overlap)
        for i, part in enumerate(parts):
            chunk_id = doc["id"] if len(parts) == 1 else f"{doc['id']}_part{i}"
            chunks.append(Chunk(
                id=chunk_id,
                text=part,
                metadata={**doc["metadata"], "chunk_size": chunk_size},
                chunk_size=chunk_size,
            ))
    return chunks


def chunk_all_sizes(docs: list[dict]) -> dict[int, list[Chunk]]:
    """Return chunks for all three required sizes."""
    return {size: chunk_documents(docs, size) for size in (500, 1000, 2000)}


def print_chunk_stats(chunks_by_size: dict[int, list]) -> None:
    print("Chunk size comparison:")
    print(f"{'Size':>6}  {'# Chunks':>9}  {'Avg len':>8}  {'Min len':>8}  {'Max len':>8}")
    print("-" * 50)
    for size, chunks in chunks_by_size.items():
        lengths = [len(c.text) for c in chunks]
        print(
            f"{size:>6}  {len(chunks):>9,}  "
            f"{sum(lengths)/len(lengths):>8.0f}  "
            f"{min(lengths):>8}  {max(lengths):>8}"
        )


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from loader import load_data
    from text_converter import build_all_texts

    df = load_data()
    transactions, summaries = build_all_texts(df)
    docs = transactions + summaries

    chunks_by_size = chunk_all_sizes(docs)
    print_chunk_stats(chunks_by_size)

    print("\n--- Sample chunk (size=1000, first transaction) ---")
    sample = chunks_by_size[1000][0]
    print(f"ID      : {sample.id}")
    print(f"Length  : {len(sample.text)} chars")
    print(f"Text    : {sample.text}")
    print(f"Metadata: {sample.metadata}")