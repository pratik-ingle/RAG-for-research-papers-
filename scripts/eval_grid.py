"""Run the retrieval evaluation grid and write results/grid_summary.csv.

Compares vector-store backend (FAISS vs Chroma), search type (similarity vs MMR),
a lexical BM25 baseline, a BM25 + dense hybrid, and alternative chunk sizes.
Chunk-level metrics are only reported for the default chunking, because gold
chunk IDs were labelled against it.

Usage:
    uv run python scripts/eval_grid.py
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pandas as pd
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS

from rag_papers.config import load_config
from rag_papers.embeddings import get_embeddings
from rag_papers.eval import evaluate_retriever, load_questions
from rag_papers.ingest import chunk_documents, load_chunks, load_papers
from rag_papers.retrieve import get_bm25_retriever, get_retriever
from rag_papers.store import build_store

K = 10
K_VALUES = (1, 3, 5, 10)
RESULTS_DIR = Path("results")
CHROMA_DIR = Path("indices/chroma")
ALT_CHUNKING = [(500, 100), (1500, 200)]


def summarize(name: str, results: pd.DataFrame, chunk_level: bool, **meta) -> dict:
    row = {"config": name, **meta}
    for k in K_VALUES:
        row[f"paper_hit@{k}"] = results[f"paper_hit@{k}"].mean()
    row["paper_mrr"] = results["paper_mrr"].mean()
    for k in K_VALUES:
        row[f"chunk_hit@{k}"] = results[f"chunk_hit@{k}"].mean() if chunk_level else float("nan")
    row["chunk_mrr"] = results["chunk_mrr"].mean() if chunk_level else float("nan")
    return row


def main() -> None:
    settings = load_config("configs/default.yaml")
    questions = load_questions("data/eval/questions.jsonl")
    embeddings = get_embeddings(settings.embedding_model)
    default_chunks = load_chunks("data/chunks.jsonl")
    RESULTS_DIR.mkdir(exist_ok=True)

    rows: list[dict] = []
    common = {
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }

    def run(name: str, retriever, chunk_level: bool, **meta) -> None:
        start = time.perf_counter()
        results = evaluate_retriever(questions, retriever, k_values=K_VALUES)
        elapsed = time.perf_counter() - start
        results.to_csv(RESULTS_DIR / f"retrieval_{name}.csv", index=False)
        rows.append(summarize(name, results, chunk_level, **meta, eval_seconds=round(elapsed, 1)))
        print(f"{name:28s} paper_hit@5={rows[-1]['paper_hit@5']:.3f} "
              f"chunk_hit@5={rows[-1]['chunk_hit@5']:.3f} chunk_mrr={rows[-1]['chunk_mrr']:.3f}")

    # 1. FAISS, default chunking, similarity and MMR
    faiss_store = FAISS.from_documents(default_chunks, embeddings)
    run("faiss_similarity", get_retriever(faiss_store, K, "similarity"), True,
        backend="faiss", search_type="similarity", **common)
    run("faiss_mmr", get_retriever(faiss_store, K, "mmr"), True,
        backend="faiss", search_type="mmr", **common)

    # 2. Chroma, default chunking
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    chroma_store = build_store(default_chunks, embeddings, "chroma", CHROMA_DIR,
                               settings.chroma_collection)
    run("chroma_similarity", get_retriever(chroma_store, K, "similarity"), True,
        backend="chroma", search_type="similarity", **common)

    # 3. BM25 lexical baseline and a BM25 + FAISS hybrid (reciprocal rank fusion)
    bm25 = get_bm25_retriever(default_chunks, K)
    run("bm25", bm25, True,
        backend="bm25", search_type="lexical", embedding_model="none",
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    hybrid = EnsembleRetriever(
        retrievers=[bm25, get_retriever(faiss_store, K, "similarity")], weights=[0.5, 0.5]
    )
    run("hybrid_bm25_faiss_rrf", hybrid, True,
        backend="faiss+bm25", search_type="rrf", **common)

    # 4. Alternative chunk sizes (paper-level metrics only)
    pages = load_papers(settings.papers_csv, settings.papers_dir)
    for size, overlap in ALT_CHUNKING:
        chunks = chunk_documents(pages, chunk_size=size, chunk_overlap=overlap)
        store = FAISS.from_documents(chunks, embeddings)
        run(f"faiss_similarity_c{size}_o{overlap}", get_retriever(store, K, "similarity"), False,
            backend="faiss", search_type="similarity", embedding_model=settings.embedding_model,
            chunk_size=size, chunk_overlap=overlap, n_chunks=len(chunks))

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "grid_summary.csv", index=False)
    print()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
