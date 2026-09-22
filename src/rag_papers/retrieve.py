"""Retriever helpers."""

from __future__ import annotations

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


def get_retriever(
    store: VectorStore,
    k: int = 5,
    search_type: str = "similarity",
) -> BaseRetriever:
    """Create a LangChain retriever from a vector store."""

    search_kwargs = {"k": k}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = max(k * 4, 20)

    return store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def get_bm25_retriever(chunks: list[Document], k: int = 5) -> BM25Retriever:
    """Create a lexical BM25 baseline retriever."""

    if not chunks:
        raise ValueError("Cannot build a BM25 retriever with zero chunks")

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever
