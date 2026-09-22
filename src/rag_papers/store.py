"""Vector store construction and loading."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


def build_store(
    chunks: list[Document],
    embeddings: Embeddings,
    backend: str = "faiss",
    index_path: str | Path = "indices/faiss",
    collection_name: str = "research_papers",
) -> VectorStore:
    """Build and persist a FAISS or Chroma vector store."""

    if not chunks:
        raise ValueError("Cannot build a vector store with zero chunks")

    backend = backend.lower()
    path = Path(index_path)
    path.mkdir(parents=True, exist_ok=True)

    if backend == "faiss":
        store = FAISS.from_documents(chunks, embeddings)
        store.save_local(str(path))
        return store

    if backend == "chroma":
        return Chroma.from_documents(
            chunks,
            embeddings,
            collection_name=collection_name,
            persist_directory=str(path),
        )

    raise ValueError("backend must be either 'faiss' or 'chroma'")


def load_store(
    embeddings: Embeddings,
    backend: str = "faiss",
    index_path: str | Path = "indices/faiss",
    collection_name: str = "research_papers",
) -> VectorStore:
    """Load a persisted FAISS or Chroma vector store."""

    backend = backend.lower()
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Vector store path not found: {path}")

    if backend == "faiss":
        return FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    if backend == "chroma":
        return Chroma(
            collection_name=collection_name,
            persist_directory=str(path),
            embedding_function=embeddings,
        )

    raise ValueError("backend must be either 'faiss' or 'chroma'")
