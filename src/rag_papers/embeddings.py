"""Embedding model factory."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


def get_embeddings(model_name: str) -> Embeddings:
    """Return an embedding model by name.

    Use local HuggingFace models by default. Prefix with ``openai:`` to opt into
    OpenAI embeddings, for example ``openai:text-embedding-3-small``.
    """

    if model_name.startswith("openai:"):
        return OpenAIEmbeddings(model=model_name.removeprefix("openai:"))

    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
