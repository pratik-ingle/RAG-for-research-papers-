"""Configuration loading for the RAG pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    chunk_size: int = 900
    chunk_overlap: int = 150
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    store_backend: str = "faiss"
    index_path: str = "indices/faiss"
    chroma_collection: str = "research_papers"
    retrieval_k: int = 5
    search_type: str = "similarity"
    papers_dir: str = "data/papers"
    papers_csv: str = "data/papers.csv"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path = "configs/default.yaml") -> Settings:
    """Load YAML config values into a typed Settings object."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    data = yaml.safe_load(config_path.read_text()) or {}
    valid_keys = {field.name for field in fields(Settings)}
    unknown_keys = set(data) - valid_keys
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown config key(s): {unknown}")

    settings = Settings(**data)
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if settings.store_backend not in {"faiss", "chroma"}:
        raise ValueError("store_backend must be either 'faiss' or 'chroma'")
    return settings
