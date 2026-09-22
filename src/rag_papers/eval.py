"""Retrieval evaluation against hand-labelled questions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    gold_paper_ids: list[str]
    gold_chunk_ids: list[str]
    answer: str = ""
    type: str = "unknown"


def load_questions(path: str | Path) -> list[Question]:
    """Load labelled questions from JSON Lines."""

    question_path = Path(path)
    if not question_path.exists():
        raise FileNotFoundError(f"Question file not found: {question_path}")

    questions: list[Question] = []
    with question_path.open() as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            questions.append(
                Question(
                    id=record["id"],
                    question=record["question"],
                    gold_paper_ids=list(record.get("gold_paper_ids", [])),
                    gold_chunk_ids=list(record.get("gold_chunk_ids", [])),
                    answer=record.get("answer", ""),
                    type=record.get("type", "unknown"),
                )
            )
    return questions


def hit_at_k(gold_ids: Iterable[str], retrieved_ids: Iterable[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    return float(bool(gold.intersection(list(retrieved_ids)[:k])))


def recall_at_k(gold_ids: Iterable[str], retrieved_ids: Iterable[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    retrieved = set(list(retrieved_ids)[:k])
    return len(gold & retrieved) / len(gold)


def mrr(gold_ids: Iterable[str], retrieved_ids: Iterable[str]) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0

    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in gold:
            return 1.0 / rank
    return 0.0


def _metadata_ids(documents: list[Document], key: str) -> list[str]:
    return [str(doc.metadata.get(key, "")) for doc in documents]


def evaluate_retriever(
    questions: list[Question],
    retriever: BaseRetriever,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """Evaluate a retriever and return one row per question."""

    rows: list[dict[str, Any]] = []
    for item in questions:
        documents = retriever.invoke(item.question)
        chunk_ids = _metadata_ids(documents, "chunk_id")
        paper_ids = _metadata_ids(documents, "paper_id")

        row: dict[str, Any] = {
            "id": item.id,
            "question": item.question,
            "type": item.type,
            "retrieved_chunk_ids": chunk_ids,
            "retrieved_paper_ids": paper_ids,
            "chunk_mrr": mrr(item.gold_chunk_ids, chunk_ids),
            "paper_mrr": mrr(item.gold_paper_ids, paper_ids),
        }
        for k in k_values:
            row[f"chunk_hit@{k}"] = hit_at_k(item.gold_chunk_ids, chunk_ids, k)
            row[f"chunk_recall@{k}"] = recall_at_k(item.gold_chunk_ids, chunk_ids, k)
            row[f"paper_hit@{k}"] = hit_at_k(item.gold_paper_ids, paper_ids, k)
            row[f"paper_recall@{k}"] = recall_at_k(item.gold_paper_ids, paper_ids, k)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Average metric columns into a compact summary table."""

    metric_cols = [
        column
        for column in results.columns
        if "_hit@" in column or "_recall@" in column or column.endswith("_mrr")
    ]
    if not metric_cols:
        return pd.DataFrame()
    summary = results[metric_cols].mean().rename_axis("metric").reset_index(name="mean")
    return summary
