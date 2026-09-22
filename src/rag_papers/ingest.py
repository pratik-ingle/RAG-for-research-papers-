"""Load, clean, and chunk research paper PDFs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """Normalize common PDF extraction artifacts without changing meaning."""

    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_paper_metadata(papers_csv: str | Path) -> list[dict[str, str]]:
    """Read paper metadata from a CSV file."""

    path = Path(papers_csv)
    if not path.exists():
        raise FileNotFoundError(f"Paper metadata file not found: {path}")

    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    required = {"paper_id", "title", "authors", "year", "url", "filename"}
    missing = required - fieldnames
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"papers.csv missing required column(s): {missing_names}")
    return rows


def load_papers(papers_csv: str | Path, papers_dir: str | Path) -> list[Document]:
    """Load PDFs listed in papers.csv and attach stable paper metadata."""

    rows = read_paper_metadata(papers_csv)
    base_dir = Path(papers_dir)
    documents: list[Document] = []
    skipped: list[str] = []

    for row in rows:
        pdf_path = base_dir / row["filename"]
        if not pdf_path.exists():
            skipped.append(str(pdf_path))
            continue

        loader = PyMuPDFLoader(str(pdf_path))
        for page_doc in loader.load():
            page_doc.page_content = clean_text(page_doc.page_content)
            page_doc.metadata.update(
                {
                    "paper_id": row["paper_id"],
                    "title": row["title"],
                    "authors": row["authors"],
                    "year": row["year"],
                    "url": row["url"],
                    "filename": row["filename"],
                }
            )
            documents.append(page_doc)

    if not documents:
        skipped_msg = f" Missing files: {', '.join(skipped)}." if skipped else ""
        raise ValueError(f"No PDF documents were loaded.{skipped_msg}")

    return documents


def _page_number(metadata: dict) -> int:
    if "page" in metadata:
        return int(metadata["page"]) + 1
    if "page_number" in metadata:
        return int(metadata["page_number"])
    return 1


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split documents into deterministic chunks with paper/page/chunk IDs."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in documents:
        metadata = dict(doc.metadata)
        paper_id = metadata.get("paper_id") or Path(metadata.get("source", "paper")).stem
        page = _page_number(metadata)

        for chunk_index, text in enumerate(splitter.split_text(clean_text(doc.page_content))):
            chunk_metadata = {
                **metadata,
                "paper_id": paper_id,
                "page": page,
                "chunk_index": chunk_index,
                "chunk_id": f"{paper_id}_p{page}_c{chunk_index}",
            }
            chunks.append(Document(page_content=text, metadata=chunk_metadata))

    return chunks


def save_chunks(chunks: Iterable[Document], output_path: str | Path) -> None:
    """Persist chunks as JSON Lines for indexing and inspection."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    {
                        "page_content": chunk.page_content,
                        "metadata": chunk.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_chunks(path: str | Path) -> list[Document]:
    """Load chunks previously written by save_chunks."""

    chunk_path = Path(path)
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunk_path}")

    chunks: list[Document] = []
    with chunk_path.open() as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            chunks.append(
                Document(
                    page_content=record["page_content"],
                    metadata=record.get("metadata", {}),
                )
            )
    return chunks
