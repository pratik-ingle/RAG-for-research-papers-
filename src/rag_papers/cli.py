"""Command-line entry points for the RAG research papers project."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from rag_papers.config import load_config
from rag_papers.embeddings import get_embeddings
from rag_papers.eval import evaluate_retriever, load_questions, summarize_metrics
from rag_papers.ingest import chunk_documents, load_chunks, load_papers, save_chunks
from rag_papers.rag import create_rag_chain, format_docs
from rag_papers.retrieve import get_retriever
from rag_papers.store import build_store, load_store

app = typer.Typer(help="RAG over research papers.")


ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", help="Path to the YAML config file."),
]


@app.command()
def ingest(
    config: ConfigOption = Path("configs/default.yaml"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path for JSONL chunks."),
    ] = Path("data/chunks.jsonl"),
) -> None:
    """Load PDFs, chunk them, and write JSONL chunks."""

    settings = load_config(config)
    documents = load_papers(settings.papers_csv, settings.papers_dir)
    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    save_chunks(chunks, output)
    typer.echo(f"Wrote {len(chunks)} chunks to {output}")


@app.command()
def index(
    config: ConfigOption = Path("configs/default.yaml"),
    chunks: Annotated[
        Path,
        typer.Option("--chunks", help="Path to JSONL chunks produced by ingest."),
    ] = Path("data/chunks.jsonl"),
) -> None:
    """Build and persist the configured vector store."""

    settings = load_config(config)
    documents = load_chunks(chunks)
    embeddings = get_embeddings(settings.embedding_model)
    build_store(
        documents,
        embeddings,
        backend=settings.store_backend,
        index_path=settings.index_path,
        collection_name=settings.chroma_collection,
    )
    typer.echo(
        f"Indexed {len(documents)} chunks into {settings.store_backend} at {settings.index_path}"
    )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to retrieve context for.")],
    config: ConfigOption = Path("configs/default.yaml"),
    generate: Annotated[
        bool,
        typer.Option("--generate", help="Use the optional OpenAI RAG answer chain."),
    ] = False,
) -> None:
    """Retrieve relevant chunks and optionally generate an answer."""

    load_dotenv()
    settings = load_config(config)
    embeddings = get_embeddings(settings.embedding_model)
    store = load_store(
        embeddings,
        backend=settings.store_backend,
        index_path=settings.index_path,
        collection_name=settings.chroma_collection,
    )
    retriever = get_retriever(store, settings.retrieval_k, settings.search_type)

    if generate:
        chain = create_rag_chain(retriever)
        typer.echo(chain.invoke(question))
        return

    documents = retriever.invoke(question)
    typer.echo(format_docs(documents))


@app.command(name="eval")
def evaluate(
    config: ConfigOption = Path("configs/default.yaml"),
    questions: Annotated[
        Path,
        typer.Option("--questions", "-q", help="Path to labelled JSONL questions."),
    ] = Path("data/eval/questions.jsonl"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path for per-question CSV results."),
    ] = Path("results/retrieval.csv"),
) -> None:
    """Evaluate retrieval against labelled questions."""

    settings = load_config(config)
    embeddings = get_embeddings(settings.embedding_model)
    store = load_store(
        embeddings,
        backend=settings.store_backend,
        index_path=settings.index_path,
        collection_name=settings.chroma_collection,
    )
    retriever = get_retriever(store, settings.retrieval_k, settings.search_type)
    results = evaluate_retriever(load_questions(questions), retriever)

    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    typer.echo(f"Wrote per-question results to {output}")
    typer.echo(summarize_metrics(results).to_string(index=False))


if __name__ == "__main__":
    app()
