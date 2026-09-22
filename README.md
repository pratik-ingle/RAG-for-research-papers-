# RAG Research Papers

Retrieval-augmented generation over a local corpus of research papers. The project loads PDFs, chunks them with stable metadata, embeds the chunks, stores them in FAISS or Chroma, and evaluates retrieval against a hand-labelled question set.

## Setup

```bash
uv sync
cp .env.example .env
```

The default configuration uses local sentence-transformers embeddings, so an API key is not required for indexing or retrieval. Set `OPENAI_API_KEY` only if you want to use the optional answer-generation command.

## Data

Place PDFs in `data/papers/` and describe them in `data/papers.csv`:

```csv
paper_id,title,authors,year,url,filename
sample-paper,Example Paper,"Doe, Jane",2024,https://example.com/sample.pdf,sample-paper.pdf
```

Create labelled retrieval questions in `data/eval/questions.jsonl`. Each row can include gold paper IDs and, when available, exact gold chunk IDs.

## Commands

```bash
# Build chunks from PDFs and metadata
uv run rag ingest --config configs/default.yaml --output data/chunks.jsonl

# Build a vector store from chunks
uv run rag index --config configs/default.yaml --chunks data/chunks.jsonl

# Ask a retrieval-only question
uv run rag ask "What method does the paper introduce?"

# Evaluate retrieval against labelled questions
uv run rag eval --config configs/default.yaml --questions data/eval/questions.jsonl --output results/retrieval.csv
```

## Evaluation

The retrieval evaluator reports `hit@k`, `recall@k`, and MRR at the chunk and paper level. Start with the default settings, then compare chunk sizes, overlaps, embedding models, backends, and retriever modes.

| Backend | Embedding model | Chunk size | Overlap | Hit@5 | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| FAISS | BAAI/bge-small-en-v1.5 | 900 | 150 | TBD | TBD |

## Project Layout

```text
configs/default.yaml      Runtime settings for chunking, embeddings, store, and retrieval.
data/papers/             Raw PDFs, ignored by git.
data/papers.csv          Paper metadata used during ingestion.
data/eval/questions.jsonl Hand-labelled retrieval questions.
indices/                 Persisted FAISS/Chroma stores, ignored by git.
src/rag_papers/          Package source.
tests/                   Unit tests for ingestion helpers and metrics.
```
