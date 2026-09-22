# RAG Research Papers

Retrieval-augmented generation over a local corpus of research papers. The project loads PDFs, chunks them with stable metadata, embeds the chunks, stores them in FAISS or Chroma, and evaluates retrieval against a hand-labelled question set.

## Setup

```bash
uv sync
cp .env.example .env
```

The default configuration uses local sentence-transformers embeddings, so an API key is not required for indexing or retrieval. Set `OPENAI_API_KEY` only if you want to use the optional answer-generation command.

## Data

The corpus is 10 papers on soft robotic manipulation surfaces and shape displays (Deng 2016, Robertson 2019, Liu 2021, Hu 2022, Johnson 2023, Xue 2023 / ArrayBot, Jang 2024, Shin 2025, and two MANTA-RAY papers by Ingle 2025). With the default chunking this yields 728 chunks (mean 804 characters).

PDFs live in `data/papers/` and are described in `data/papers.csv` (metadata exported from Zotero):

```csv
paper_id,title,authors,year,url,filename
deng2016-soft-table,A Novel Soft Machine Table for ...,"Deng, Zhicong; Stommel, Martin; Xu, Weiliang",2016,https://doi.org/10.1109/TMECH.2016.2519333,Deng et al. - 2016 - ....pdf
```

`data/eval/questions.jsonl` contains 80 hand-labelled retrieval questions (41 numeric, 27 factual, 9 method, 3 cross-paper). Every question has gold paper IDs and gold chunk IDs (115 in total, labelled against the default 900/150 chunking). Where the answer spans the chunk overlap, both chunks are marked gold.

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

# Run the full comparison grid (writes results/grid_summary.csv and one CSV per config)
uv run python scripts/eval_grid.py
```

## Evaluation

The retrieval evaluator reports `hit@k`, `recall@k`, and MRR at the chunk and paper level. All rows below retrieve k=10 and use `BAAI/bge-small-en-v1.5` (local, normalized) unless noted. Chunk-level metrics are only reported for the 900/150 chunking that the gold chunk IDs were labelled against; the other chunk sizes are compared at paper level.

| Backend | Search | Chunk size | Overlap | Paper Hit@1 | Paper Hit@5 | Paper MRR | Chunk Hit@1 | Chunk Hit@5 | Chunk Hit@10 | Chunk MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FAISS | similarity | 900 | 150 | 0.925 | 0.988 | 0.954 | 0.362 | 0.662 | 0.762 | 0.498 |
| FAISS | MMR | 900 | 150 | 0.925 | 0.988 | 0.949 | 0.362 | 0.512 | 0.662 | 0.437 |
| Chroma | similarity | 900 | 150 | 0.925 | 0.988 | 0.954 | 0.362 | 0.662 | 0.762 | 0.498 |
| BM25 (lexical, no embeddings) | - | 900 | 150 | 0.912 | 0.963 | 0.939 | 0.487 | 0.750 | 0.787 | 0.600 |
| **BM25 + FAISS hybrid (RRF, 0.5/0.5)** | rrf | 900 | 150 | **0.963** | 0.988 | **0.971** | **0.487** | **0.775** | **0.825** | **0.607** |
| FAISS | similarity | 500 | 100 | 0.950 | **1.000** | 0.973 | - | - | - | - |
| FAISS | similarity | 1500 | 200 | 0.912 | 0.975 | 0.938 | - | - | - | - |

Findings:

- FAISS and Chroma give identical results because they store the same embeddings; the backend choice is an engineering decision, not a quality one.
- Paper-level retrieval is essentially solved on this corpus (Hit@5 of 0.96 to 1.00 for every config). Chunk-level retrieval is the hard part: dense similarity finds the exact gold chunk in the top 5 only 66% of the time.
- BM25 beats dense retrieval at chunk level (0.750 vs 0.662 Hit@5, 0.600 vs 0.498 MRR). Split by question type, BM25 gets 0.854 chunk Hit@5 on numeric questions versus 0.634 for dense, because these questions contain exact tokens (part numbers such as `LIS3MDL`, values such as `12.48`) that a small embedding model does not weight strongly. Dense retrieval is better on cross-paper questions (0.667 vs 0.333).
- The hybrid combines both strengths and is the best configuration overall (0.775 chunk Hit@5, 0.607 chunk MRR, 0.971 paper MRR).
- MMR hurts chunk-level metrics (0.512 Hit@5): de-duplicating similar chunks pushes out overlapping gold chunks from the same passage.
- Smaller chunks help paper-level retrieval (500/100: 1.000 Hit@5, 0.973 MRR) and larger chunks hurt (1500/200: 0.975, 0.938), consistent with fact-lookup questions matching more precisely against shorter passages.

Per-question results for every configuration are written to `results/retrieval_<config>.csv`, and the summary to `results/grid_summary.csv`.

## Project Layout

```text
configs/default.yaml      Runtime settings for chunking, embeddings, store, and retrieval.
data/papers/             Raw PDFs, ignored by git.
data/papers.csv          Paper metadata used during ingestion.
data/eval/questions.jsonl Hand-labelled retrieval questions.
indices/                 Persisted FAISS/Chroma stores, ignored by git.
results/                 Evaluation CSVs, ignored by git.
scripts/eval_grid.py     Comparison grid: FAISS vs Chroma, similarity vs MMR, BM25, hybrid, chunk sizes.
src/rag_papers/          Package source.
tests/                   Unit tests for ingestion helpers and metrics.
```
