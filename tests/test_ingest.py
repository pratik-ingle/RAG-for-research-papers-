from pathlib import Path

from langchain_core.documents import Document

from rag_papers.ingest import chunk_documents, clean_text, read_paper_metadata


def test_clean_text_repairs_hyphenated_line_breaks() -> None:
    text = "retrieval-\naugmented   generation\n\n\nworks"

    assert clean_text(text) == "retrievalaugmented generation\n\nworks"


def test_chunk_documents_adds_deterministic_metadata() -> None:
    document = Document(
        page_content="This is the first sentence. This is the second sentence. This is the third.",
        metadata={"paper_id": "paper-a", "page": 0, "title": "Paper A"},
    )

    chunks = chunk_documents([document], chunk_size=45, chunk_overlap=5)

    assert chunks
    assert chunks[0].metadata["paper_id"] == "paper-a"
    assert chunks[0].metadata["page"] == 1
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["chunk_id"] == "paper-a_p1_c0"
    assert all(chunk.page_content for chunk in chunks)


def test_read_paper_metadata_accepts_header_only_csv(tmp_path: Path) -> None:
    path = tmp_path / "papers.csv"
    path.write_text("paper_id,title,authors,year,url,filename\n")

    assert read_paper_metadata(path) == []
