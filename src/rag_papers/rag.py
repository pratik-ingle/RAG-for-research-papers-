"""RAG chain assembly."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using only the provided research-paper context. "
            "If the answer is not in the context, say you do not know. "
            "Cite sources with paper_id and page.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)


def format_docs(documents: list[Document]) -> str:
    """Format retrieved documents for a citation-friendly prompt."""

    formatted: list[str] = []
    for doc in documents:
        metadata = doc.metadata
        source = metadata.get("paper_id", "unknown-paper")
        page = metadata.get("page", "?")
        chunk_id = metadata.get("chunk_id", "unknown-chunk")
        formatted.append(
            f"[paper_id={source}; page={page}; chunk_id={chunk_id}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def create_rag_chain(
    retriever: BaseRetriever,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
):
    """Create a simple LCEL RAG chain backed by an OpenAI chat model."""

    llm = ChatOpenAI(model=model_name, temperature=temperature)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
