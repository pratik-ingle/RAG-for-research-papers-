from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_papers.eval import Question, evaluate_retriever, hit_at_k, mrr, recall_at_k, summarize_metrics


class StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return self.documents


def test_metric_functions() -> None:
    retrieved = ["a", "b", "c"]

    assert hit_at_k(["b"], retrieved, 1) == 0.0
    assert hit_at_k(["b"], retrieved, 2) == 1.0
    assert recall_at_k(["b", "c"], retrieved, 2) == 0.5
    assert mrr(["c"], retrieved) == 1 / 3
    assert mrr(["missing"], retrieved) == 0.0


def test_evaluate_retriever_scores_chunk_and_paper_matches() -> None:
    retriever = StaticRetriever(
        documents=[
            Document(page_content="first", metadata={"chunk_id": "c1", "paper_id": "p1"}),
            Document(page_content="second", metadata={"chunk_id": "c2", "paper_id": "p2"}),
        ]
    )
    questions = [
        Question(
            id="q1",
            question="Where is the answer?",
            gold_paper_ids=["p2"],
            gold_chunk_ids=["c2"],
        )
    ]

    results = evaluate_retriever(questions, retriever, k_values=(1, 2))
    summary = summarize_metrics(results)

    assert results.loc[0, "chunk_hit@1"] == 0.0
    assert results.loc[0, "chunk_hit@2"] == 1.0
    assert results.loc[0, "paper_mrr"] == 0.5
    assert set(summary["metric"]).issuperset({"chunk_hit@2", "paper_mrr"})
