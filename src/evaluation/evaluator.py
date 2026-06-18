"""RAGAS evaluator wrapper and experiment runner."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from src.evaluation.metrics import mean_reciprocal_rank, hit_at_k
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalResult:
    """Single experiment configuration result."""
    config_name: str
    chunk_size: int
    top_k: int
    reranker: str
    mrr: float = 0.0
    hit_at_5: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0


@dataclass
class EvalReport:
    """Collection of evaluation results."""
    results: List[EvalResult] = field(default_factory=list)
    best_config: Optional[str] = None

    def to_table(self) -> str:
        """Format as markdown table."""
        if not self.results:
            return "No results."

        header = (
            "| Config | Chunk | K | Reranker | MRR | Hit@5 | "
            "Precision | Recall | Faithfulness | Relevancy |"
        )
        sep = "|" + "|".join([" --- " for _ in range(10)]) + "|"
        rows = [header, sep]

        for r in self.results:
            rows.append(
                f"| {r.config_name} | {r.chunk_size} | {r.top_k} | {r.reranker} | "
                f"{r.mrr:.3f} | {r.hit_at_5:.3f} | "
                f"{r.context_precision:.3f} | {r.context_recall:.3f} | "
                f"{r.faithfulness:.3f} | {r.answer_relevancy:.3f} |"
            )
        return "\n".join(rows)


class RAGEvaluator:
    """Evaluates RAG pipeline using RAGAS and custom metrics."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def evaluate_retrieval(
        self,
        test_queries: List[str],
        relevant_doc_ids: List[List[str]],
        top_k: int = 10,
    ) -> Dict[str, float]:
        """Evaluate retrieval quality with MRR and Hit@k.

        Args:
            test_queries: List of query strings.
            relevant_doc_ids: For each query, list of relevant chunk IDs.
            top_k: Number of results to retrieve per query.

        Returns:
            Dict with MRR and Hit@5 scores.
        """
        retrieved = []
        for query in test_queries:
            answer = self.pipeline.query(query, top_k=top_k)
            chunk_ids = [s["chunk_id"] for s in answer.sources]
            retrieved.append(chunk_ids)

        mrr = mean_reciprocal_rank(test_queries, relevant_doc_ids, retrieved)
        h5 = hit_at_k(test_queries, relevant_doc_ids, retrieved, k=5)

        return {"mrr": mrr, "hit_at_5": h5}

    def run_ragas_evaluation(
        self,
        test_dataset_path: Path,
    ) -> Dict[str, Any]:
        """Run RAGAS evaluation on a test dataset.

        Test dataset format (JSON):
        [
            {
                "question": "...",
                "ground_truth": "...",
                "relevant_chunk_ids": ["..."]
            }
        ]

        Returns RAGAS metrics dict.
        """
        try:
            from ragas import evaluate
            from ragas.metrics import (
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            )
            from datasets import Dataset
        except ImportError:
            logger.warning("RAGAS not installed. Install: pip install ragas datasets")
            return {"error": "RAGAS not installed"}

        with open(test_dataset_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        # Build dataset for RAGAS
        questions = []
        answers = []
        contexts_list = []
        ground_truths = []

        for item in test_data:
            question = item["question"]
            answer = self.pipeline.query(question)
            questions.append(question)
            answers.append(answer.answer)
            contexts_list.append([s["content_snippet"] for s in answer.sources])
            ground_truths.append(item.get("ground_truth", ""))

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        })

        try:
            result = evaluate(
                dataset,
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy,
                ],
            )
            return dict(result)
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {"error": str(e)}


class ExperimentRunner:
    """Runs grid search experiments over chunk sizes, top-k, and rerankers."""

    def __init__(self, evaluator: RAGEvaluator):
        self.evaluator = evaluator
        self.results: List[EvalResult] = []

    def run_grid(
        self,
        chunk_sizes: List[int],
        top_k_values: List[int],
        reranker_names: List[str],
        test_queries: List[str],
        relevant_doc_ids: List[List[str]],
    ) -> EvalReport:
        """Run evaluation across all parameter combinations."""
        for chunk_size in chunk_sizes:
            for top_k in top_k_values:
                for reranker in reranker_names:
                    config_name = (
                        f"chunk={chunk_size}_k={top_k}_rerank={reranker}"
                    )
                    logger.info(f"Evaluating: {config_name}")

                    try:
                        metrics = self.evaluator.evaluate_retrieval(
                            test_queries, relevant_doc_ids, top_k=top_k
                        )

                        result = EvalResult(
                            config_name=config_name,
                            chunk_size=chunk_size,
                            top_k=top_k,
                            reranker=reranker,
                            mrr=metrics.get("mrr", 0.0),
                            hit_at_5=metrics.get("hit_at_5", 0.0),
                        )
                        self.results.append(result)
                    except Exception as e:
                        logger.error(f"Eval failed for {config_name}: {e}")

        # Find best config by MRR
        report = EvalReport(results=self.results)
        if self.results:
            best = max(self.results, key=lambda r: r.mrr)
            report.best_config = best.config_name

        return report
