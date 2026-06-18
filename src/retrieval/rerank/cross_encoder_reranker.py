"""Cross-encoder reranker using BGE-Reranker."""

from typing import List

from src.core.reranker import BaseReranker
from src.core.schemas import SearchResult
from src.registry import ModuleRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@ModuleRegistry.rerankers.register("cross-encoder")
class CrossEncoderReranker(BaseReranker):
    """Re-rank using a cross-encoder model (BGE-Reranker-v2-m3)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self._model = None
        self._device = device

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                self.model_name,
                use_fp16=(self._device != "cpu"),
                device=self._device,
            )
            logger.info(f"Cross-encoder loaded: {self.model_name}")
        except ImportError:
            logger.warning("FlagEmbedding not installed. Install: pip install FlagEmbedding")
            self._model = False

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        self._load_model()
        if not self._model or self._model is False:
            return candidates[:top_k]

        # Build pairs
        pairs = [[query, c.content] for c in candidates]

        try:
            scores = self._model.compute_score(pairs, normalize=True)
            if isinstance(scores, float):
                scores = [scores]

            # Assign new scores
            for c, score in zip(candidates, scores):
                c.score = float(score)

            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates[:top_k]
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return candidates[:top_k]
