"""Confidence scoring and secondary retrieval trigger."""

from typing import List, Optional, Callable

from src.core.verifier import BaseVerifier
from src.core.schemas import SearchResult, VerificationResult
from src.registry import ModuleRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@ModuleRegistry.verifiers.register("confidence")
class ConfidenceVerifier(BaseVerifier):
    """Verify retrieval quality via confidence scoring.

    If top result confidence < threshold, triggers secondary retrieval.
    """

    def __init__(
        self,
        threshold: float = 0.4,
        secondary_multiplier: float = 2.0,
        secondary_retrieval_fn: Optional[Callable] = None,
    ):
        self.threshold = threshold
        self.secondary_multiplier = secondary_multiplier
        self._secondary_fn = secondary_retrieval_fn

    def verify(
        self,
        query: str,
        results: List[SearchResult],
    ) -> VerificationResult:
        if not results:
            return VerificationResult(
                confidence_score=0.0,
                needs_secondary_retrieval=True,
                reason="No results from primary retrieval",
                verified_results=[],
            )

        # Compute confidence from score distribution
        top_score = results[0].score
        avg_score = sum(r.score for r in results) / len(results)
        score_std = (
            sum((r.score - avg_score) ** 2 for r in results) / len(results)
        ) ** 0.5

        # Confidence = top_score adjusted by score concentration
        # High top score + large gap to others = high confidence
        confidence = top_score * (1.0 + min(score_std, 0.5))

        needs_secondary = confidence < self.threshold

        reason = (
            f"Confidence {confidence:.3f} {'< ' if needs_secondary else '>= '}"
            f"threshold {self.threshold}. "
            f"Top score: {top_score:.3f}, Avg: {avg_score:.3f}, Std: {score_std:.3f}"
        )

        if needs_secondary:
            logger.info(f"Low confidence: {reason}")

        return VerificationResult(
            confidence_score=confidence,
            needs_secondary_retrieval=needs_secondary,
            reason=reason,
            verified_results=results,
        )
