"""BaseVerifier ABC for retrieval quality verification."""

from abc import ABC, abstractmethod
from typing import List

from .schemas import SearchResult, VerificationResult


class BaseVerifier(ABC):
    """Pluggable verifier. Checks retrieval quality and triggers re-retrieval."""

    @abstractmethod
    def verify(
        self,
        query: str,
        results: List[SearchResult],
    ) -> VerificationResult:
        """Verify retrieval quality. Returns confidence + re-retrieval decision."""
        ...
