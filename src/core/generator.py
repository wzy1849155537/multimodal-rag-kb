"""BaseGenerator ABC for answer generation."""

from abc import ABC, abstractmethod
from typing import List

from .schemas import SearchResult, GeneratedAnswer


class BaseGenerator(ABC):
    """Pluggable generator. Produces the final answer from retrieved context."""

    @abstractmethod
    def generate(
        self,
        query: str,
        context_chunks: List[SearchResult],
    ) -> GeneratedAnswer:
        """Generate an answer grounded in the provided context chunks."""
        ...
