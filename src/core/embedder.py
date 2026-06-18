"""BaseEmbedder ABC for text embedding."""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """Pluggable embedder. Encodes text into dense vectors."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Returns embeddings in same order."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output vector dimension."""
        ...
