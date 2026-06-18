"""BaseChunker ABC for text chunking."""

from abc import ABC, abstractmethod
from typing import List

from .schemas import RawDocument, Chunk


class BaseChunker(ABC):
    """Pluggable chunker. Splits RawDocument into Chunks."""

    @abstractmethod
    def chunk(self, document: RawDocument) -> List[Chunk]:
        """Split a RawDocument into chunks."""
        ...

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the strategy name, e.g. 'recursive-512', 'semantic'."""
        ...
