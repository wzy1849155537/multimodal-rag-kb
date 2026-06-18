"""BaseQueryRewriter ABC for query rewriting/expansion."""

from abc import ABC, abstractmethod
from typing import List

from .schemas import RewrittenQuery


class BaseQueryRewriter(ABC):
    """Pluggable query rewriter. Expands or decomposes user queries."""

    @abstractmethod
    def rewrite(self, query: str) -> List[RewrittenQuery]:
        """Return one or more rewritten query variants."""
        ...
