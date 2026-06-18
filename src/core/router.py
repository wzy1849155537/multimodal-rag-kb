"""BaseRouter ABC for query routing."""

from abc import ABC, abstractmethod

from .schemas import RouteDecision


class BaseRouter(ABC):
    """Pluggable router. Classifies query and decides retrieval strategy."""

    @abstractmethod
    def route(self, query: str) -> RouteDecision:
        """Analyze the query and return a routing decision."""
        ...
