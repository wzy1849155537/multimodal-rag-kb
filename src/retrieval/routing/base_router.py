"""Default router: all queries use the same retrieval strategy."""

from src.core.router import BaseRouter
from src.core.schemas import RouteDecision
from src.registry import ModuleRegistry


@ModuleRegistry.routers.register("default")
class DefaultRouter(BaseRouter):
    """Routes all queries to the default collection with fixed parameters."""

    def __init__(
        self,
        collection_name: str = "default",
        top_k: int = 10,
        bm25_weight: float = 0.3,
    ):
        self.collection_name = collection_name
        self._top_k = top_k
        self.bm25_weight = bm25_weight

    def route(self, query: str) -> RouteDecision:
        return RouteDecision(
            target_collection=self.collection_name,
            top_k=self._top_k,
            bm25_weight=self.bm25_weight,
            use_rewrite=False,
        )
