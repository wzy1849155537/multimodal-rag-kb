"""No-op query rewriter (passes query through unchanged)."""

from typing import List

from src.core.query_rewriter import BaseQueryRewriter
from src.core.schemas import RewrittenQuery
from src.registry import ModuleRegistry


@ModuleRegistry.rewriters.register("noop")
class NoOpRewriter(BaseQueryRewriter):
    """Returns the query as-is."""

    def rewrite(self, query: str) -> List[RewrittenQuery]:
        return [RewrittenQuery(
            original=query,
            rewritten=query,
            rewrite_type="noop",
        )]
