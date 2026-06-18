"""Core abstract interfaces and shared schemas."""

from .schemas import (
    ChangeType, SearchSource, DocumentImage, RawDocument, Chunk,
    SearchResult, RewrittenQuery, RouteDecision, VerificationResult,
    GeneratedAnswer, IndexStats, FileChange, EvalResult,
)
from .parser import BaseParser
from .chunker import BaseChunker
from .embedder import BaseEmbedder
from .indexer import BaseIndexer
from .query_rewriter import BaseQueryRewriter
from .router import BaseRouter
from .retriever import BaseRetriever
from .reranker import BaseReranker
from .verifier import BaseVerifier
from .generator import BaseGenerator

__all__ = [
    # schemas
    "ChangeType", "SearchSource", "DocumentImage", "RawDocument", "Chunk",
    "SearchResult", "RewrittenQuery", "RouteDecision", "VerificationResult",
    "GeneratedAnswer", "IndexStats", "FileChange", "EvalResult",
    # ABCs
    "BaseParser", "BaseChunker", "BaseEmbedder", "BaseIndexer",
    "BaseQueryRewriter", "BaseRouter", "BaseRetriever",
    "BaseReranker", "BaseVerifier", "BaseGenerator",
]
