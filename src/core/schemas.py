"""Shared dataclasses used across all modules."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
from enum import Enum


class ChangeType(str, Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


class SearchSource(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


@dataclass
class DocumentImage:
    """An image extracted from a document."""
    image_id: str
    image_path: Optional[Path] = None
    image_bytes: Optional[bytes] = None
    page_num: int = 0
    caption: str = ""
    ocr_text: str = ""
    vlm_description: str = ""


@dataclass
class RawDocument:
    """Output of a parser, before chunking."""
    doc_id: str
    source_path: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    images: List[DocumentImage] = field(default_factory=list)


@dataclass
class Chunk:
    """A single indexed chunk."""
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __hash__(self):
        return hash(self.chunk_id)


@dataclass
class SearchResult:
    """A single search result from the index."""
    chunk_id: str
    content: str
    score: float
    source: SearchSource = SearchSource.HYBRID
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewrittenQuery:
    """A rewritten/expanded variant of the user query."""
    original: str
    rewritten: str
    rewrite_type: str  # 'noop' | 'llm_expand' | 'llm_decompose' | 'hyde'


@dataclass
class RouteDecision:
    """Output of the query router."""
    target_collection: str = "default"
    top_k: int = 10
    bm25_weight: float = 0.3
    metadata_filter: Optional[Dict[str, Any]] = None
    use_rewrite: bool = True


@dataclass
class VerificationResult:
    """Output of the retrieval verifier."""
    confidence_score: float
    needs_secondary_retrieval: bool = False
    reason: str = ""
    verified_results: List[SearchResult] = field(default_factory=list)


@dataclass
class GeneratedAnswer:
    """Final answer from the RAG pipeline."""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0


@dataclass
class IndexStats:
    """Statistics about the current index state."""
    total_chunks: int = 0
    total_documents: int = 0
    dense_index_size_bytes: int = 0
    sparse_index_size_bytes: int = 0
    collection_names: List[str] = field(default_factory=list)


@dataclass
class FileChange:
    """Represents a detected change to a document file."""
    file_path: str
    change_type: ChangeType
    old_hash: str = ""
    new_hash: str = ""


@dataclass
class EvalResult:
    """A single evaluation result row."""
    config_name: str
    chunk_size: int
    top_k: int
    reranker: str
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    mrr: float = 0.0
    hit_at_k: float = 0.0
