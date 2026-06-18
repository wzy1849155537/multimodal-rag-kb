"""Factory pattern module registry. Maps string names to classes."""

from typing import Any, Dict, Type, TypeVar, Generic
from src.core.parser import BaseParser
from src.core.chunker import BaseChunker
from src.core.embedder import BaseEmbedder
from src.core.indexer import BaseIndexer
from src.core.query_rewriter import BaseQueryRewriter
from src.core.router import BaseRouter
from src.core.retriever import BaseRetriever
from src.core.reranker import BaseReranker
from src.core.verifier import BaseVerifier
from src.core.generator import BaseGenerator

T = TypeVar("T")


class _Registry(Generic[T]):
    """Generic registry for one component type."""

    def __init__(self):
        self._items: Dict[str, Type[T]] = {}

    def register(self, name: str):
        """Decorator to register a class under a name."""
        def decorator(klass: Type[T]):
            self._items[name] = klass
            return klass
        return decorator

    def get(self, name: str) -> Type[T]:
        """Get a registered class by name."""
        if name not in self._items:
            available = list(self._items.keys())
            raise KeyError(
                f"Unknown component '{name}'. Available: {available}"
            )
        return self._items[name]

    def build(self, name: str, **kwargs) -> T:
        """Instantiate a registered class with kwargs."""
        return self.get(name)(**kwargs)

    def list(self) -> list:
        return list(self._items.keys())


class ModuleRegistry:
    """Central registry for all pluggable module types."""

    parsers = _Registry[BaseParser]()
    chunkers = _Registry[BaseChunker]()
    embedders = _Registry[BaseEmbedder]()
    indexers = _Registry[BaseIndexer]()
    rewriters = _Registry[BaseQueryRewriter]()
    routers = _Registry[BaseRouter]()
    retrievers = _Registry[BaseRetriever]()
    rerankers = _Registry[BaseReranker]()
    verifiers = _Registry[BaseVerifier]()
    generators = _Registry[BaseGenerator]()

    # Map component type strings to their registries
    _registry_map: Dict[str, _Registry] = {
        "parser": parsers,
        "chunker": chunkers,
        "embedder": embedders,
        "indexer": indexers,
        "rewriter": rewriters,
        "router": routers,
        "retriever": retrievers,
        "reranker": rerankers,
        "verifier": verifiers,
        "generator": generators,
    }

    @classmethod
    def build(cls, component_type: str, name: str, **kwargs) -> Any:
        """Factory method: build a component by type + name."""
        if component_type not in cls._registry_map:
            raise KeyError(
                f"Unknown component type '{component_type}'. "
                f"Available: {list(cls._registry_map.keys())}"
            )
        return cls._registry_map[component_type].build(name, **kwargs)
