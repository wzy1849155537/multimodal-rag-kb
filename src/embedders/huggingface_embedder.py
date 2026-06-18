"""HuggingFace sentence-transformers embedder wrapper."""

from typing import List

from sentence_transformers import SentenceTransformer

from src.core.embedder import BaseEmbedder
from src.registry import ModuleRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@ModuleRegistry.embedders.register("bge-m3")
@ModuleRegistry.embedders.register("bge-small")
class HuggingFaceEmbedder(BaseEmbedder):
    """Embedder using sentence-transformers models (BGE-M3 default)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 32,
    ):
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model_name = model_name
        self._batch_size = batch_size
        self._normalize = normalize
        self._model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
        )
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self._dim}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        embedding = self._model.encode(
            [query],
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        return self._dim
