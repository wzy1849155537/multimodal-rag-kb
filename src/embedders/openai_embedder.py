"""OpenAI-compatible API embedder (works with SiliconFlow, OpenAI, etc.)."""

from typing import List

from openai import OpenAI

from src.core.embedder import BaseEmbedder
from src.registry import ModuleRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@ModuleRegistry.embedders.register("openai")
@ModuleRegistry.embedders.register("siliconflow")
class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI-compatible API (SiliconFlow, OpenAI, etc.)."""

    def __init__(
        self,
        api_base: str = "https://api.siliconflow.cn/v1",
        api_key: str = "",
        model_name: str = "BAAI/bge-m3",
        dimension: int = 1024,
        batch_size: int = 32,
        **kwargs,  # Accept and ignore extra config (device, normalize, etc.)
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self._dim = dimension
        self._batch_size = batch_size
        self._client = OpenAI(base_url=api_base, api_key=api_key)
        logger.info(
            f"OpenAIEmbedder initialized: {model_name} via {api_base}"
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            try:
                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                )
                batch_embeddings = [
                    r.embedding for r in response.data
                ]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Embedding batch failed: {e}")
                # Return zeros as fallback
                for _ in batch:
                    all_embeddings.append([0.0] * self._dim)

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        try:
            response = self._client.embeddings.create(
                model=self.model_name,
                input=[query],
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return [0.0] * self._dim

    @property
    def dimension(self) -> int:
        return self._dim
