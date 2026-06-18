"""ChromaDB vector store wrapper."""

from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from src.core.indexer import BaseIndexer
from src.core.schemas import Chunk, SearchResult, SearchSource
from src.registry import ModuleRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@ModuleRegistry.indexers.register("dense")
class ChromaVectorStore(BaseIndexer):
    """ChromaDB-backed dense vector store."""

    def __init__(
        self,
        persist_directory: str = "./data/index/chroma",
        collection_name: str = "rag_kb_default",
        distance_metric: str = "cosine",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )
        logger.info(
            f"ChromaDB collection '{collection_name}' ready "
            f"at {persist_directory}"
        )

    def add_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        embeddings = [c.embedding for c in chunks]
        metadatas = [
            {k: str(v) for k, v in c.metadata.items()}
            for c in chunks
        ]

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(chunks)} chunks to ChromaDB")

    def delete_by_doc_id(self, doc_id: str) -> int:
        results = self._collection.get(
            where={"doc_id": doc_id},
            include=[],
        )
        ids_to_delete = results["ids"]
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for doc {doc_id}")
        return len(ids_to_delete)

    def search_dense(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None,
    ) -> List[SearchResult]:
        where = None
        if filter_metadata:
            where = {k: str(v) for k, v in filter_metadata.items()}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = (
                    results["distances"][0][i]
                    if results["distances"]
                    else 1.0
                )
                # Convert distance to similarity score (for cosine distance)
                score = 1.0 - min(distance, 2.0) / 2.0
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    score=score,
                    source=SearchSource.DENSE,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                ))

        return search_results

    def search_sparse(
        self,
        query_text: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Fallback: use ChromaDB's built-in text search."""
        where = None
        if filter_metadata:
            where = {k: str(v) for k, v in filter_metadata.items()}

        results = self._collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = (
                    results["distances"][0][i]
                    if results["distances"]
                    else 1.0
                )
                score = 1.0 - min(distance, 2.0) / 2.0
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    score=score,
                    source=SearchSource.SPARSE,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                ))

        return search_results

    def get_stats(self) -> Dict:
        count = self._collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_directory": self.persist_directory,
        }

    def clear(self) -> None:
        """Delete the entire collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
        )
        logger.info(f"Cleared collection '{self.collection_name}'")
