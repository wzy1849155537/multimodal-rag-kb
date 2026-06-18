"""Main RAG pipeline orchestrator. Ties together all pluggable modules."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import ConfigLoader
from src.core.schemas import (
    RawDocument, Chunk, SearchResult, GeneratedAnswer, IndexStats,
)
from src.core.parser import BaseParser
from src.core.chunker import BaseChunker
from src.core.embedder import BaseEmbedder
from src.core.indexer import BaseIndexer
from src.core.generator import BaseGenerator
from src.chunkers.chunk_cleaner import ChunkCleaner
from src.registry import ModuleRegistry
from src.utils.file_utils import find_files
from src.utils.timer import Timer
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Import implementations so they self-register with ModuleRegistry
import src.parsers.pdf_parser       # noqa: F401
import src.parsers.markdown_parser  # noqa: F401
import src.parsers.text_parser      # noqa: F401
import src.parsers.video_parser     # noqa: F401
import src.chunkers.recursive_chunker  # noqa: F401
import src.embedders.huggingface_embedder  # noqa: F401
import src.embedders.openai_embedder       # noqa: F401
import src.index.vector_store       # noqa: F401
import src.index.hybrid_index       # noqa: F401
import src.generation.rag_generator # noqa: F401


class RAGPipeline:
    """Top-level orchestrator for the RAG system."""

    def __init__(self, config: Optional[ConfigLoader] = None):
        self.config = config or ConfigLoader().load(
            "default.yaml", "models.yaml", "pipeline.yaml"
        )
        self._parser_cache: Dict[str, BaseParser] = {}
        self._chunker: Optional[BaseChunker] = None
        self._embedder: Optional[BaseEmbedder] = None
        self._indexer: Optional[BaseIndexer] = None
        self._generator: Optional[BaseGenerator] = None
        self._initialized = False

    def initialize(self) -> None:
        """Lazy-init all modules from config."""
        if self._initialized:
            return

        pipe = self.config.data.get("pipeline", self.config.data)

        # Embedder
        embedder_name = pipe.get("embedder", "bge-m3")
        embedder_cfg = self.config.data.get("embedding", {}).get("models", {}).get(embedder_name, {})
        self._embedder = ModuleRegistry.embedders.build(
            embedder_name,
            **embedder_cfg,
        )

        # Indexer
        indexer_name = pipe.get("indexer", "dense")
        idx_cfg = self.config.data.get("index", {})
        self._indexer = ModuleRegistry.indexers.build(
            indexer_name,
            persist_directory=idx_cfg.get("index_dir", "./data/index/chroma"),
            collection_name=idx_cfg.get("collection_name", "rag_kb_default"),
            distance_metric=idx_cfg.get("distance_metric", "cosine"),
        )

        # Chunker
        chunker_name = pipe.get("chunker", "recursive")
        chunk_cfg = self.config.data.get("chunking", {})
        self._chunker = ModuleRegistry.chunkers.build(
            chunker_name,
            chunk_size=chunk_cfg.get("chunk_size", 512),
            chunk_overlap=chunk_cfg.get("chunk_overlap", 64),
            separators=chunk_cfg.get("separators"),
        )

        # Generator
        gen_name = pipe.get("generator", "rag")
        gen_cfg = self.config.data.get("generation", {})
        llm_cfg = self._get_llm_config()
        self._generator = ModuleRegistry.generators.build(
            gen_name,
            api_base=llm_cfg.get("api_base", "https://api.openai.com/v1"),
            api_key=llm_cfg.get("api_key", os.environ.get("OPENAI_API_KEY", "")),
            model=llm_cfg.get("model", "gpt-4o-mini"),
            temperature=gen_cfg.get("llm_temperature", 0.3),
            max_context_tokens=gen_cfg.get("max_context_tokens", 4096),
        )

        self._initialized = True
        logger.info("RAGPipeline initialized")

    def _get_llm_config(self) -> Dict[str, Any]:
        """Extract LLM config from models config."""
        models_cfg = self.config.data.get("llm", {})
        default_provider = models_cfg.get("default", "qwen")
        providers = models_cfg.get("providers", {})
        if default_provider in providers:
            return providers[default_provider]
        # Fallback: first provider
        for prov in providers.values():
            return prov
        return {"model": "gpt-4o-mini"}

    def _get_parser(self, file_path: Path) -> Optional[BaseParser]:
        """Find the right parser for a file."""
        ext = file_path.suffix.lower()
        pipe = self.config.data.get("pipeline", self.config.data)
        parsers_cfg = pipe.get("parsers", {})

        parser_name = parsers_cfg.get(ext, "")
        if not parser_name:
            return None

        if parser_name not in self._parser_cache:
            # Pass VLM/OCR/ASR config to parsers that need it
            kwargs = {}
            if parser_name in ("pdf", "video"):
                llm_cfg = self._get_llm_config()
                kwargs["vlm_api_key"] = llm_cfg.get("api_key", "")
                kwargs["vlm_api_base"] = llm_cfg.get("api_base", "https://api.siliconflow.cn/v1")
                kwargs["asr_api_key"] = llm_cfg.get("api_key", "")
                kwargs["asr_api_base"] = llm_cfg.get("api_base", "https://api.siliconflow.cn/v1")
            self._parser_cache[parser_name] = ModuleRegistry.parsers.build(
                parser_name, **kwargs
            )
        return self._parser_cache[parser_name]

    # ========================================================================
    # Ingestion
    # ========================================================================

    def ingest_file(self, file_path: Path) -> int:
        """Ingest a single file. Returns number of chunks indexed."""
        self.initialize()

        parser = self._get_parser(file_path)
        if parser is None:
            logger.warning(f"No parser for {file_path.suffix}, skipping: {file_path}")
            return 0

        # Parse
        with Timer(f"parse {file_path.name}") as t:
            doc = parser.parse(file_path)
        logger.info(f"Parsed in {t.elapsed_ms:.0f}ms")

        return self._index_document(doc)

    def ingest_directory(self, directory: Path, extensions: Optional[List[str]] = None) -> int:
        """Ingest all supported files in a directory."""
        self.initialize()

        if extensions is None:
            pipe = self.config.data.get("pipeline", self.config.data)
            parsers_cfg = pipe.get("parsers", {})
            extensions = list(parsers_cfg.keys())

        files = find_files(directory, extensions=extensions)
        logger.info(f"Found {len(files)} files to ingest in {directory}")

        total_chunks = 0
        for file_path in files:
            try:
                chunks = self.ingest_file(file_path)
                total_chunks += chunks
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")

        logger.info(f"Ingestion complete: {total_chunks} total chunks")
        return total_chunks

    def _index_document(self, doc: RawDocument) -> int:
        """Parse → chunk → clean → embed → index for a document."""
        self.initialize()

        # Chunk
        chunks = self._chunker.chunk(doc)

        # Clean
        chunks = ChunkCleaner.clean(chunks)

        if not chunks:
            logger.info(f"No chunks produced for {doc.doc_id}")
            return 0

        # Embed
        texts = [c.content for c in chunks]
        with Timer(f"embed {len(texts)} chunks") as t:
            embeddings = self._embedder.embed_texts(texts)
        logger.info(f"Embedded {len(texts)} chunks in {t.elapsed_ms:.0f}ms")

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # Index
        self._indexer.add_chunks(chunks)
        return len(chunks)

    # ========================================================================
    # Query
    # ========================================================================

    def query(
        self,
        question: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None,
    ) -> GeneratedAnswer:
        """Run the full RAG query pipeline."""
        self.initialize()

        with Timer("total_query") as total_timer:
            # Embed query
            with Timer("embed_query") as t:
                query_embedding = self._embedder.embed_query(question)
            logger.debug(f"Query embedding: {t.elapsed_ms:.0f}ms")

            # Retrieve
            with Timer("retrieve") as t:
                results = self._indexer.search_dense(
                    query_embedding, top_k=top_k, filter_metadata=filter_metadata
                )
            logger.debug(f"Retrieved {len(results)} results in {t.elapsed_ms:.0f}ms")

            # Generate
            with Timer("generate") as t:
                answer = self._generator.generate(question, results)
            logger.debug(f"Generation: {t.elapsed_ms:.0f}ms")

        answer.latency_ms = total_timer.elapsed_ms
        return answer

    # ========================================================================
    # Management
    # ========================================================================

    def get_stats(self) -> Dict:
        """Get index statistics."""
        self.initialize()
        return self._indexer.get_stats()

    def clear_index(self) -> None:
        """Clear the entire index."""
        self.initialize()
        if hasattr(self._indexer, 'clear'):
            self._indexer.clear()
            logger.info("Index cleared")

    def delete_document(self, doc_id: str) -> int:
        """Delete a document from the index."""
        self.initialize()
        return self._indexer.delete_by_doc_id(doc_id)
