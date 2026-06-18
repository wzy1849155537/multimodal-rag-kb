"""Recursive character text splitter chunker (wraps LangChain)."""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.chunker import BaseChunker
from src.core.schemas import RawDocument, Chunk
from src.registry import ModuleRegistry
from src.utils.text_utils import chunk_id_from, clean_chunk_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


@ModuleRegistry.chunkers.register("recursive")
class RecursiveChunker(BaseChunker):
    """Split documents using LangChain's RecursiveCharacterTextSplitter."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: List[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ".", " ", ""]
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, document: RawDocument) -> List[Chunk]:
        logger.info(
            f"Chunking doc {document.doc_id} "
            f"({len(document.content)} chars) with {self.get_strategy_name()}"
        )
        texts = self._splitter.split_text(document.content)

        chunks = []
        for i, text in enumerate(texts):
            cleaned = clean_chunk_text(text)
            if len(cleaned) < 10:  # Skip empty/near-empty chunks
                continue
            chunk = Chunk(
                chunk_id=chunk_id_from(document.doc_id, i),
                doc_id=document.doc_id,
                content=cleaned,
                chunk_index=i,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "chunk_count": len(texts),
                },
            )
            chunks.append(chunk)

        logger.info(f"Chunking produced {len(chunks)} chunks")
        return chunks

    def get_strategy_name(self) -> str:
        return f"recursive-{self.chunk_size}"

    def __repr__(self) -> str:
        return (
            f"RecursiveChunker(chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap})"
        )
