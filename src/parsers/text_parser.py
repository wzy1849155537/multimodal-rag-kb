"""Plain text parser (fallback for unsupported file types)."""

import uuid
from pathlib import Path
from typing import List

from src.core.parser import BaseParser
from src.core.schemas import RawDocument
from src.registry import ModuleRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@ModuleRegistry.parsers.register("text")
class TextParser(BaseParser):
    """Parse plain text files."""

    supported_extensions: List[str] = [".txt", ".log", ".csv", ".json", ".xml"]

    def parse(self, file_path: Path) -> RawDocument:
        logger.info(f"Parsing text: {file_path.name}")
        doc_id = str(uuid.uuid4())[:8]

        # Try UTF-8 first, then fall back to other encodings
        content = ""
        for encoding in ["utf-8", "gbk", "gb2312", "latin-1"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if not content and content != "":
            # Last resort: read as bytes and decode with errors='replace'
            with open(file_path, "rb") as f:
                content = f.read().decode("utf-8", errors="replace")

        metadata = {
            "file_name": file_path.name,
            "file_stem": file_path.stem,
            "file_type": "text",
        }

        logger.info(f"Text parsed: {len(content)} chars from {file_path.name}")
        return RawDocument(
            doc_id=doc_id,
            source_path=str(file_path),
            content=content,
            metadata=metadata,
        )
