"""Markdown parser with frontmatter support."""

import re
import uuid
from pathlib import Path
from typing import List

from src.core.parser import BaseParser
from src.core.schemas import RawDocument
from src.registry import ModuleRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@ModuleRegistry.parsers.register("markdown")
class MarkdownParser(BaseParser):
    """Parse Markdown files. Extracts text and optional YAML frontmatter."""

    supported_extensions: List[str] = [".md", ".markdown"]

    def parse(self, file_path: Path) -> RawDocument:
        logger.info(f"Parsing Markdown: {file_path.name}")
        doc_id = str(uuid.uuid4())[:8]

        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Extract frontmatter
        frontmatter = {}
        content = raw_text
        match = _FRONTMATTER_RE.match(raw_text)
        if match:
            import yaml
            try:
                frontmatter = yaml.safe_load(match.group(1)) or {}
            except yaml.YAMLError:
                pass
            content = raw_text[match.end():]

        metadata = {
            "file_name": file_path.name,
            "file_stem": file_path.stem,
            "file_type": "markdown",
            "frontmatter": frontmatter,
        }

        logger.info(
            f"Markdown parsed: {len(content)} chars from {file_path.name}"
        )
        return RawDocument(
            doc_id=doc_id,
            source_path=str(file_path),
            content=content,
            metadata=metadata,
        )
