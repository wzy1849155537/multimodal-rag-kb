"""BaseParser ABC for document parsing."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from .schemas import RawDocument


class BaseParser(ABC):
    """Pluggable document parser. Implementations handle specific file types."""

    @abstractmethod
    def parse(self, file_path: Path) -> RawDocument:
        """Parse a document file into a RawDocument with extracted text and images."""
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """File extensions this parser handles, e.g. ['.pdf', '.md']."""
        ...

    def can_handle(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file."""
        return file_path.suffix.lower() in self.supported_extensions
