"""SHA256 file hash computation and storage."""

import hashlib
from pathlib import Path
from typing import Dict, Optional

from src.utils.file_utils import find_files, load_json, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HashManager:
    """Manages file hashes for incremental indexing."""

    def __init__(
        self,
        store_path: Path,
        algorithm: str = "sha256",
    ):
        self.store_path = Path(store_path)
        self.algorithm = algorithm
        self._store: Dict[str, str] = {}

    def load(self) -> Dict[str, str]:
        """Load the stored hash map."""
        self._store = load_json(self.store_path)
        return self._store

    def save(self, hashes: Optional[Dict[str, str]] = None) -> None:
        """Save hashes to the store."""
        if hashes is not None:
            self._store = hashes
        save_json(self.store_path, self._store)

    def compute(self, file_path: Path) -> str:
        """Compute the hash of a single file."""
        h = hashlib.new(self.algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def compute_all(
        self,
        directory: Path,
        extensions: Optional[list] = None,
    ) -> Dict[str, str]:
        """Compute hashes for all files in a directory.

        Returns {relative_path: hash}.
        """
        files = find_files(directory, extensions=extensions)
        hashes = {}
        for f in files:
            rel_path = str(f.relative_to(directory))
            hashes[rel_path] = self.compute(f)
        logger.info(f"Computed {len(hashes)} hashes from {directory}")
        return hashes
