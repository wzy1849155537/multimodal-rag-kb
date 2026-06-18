"""Change detection: compare current vs stored hashes."""

from typing import Dict, List

from src.core.schemas import FileChange, ChangeType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChangeDetector:
    """Detects file changes by comparing hash maps."""

    @staticmethod
    def detect(
        current_hashes: Dict[str, str],
        stored_hashes: Dict[str, str],
    ) -> List[FileChange]:
        """Compare current vs stored hashes and classify changes.

        Returns list of FileChange objects for ADDED, MODIFIED, and DELETED files.
        UNCHANGED files are NOT returned (no action needed).
        """
        changes: List[FileChange] = []
        current_files = set(current_hashes.keys())
        stored_files = set(stored_hashes.keys())

        # ADDED: files in current but not in stored
        for path in current_files - stored_files:
            changes.append(FileChange(
                file_path=path,
                change_type=ChangeType.ADDED,
                old_hash="",
                new_hash=current_hashes[path],
            ))

        # MODIFIED: files in both but hash differs
        for path in current_files & stored_files:
            if current_hashes[path] != stored_hashes[path]:
                changes.append(FileChange(
                    file_path=path,
                    change_type=ChangeType.MODIFIED,
                    old_hash=stored_hashes[path],
                    new_hash=current_hashes[path],
                ))

        # DELETED: files in stored but not in current
        for path in stored_files - current_files:
            changes.append(FileChange(
                file_path=path,
                change_type=ChangeType.DELETED,
                old_hash=stored_hashes[path],
                new_hash="",
            ))

        logger.info(
            f"Change detection: {len(changes)} changes "
            f"(A:{sum(1 for c in changes if c.change_type == ChangeType.ADDED)} "
            f"M:{sum(1 for c in changes if c.change_type == ChangeType.MODIFIED)} "
            f"D:{sum(1 for c in changes if c.change_type == ChangeType.DELETED)})"
        )
        return changes
