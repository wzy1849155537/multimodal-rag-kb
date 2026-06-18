"""Sync engine: applies detected changes to the index."""

from pathlib import Path
from typing import Callable, List, Optional

from src.core.schemas import FileChange, ChangeType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyncEngine:
    """Applies incremental changes to the index."""

    def __init__(
        self,
        doc_dir: Path,
        ingest_fn: Callable[[Path], int],
        delete_fn: Callable[[str], int],
    ):
        self.doc_dir = doc_dir
        self._ingest = ingest_fn
        self._delete = delete_fn

    def apply(self, changes: List[FileChange]) -> dict:
        """Apply a list of file changes to the index.

        Returns stats: {added: N, modified: N, deleted: N, errors: N}
        """
        stats = {"added": 0, "modified": 0, "deleted": 0, "errors": 0}

        for change in changes:
            try:
                if change.change_type == ChangeType.DELETED:
                    # Delete by reconstructing doc_id from file path
                    doc_name = Path(change.file_path).name
                    count = self._delete(doc_name)
                    if count > 0:
                        stats["deleted"] += 1
                    logger.info(f"Deleted: {change.file_path} ({count} chunks)")

                elif change.change_type in (ChangeType.ADDED, ChangeType.MODIFIED):
                    file_path = self.doc_dir / change.file_path
                    if not file_path.exists():
                        logger.warning(f"File not found, skipping: {file_path}")
                        stats["errors"] += 1
                        continue

                    if change.change_type == ChangeType.MODIFIED:
                        # Delete old chunks first (by file stem)
                        doc_name = Path(change.file_path).name
                        self._delete(doc_name)

                    # Ingest new version
                    count = self._ingest(file_path)
                    if count > 0:
                        if change.change_type == ChangeType.ADDED:
                            stats["added"] += 1
                        else:
                            stats["modified"] += 1
                    logger.info(
                        f"{'Added' if change.change_type == ChangeType.ADDED else 'Modified'}: "
                        f"{change.file_path} ({count} chunks)"
                    )

            except Exception as e:
                logger.error(f"Sync error for {change.file_path}: {e}")
                stats["errors"] += 1

        logger.info(
            f"Sync complete: A={stats['added']} M={stats['modified']} "
            f"D={stats['deleted']} E={stats['errors']}"
        )
        return stats
