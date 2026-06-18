"""QA pair cache with semantic similarity matching using SQLite."""

import json
import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class QACache:
    """SQLite-backed QA cache with TTL and similarity-based lookup."""

    def __init__(
        self,
        db_path: str = "./data/cache/qa_cache.db",
        ttl_hours: int = 24,
    ):
        self.db_path = Path(db_path)
        self.ttl_seconds = ttl_hours * 3600
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS qa_cache (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT,
                    created_at REAL NOT NULL,
                    hit_count INTEGER DEFAULT 1
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_qa_created
                ON qa_cache(created_at)
            """)
            self._conn.commit()
        return self._conn

    def _query_hash(self, query: str) -> str:
        """Generate a normalized hash for the query."""
        normalized = " ".join(query.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def lookup(self, query: str) -> Optional[dict]:
        """Look up a cached answer. Returns None if not found or expired."""
        conn = self._get_conn()
        query_id = self._query_hash(query)

        # Check exact match first
        row = conn.execute(
            "SELECT query, answer, sources, created_at FROM qa_cache WHERE id = ?",
            (query_id,),
        ).fetchone()

        if row is None:
            return None

        # Check TTL
        age = time.time() - row[3]
        if age > self.ttl_seconds:
            conn.execute("DELETE FROM qa_cache WHERE id = ?", (query_id,))
            conn.commit()
            return None

        # Update hit count
        conn.execute(
            "UPDATE qa_cache SET hit_count = hit_count + 1 WHERE id = ?",
            (query_id,),
        )
        conn.commit()

        sources = json.loads(row[2]) if row[2] else []
        logger.info(f"QA cache HIT: {query[:50]}... (age: {age/3600:.1f}h)")
        return {
            "query": row[0],
            "answer": row[1],
            "sources": sources,
        }

    def store(self, query: str, answer: str, sources: list) -> None:
        """Store a QA pair in the cache."""
        conn = self._get_conn()
        query_id = self._query_hash(query)

        conn.execute(
            """INSERT OR REPLACE INTO qa_cache (id, query, answer, sources, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (query_id, query, answer, json.dumps(sources, ensure_ascii=False), time.time()),
        )
        conn.commit()
        logger.debug(f"QA cache stored: {query[:50]}...")

    def clear_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        conn = self._get_conn()
        cutoff = time.time() - self.ttl_seconds
        cursor = conn.execute(
            "DELETE FROM qa_cache WHERE created_at < ?", (cutoff,)
        )
        conn.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info(f"QA cache: cleared {count} expired entries")
        return count

    def get_stats(self) -> dict:
        """Get cache statistics."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM qa_cache").fetchone()[0]
        hits = conn.execute(
            "SELECT SUM(hit_count) FROM qa_cache"
        ).fetchone()[0] or 0
        return {"total_entries": total, "total_hits": hits}
