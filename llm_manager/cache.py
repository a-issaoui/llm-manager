"""
Persistent disk cache for token estimation.

Uses SQLite for fast, reliable, dependency-free caching.
"""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any

logger = logging.getLogger(__name__)


class DiskCache:
    """
    SQLite-based persistent cache with TTL and LRU eviction.

    Thread-safe implementation for cross-session token estimation caching.

    Attributes:
        path: Path to SQLite database file
        max_size: Maximum number of entries
        default_ttl: Default time-to-live in seconds

    Examples:
        >>> cache = DiskCache(Path("./cache.db"))
        >>> cache.set("key1", {"tokens": 100})
        >>> cache.get("key1")
        {'tokens': 100}
    """

    def __init__(
        self,
        path: Path,
        max_size: int = 10000,
        default_ttl: int = 86400 * 7,  # 7 days
    ):
        """
        Initialize disk cache.

        Args:
            path: Path to SQLite database file
            max_size: Maximum number of cached entries
            default_ttl: Default TTL in seconds (7 days)
        """
        self.path = Path(path)
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.disabled = False

        # Connection pool with bounded concurrency
        self._pool: Queue[sqlite3.Connection] = Queue(maxsize=10)
        self._max_connections = 10
        self._active_connections = 0
        self._pool_lock = threading.Lock()

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        try:
            self._init_db()
        except Exception as e:
            logger.error(f"Failed to initialize disk cache: {e}")
            self.disabled = True

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires
                ON cache(expires_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_accessed
                ON cache(accessed_at)
            """)
            conn.commit()
            self._release_connection(conn)

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a connection from the pool."""
        try:
            return self._pool.get_nowait()
        except Empty:
            with self._pool_lock:
                if self._active_connections >= self._max_connections:
                    # Pool exhausted, wait for a connection with timeout
                    try:
                        return self._pool.get(timeout=5.0)
                    except Empty:
                        raise RuntimeError(
                            "Connection pool exhausted - too many concurrent operations"
                        ) from None

                self._active_connections += 1

            conn = sqlite3.connect(str(self.path), timeout=5.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            return conn

    def _release_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        from queue import Full

        try:
            self._pool.put_nowait(conn)
        except Full:
            # Pool is full, close this connection and decrement counter
            with self._pool_lock:
                self._active_connections = max(0, self._active_connections - 1)
            conn.close()

    def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        now = time.time()

        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
                ).fetchone()

                if row is None:
                    self._stats["misses"] += 1
                    self._release_connection(conn)
                    return None

                if row["expires_at"] < now:
                    # Expired - delete and return None
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()
                    self._stats["misses"] += 1
                    self._release_connection(conn)
                    return None

                # Update access time
                conn.execute("UPDATE cache SET accessed_at = ? WHERE key = ?", (now, key))
                conn.commit()

                self._stats["hits"] += 1
                result = json.loads(row["value"])
                self._release_connection(conn)
                return result

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Optional TTL override in seconds
        """
        now = time.time()
        expires_at = now + (ttl if ttl is not None else self.default_ttl)
        value_json = json.dumps(value)

        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache
                    (key, value, expires_at, created_at, accessed_at)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (key, value_json, expires_at, now, now),
                )
                conn.commit()

                # Check if eviction needed
                self._maybe_evict(conn)
                self._release_connection(conn)

    def _maybe_evict(self, conn: sqlite3.Connection) -> None:
        """Evict entries if over max size."""
        count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]

        if count > self.max_size:
            # Delete oldest accessed entries
            to_delete = count - self.max_size + (self.max_size // 10)  # 10% buffer
            conn.execute(
                """
                DELETE FROM cache WHERE key IN (
                    SELECT key FROM cache
                    ORDER BY accessed_at ASC
                    LIMIT ?
                )
            """,
                (to_delete,),
            )
            conn.commit()
            self._stats["evictions"] += to_delete
            logger.debug(f"Evicted {to_delete} cache entries")

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                result = cursor.rowcount > 0
                self._release_connection(conn)
                return result

    def clear(self) -> int:
        """
        Clear all entries from cache.

        Returns:
            Number of entries deleted
        """
        with self._lock:
            with self._get_connection() as conn:
                count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
                conn.execute("DELETE FROM cache")
                conn.commit()
                self._release_connection(conn)
                return int(count)

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        now = time.time()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM cache WHERE expires_at < ?", (now,))
                conn.commit()
                self._release_connection(conn)
                return int(cursor.rowcount) if cursor.rowcount is not None else 0

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with hits, misses, evictions, and size
        """
        with self._lock:
            with self._get_connection() as conn:
                size = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
                self._release_connection(conn)

        return {
            **self._stats,
            "size": size,
            "max_size": self.max_size,
            "hit_rate": (self._stats["hits"] / max(1, self._stats["hits"] + self._stats["misses"])),
        }

    def __len__(self) -> int:
        """Return number of cached entries."""
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute("SELECT COUNT(*) FROM cache").fetchone()
                self._release_connection(conn)
                return int(row[0]) if row else 0
