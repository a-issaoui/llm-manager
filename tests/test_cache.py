"""
Tests for llm_manager/cache.py - Persistent disk cache.
"""

import json
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from queue import Full

import pytest

from unittest.mock import patch, Mock, MagicMock
from llm_manager.cache import DiskCache


class TestDiskCacheBasics:
    """Basic DiskCache operations."""

    def test_init_creates_db(self, tmp_path):
        """Test initialization creates database file."""
        db_path = tmp_path / "test_cache.db"
        cache = DiskCache(db_path)
        assert db_path.exists()

    def test_init_creates_parent_dirs(self, tmp_path):
        """Test initialization creates parent directories."""
        db_path = tmp_path / "nested" / "dirs" / "cache.db"
        cache = DiskCache(db_path)
        assert db_path.parent.exists()

    def test_cache_init_error(self):
        """Cover cache initialization error."""
        # Path where it is USED
        with patch('llm_manager.cache.sqlite3.connect', side_effect=Exception("SQL Error")):
            # DiskCache should handle the error during init
            cache = DiskCache(":memory:")
            assert cache.disabled is True

    def test_set_and_get(self, tmp_path):
        """Test basic set and get operations."""
        cache = DiskCache(tmp_path / "cache.db")
        cache.set("key1", {"value": 100})
        result = cache.get("key1")
        assert result == {"value": 100}

    def test_get_nonexistent(self, tmp_path):
        """Test get returns None for non-existent key."""
        cache = DiskCache(tmp_path / "cache.db")
        result = cache.get("nonexistent")
        assert result is None

    def test_get_expired(self, tmp_path):
        """Test get returns None for expired key."""
        cache = DiskCache(tmp_path / "cache.db")
        cache.set("key1", {"value": 100}, ttl=0)
        time.sleep(0.01)
        result = cache.get("key1")
        assert result is None

    def test_set_overwrites(self, tmp_path):
        """Test set overwrites existing value."""
        cache = DiskCache(tmp_path / "cache.db")
        cache.set("key1", {"value": 100})
        cache.set("key1", {"value": 200})
        result = cache.get("key1")
        assert result == {"value": 200}

    def test_delete_existing(self, tmp_path):
        """Test delete removes existing key."""
        cache = DiskCache(tmp_path / "cache.db")
        cache.set("key1", {"value": 100})
        deleted = cache.delete("key1")
        assert deleted is True
        assert cache.get("key1") is None

    def test_delete_nonexistent(self, tmp_path):
        """Test delete returns False for non-existent key."""
        cache = DiskCache(tmp_path / "cache.db")
        deleted = cache.delete("nonexistent")
        assert deleted is False

    def test_clear(self, tmp_path):
        """Test clear removes all entries."""
        cache = DiskCache(tmp_path / "cache.db")
        cache.set("key1", {"value": 1})
        cache.set("key2", {"value": 2})
        count = cache.clear()
        assert count == 2
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_clear_empty(self, tmp_path):
        """Test clear on empty cache."""
        cache = DiskCache(tmp_path / "cache.db")
        count = cache.clear()
        assert count == 0


class TestDiskCacheTTL:
    """TTL-related tests."""

    def test_custom_ttl(self, tmp_path):
        """Test custom TTL is respected."""
        cache = DiskCache(tmp_path / "cache.db")
        cache.set("key1", {"value": 100}, ttl=3600)
        time.sleep(0.01)
        result = cache.get("key1")
        assert result == {"value": 100}

    def test_default_ttl_long(self, tmp_path):
        """Test default TTL (7 days)."""
        cache = DiskCache(tmp_path / "cache.db")
        cache.set("key1", {"value": 100})
        result = cache.get("key1")
        assert result == {"value": 100}

    def test_cleanup_expired(self, tmp_path):
        """Test cleanup_expired removes expired entries."""
        cache = DiskCache(tmp_path / "cache.db")
        cache.set("key1", {"value": 1}, ttl=0)
        cache.set("key2", {"value": 2}, ttl=3600)
        time.sleep(0.01)
        removed = cache.cleanup_expired()
        assert removed == 1
        assert cache.get("key1") is None
        assert cache.get("key2") == {"value": 2}


class TestDiskCacheEviction:
    """LRU eviction tests."""

    def test_eviction_when_over_max(self, tmp_path):
        """Test eviction when exceeding max_size."""
        cache = DiskCache(tmp_path / "cache.db", max_size=5)
        for i in range(7):
            cache.set(f"key{i}", {"value": i})
            time.sleep(0.01)

        assert cache.get("key0") is None
        assert cache.get("key1") is None
        assert cache.get("key5") == {"value": 5}
        assert cache.get("key6") == {"value": 6}

    def test_access_time_updated_on_get(self, tmp_path):
        """Test that get updates access time."""
        cache = DiskCache(tmp_path / "cache.db", max_size=3)
        cache.set("key1", {"value": 1})
        time.sleep(0.01)
        cache.set("key2", {"value": 2})
        time.sleep(0.01)
        cache.set("key3", {"value": 3})

        cache.get("key1")
        time.sleep(0.01)
        cache.set("key4", {"value": 4})

        assert cache.get("key1") == {"value": 1}


class TestDiskCacheStats:
    """Statistics tests."""

    def test_stats_initial(self, tmp_path):
        """Test initial stats are zero."""
        cache = DiskCache(tmp_path / "cache.db")
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["size"] == 0

    def test_stats_hits(self, tmp_path):
        """Test hit tracking."""
        cache = DiskCache(tmp_path / "cache.db")
        cache.set("key1", {"value": 100})
        cache.get("key1")
        cache.get("key1")
        stats = cache.get_stats()
        assert stats["hits"] == 2

    def test_stats_misses(self, tmp_path):
        """Test miss tracking."""
        cache = DiskCache(tmp_path / "cache.db")
        cache.get("nonexistent")
        cache.get("also_nonexistent")
        stats = cache.get_stats()
        assert stats["misses"] == 2

    def test_stats_evictions(self, tmp_path):
        """Test eviction tracking."""
        cache = DiskCache(tmp_path / "cache.db", max_size=3)
        for i in range(5):
            cache.set(f"key{i}", {"value": i})
            time.sleep(0.01)
        stats = cache.get_stats()
        assert stats["evictions"] >= 1

    def test_stats_hit_rate(self, tmp_path):
        """Test hit rate calculation."""
        cache = DiskCache(tmp_path / "cache.db")
        cache.set("key1", {"value": 100})
        cache.get("key1")
        cache.get("key1")
        cache.get("nonexistent")
        stats = cache.get_stats()
        assert stats["hit_rate"] == 2 / 3


class TestDiskCacheConcurrency:
    """Thread safety tests."""

    def test_concurrent_reads(self, tmp_path):
        """Test concurrent reads are safe."""
        cache = DiskCache(tmp_path / "cache.db")
        cache.set("key1", {"value": 100})

        results = []
        threads = []

        def read_cache():
            for _ in range(10):
                results.append(cache.get("key1"))

        for _ in range(5):
            t = threading.Thread(target=read_cache)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert all(r == {"value": 100} for r in results if r is not None)

    def test_concurrent_writes(self, tmp_path):
        """Test concurrent writes are safe."""
        cache = DiskCache(tmp_path / "cache.db")
        threads = []

        def write_cache(i):
            cache.set(f"key{i}", {"value": i})

        for i in range(10):
            t = threading.Thread(target=write_cache, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        for i in range(10):
            assert cache.get(f"key{i}") == {"value": i}


class TestDiskCacheLen:
    """__len__ tests."""

    def test_len_empty(self, tmp_path):
        """Test len on empty cache."""
        cache = DiskCache(tmp_path / "cache.db")
        assert len(cache) == 0

    def test_len_with_items(self, tmp_path):
        """Test len with items."""
        cache = DiskCache(tmp_path / "cache.db")
        cache.set("key1", {"value": 1})
        cache.set("key2", {"value": 2})
        assert len(cache) == 2


class TestDiskCacheEdgeCases:
    """Edge case tests."""

    def test_empty_string_key(self, tmp_path):
        """Test empty string as key."""
        cache = DiskCache(tmp_path / "cache.db")
        cache.set("", {"value": 100})
        assert cache.get("") == {"value": 100}

    def test_nested_dict(self, tmp_path):
        """Test nested dictionary storage."""
        cache = DiskCache(tmp_path / "cache.db")
        data = {"outer": {"inner": [1, 2, 3], "value": "test"}}
        cache.set("key1", data)
        assert cache.get("key1") == data

    def test_large_value(self, tmp_path):
        """Test large value storage."""
        cache = DiskCache(tmp_path / "cache.db")
        large_data = {"data": "x" * 10000}
        cache.set("key1", large_data)
        assert cache.get("key1") == large_data

    def test_unicode_key_and_value(self, tmp_path):
        """Test unicode in key and value."""
        cache = DiskCache(tmp_path / "cache.db")
        cache.set("日本語キー", {"value": "日本語値"})
        assert cache.get("日本語キー") == {"value": "日本語値"}

    def test_cache_connection_pool_full(self, tmp_path):
        """Cover cache connection pool full exception."""
        cache = DiskCache(tmp_path / "cache.db")
        conn = Mock()
        cache._pool = MagicMock()
        cache._pool.put_nowait.side_effect = Full

        cache._release_connection(conn)
        conn.close.assert_called_once()
