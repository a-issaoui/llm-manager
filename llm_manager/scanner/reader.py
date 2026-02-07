"""
GGUF file reader and hashing utilities.
"""

import hashlib
import logging
import struct
from pathlib import Path
from typing import IO, Any

from .types import GGUFConstants

logger = logging.getLogger(__name__)


class GGUFReader:
    """Minimal GGUF reader to extract metadata"""

    @classmethod
    def read_value(cls, f: IO[bytes], value_type: int) -> Any:
        """Read a GGUF value based on type"""
        if value_type == GGUFConstants.UINT8:
            return struct.unpack("<B", f.read(1))[0]
        elif value_type == GGUFConstants.INT8:
            return struct.unpack("<b", f.read(1))[0]
        elif value_type == GGUFConstants.UINT16:
            return struct.unpack("<H", f.read(2))[0]
        elif value_type == GGUFConstants.INT16:
            return struct.unpack("<h", f.read(2))[0]
        elif value_type == GGUFConstants.UINT32:
            return struct.unpack("<I", f.read(4))[0]
        elif value_type == GGUFConstants.INT32:
            return struct.unpack("<i", f.read(4))[0]
        elif value_type == GGUFConstants.FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        elif value_type == GGUFConstants.BOOL:
            return struct.unpack("?", f.read(1))[0]
        elif value_type == GGUFConstants.STRING:
            length = struct.unpack("<Q", f.read(8))[0]
            return f.read(length).decode("utf-8", errors="ignore")
        elif value_type == GGUFConstants.ARRAY:
            item_type = struct.unpack("<I", f.read(4))[0]
            count = struct.unpack("<Q", f.read(8))[0]

            # If too large, we still MUST consume the data to stay in sync
            if count > 10000: # Increased limit slightly, still safety truncate
                # For fixed-size types, we can seek
                item_size = GGUFConstants.TYPE_SIZES.get(item_type)
                if item_size is not None:
                    f.seek(count * item_size, 1)
                    return f"<array:{count} (skipped)>"

                # For variable-size (STRING/ARRAY), we must read them one by one
                # to know how much to skip.
                for _ in range(count):
                    cls.read_value(f, item_type)
                return f"<array:{count} (skipped)>"

            values = []
            for _ in range(count):
                values.append(cls.read_value(f, item_type))
            return values
        elif value_type == GGUFConstants.UINT64:
            return struct.unpack("<Q", f.read(8))[0]
        elif value_type == GGUFConstants.INT64:
            return struct.unpack("<q", f.read(8))[0]
        elif value_type == GGUFConstants.FLOAT64:
            return struct.unpack("<d", f.read(8))[0]

        raise ValueError(f"Unsupported GGUF value type: {value_type}")

    @classmethod
    def extract_metadata(cls, filepath: str) -> dict[str, Any] | None:
        """Read metadata key table from GGUF file"""
        metadata = {}
        try:
            with open(filepath, "rb") as f:
                # Magic number
                magic = f.read(4)
                if magic != b"GGUF":
                    return None

                # Version
                f.read(4)

                # Tensor count
                f.read(8)

                # Metadata key-value count
                kv_count = struct.unpack("<Q", f.read(8))[0]

                for i in range(kv_count):
                    # Key
                    key_len = struct.unpack("<Q", f.read(8))[0]
                    key = f.read(key_len).decode("utf-8", errors="ignore")

                    # Value type
                    value_type = struct.unpack("<I", f.read(4))[0]

                    # Value
                    value = cls.read_value(f, value_type)
                    metadata[key] = value

        except Exception as e:
            logger.debug(f"Error reading GGUF header for {filepath}: {e}")
            return None

        return metadata


def get_file_hash(filepath: str) -> str:
    """Quick hash of first/last 1MB for change detection"""
    try:
        path = Path(filepath)
        size = path.stat().st_size
        sha256 = hashlib.sha256()

        with open(filepath, "rb") as f:
            # First 1MB
            sha256.update(f.read(1024 * 1024))
            # Last 1MB
            if size > 2 * 1024 * 1024:
                f.seek(-1024 * 1024, 2)
                sha256.update(f.read(1024 * 1024))
            elif size > 1024 * 1024:
                # Overlap but simple logic
                f.seek(0, 0)
                sha256.update(f.read())

        # Include size and mtime for speed
        sha256.update(str(size).encode())
        sha256.update(str(path.stat().st_mtime).encode())

        return sha256.hexdigest()[:16]

    except Exception:
        return ""
