"""
Unified Scanner for llm_manager.

Combines metadata extraction, GPU context testing, and llm_manager integration.
"""

from .constants import MAX_REASONABLE_CONTEXT, __version__
from .core import (
    ModelScanner,
    PerfectScanner,
    main,
    scan_models,
    scan_models_async,
)
from .detector import CapabilityDetector
from .metadata import MetadataExtractor
from .reader import GGUFReader, get_file_hash
from .tester import ContextTester, cleanup
from .types import (
    ArchitectureDefaults,
    ContextTestResult,
    GGUFConstants,
    ModelCapabilities,
    ModelEntry,
    ModelSpecs,
    QuantizationType,
)

__all__ = [
    "MAX_REASONABLE_CONTEXT",
    "ArchitectureDefaults",
    "CapabilityDetector",
    "ContextTestResult",
    "ContextTester",
    "GGUFConstants",
    "GGUFReader",
    "MetadataExtractor",
    "ModelCapabilities",
    "ModelEntry",
    "ModelScanner",
    "ModelSpecs",
    "PerfectScanner",
    "QuantizationType",
    "__version__",
    "cleanup",
    "get_file_hash",
    "main",
    "scan_models",
    "scan_models_async",
]
