#!/usr/bin/env python3
"""Additional tests for scanner module."""

import pytest

pytest.importorskip("transformers")

from llm_manager.scanner import CapabilityDetector, MetadataExtractor, PerfectScanner


class TestPerfectScanner:
    """Test PerfectScanner class."""

    def test_perfect_scanner_init(self):
        """Test PerfectScanner initialization."""
        scanner = PerfectScanner()
        assert scanner is not None
        assert scanner.results == {}
        assert scanner.stats["total"] == 0

class TestMetadataExtractor:
    """Test MetadataExtractor class."""

    def test_parse_parameters_unknown(self):
        """Test parse_parameters with unknown model."""
        extractor = MetadataExtractor()
        result = extractor.parse_parameters("Unknown", "unknown_model.gguf", {})
        assert result is None

    def test_detect_quantization(self):
        """Test detect_quantization."""
        extractor = MetadataExtractor()
        # Test with metadata
        metadata = {"general.file_type": 2}  # Q4_0
        result = extractor.detect_quantization(metadata, "model.gguf")
        assert isinstance(result, str)

    def test_detect_quantization_from_filename(self):
        """Test detect_quantization from filename."""
        extractor = MetadataExtractor()
        result = extractor.detect_quantization({}, "model-Q4_K_M.gguf")
        assert "Q4_K_M" in result

class TestCapabilityDetector:
    """Test CapabilityDetector class."""

    def test_detect_capabilities(self):
        """Test detect_capabilities."""
        caps = CapabilityDetector.detect("test.gguf", "llama", {})
        assert caps is not None

    def test_detect_embedding(self):
        """Test detect_capabilities for embedding model."""
        caps = CapabilityDetector.detect("nomic-embed.gguf", "nomic", {})
        assert caps.embed

    def test_detect_vision(self):
        """Test detect_capabilities for vision model."""
        caps = CapabilityDetector.detect("llava.gguf", "llava", {})
        assert caps.vision

    def test_is_custom_template(self):
        """Test is_custom_template detection."""
        # Empty or short template
        assert not CapabilityDetector.is_custom_template("")
        assert not CapabilityDetector.is_custom_template("short")

        # Long template is considered custom
        long_template = "a" * 100
        assert CapabilityDetector.is_custom_template(long_template)
