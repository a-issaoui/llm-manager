"""
Tests for llm_manager/scanner.py - GGUF Scanner Integration
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from llm_manager.scanner import ModelScanner, scan_models
from llm_manager.exceptions import ValidationError


class TestModelScanner:
    """Tests for ModelScanner class."""

    def test_init_default(self, tmp_path):
        """Test scanner initialization with defaults."""
        scanner = ModelScanner(tmp_path)
        
        assert scanner.models_dir == tmp_path
        assert scanner.registry_file == tmp_path / "models.json"
        assert scanner.config_file == tmp_path / "llm_manager.yaml"

    def test_init_custom_files(self, tmp_path):
        """Test scanner initialization with custom file names."""
        scanner = ModelScanner(
            tmp_path,
            registry_file="custom.json",
            config_file="custom.yaml"
        )
        
        assert scanner.registry_file == tmp_path / "custom.json"
        assert scanner.config_file == tmp_path / "custom.yaml"

    def test_scan_and_save_validation_error(self, tmp_path):
        """Test scan fails when directory doesn't exist."""
        nonexistent = tmp_path / "nonexistent"
        scanner = ModelScanner(nonexistent)
        
        with pytest.raises(ValidationError) as exc_info:
            scanner.scan_and_save()
        
        assert "not found" in str(exc_info.value).lower()

    def test_get_registry_not_found(self, tmp_path):
        """Test get_registry fails when registry doesn't exist."""
        scanner = ModelScanner(tmp_path)
        
        with pytest.raises(ValidationError) as exc_info:
            scanner.get_registry()
        
        assert "not found" in str(exc_info.value).lower()

    def test_quick_scan_calls_scan_and_save(self, tmp_path):
        """Test quick_scan calls scan_and_save with test_context=False."""
        scanner = ModelScanner(tmp_path)
        
        with patch.object(scanner, 'scan_and_save') as mock_scan:
            mock_scan.return_value = {"models_found": 5}
            result = scanner.quick_scan()
        
        mock_scan.assert_called_once_with(test_context=False)
        assert result["models_found"] == 5


class TestScanModelsFunction:
    """Tests for scan_models convenience function."""

    def test_scan_models_basic(self, tmp_path):
        """Test scan_models convenience function."""
        with patch('llm_manager.scanner.ModelScanner') as mock_scanner_class:
            mock_scanner = Mock()
            mock_scanner.scan_and_save.return_value = {"models_found": 3}
            mock_scanner_class.return_value = mock_scanner
            
            result = scan_models(tmp_path, test_context=True)
        
        mock_scanner_class.assert_called_once_with(tmp_path)
        mock_scanner.scan_and_save.assert_called_once_with(test_context=True)
        assert result["models_found"] == 3


class TestScannerIntegration:
    """Integration tests for scanner with llm_manager."""

    def test_scanner_with_manager(self, tmp_path):
        """Test scanner integration with LLMManager."""
        from llm_manager import LLMManager
        
        # Create a mock GGUF file
        mock_gguf = tmp_path / "test-model.gguf"
        mock_gguf.write_bytes(b"GGUF\x00" * 100)
        
        with patch('llm_manager.core.ModelRegistry'):
            manager = LLMManager(models_dir=str(tmp_path))
            
            with patch.object(manager, 'scan_models') as mock_scan:
                mock_scan.return_value = {"models_found": 1}
                result = manager.scan_models()
            
            assert result["models_found"] == 1

    def test_scanner_registry_reload(self, tmp_path):
        """Test that scanner reloads registry after scan."""
        from llm_manager import LLMManager
        
        # Create models.json
        registry_file = tmp_path / "models.json"
        registry_file.write_text('{"test-model.gguf": {}}')
        
        with patch('llm_manager.core.ModelRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry
            
            manager = LLMManager(models_dir=str(tmp_path))
            
            with patch.object(manager, 'scan_models') as mock_scan:
                mock_scan.return_value = {"models_found": 1}
                manager.scan_models()


class TestConfigGeneration:
    """Tests for llm_manager.yaml generation."""

    def test_generate_config_no_yaml(self, tmp_path):
        """Test config generation when yaml not available."""
        scanner = ModelScanner(tmp_path)
        
        with patch('llm_manager.scanner.YAML_AVAILABLE', False):
            # Should not raise
            scanner._generate_config()

    def test_generate_config_no_results(self, tmp_path):
        """Test config generation with no scan results."""
        scanner = ModelScanner(tmp_path)
        
        with patch('llm_manager.scanner.YAML_AVAILABLE', True):
            with patch('llm_manager.scanner.yaml') as mock_yaml:
                # Should not write anything with no results
                scanner._generate_config()
                mock_yaml.dump.assert_not_called()
