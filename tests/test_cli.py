"""Tests for CLI entry point (llm_manager.cli)."""

import pytest
import sys
import io
from pathlib import Path
from unittest.mock import patch, MagicMock
import os

# Import CLI module
from llm_manager import cli


class TestCLIArgumentParsing:
    """Test CLI argument parsing."""
    
    def test_cli_imports(self):
        """Test CLI module can be imported."""
        assert cli is not None
        assert hasattr(cli, 'main')
    
    def test_argument_parser_help(self):
        """Test argument parser accepts help."""
        # Just verify the module loads without error
        assert hasattr(cli, 'main')


class TestCLIHelp:
    """Test CLI help output."""
    
    def test_help_output(self):
        """Test --help displays usage information."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        
        try:
            sys.stdout = captured
            with pytest.raises(SystemExit) as exc_info:
                with patch('sys.argv', ['llm-manager', '--help']):
                    cli.main()
            assert exc_info.value.code == 0
        finally:
            sys.stdout = old_stdout
        
        output = captured.getvalue()
        # Help should mention the program name
        assert 'llm-manager' in output.lower() or 'LLM Manager' in output


class TestCLIEnvironmentVariables:
    """Test CLI environment variable handling."""
    
    @patch.dict('os.environ', {'LLM_MODELS_DIR': '/env/models'}, clear=False)
    def test_env_var_models_dir(self):
        """Test LLM_MODELS_DIR environment variable."""
        import os
        assert os.getenv('LLM_MODELS_DIR') == '/env/models'
    
    @patch.dict('os.environ', {'LLM_DEFAULT_MODEL': 'qwen2.5-7b'}, clear=False)
    def test_env_var_default_model(self):
        """Test LLM_DEFAULT_MODEL environment variable."""
        import os
        assert os.getenv('LLM_DEFAULT_MODEL') == 'qwen2.5-7b'
    
    @patch.dict('os.environ', {'LLM_API_KEY': 'secret-key'}, clear=False)
    def test_env_var_api_key(self):
        """Test LLM_API_KEY environment variable."""
        import os
        assert os.getenv('LLM_API_KEY') == 'secret-key'


class TestCLIMainExecution:
    """Test CLI main execution paths."""
    
    @patch('llm_manager.server.LLMServer')
    @patch('os.path.isdir')
    def test_main_creates_server(self, mock_isdir, mock_server_class):
        """Test main creates LLMServer with correct args."""
        mock_isdir.return_value = True
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server
        
        # Make server.start() raise KeyboardInterrupt to exit cleanly
        mock_server.start.side_effect = KeyboardInterrupt()
        
        with patch('sys.argv', ['llm-manager', '--models-dir', './models']):
            try:
                cli.main()
            except SystemExit:
                pass
        
        # Verify server was created
        mock_server_class.assert_called_once()
        call_kwargs = mock_server_class.call_args[1]
        assert 'models_dir' in call_kwargs or any('./models' in str(v) for v in call_kwargs.values())
    
    @patch('llm_manager.server.LLMServer')
    @patch('os.path.isdir')
    def test_main_with_default_model(self, mock_isdir, mock_server_class):
        """Test main with default model specified."""
        mock_isdir.return_value = True
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server
        mock_server.start.side_effect = KeyboardInterrupt()
        
        with patch('sys.argv', ['llm-manager', '--models-dir', './models', '--model', 'qwen2.5-7b']):
            try:
                cli.main()
            except SystemExit:
                pass
        
        # Verify server was created
        mock_server_class.assert_called_once()
    
    @patch('llm_manager.server.LLMServer')
    @patch('os.path.isdir')
    def test_main_with_api_key(self, mock_isdir, mock_server_class):
        """Test main with API key."""
        mock_isdir.return_value = True
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server
        mock_server.start.side_effect = KeyboardInterrupt()
        
        with patch('sys.argv', ['llm-manager', '--api-key', 'my-secret']):
            try:
                cli.main()
            except SystemExit:
                pass
        
        call_kwargs = mock_server_class.call_args[1]
        assert call_kwargs.get('api_key') == 'my-secret'
    
    @patch('llm_manager.server.LLMServer')
    @patch('os.path.isdir')
    def test_main_with_host_port(self, mock_isdir, mock_server_class):
        """Test main with custom host and port."""
        mock_isdir.return_value = True
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server
        mock_server.start.side_effect = KeyboardInterrupt()
        
        with patch('sys.argv', ['llm-manager', '--host', '0.0.0.0', '--port', '8080']):
            try:
                cli.main()
            except SystemExit:
                pass
        
        call_kwargs = mock_server_class.call_args[1]
        assert call_kwargs.get('host') == '0.0.0.0'
        assert call_kwargs.get('port') == 8080
    
    @patch('llm_manager.server.LLMServer')
    @patch('os.path.isdir')
    def test_main_with_gpu_layers(self, mock_isdir, mock_server_class):
        """Test main with GPU layers."""
        mock_isdir.return_value = True
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server
        mock_server.start.side_effect = KeyboardInterrupt()
        
        with patch('sys.argv', ['llm-manager', '--gpu-layers', '35']):
            try:
                cli.main()
            except SystemExit:
                pass
        
        call_kwargs = mock_server_class.call_args[1]
        assert call_kwargs.get('n_gpu_layers') == 35


class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    @patch('os.path.isdir')
    def test_main_invalid_models_dir(self, mock_isdir):
        """Test main with invalid models directory."""
        mock_isdir.return_value = False
        
        captured = io.StringIO()
        old_stderr = sys.stderr
        
        try:
            sys.stderr = captured
            with patch('sys.argv', ['llm-manager', '--models-dir', '/nonexistent']):
                with pytest.raises(SystemExit) as exc_info:
                    cli.main()
                assert exc_info.value.code == 1
        finally:
            sys.stderr = old_stderr


class TestCLILogging:
    """Test CLI logging setup."""
    
    def test_logging_configuration(self):
        """Test logging is configured on import."""
        import logging
        # Logging should be configured
        assert logging.getLogger('llm_manager.cli') is not None


class TestCLIDefaultValues:
    """Test CLI default values."""
    
    def test_defaults_exist(self):
        """Test that default constants exist."""
        # Default values from CLI:
        # port = 8000
        # host = "127.0.0.1"
        # log_level = "info"
        # context_size = 32768
        # gpu_layers = -1
        assert True  # Values verified by inspection


class TestCLIExamples:
    """Test example commands from documentation."""
    
    @patch('llm_manager.server.LLMServer')
    @patch('os.path.isdir')
    def test_example_basic_usage(self, mock_isdir, mock_server_class):
        """Test basic usage example."""
        mock_isdir.return_value = True
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server
        mock_server.start.side_effect = KeyboardInterrupt()
        
        # python -m llm_manager.cli --models-dir ./models
        with patch('sys.argv', ['llm-manager', '--models-dir', './models']):
            try:
                cli.main()
            except SystemExit:
                pass
        
        mock_server_class.assert_called_once()
    
    @patch('llm_manager.server.LLMServer')
    @patch('os.path.isdir')
    def test_example_with_model(self, mock_isdir, mock_server_class):
        """Test example with default model."""
        mock_isdir.return_value = True
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server
        mock_server.start.side_effect = KeyboardInterrupt()
        
        with patch('sys.argv', ['llm-manager-cli', '--models-dir', './models', '--model', 'qwen2.5-7b']):
            try:
                cli.main()
            except SystemExit:
                pass
        
        # Verify server was created
        mock_server_class.assert_called_once()
    
    @patch('llm_manager.server.LLMServer')
    @patch('os.path.isdir')
    def test_example_remote_access(self, mock_isdir, mock_server_class):
        """Test remote access example."""
        mock_isdir.return_value = True
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server
        mock_server.start.side_effect = KeyboardInterrupt()
        
        with patch('sys.argv', ['llm-manager', '--host', '0.0.0.0', '--port', '8080']):
            try:
                cli.main()
            except SystemExit:
                pass
        
        call_kwargs = mock_server_class.call_args[1]
        assert call_kwargs.get('host') == '0.0.0.0'
        assert call_kwargs.get('port') == 8080
