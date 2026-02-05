"""
Tests for llm_manager/core.py - Agent Features
(switch_model, generate_batch, generate_variations)
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from llm_manager import LLMManager
from llm_manager.exceptions import GenerationError


class TestSwitchModel:
    """Tests for switch_model method."""
    
    def test_switch_model_success(self, tmp_path):
        """Test successful model switch."""
        model_file = tmp_path / "model1.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 100)
        
        manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)
        
        with patch.object(manager, 'unload_model') as mock_unload:
            with patch.object(manager, 'load_model', return_value=True) as mock_load:
                result = manager.switch_model("model1.gguf")
                
                assert result is True
                mock_unload.assert_called_once()
                mock_load.assert_called_once_with(
                    model_path="model1.gguf",
                    n_ctx=None,
                    n_gpu_layers=-1,
                    auto_context=True
                )
    
    def test_switch_model_with_params(self, tmp_path):
        """Test model switch with custom parameters."""
        manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)
        
        with patch.object(manager, 'unload_model'):
            with patch.object(manager, 'load_model', return_value=True) as mock_load:
                manager.switch_model(
                    "model.gguf",
                    n_ctx=8192,
                    n_gpu_layers=20,
                    auto_context=False,
                    flash_attn=True
                )
                
                mock_load.assert_called_once()
                call_kwargs = mock_load.call_args.kwargs
                assert call_kwargs['n_ctx'] == 8192
                assert call_kwargs['n_gpu_layers'] == 20
                assert call_kwargs['auto_context'] is False
                assert call_kwargs['flash_attn'] is True
    
    def test_switch_model_clears_vram(self, tmp_path):
        """Test that switch_model clears VRAM cache."""
        manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)
        
        with patch('llm_manager.core.TORCH_AVAILABLE', True):
            mock_torch = Mock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.empty_cache = Mock()
            
            with patch('llm_manager.core.torch', mock_torch):
                with patch.object(manager, 'unload_model'):
                    with patch.object(manager, 'load_model', return_value=True):
                        manager.switch_model("model.gguf")
                        
                        mock_torch.cuda.empty_cache.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_switch_model_async(self, tmp_path):
        """Test async model switch."""
        manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)
        
        with patch.object(manager, 'unload_model') as mock_unload:
            with patch.object(manager, 'load_model_async', return_value=True) as mock_load:
                result = await manager.switch_model_async("model.gguf")
                
                assert result is True
                mock_unload.assert_called_once()
                mock_load.assert_called_once()


class TestGenerateBatch:
    """Tests for generate_batch method."""
    
    @pytest.mark.asyncio
    async def test_generate_batch_empty_prompts(self, tmp_path):
        """Test batch generation with empty prompts list."""
        manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)
        
        # Mock is_loaded to return True
        with patch.object(manager, 'is_loaded', return_value=True):
            results = await manager.generate_batch([])
            
            assert results == []
    
    @pytest.mark.asyncio
    async def test_generate_batch_not_loaded(self, tmp_path):
        """Test batch generation when no model loaded."""
        manager = LLMManager(models_dir=str(tmp_path))
        
        with pytest.raises(GenerationError) as exc_info:
            await manager.generate_batch([[{"role": "user", "content": "Hi"}]])
        
        assert "No model loaded" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_batch_success(self, tmp_path):
        """Test successful batch generation."""
        manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)
        manager.model = Mock()  # Pretend model is loaded
        
        prompts = [
            [{"role": "user", "content": "Prompt 1"}],
            [{"role": "user", "content": "Prompt 2"}],
        ]
        
        expected_responses = [
            {"choices": [{"message": {"content": "Response 1"}}]},
            {"choices": [{"message": {"content": "Response 2"}}]},
        ]
        
        with patch.object(manager, 'generate_async', side_effect=expected_responses):
            results = await manager.generate_batch(prompts)
        
        assert len(results) == 2
        assert results[0]["choices"][0]["message"]["content"] == "Response 1"
        assert results[1]["choices"][0]["message"]["content"] == "Response 2"
    
    @pytest.mark.asyncio
    async def test_generate_batch_with_errors(self, tmp_path):
        """Test batch generation with some failures."""
        manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)
        manager.model = Mock()
        
        prompts = [
            [{"role": "user", "content": "Prompt 1"}],
            [{"role": "user", "content": "Prompt 2"}],
        ]
        
        # First succeeds, second fails
        async def side_effect(*args, **kwargs):
            if "Prompt 1" in str(args):
                return {"choices": [{"message": {"content": "Response 1"}}]}
            else:
                raise GenerationError("Generation failed")
        
        with patch.object(manager, 'generate_async', side_effect=side_effect):
            results = await manager.generate_batch(prompts)
        
        assert len(results) == 2
        assert "error" in results[1]
        assert results[1]["error_type"] == "GenerationError"
    
    @pytest.mark.asyncio
    async def test_generate_batch_parallel_execution(self, tmp_path):
        """Test that batch generation runs in parallel."""
        manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)
        manager.model = Mock()
        
        prompts = [[{"role": "user", "content": f"Prompt {i}"}] for i in range(5)]
        
        call_order = []
        
        async def mock_generate(*args, **kwargs):
            call_order.append("start")
            await asyncio.sleep(0.01)  # Small delay
            call_order.append("end")
            return {"choices": [{"message": {"content": "Response"}}]}
        
        with patch.object(manager, 'generate_async', side_effect=mock_generate):
            start_time = asyncio.get_event_loop().time()
            results = await manager.generate_batch(prompts)
            end_time = asyncio.get_event_loop().time()
        
        # All should complete in parallel (less than 0.05s sequential)
        assert end_time - start_time < 0.05
        assert len(results) == 5


class TestGenerateVariations:
    """Tests for generate_variations method."""
    
    @pytest.mark.asyncio
    async def test_generate_variations_basic(self, tmp_path):
        """Test generating variations."""
        manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)
        manager.model = Mock()
        
        prompt = [{"role": "user", "content": "Solve this puzzle"}]
        
        expected_responses = [
            {"choices": [{"message": {"content": f"Variation {i}"}}]}
            for i in range(3)
        ]
        
        with patch.object(manager, 'generate_async', side_effect=expected_responses):
            variations = await manager.generate_variations(prompt, n_variations=3)
        
        assert len(variations) == 3
        assert variations[0] == "Variation 0"
        assert variations[1] == "Variation 1"
        assert variations[2] == "Variation 2"
    
    @pytest.mark.asyncio
    async def test_generate_variations_with_temperature(self, tmp_path):
        """Test generating variations with custom temperature."""
        manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)
        manager.model = Mock()
        
        prompt = [{"role": "user", "content": "Hi"}]
        
        with patch.object(manager, 'generate_async', return_value={
            "choices": [{"message": {"content": "Response"}}]
        }) as mock_generate:
            await manager.generate_variations(prompt, n_variations=2, temperature=0.9)
            
            # Should pass temperature to generate_async
            assert mock_generate.call_args.kwargs['temperature'] == 0.9
    
    @pytest.mark.asyncio
    async def test_generate_variations_empty_on_error(self, tmp_path):
        """Test that errors result in empty strings."""
        manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)
        manager.model = Mock()
        
        prompt = [{"role": "user", "content": "Hi"}]
        
        call_count = [0]
        async def fail_on_second(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise GenerationError("Failed")
            return {"choices": [{"message": {"content": "Success"}}]}
        
        with patch.object(manager, 'generate_async', side_effect=fail_on_second):
            variations = await manager.generate_variations(prompt, n_variations=2)
        
        assert variations[0] == "Success"
        assert variations[1] == ""  # Empty on error
    
    @pytest.mark.asyncio
    async def test_generate_variations_invalid_response(self, tmp_path):
        """Test handling of invalid response structure."""
        manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)
        manager.model = Mock()
        
        prompt = [{"role": "user", "content": "Hi"}]
        
        # Invalid response structure
        with patch.object(manager, 'generate_async', return_value={"invalid": "response"}):
            variations = await manager.generate_variations(prompt, n_variations=1)
        
        assert variations[0] == ""  # Empty on invalid structure
    
    @pytest.mark.asyncio
    async def test_generate_variations_not_loaded(self, tmp_path):
        """Test variations when no model loaded."""
        manager = LLMManager(models_dir=str(tmp_path))
        
        with pytest.raises(GenerationError):
            await manager.generate_variations([{"role": "user", "content": "Hi"}])


class TestAgentFeaturesIntegration:
    """Integration tests for agent features."""
    
    @pytest.mark.asyncio
    async def test_agent_workflow(self, tmp_path):
        """Test complete agent workflow with new features."""
        manager = LLMManager(models_dir=str(tmp_path), use_subprocess=False)
        
        # Mock loaded state
        manager.model = Mock()
        manager.model_name = "test-model"
        
        # 1. Generate initial response
        with patch.object(manager, 'generate_async', return_value={
            "choices": [{"message": {"content": "Initial response"}}]
        }):
            response = await manager.generate_async([{"role": "user", "content": "Hi"}])
            assert response["choices"][0]["message"]["content"] == "Initial response"
        
        # 2. Generate variations
        with patch.object(manager, 'generate_async', side_effect=[
            {"choices": [{"message": {"content": f"V{i}"}}]} for i in range(3)
        ]):
            variations = await manager.generate_variations(
                [{"role": "user", "content": "Solve"}],
                n_variations=3
            )
            assert len(variations) == 3
        
        # 3. Switch model
        with patch.object(manager, 'unload_model'):
            with patch.object(manager, 'load_model_async', return_value=True) as mock_load:
                await manager.switch_model_async("new-model.gguf")
                mock_load.assert_called_once()
