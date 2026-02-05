"""Tests package for llm_manager."""

import warnings

# Suppress AsyncMock RuntimeWarning globally
# This is a known issue with unittest.mock.AsyncMock when coroutines are garbage collected
warnings.filterwarnings(
    "ignore",
    message="coroutine 'AsyncMockMixin._execute_mock_call' was never awaited",
    category=RuntimeWarning
)
