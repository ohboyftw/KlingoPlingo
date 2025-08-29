import pytest
import asyncio
import sys
import os
from unittest.mock import patch

# Add src to Python path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_env():
    """Mock environment variables for testing.""" 
    env_vars = {
        'OPENAI_API_KEY': 'sk-test-key-for-testing-1234567890abcdef',
        'OPENAI_API_BASE': 'wss://api.test.com/v1'
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars

@pytest.fixture
def sample_audio_data():
    """Provide sample audio data for tests."""
    import numpy as np
    
    # Generate 1 second of test audio at 24kHz
    sample_rate = 24000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simple sine wave
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    audio_int16 = (audio * 32767 * 0.5).astype(np.int16)
    
    return {
        'pcm16_bytes': audio_int16.tobytes(),
        'gradio_format': (sample_rate, audio.astype(np.float32) * 0.5),
        'numpy_array': audio_int16,
        'sample_rate': sample_rate,
        'duration': duration
    }

@pytest.fixture
def mock_websocket_responses():
    """Provide mock WebSocket responses for different scenarios."""
    import base64
    
    return {
        'success': [
            '{"type": "response.audio.delta", "delta": "' + base64.b64encode(b'chunk1').decode() + '"}',
            '{"type": "response.audio.delta", "delta": "' + base64.b64encode(b'chunk2').decode() + '"}',
            '{"type": "response.audio.done"}'
        ],
        'error': [
            '{"type": "error", "error": {"type": "rate_limit_error", "message": "Rate limit exceeded"}}'
        ],
        'auth_error': [
            '{"type": "error", "error": {"type": "authentication_error", "message": "Invalid API key"}}'
        ],
        'timeout': []  # Empty responses to simulate timeout
    }

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)