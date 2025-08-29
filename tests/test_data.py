import numpy as np
import wave
import tempfile
import os

class AudioTestData:
    """Generate test audio data for unit tests."""
    
    @staticmethod
    def create_test_audio_samples(duration: float = 1.0, sample_rate: int = 24000) -> np.ndarray:
        """
        Create synthetic audio samples for testing.
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Numpy array with test audio samples
        """
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Generate a simple sine wave with some harmonics
        frequency = 440  # A4 note
        audio = (
            0.3 * np.sin(2 * np.pi * frequency * t) +
            0.1 * np.sin(2 * np.pi * frequency * 2 * t) +
            0.05 * np.sin(2 * np.pi * frequency * 3 * t)
        )
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.01, num_samples)
        audio += noise
        
        # Convert to int16 range
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767).astype(np.int16)
    
    @staticmethod
    def create_test_pcm16_bytes(duration: float = 1.0, sample_rate: int = 24000) -> bytes:
        """Create PCM16 audio bytes for testing."""
        samples = AudioTestData.create_test_audio_samples(duration, sample_rate)
        return samples.tobytes()
    
    @staticmethod
    def create_test_gradio_audio(duration: float = 1.0, sample_rate: int = 24000) -> tuple:
        """Create Gradio audio format for testing."""
        samples = AudioTestData.create_test_audio_samples(duration, sample_rate)
        # Convert to float32 normalized format for Gradio
        audio_float = samples.astype(np.float32) / 32768.0
        return (sample_rate, audio_float)
    
    @staticmethod
    def create_test_wav_file(duration: float = 1.0, sample_rate: int = 24000) -> str:
        """
        Create a temporary WAV file for testing.
        
        Returns:
            Path to temporary WAV file (caller should clean up)
        """
        samples = AudioTestData.create_test_audio_samples(duration, sample_rate)
        
        # Create temporary file
        fd, path = tempfile.mkstemp(suffix='.wav')
        
        try:
            with wave.open(path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples.tobytes())
            
            return path
            
        except Exception:
            os.close(fd)
            if os.path.exists(path):
                os.unlink(path)
            raise
        finally:
            os.close(fd)
    
    @staticmethod
    def create_silence_audio(duration: float = 0.5, sample_rate: int = 24000) -> np.ndarray:
        """Create silent audio for testing edge cases."""
        num_samples = int(duration * sample_rate)
        return np.zeros(num_samples, dtype=np.int16)
    
    @staticmethod
    def create_noisy_audio(duration: float = 1.0, sample_rate: int = 24000, noise_level: float = 0.1) -> np.ndarray:
        """Create noisy audio for testing robustness."""
        num_samples = int(duration * sample_rate)
        noise = np.random.normal(0, noise_level, num_samples)
        audio = np.clip(noise, -1.0, 1.0)
        return (audio * 32767).astype(np.int16)

# Test data constants
TEST_AUDIO_SAMPLES = {
    'short': AudioTestData.create_test_pcm16_bytes(0.5),  # 0.5 seconds
    'medium': AudioTestData.create_test_pcm16_bytes(2.0),  # 2 seconds  
    'long': AudioTestData.create_test_pcm16_bytes(5.0),   # 5 seconds
    'silence': AudioTestData.create_silence_audio(1.0).tobytes(),
    'noisy': AudioTestData.create_noisy_audio(1.0).tobytes()
}

GRADIO_TEST_AUDIO = {
    'english_sample': AudioTestData.create_test_gradio_audio(1.5),
    'french_sample': AudioTestData.create_test_gradio_audio(2.0, 22050),  # Different sample rate
    'short_sample': AudioTestData.create_test_gradio_audio(0.3)
}

# Mock WebSocket responses for different scenarios
MOCK_WEBSOCKET_RESPONSES = {
    'successful_translation': [
        '{"type": "session.updated", "session": {"model": "gpt-realtime", "type": "realtime"}}',
        '{"type": "response.audio.delta", "delta": "' + 
        AudioTestData.create_test_pcm16_bytes(0.5).hex() + '"}',
        '{"type": "response.audio.done"}'
    ],
    
    'streaming_translation': [
        '{"type": "session.updated", "session": {"model": "gpt-realtime", "type": "realtime"}}',
        '{"type": "response.audio.delta", "delta": "chunk1"}',
        '{"type": "response.audio.delta", "delta": "chunk2"}', 
        '{"type": "response.audio.delta", "delta": "chunk3"}',
        '{"type": "response.audio.done"}'
    ],
    
    'api_error': [
        '{"type": "error", "error": {"type": "rate_limit_error", "message": "Rate limit exceeded"}}'
    ],
    
    'authentication_error': [
        '{"type": "error", "error": {"type": "authentication_error", "message": "Invalid API key"}}'
    ]
}