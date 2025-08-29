import pytest
import numpy as np
import tempfile
import os
import wave
import asyncio
from unittest.mock import patch, MagicMock
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.audio_handler import AudioProcessor

class TestAudioProcessor:
    """Test cases for AudioProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create an AudioProcessor instance for testing."""
        return AudioProcessor(sample_rate=24000, chunk_duration=0.1)
    
    def test_init(self, processor):
        """Test processor initialization."""
        assert processor.sample_rate == 24000
        assert processor.chunk_duration == 0.1
        assert processor.chunk_size == 2400  # 24000 * 0.1
    
    def test_convert_to_gradio_format(self, processor):
        """Test conversion of PCM16 bytes to Gradio format."""
        # Create test PCM16 audio data
        test_samples = np.array([1000, -1000, 2000, -2000], dtype=np.int16)
        test_bytes = test_samples.tobytes()
        
        sample_rate, audio_array = processor.convert_to_gradio_format(test_bytes)
        
        assert sample_rate == 24000
        assert len(audio_array) == 4
        assert audio_array.dtype == np.float32
        
        # Check normalization (int16 to float32 [-1, 1])
        expected = test_samples.astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(audio_array, expected)
    
    def test_convert_from_gradio_format(self, processor):
        """Test conversion from Gradio format to PCM16 bytes."""
        # Create test Gradio audio data
        sample_rate = 24000
        audio_array = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
        gradio_data = (sample_rate, audio_array)
        
        result_bytes = processor.convert_from_gradio_format(gradio_data)
        
        # Convert back to verify
        result_array = np.frombuffer(result_bytes, dtype=np.int16)
        expected = (audio_array * 32767).astype(np.int16)
        
        np.testing.assert_array_equal(result_array, expected)
    
    def test_convert_from_gradio_format_invalid_data(self, processor):
        """Test conversion with invalid Gradio data."""
        with pytest.raises(Exception) as exc_info:
            processor.convert_from_gradio_format(None)
        assert "Invalid audio data format" in str(exc_info.value)
        
        with pytest.raises(Exception) as exc_info:
            processor.convert_from_gradio_format((24000,))  # Missing audio array
        assert "Invalid audio data format" in str(exc_info.value)
    
    def test_convert_from_gradio_format_resampling(self, processor):
        """Test audio resampling when sample rates don't match."""
        # Test with different sample rate
        audio_array = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        gradio_data = (48000, audio_array)  # Different sample rate
        
        with patch('src.audio_handler.logger'):  # Mock logger to suppress warning
            result_bytes = processor.convert_from_gradio_format(gradio_data)
        
        # Should still produce valid PCM16 data
        result_array = np.frombuffer(result_bytes, dtype=np.int16)
        assert len(result_array) > 0
        assert result_array.dtype == np.int16
    
    @pytest.mark.asyncio
    async def test_chunk_audio_for_streaming(self, processor):
        """Test audio chunking for streaming."""
        # Create test audio (1 second at 24kHz, 16-bit)
        audio_samples = np.random.randint(-1000, 1000, 24000, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()
        
        chunks = []
        start_time = asyncio.get_event_loop().time()
        
        async for chunk in processor.chunk_audio_for_streaming(audio_bytes):
            chunks.append(chunk)
            if len(chunks) >= 3:  # Test first few chunks
                break
        
        end_time = asyncio.get_event_loop().time()
        
        # Verify chunking
        assert len(chunks) >= 3
        assert all(isinstance(chunk, bytes) for chunk in chunks)
        
        # Verify timing (should take at least 0.2 seconds for 3 chunks with 0.1s delay)
        assert end_time - start_time >= 0.2
    
    def test_save_audio_file(self, processor):
        """Test saving audio to file."""
        # Create test PCM16 audio
        audio_samples = np.random.randint(-1000, 1000, 2400, dtype=np.int16) 
        audio_bytes = audio_samples.tobytes()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            result_path = processor.save_audio_file(audio_bytes, output_path)
            
            assert result_path == output_path
            assert os.path.exists(output_path)
            
            # Verify the saved file is valid WAV
            with wave.open(output_path, 'rb') as wav_file:
                assert wav_file.getframerate() == 24000
                assert wav_file.getnchannels() == 1  # Mono
                assert wav_file.getsampwidth() == 2  # 16-bit
        
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_save_audio_file_error_handling(self, processor):
        """Test audio file saving with invalid data."""
        with pytest.raises(Exception) as exc_info:
            processor.save_audio_file(b'', '/invalid/path/file.wav')
        
        assert "Failed to save audio" in str(exc_info.value)

class TestAudioProcessorIntegration:
    """Integration tests for AudioProcessor."""
    
    @pytest.fixture
    def processor(self):
        return AudioProcessor()
    
    def test_round_trip_conversion(self, processor):
        """Test converting audio through Gradio format and back."""
        # Create original PCM16 data
        original_samples = np.array([1000, -1000, 2000, -2000], dtype=np.int16)
        original_bytes = original_samples.tobytes()
        
        # Convert to Gradio format
        gradio_audio = processor.convert_to_gradio_format(original_bytes)
        
        # Convert back to PCM16
        result_bytes = processor.convert_from_gradio_format(gradio_audio)
        result_samples = np.frombuffer(result_bytes, dtype=np.int16)
        
        # Should be very close (allowing for small floating-point errors)
        np.testing.assert_array_almost_equal(result_samples, original_samples, decimal=0)
    
    @pytest.mark.asyncio
    async def test_chunking_complete_audio(self, processor):
        """Test that chunking preserves complete audio data."""
        # Create test audio
        audio_samples = np.random.randint(-1000, 1000, 4800, dtype=np.int16)  # 0.2 seconds
        audio_bytes = audio_samples.tobytes()
        
        # Collect all chunks
        chunks = []
        async for chunk in processor.chunk_audio_for_streaming(audio_bytes):
            chunks.append(chunk)
        
        # Verify all audio is preserved
        reconstructed = b"".join(chunks)
        assert reconstructed == audio_bytes