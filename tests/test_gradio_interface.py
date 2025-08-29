import pytest
import numpy as np
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.speech_gradio_interface import SpeechTranslationInterface

class TestSpeechTranslationInterface:
    """Test cases for SpeechTranslationInterface."""
    
    @pytest.fixture
    def interface(self):
        """Create interface instance for testing."""
        with patch('src.speech_gradio_interface.translation_service') as mock_service:
            with patch('src.speech_gradio_interface.audio_processor') as mock_processor:
                return SpeechTranslationInterface()
    
    def test_init(self, interface):
        """Test interface initialization."""
        assert interface.service is not None
        assert interface.audio_processor is not None
        assert len(interface.language_pairs) == 4
    
    def test_language_pairs(self, interface):
        """Test language pair configurations."""
        pairs = interface.language_pairs
        
        expected_pairs = [
            ("English → French", "en", "fr"),
            ("French → English", "fr", "en"),
            ("Auto-detect → English", "auto", "en"), 
            ("Auto-detect → French", "auto", "fr")
        ]
        
        assert pairs == expected_pairs
    
    def test_swap_language_pair(self, interface):
        """Test language pair swapping functionality."""
        test_cases = [
            ("English → French", "French → English"),
            ("French → English", "English → French"),
            ("Auto-detect → English", "Auto-detect → French"),
            ("Auto-detect → French", "Auto-detect → English"),
            ("Invalid → Pair", "Invalid → Pair")  # Should remain unchanged
        ]
        
        for input_pair, expected_output in test_cases:
            result = interface.swap_language_pair(input_pair)
            assert result == expected_output
    
    @patch('src.speech_gradio_interface.asyncio')
    def test_single_shot_translation_success(self, mock_asyncio, interface):
        """Test successful single-shot translation."""
        # Mock audio input
        sample_rate = 24000
        audio_array = np.array([0.1, 0.2, -0.1, -0.2], dtype=np.float32)
        audio_input = (sample_rate, audio_array)
        
        # Mock processors
        interface.audio_processor.convert_from_gradio_format.return_value = b'input_audio'
        interface.audio_processor.convert_to_gradio_format.return_value = (24000, np.array([0.3, 0.4]))
        
        # Mock async service call
        mock_loop = MagicMock()
        mock_asyncio.new_event_loop.return_value = mock_loop
        mock_asyncio.set_event_loop.return_value = None
        mock_loop.run_until_complete.return_value = b'translated_audio'
        
        # Call method
        result_audio, status = interface.translate_audio_single_shot(
            audio_input=audio_input,
            language_pair="English → French",
            voice="alloy",
            voice_mode="preserve", 
            processing_mode="single_shot"
        )
        
        # Verify results
        assert result_audio is not None
        assert result_audio[0] == 24000
        np.testing.assert_array_equal(result_audio[1], np.array([0.3, 0.4]))
        assert "✅ Translation completed" in status
        assert "preserve" in status.lower()
        
        # Verify service was called correctly
        mock_loop.run_until_complete.assert_called_once()
    
    @patch('src.speech_gradio_interface.asyncio')
    def test_single_shot_translation_no_audio(self, mock_asyncio, interface):
        """Test single-shot translation with no audio input."""
        
        result_audio, status = interface.translate_audio_single_shot(
            audio_input=None,
            language_pair="French → English",
            voice="echo",
            voice_mode="neutral",
            processing_mode="single_shot"
        )
        
        assert result_audio is None
        assert "❌ Please record or upload audio first" == status
    
    @patch('src.speech_gradio_interface.asyncio')
    def test_single_shot_translation_error(self, mock_asyncio, interface):
        """Test single-shot translation with service error."""
        # Mock audio input  
        audio_input = (24000, np.array([0.1, 0.2]))
        
        interface.audio_processor.convert_from_gradio_format.return_value = b'input_audio'
        
        # Mock async error
        mock_loop = MagicMock()
        mock_asyncio.new_event_loop.return_value = mock_loop
        mock_loop.run_until_complete.side_effect = Exception("Translation service error")
        
        result_audio, status = interface.translate_audio_single_shot(
            audio_input=audio_input,
            language_pair="English → French", 
            voice="onyx",
            voice_mode="preserve",
            processing_mode="single_shot"
        )
        
        assert result_audio is None
        assert "❌ Translation failed" in status
        assert "Translation service error" in status
    
    @patch('src.speech_gradio_interface.asyncio')
    def test_streaming_translation_success(self, mock_asyncio, interface):
        """Test successful streaming translation."""
        # Mock audio input
        audio_input = (24000, np.array([0.5, -0.5]))
        
        interface.audio_processor.convert_from_gradio_format.return_value = b'stream_audio'
        interface.audio_processor.convert_to_gradio_format.return_value = (24000, np.array([0.7, 0.8]))
        
        # Mock streaming response
        mock_loop = MagicMock()
        mock_asyncio.new_event_loop.return_value = mock_loop
        
        # Mock async generator result
        async def mock_streaming_result():
            yield b'chunk1'
            yield b'chunk2'
        
        mock_loop.run_until_complete.return_value = mock_streaming_result()
        
        # Mock the async generator consumption
        def mock_run_until_complete(coro):
            if hasattr(coro, '__aiter__'):
                return [b'chunk1', b'chunk2']  # Simulate collected chunks
            return coro
        
        mock_loop.run_until_complete.side_effect = mock_run_until_complete
        
        result_audio, status = interface.translate_audio_streaming(
            audio_input=audio_input,
            language_pair="Auto-detect → French",
            voice="cedar",
            voice_mode="enhanced",
            processing_mode="streaming"
        )
        
        assert result_audio is not None
        assert result_audio[0] == 24000
        np.testing.assert_array_equal(result_audio[1], np.array([0.7, 0.8]))
        assert "✅ Streaming translation completed" in status
        assert "enhanced" in status.lower()

class TestGradioInterfaceHelpers:
    """Test helper functions in the Gradio interface."""
    
    @pytest.fixture
    def interface(self):
        with patch('src.speech_gradio_interface.translation_service'):
            with patch('src.speech_gradio_interface.audio_processor'):
                return SpeechTranslationInterface()
    
    def test_language_pair_parsing(self, interface):
        """Test parsing language pairs correctly."""
        # Test each language pair
        for display_name, source, target in interface.language_pairs:
            # Find matching pair
            found_pair = next(
                (pair for pair in interface.language_pairs if pair[0] == display_name),
                (None, None, None)
            )
            
            assert found_pair[1] == source
            assert found_pair[2] == target
    
    def test_voice_options_structure(self, interface):
        """Test voice options are properly structured."""
        # Mock service voice options
        mock_voices = {
            'alloy': {'name': 'Alloy', 'description': 'Neutral'},
            'cedar': {'name': 'Cedar', 'description': 'Natural'}
        }
        
        interface.service.get_voice_options.return_value = mock_voices
        
        # Test voice option formatting (would be used in actual interface)
        voice_options = [(f"{info['name']} - {info['description']}", voice) 
                        for voice, info in mock_voices.items()]
        
        assert voice_options == [
            ('Alloy - Neutral', 'alloy'),
            ('Cedar - Natural', 'cedar')
        ]
    
    def test_voice_mode_options_structure(self, interface):
        """Test voice mode options are properly structured.""" 
        mock_modes = {
            'preserve': 'Preserve original speaker voice',
            'neutral': 'Use selected voice profile'
        }
        
        interface.service.get_voice_mode_options.return_value = mock_modes
        
        # Test mode option formatting
        mode_options = [(desc, mode) for mode, desc in mock_modes.items()]
        
        assert mode_options == [
            ('Preserve original speaker voice', 'preserve'),
            ('Use selected voice profile', 'neutral')
        ]

@pytest.mark.asyncio 
class TestInterfaceIntegration:
    """Integration tests between interface and services."""
    
    @pytest.fixture
    def interface(self):
        """Create interface with mocked services."""
        with patch('src.speech_gradio_interface.translation_service') as mock_service:
            with patch('src.speech_gradio_interface.audio_processor') as mock_processor:
                interface = SpeechTranslationInterface()
                interface.service = mock_service
                interface.audio_processor = mock_processor
                return interface
    
    def test_end_to_end_single_shot_flow(self, interface):
        """Test complete single-shot translation flow.""" 
        # Setup mocks
        audio_input = (24000, np.array([0.1, 0.2]))
        interface.audio_processor.convert_from_gradio_format.return_value = b'pcm_data'
        interface.audio_processor.convert_to_gradio_format.return_value = (24000, np.array([0.3, 0.4]))
        
        # Mock the async service method as a coroutine that returns the expected value
        async def mock_translate_single_shot(audio_data, voice, voice_mode, target_lang):
            return b'translated_pcm'
        
        interface.service.translate_audio_single_shot = mock_translate_single_shot
        
        result_audio, status = interface.translate_audio_single_shot(
            audio_input=audio_input,
            language_pair="English → French",
            voice="fable", 
            voice_mode="preserve",
            processing_mode="single_shot"
        )
        
        assert result_audio is not None
        assert result_audio[0] == 24000
        np.testing.assert_array_equal(result_audio[1], np.array([0.3, 0.4]))
        assert "✅" in status