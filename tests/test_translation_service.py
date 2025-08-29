import pytest
import asyncio
import json
import base64
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.translation_service import RealtimeTranslationService

class TestRealtimeTranslationService:
    """Test cases for RealtimeTranslationService."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'}):
            return RealtimeTranslationService()
    
    def test_init(self, service):
        """Test service initialization."""
        assert service.api_key == 'test-api-key'
        assert service.base_url == 'wss://api.openai.com/v1'
        assert len(service.voices) == 8
        assert len(service.languages) == 2
        assert len(service.voice_modes) == 3
    
    def test_voice_options(self, service):
        """Test voice options retrieval."""
        voices = service.get_voice_options()
        
        assert 'alloy' in voices
        assert 'cedar' in voices
        assert 'marin' in voices
        assert voices['alloy']['name'] == 'Alloy'
    
    def test_language_info(self, service):
        """Test language information retrieval."""
        en_info = service.get_language_info('en')
        fr_info = service.get_language_info('fr')
        unknown_info = service.get_language_info('unknown')
        
        assert en_info['name'] == 'English'
        assert en_info['flag'] == '🇺🇸'
        assert fr_info['name'] == 'French'
        assert fr_info['flag'] == '🇫🇷'
        assert unknown_info['name'] == 'unknown'
    
    def test_voice_mode_options(self, service):
        """Test voice mode options."""
        modes = service.get_voice_mode_options()
        
        assert 'preserve' in modes
        assert 'neutral' in modes
        assert 'enhanced' in modes
        assert 'Preserve original speaker voice' in modes['preserve']
    
    def test_translation_instructions_preserve_mode(self, service):
        """Test translation instructions for preserve mode."""
        instructions = service._get_translation_instructions('preserve', 'fr')
        
        assert 'professional speech translator' in instructions.lower()
        assert 'preserve' in instructions.lower()
        assert 'vocal tone' in instructions.lower()
        assert 'french' in instructions.lower()
    
    def test_translation_instructions_neutral_mode(self, service):
        """Test translation instructions for neutral mode."""
        instructions = service._get_translation_instructions('neutral', 'en')
        
        assert 'professional speech translator' in instructions.lower()
        assert 'english' in instructions.lower()
        assert 'preserve' not in instructions.lower()
    
    def test_translation_instructions_enhanced_mode(self, service):
        """Test translation instructions for enhanced mode."""
        instructions = service._get_translation_instructions('enhanced', 'fr')
        
        assert 'enhance' in instructions.lower()
        assert 'preserve core characteristics' in instructions.lower()
        assert 'french' in instructions.lower()
    
    @pytest.mark.asyncio
    async def test_connect_websocket_success(self, service):
        """Test successful WebSocket connection."""
        mock_websocket = AsyncMock()
        
        with patch('websockets.connect', return_value=mock_websocket):
            result = await service.connect_websocket()
            
            assert result is True
            assert service.websocket == mock_websocket
    
    @pytest.mark.asyncio
    async def test_connect_websocket_failure(self, service):
        """Test WebSocket connection failure."""
        with patch('websockets.connect', side_effect=Exception("Connection failed")):
            result = await service.connect_websocket()
            
            assert result is False
            assert service.websocket is None
    
    @pytest.mark.asyncio
    async def test_configure_session(self, service):
        """Test session configuration."""
        mock_websocket = AsyncMock()
        service.websocket = mock_websocket
        
        await service.configure_session(voice='alloy', voice_mode='preserve', target_lang='fr')
        
        # Verify session config was sent
        mock_websocket.send.assert_called_once()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        
        assert sent_data['type'] == 'session.update'
        assert sent_data['session']['model'] == 'gpt-4o-realtime-preview'
        assert sent_data['session']['voice'] == 'alloy'
        assert sent_data['session']['input_audio_format'] == 'pcm16'
        assert 'preserve' in sent_data['session']['instructions'].lower()
    
    @pytest.mark.asyncio
    async def test_single_shot_translation_success(self, service):
        """Test successful single-shot audio translation."""
        test_audio = b'fake_audio_data'
        mock_websocket = AsyncMock()
        
        # Mock WebSocket messages
        mock_responses = [
            json.dumps({"type": "response.audio.delta", "delta": base64.b64encode(b"chunk1").decode()}),
            json.dumps({"type": "response.audio.delta", "delta": base64.b64encode(b"chunk2").decode()}),
            json.dumps({"type": "response.audio.done"})
        ]
        
        mock_websocket.__aiter__.return_value = iter(mock_responses)
        
        with patch('websockets.connect', return_value=mock_websocket):
            service.websocket = mock_websocket
            
            result = await service.translate_audio_single_shot(
                audio_data=test_audio,
                voice='alloy',
                voice_mode='preserve',
                target_lang='fr'
            )
            
            assert result == b'chunk1chunk2'
            assert mock_websocket.send.call_count >= 2  # Config + audio
    
    @pytest.mark.asyncio 
    async def test_single_shot_translation_error(self, service):
        """Test single-shot translation with API error."""
        test_audio = b'fake_audio_data'
        mock_websocket = AsyncMock()
        
        # Mock error response
        error_response = json.dumps({
            "type": "error",
            "error": {"message": "API rate limit exceeded"}
        })
        
        mock_websocket.__aiter__.return_value = iter([error_response])
        
        with patch('websockets.connect', return_value=mock_websocket):
            service.websocket = mock_websocket
            
            with pytest.raises(Exception) as exc_info:
                await service.translate_audio_single_shot(
                    audio_data=test_audio,
                    voice='alloy', 
                    voice_mode='preserve',
                    target_lang='fr'
                )
            
            assert "API rate limit exceeded" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_streaming_translation(self, service):
        """Test streaming audio translation."""
        
        # Mock audio generator
        async def mock_audio_generator():
            yield b'chunk1'
            yield b'chunk2'
        
        mock_websocket = AsyncMock()
        
        # Mock streaming responses
        mock_responses = [
            json.dumps({"type": "response.audio.delta", "delta": base64.b64encode(b"translated1").decode()}),
            json.dumps({"type": "response.audio.delta", "delta": base64.b64encode(b"translated2").decode()}),
            json.dumps({"type": "response.audio.done"})
        ]
        
        # Setup response handler
        service.response_queue.put(b"translated1")
        service.response_queue.put(b"translated2") 
        service.response_queue.put(None)  # End signal
        
        with patch('websockets.connect', return_value=mock_websocket):
            service.websocket = mock_websocket
            
            results = []
            async for chunk in service.translate_audio_streaming(
                mock_audio_generator(),
                voice='nova',
                voice_mode='enhanced',
                target_lang='en'
            ):
                results.append(chunk)
            
            assert results == [b"translated1", b"translated2"]
    
    @pytest.mark.asyncio
    async def test_websocket_connection_cleanup(self, service):
        """Test WebSocket connection is properly cleaned up."""
        mock_websocket = AsyncMock()
        service.websocket = mock_websocket
        
        try:
            await service.translate_audio_single_shot(b'test', 'alloy', 'preserve', 'fr')
        except:
            pass  # Expected to fail in test
        
        # Verify cleanup happened
        mock_websocket.close.assert_called_once()
        assert service.websocket is None

class TestTranslationServiceEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def service(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return RealtimeTranslationService()
    
    def test_missing_api_key(self):
        """Test initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            service = RealtimeTranslationService()
            assert service.api_key is None
    
    def test_invalid_voice_mode(self, service):
        """Test with invalid voice mode."""
        instructions = service._get_translation_instructions('invalid_mode', 'fr')
        
        # Should default to neutral mode behavior
        assert 'professional speech translator' in instructions.lower()
        assert 'preserve' not in instructions.lower()
    
    def test_invalid_language(self, service):
        """Test with invalid target language."""
        instructions = service._get_translation_instructions('preserve', 'invalid_lang')
        
        # Should handle gracefully
        assert 'professional speech translator' in instructions.lower()