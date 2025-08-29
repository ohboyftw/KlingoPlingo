import pytest
import json
import base64
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.translation_service import RealtimeTranslationService

class MockWebSocketServer:
    """Mock WebSocket server for testing GPT-Realtime API responses."""
    
    def __init__(self, responses=None):
        self.responses = responses or []
        self.received_messages = []
        self.closed = False
        self._response_iter = None
    
    async def send(self, message):
        """Mock send method."""
        self.received_messages.append(json.loads(message))
    
    async def close(self):
        """Mock close method."""
        self.closed = True
    
    def __aiter__(self):
        """Mock async iteration over responses."""
        self._response_iter = iter(self.responses)
        return self
    
    async def __anext__(self):
        """Mock async next for responses."""
        try:
            return next(self._response_iter)
        except StopIteration:
            raise StopAsyncIteration
    
    # Make it awaitable for websockets.connect
    def __await__(self):
        async def _await():
            return self
        return _await().__await__()

class TestGPTRealtimeAPIMocks:
    """Test cases using mocked GPT Realtime API responses."""
    
    @pytest.fixture
    def service(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return RealtimeTranslationService()
    
    @pytest.mark.asyncio
    async def test_successful_translation_response(self, service):
        """Test successful translation with realistic API responses."""
        
        # Create realistic GPT Realtime response sequence
        responses = [
            json.dumps({
                "type": "session.update", 
                "session": {"model": "gpt-realtime"}
            }),
            json.dumps({
                "type": "response.audio.delta",
                "delta": base64.b64encode(b"bonjour").decode()
            }),
            json.dumps({
                "type": "response.audio.delta", 
                "delta": base64.b64encode(b" comment").decode()
            }),
            json.dumps({
                "type": "response.audio.delta",
                "delta": base64.b64encode(b" allez-vous").decode()  
            }),
            json.dumps({"type": "response.audio.done"})
        ]
        
        mock_ws = MockWebSocketServer(responses)
        
        with patch('websockets.connect', return_value=mock_ws):
            result = await service.translate_audio_single_shot(
                audio_data=b'hello_how_are_you_audio',
                voice='alloy',
                voice_mode='preserve',
                target_lang='fr'
            )
            
            assert result == b'bonjour comment allez-vous'
            assert mock_ws.closed is True
    
    @pytest.mark.asyncio
    async def test_api_error_response(self, service):
        """Test handling of API error responses."""
        
        error_response = json.dumps({
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "Invalid API key provided"
            }
        })
        
        mock_ws = MockWebSocketServer([error_response])
        
        with patch('websockets.connect', return_value=mock_ws):
            with pytest.raises(Exception) as exc_info:
                await service.translate_audio_single_shot(
                    audio_data=b'test_audio',
                    voice='nova',
                    voice_mode='neutral',
                    target_lang='en'
                )
            
            assert "Invalid API key provided" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_rate_limit_error(self, service):
        """Test rate limit error handling."""
        
        rate_limit_response = json.dumps({
            "type": "error", 
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit exceeded. Please try again later."
            }
        })
        
        mock_ws = MockWebSocketServer([rate_limit_response])
        
        with patch('websockets.connect', return_value=mock_ws):
            with pytest.raises(Exception) as exc_info:
                await service.translate_audio_single_shot(
                    audio_data=b'test_audio',
                    voice='shimmer',
                    voice_mode='enhanced', 
                    target_lang='fr'
                )
            
            assert "rate limit exceeded" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_streaming_translation_mock(self, service):
        """Test streaming translation with mocked responses."""
        
        # Mock audio generator
        async def mock_audio_gen():
            yield b'audio_chunk_1'
            yield b'audio_chunk_2'
        
        # Create streaming responses
        responses = [
            json.dumps({
                "type": "response.audio.delta",
                "delta": base64.b64encode(b"translated_1").decode()
            }),
            json.dumps({
                "type": "response.audio.delta", 
                "delta": base64.b64encode(b"translated_2").decode()
            }),
            json.dumps({"type": "response.audio.done"})
        ]
        
        # Setup response queue
        service.response_queue.put(b"translated_1")
        service.response_queue.put(b"translated_2")
        service.response_queue.put(None)
        
        mock_ws = MockWebSocketServer()
        
        with patch('websockets.connect', return_value=mock_ws):
            with patch.object(service, '_handle_responses'):
                
                chunks = []
                async for chunk in service.translate_audio_streaming(
                    mock_audio_gen(),
                    voice='echo', 
                    voice_mode='preserve',
                    target_lang='en'
                ):
                    chunks.append(chunk)
                
                assert chunks == [b"translated_1", b"translated_2"]
                
                # Verify audio was sent
                audio_messages = [msg for msg in mock_ws.received_messages 
                                if msg.get('type') == 'input_audio_buffer.append']
                assert len(audio_messages) == 2
    
    @pytest.mark.asyncio
    async def test_session_update_configurations(self, service):
        """Test different session configurations are sent correctly."""
        
        test_cases = [
            ('cedar', 'preserve', 'fr'),
            ('marin', 'enhanced', 'en'), 
            ('fable', 'neutral', 'fr')
        ]
        
        for voice, voice_mode, target_lang in test_cases:
            mock_ws = MockWebSocketServer()
            service.websocket = mock_ws
            
            await service.configure_session(voice, voice_mode, target_lang)
            
            # Verify session config message
            session_msg = mock_ws.received_messages[0]
            assert session_msg['type'] == 'session.update'
            
            session_config = session_msg['session']
            assert session_config['audio']['output']['voice'] == voice
            assert session_config['model'] == 'gpt-realtime'
            assert session_config['type'] == 'realtime'
            assert session_config['audio']['input']['format'] == 'pcm16'
            assert session_config['audio']['output']['format'] == 'pcm16'
            
            # Check instructions contain expected voice mode keywords and structure
            instructions = session_config['instructions'].lower()
            assert '# role & objective' in instructions  # Check structured format
            assert '# language' in instructions
            
            if voice_mode == 'preserve':
                assert 'preserve' in instructions
                assert 'vocal tone' in instructions
                assert 'voice preservation (critical)' in instructions
            elif voice_mode == 'enhanced':
                assert 'enhance' in instructions
                assert 'voice enhancement' in instructions
            else:  # neutral
                assert 'voice processing' in instructions
                assert 'preserve' not in instructions

class TestRealtimeAPIScenarios:
    """Test realistic API interaction scenarios."""
    
    @pytest.fixture
    def service(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return RealtimeTranslationService()
    
    @pytest.mark.asyncio
    async def test_french_to_english_preserve_voice(self, service):
        """Test French to English translation preserving voice."""
        
        # French audio input mock
        french_audio = b'bonjour_audio_data'
        
        # English response with preserved voice characteristics
        responses = [
            json.dumps({
                "type": "response.audio.delta",
                "delta": base64.b64encode(b"hello").decode()
            }),
            json.dumps({
                "type": "response.audio.delta",
                "delta": base64.b64encode(b"_preserved_voice").decode()
            }),
            json.dumps({"type": "response.audio.done"})
        ]
        
        mock_ws = MockWebSocketServer(responses)
        
        with patch('websockets.connect', return_value=mock_ws):
            result = await service.translate_audio_single_shot(
                audio_data=french_audio,
                voice='alloy',
                voice_mode='preserve', 
                target_lang='en'
            )
            
            assert result == b'hello_preserved_voice'
            
            # Verify preserve mode instructions were sent
            session_msg = mock_ws.received_messages[0]
            instructions = session_msg['session']['instructions'].lower()
            assert 'preserve' in instructions
            assert 'vocal tone' in instructions
            assert '# role & objective' in instructions
    
    @pytest.mark.asyncio
    async def test_english_to_german_translation(self, service):
        """Test English to German translation."""
        
        # English audio input mock
        english_audio = b'hello_world_audio_data'
        
        # German response
        responses = [
            json.dumps({
                "type": "response.audio.delta",
                "delta": base64.b64encode(b"hallo").decode()
            }),
            json.dumps({
                "type": "response.audio.delta", 
                "delta": base64.b64encode(b"_welt").decode()
            }),
            json.dumps({"type": "response.audio.done"})
        ]
        
        mock_ws = MockWebSocketServer(responses)
        
        with patch('websockets.connect', return_value=mock_ws):
            result = await service.translate_audio_single_shot(
                audio_data=english_audio,
                voice='cedar',
                voice_mode='preserve',
                target_lang='de'
            )
            
            assert result == b'hallo_welt'
            
            # Verify German instructions were sent
            session_msg = mock_ws.received_messages[0]
            instructions = session_msg['session']['instructions'].lower()
            assert 'german' in instructions
    
    @pytest.mark.asyncio
    async def test_connection_failure_recovery(self, service):
        """Test graceful handling of connection failures."""
        
        # First connection attempt fails
        with patch('websockets.connect', side_effect=ConnectionError("Network error")):
            result = await service.connect_websocket()
            assert result is False
        
        # Subsequent translation should handle the failure
        with pytest.raises(Exception) as exc_info:
            await service.translate_audio_single_shot(
                audio_data=b'test',
                voice='nova',
                voice_mode='neutral',
                target_lang='fr'
            )
        
        assert "Failed to establish WebSocket connection" in str(exc_info.value)