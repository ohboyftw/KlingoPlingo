import pytest
import asyncio
import json
import base64
import os
import sys
from unittest.mock import AsyncMock, patch, MagicMock
import websockets

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.translation_service import RealtimeTranslationService

class TestWebSocketIntegration:
    """Integration tests for WebSocket functionality."""
    
    @pytest.fixture
    def service(self):
        """Create service with test environment."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'sk-test-key-1234567890abcdef',
            'OPENAI_API_BASE': 'wss://api.test.com/v1'
        }):
            return RealtimeTranslationService()
    
    @pytest.mark.asyncio
    async def test_websocket_connection_flow(self, service):
        """Test complete WebSocket connection flow."""
        mock_websocket = AsyncMock()
        
        with patch('websockets.connect') as mock_connect:
            mock_connect.return_value = mock_websocket
            
            # Test connection
            connected = await service.connect_websocket()
            
            assert connected is True
            assert service.websocket == mock_websocket
            
            # Verify connection parameters
            mock_connect.assert_called_once_with(
                "wss://api.test.com/v1/realtime",
                extra_headers={
                    "Authorization": "Bearer sk-test-key-1234567890abcdef",
                    "OpenAI-Beta": "realtime=v1"
                }
            )
    
    @pytest.mark.asyncio
    async def test_session_configuration_flow(self, service):
        """Test session configuration with different parameters."""
        mock_websocket = AsyncMock()
        service.websocket = mock_websocket
        
        # Test preserve mode configuration
        await service.configure_session(
            voice='cedar',
            voice_mode='preserve', 
            target_lang='fr'
        )
        
        # Verify session config
        mock_websocket.send.assert_called_once()
        sent_message = json.loads(mock_websocket.send.call_args[0][0])
        
        assert sent_message['type'] == 'session.update'
        session = sent_message['session']
        assert session['model'] == 'gpt-4o-realtime-preview'
        assert session['voice'] == 'cedar'
        assert session['input_audio_format'] == 'pcm16'
        assert session['output_audio_format'] == 'pcm16'
        assert 'preserve' in session['instructions'].lower()
        assert 'french' in session['instructions'].lower()
    
    @pytest.mark.asyncio
    async def test_audio_buffer_management(self, service):
        """Test audio buffer append and commit operations."""
        mock_websocket = AsyncMock()
        service.websocket = mock_websocket
        
        test_audio = b'test_audio_data'
        
        # Test single-shot flow
        with patch.object(service, 'configure_session') as mock_config:
            with patch.object(service, 'connect_websocket', return_value=True):
                
                # Mock response flow
                mock_responses = [
                    json.dumps({"type": "response.audio.delta", "delta": base64.b64encode(b"response1").decode()}),
                    json.dumps({"type": "response.audio.done"})
                ]
                mock_websocket.__aiter__.return_value = iter(mock_responses)
                
                result = await service.translate_audio_single_shot(
                    audio_data=test_audio,
                    voice='alloy',
                    voice_mode='neutral',
                    target_lang='en'
                )
                
                # Verify audio buffer operations
                calls = [call[0][0] for call in mock_websocket.send.call_args_list]
                
                # Should have audio append and commit
                audio_append = next((json.loads(call) for call in calls 
                                   if json.loads(call)['type'] == 'input_audio_buffer.append'), None)
                audio_commit = next((json.loads(call) for call in calls 
                                   if json.loads(call)['type'] == 'input_audio_buffer.commit'), None)
                
                assert audio_append is not None
                assert audio_commit is not None
                assert audio_append['audio'] == base64.b64encode(test_audio).decode()
                assert result == b'response1'
    
    @pytest.mark.asyncio
    async def test_error_handling_in_websocket(self, service):
        """Test error handling during WebSocket communication."""
        mock_websocket = AsyncMock()
        
        # Test connection error
        with patch('websockets.connect', side_effect=websockets.ConnectionClosed(None, None)):
            result = await service.connect_websocket()
            assert result is False
        
        # Test API error response
        service.websocket = mock_websocket
        error_response = json.dumps({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid audio format"
            }
        })
        
        mock_websocket.__aiter__.return_value = iter([error_response])
        
        with patch.object(service, 'connect_websocket', return_value=True):
            with patch.object(service, 'configure_session'):
                with pytest.raises(Exception) as exc_info:
                    await service.translate_audio_single_shot(b'test', 'alloy', 'preserve', 'fr')
                
                assert "Invalid audio format" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_streaming_response_handling(self, service):
        """Test streaming response handling and queue management."""
        mock_websocket = AsyncMock()
        service.websocket = mock_websocket
        
        # Create streaming audio generator
        async def test_audio_gen():
            yield b'chunk1'
            yield b'chunk2'
        
        # Mock streaming responses
        mock_responses = [
            json.dumps({"type": "response.audio.delta", "delta": base64.b64encode(b"out1").decode()}),
            json.dumps({"type": "response.audio.delta", "delta": base64.b64encode(b"out2").decode()}),
            json.dumps({"type": "response.audio.delta", "delta": base64.b64encode(b"out3").decode()}),
            json.dumps({"type": "response.audio.done"})
        ]
        
        # Setup response queue manually for testing
        for response in mock_responses:
            event = json.loads(response)
            if event["type"] == "response.audio.delta":
                service.response_queue.put(base64.b64decode(event["delta"]))
            elif event["type"] == "response.audio.done":
                service.response_queue.put(None)
        
        with patch.object(service, 'connect_websocket', return_value=True):
            with patch.object(service, 'configure_session'):
                with patch.object(service, '_handle_responses'):
                    
                    # Collect streaming results
                    results = []
                    async for chunk in service.translate_audio_streaming(
                        test_audio_gen(),
                        voice='nova',
                        voice_mode='enhanced', 
                        target_lang='fr'
                    ):
                        results.append(chunk)
                    
                    assert results == [b"out1", b"out2", b"out3"]
    
    @pytest.mark.asyncio
    async def test_websocket_cleanup_on_exception(self, service):
        """Test WebSocket cleanup when exceptions occur."""
        mock_websocket = AsyncMock()
        service.websocket = mock_websocket
        
        with patch.object(service, 'connect_websocket', return_value=True):
            with patch.object(service, 'configure_session', side_effect=Exception("Config failed")):
                
                try:
                    await service.translate_audio_single_shot(b'test', 'alloy', 'preserve', 'fr')
                except Exception:
                    pass  # Expected
                
                # Verify cleanup
                mock_websocket.close.assert_called_once()
                assert service.websocket is None
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, service):
        """Test timeout handling in single-shot translation."""
        mock_websocket = AsyncMock()
        
        # Create infinite generator that never sends response.audio.done
        async def infinite_responses():
            while True:
                yield json.dumps({"type": "response.audio.delta", "delta": base64.b64encode(b"chunk").decode()})
                await asyncio.sleep(0.1)
        
        mock_websocket.__aiter__.return_value = infinite_responses()
        
        service.websocket = mock_websocket
        
        with patch.object(service, 'connect_websocket', return_value=True):
            with patch.object(service, 'configure_session'):
                
                # Should timeout after 30 seconds (but we'll patch time for testing)
                with patch('time.time', side_effect=[0, 35]):  # Simulate timeout
                    with pytest.raises(Exception) as exc_info:
                        await service.translate_audio_single_shot(b'test', 'alloy', 'preserve', 'fr')
                    
                    assert "timeout" in str(exc_info.value).lower()

class TestWebSocketEventHandling:
    """Test WebSocket event handling and message parsing."""
    
    @pytest.fixture 
    def service(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return RealtimeTranslationService()
    
    @pytest.mark.asyncio
    async def test_response_handler_audio_events(self, service):
        """Test response handler processes audio events correctly."""
        mock_websocket = AsyncMock()
        service.websocket = mock_websocket
        
        # Mock WebSocket message stream
        messages = [
            json.dumps({"type": "response.audio.delta", "delta": base64.b64encode(b"chunk1").decode()}),
            json.dumps({"type": "response.audio.delta", "delta": base64.b64encode(b"chunk2").decode()}),
            json.dumps({"type": "response.audio.done"})
        ]
        
        mock_websocket.__aiter__.return_value = iter(messages)
        
        # Run response handler
        await service._handle_responses()
        
        # Check queue contents
        chunk1 = service.response_queue.get_nowait()
        chunk2 = service.response_queue.get_nowait()
        end_signal = service.response_queue.get_nowait()
        
        assert chunk1 == b"chunk1"
        assert chunk2 == b"chunk2"
        assert end_signal is None  # End signal
    
    @pytest.mark.asyncio
    async def test_response_handler_error_events(self, service):
        """Test response handler handles error events."""
        mock_websocket = AsyncMock()
        service.websocket = mock_websocket
        
        error_message = json.dumps({
            "type": "error",
            "error": {"message": "Rate limit exceeded"}
        })
        
        mock_websocket.__aiter__.return_value = iter([error_message])
        
        # Run response handler
        await service._handle_responses()
        
        # Should have end signal in queue
        end_signal = service.response_queue.get_nowait()
        assert end_signal is None