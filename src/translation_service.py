import os
import asyncio
import json
import logging
import base64
import threading
import queue
import time
from typing import Optional, Tuple, Generator, AsyncGenerator
import websockets
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

class RealtimeTranslationService:
    """Real-time speech-to-speech translation service using GPT-4o Realtime API."""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = os.getenv('OPENAI_API_BASE', 'wss://api.openai.com/v1')
        self.websocket = None
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Available voices with characteristics
        self.voices = {
            'alloy': {'name': 'Alloy', 'description': 'Neutral, clear'},
            'echo': {'name': 'Echo', 'description': 'Deep, resonant'},
            'fable': {'name': 'Fable', 'description': 'Warm, expressive'},
            'onyx': {'name': 'Onyx', 'description': 'Strong, confident'},
            'nova': {'name': 'Nova', 'description': 'Bright, energetic'},
            'shimmer': {'name': 'Shimmer', 'description': 'Soft, gentle'},
            'cedar': {'name': 'Cedar', 'description': 'Natural, grounded'},
            'marin': {'name': 'Marin', 'description': 'Coastal, fresh'}
        }
        
        # Language configurations
        self.languages = {
            'en': {'name': 'English', 'flag': '🇺🇸'},
            'fr': {'name': 'French', 'flag': '🇫🇷'}
        }
        
        # Voice preservation modes
        self.voice_modes = {
            'preserve': 'Preserve original speaker voice and nuances',
            'neutral': 'Use selected voice without preservation',
            'enhanced': 'Enhance voice while preserving characteristics'
        }
    
    async def connect_websocket(self) -> bool:
        """Establish WebSocket connection to GPT-Realtime API."""
        try:
            # Use correct WebSocket URL format
            ws_url = f"{self.base_url}/realtime?model=gpt-realtime"
            
            self.websocket = await websockets.connect(
                ws_url,
                extra_headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "realtime=v1"
                }
            )
            logger.info("WebSocket connection established to GPT-Realtime")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    async def configure_session(self, voice: str = 'alloy', voice_mode: str = 'neutral', 
                              target_lang: str = 'fr'):
        """Configure the realtime session with translation instructions."""
        
        # Create instruction based on voice preservation mode
        instructions = self._get_translation_instructions(voice_mode, target_lang)
        
        session_config = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": "gpt-realtime",
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": "pcm16",
                        "turn_detection": {"type": "semantic_vad", "create_response": True}
                    },
                    "output": {
                        "format": "pcm16",
                        "voice": voice,
                        "speed": 1.0
                    }
                },
                "input_audio_transcription": {"model": "whisper-1"},
                "instructions": instructions,
                "temperature": 0.6,
                "max_response_output_tokens": 4096
            }
        }
        
        await self.websocket.send(json.dumps(session_config))
        logger.info(f"Session configured with voice: {voice}, mode: {voice_mode}")
    
    def _get_translation_instructions(self, voice_mode: str, target_lang: str) -> str:
        """Generate translation instructions following OpenAI's best practices."""
        
        # Handle invalid language gracefully
        lang_info = self.languages.get(target_lang, {'name': target_lang.title()})
        lang_name = lang_info['name']
        
        # Base instruction following recommended structure
        base_instruction = f"""# Role & Objective
You are a professional speech translator specializing in French and English.
Your task is to translate spoken input to {lang_name} while maintaining natural conversation flow.

# Personality & Tone
- Warm, natural, and expressive
- Match the original speaker's emotional tone
- 2-3 sentences per turn maximum
- Deliver audio response at natural speaking pace

# Language
- ALWAYS respond in {lang_name} only
- Do not respond in any other language
- Maintain the same accent/dialect style when appropriate

# Unclear Audio
- Only respond to clear audio input
- If audio is unclear/partial/noisy/silent, ask for clarification in {lang_name}
- Sample clarification phrases:
  • "Sorry, I didn't catch that—could you say it again?"
  • "There's some background noise. Please repeat that."
  • "I only heard part of that. What did you say?"

# Variety
- Do not repeat the same sentence twice
- Vary your responses so it doesn't sound robotic"""

        # Add voice mode specific instructions
        if voice_mode == 'preserve':
            voice_instructions = """

# Voice Preservation (CRITICAL)
- PRESERVE the original speaker's voice characteristics including:
  • Vocal tone, pitch, and speaking rhythm
  • Emotional nuances and inflections
  • Speaking style and personality  
  • Age and gender vocal characteristics
  • Accent patterns when translating
- TRANSLATE the meaning while keeping the speaker's unique vocal identity intact"""
        
        elif voice_mode == 'enhanced':
            voice_instructions = """

# Voice Enhancement
- ENHANCE the original speaker's voice while preserving core characteristics:
  • Maintain emotional tone and speaking style
  • Preserve personality and inflections
  • Slightly improve clarity and naturalness
  • Keep recognizable vocal identity
- BALANCE preservation with enhancement for optimal clarity"""
        
        else:  # neutral mode
            voice_instructions = """

# Voice Processing
- Use the selected voice profile for clear, natural speech
- Focus on accurate meaning transfer and natural speech patterns
- Maintain professional, consistent vocal delivery"""

        return base_instruction + voice_instructions
    
    async def translate_audio_streaming(self, audio_generator, voice: str = 'alloy', 
                                       voice_mode: str = 'preserve', target_lang: str = 'fr'):
        """
        Stream audio translation using GPT-4o Realtime API.
        
        Args:
            audio_generator: Generator yielding audio chunks (bytes)
            voice: Voice to use for output
            voice_mode: Voice preservation mode ('preserve', 'neutral', 'enhanced')  
            target_lang: Target language ('en' or 'fr')
            
        Yields:
            Translated audio chunks
        """
        if not await self.connect_websocket():
            raise Exception("Failed to establish WebSocket connection")
        
        try:
            await self.configure_session(voice, voice_mode, target_lang)
            
            # Start listening for responses
            response_task = asyncio.create_task(self._handle_responses())
            
            # Stream audio input
            async for audio_chunk in audio_generator:
                if audio_chunk:
                    message = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(audio_chunk).decode()
                    }
                    await self.websocket.send(json.dumps(message))
            
            # Commit audio for processing
            await self.websocket.send(json.dumps({
                "type": "input_audio_buffer.commit"
            }))
            
            # Yield translated audio chunks
            while True:
                try:
                    audio_chunk = self.response_queue.get(timeout=1.0)
                    if audio_chunk is None:  # End signal
                        break
                    yield audio_chunk
                except queue.Empty:
                    continue
                    
        except Exception as e:
            logger.error(f"Streaming translation error: {e}")
            raise
        finally:
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
    
    async def translate_audio_single_shot(self, audio_data: bytes, voice: str = 'alloy',
                                         voice_mode: str = 'preserve', target_lang: str = 'fr') -> bytes:
        """
        Single-shot audio translation using GPT-4o Realtime API.
        
        Args:
            audio_data: Complete audio data as bytes
            voice: Voice to use for output
            voice_mode: Voice preservation mode
            target_lang: Target language
            
        Returns:
            Complete translated audio as bytes
        """
        if not await self.connect_websocket():
            raise Exception("Failed to establish WebSocket connection")
        
        try:
            await self.configure_session(voice, voice_mode, target_lang)
            
            # Send entire audio at once
            message = {
                "type": "input_audio_buffer.append", 
                "audio": base64.b64encode(audio_data).decode()
            }
            await self.websocket.send(json.dumps(message))
            
            # Commit and trigger processing
            await self.websocket.send(json.dumps({
                "type": "input_audio_buffer.commit"
            }))
            
            # Collect all response audio
            translated_audio = b""
            timeout = time.time() + 30  # 30 second timeout
            
            async for message in self.websocket:
                if time.time() > timeout:
                    raise Exception("Translation timeout")
                    
                event = json.loads(message)
                
                if event["type"] == "response.audio.delta":
                    audio_chunk = base64.b64decode(event["delta"])
                    translated_audio += audio_chunk
                    
                elif event["type"] == "response.audio.done":
                    break
                    
                elif event["type"] == "error":
                    raise Exception(f"API error: {event.get('error', {}).get('message', 'Unknown error')}")
            
            return translated_audio
            
        except Exception as e:
            logger.error(f"Single-shot translation error: {e}")
            raise
        finally:
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
    
    async def _handle_responses(self):
        """Handle incoming WebSocket responses and queue audio chunks."""
        try:
            async for message in self.websocket:
                event = json.loads(message)
                
                if event["type"] == "response.audio.delta":
                    audio_chunk = base64.b64decode(event["delta"])
                    self.response_queue.put(audio_chunk)
                    
                elif event["type"] == "response.audio.done":
                    self.response_queue.put(None)  # End signal
                    break
                    
                elif event["type"] == "error":
                    logger.error(f"WebSocket error: {event}")
                    self.response_queue.put(None)
                    break
                    
        except Exception as e:
            logger.error(f"Response handler error: {e}")
            self.response_queue.put(None)
    
    def get_language_info(self, lang_code: str) -> dict:
        """Get language information including name and flag."""
        return self.languages.get(lang_code, {'name': lang_code, 'flag': '🌐'})
    
    def get_voice_options(self) -> dict:
        """Get available voice options."""
        return self.voices
    
    def get_voice_mode_options(self) -> dict:
        """Get voice preservation mode options."""
        return self.voice_modes

# Global service instance
translation_service = RealtimeTranslationService()