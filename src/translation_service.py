import os
import asyncio
import json
import logging
import base64
import threading
import queue
import time
import numpy as np
from typing import Optional, Tuple, Generator, AsyncGenerator
import websockets
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

class RealtimeTranslationService:
    """Real-time speech-to-speech translation service using GPT Realtime API."""
    
    def __init__(self, model: str = "gpt-realtime"):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = os.getenv('OPENAI_API_BASE', 'wss://api.openai.com/v1')
        self.model = model
        self.websocket = None
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.backpressure_event = asyncio.Event()
        self.backpressure_event.set()  # Start with no backpressure

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
            'fr': {'name': 'French', 'flag': '🇫🇷'},
            'de': {'name': 'German', 'flag': '🇩🇪'}
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
            if not self.api_key:
                logger.error("OpenAI API key is not set")
                return False
            
            ws_url = f"{self.base_url}/realtime?model={self.model}"
            
            logger.info(f"Connecting to WebSocket URL: {ws_url}")
            
            self.websocket = await websockets.connect(
                ws_url,
                additional_headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "realtime=v1"
                }
            )
            logger.info("WebSocket connection established")

            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False

    async def configure_session(self, voice: str = 'alloy', voice_mode: str = 'neutral', 
                              target_lang: str = 'fr'):
        """Configure the realtime session with translation instructions."""
        instructions = self._get_translation_instructions(voice_mode, target_lang)
        
        # session_config = {
        #     "type": "session.update",
        #     "session": {
        #         "modalities": ["text", "audio"],
        #         "instructions": instructions,
        #         "voice": voice,
        #         "input_audio_format": "pcm16",
        #         "output_audio_format": "pcm16",
        #         "input_audio_transcription": {"model": "whisper-1"},
        #         "turn_detection": {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 200},
        #         "tools": [],
        #         "tool_choice": "auto",
        #         "temperature": 0.6,
        #         "max_response_output_tokens": 4096
        #     }
        # }

        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": instructions,
                "voice": voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                #"turn_detection": {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 200},
                "turn_detection": {
                "type": "server_vad",
                "threshold": 0.4,           
                "prefix_padding_ms": 200,   
                "silence_duration_ms": 1000  
                },
               #"turn_detection": "null",
                "temperature": 0.6,
                "max_response_output_tokens": 4096
            }
        }
        
        await self.websocket.send(json.dumps(session_config))
        logger.info(f"Session configured with voice: {voice}, mode: {voice_mode}")

    def _get_translation_instructions(self, voice_mode: str, target_lang: str) -> str:
        lang_info = self.languages.get(target_lang, {'name': target_lang.title()})
        lang_name = lang_info['name']
        
        base_instruction = f"""# Role & Objective
You are a professional speech translator specializing in French, English, and German.
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

    async def send_text(self, text: str):
        """Sends text to the realtime session."""
        if not self.websocket:
            raise Exception("WebSocket connection not established")
        
        message = {"type": "input_text_buffer.append", "text": text}
        await self.websocket.send(json.dumps(message))
        await self.websocket.send(json.dumps({"type": "input_text_buffer.commit"}))

    async def translate_audio_streaming(self, audio_generator, voice: str = 'alloy', 
                                       voice_mode: str = 'preserve', target_lang: str = 'fr'):
        if not await self.connect_websocket():
            raise Exception("Failed to establish WebSocket connection")
        
        try:
            await self.configure_session(voice, voice_mode, target_lang)
            
            response_task = asyncio.create_task(self._handle_responses())
            
            total_audio_bytes = 0
            
            async for audio_chunk in audio_generator:
                await self.backpressure_event.wait()
                if audio_chunk:
                    total_audio_bytes += len(audio_chunk)
                    message = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(audio_chunk).decode()
                    }
                    await self.websocket.send(json.dumps(message))
            
            if total_audio_bytes == 0:
                raise Exception("No audio data provided in streaming mode")

            await self.websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
            
            # Trigger response generation for streaming
            await self.websocket.send(json.dumps({"type": "response.create"}))
            
            while True:
                try:
                    response = self.response_queue.get(timeout=1.0)
                    if response is None:
                        break
                    yield response
                except queue.Empty:
                    continue
                    
        except Exception as e:
            logger.error(f"Streaming translation error: {e}")
            raise
        finally:
            if self.websocket:
                await self.websocket.close()
                self.websocket = None

    def _validate_audio_buffer(self, audio_data: bytes, context: str = "") -> None:
        if audio_data is None or len(audio_data) == 0:
            raise Exception(f"No audio data provided {context}")
        
        min_samples = int(0.1 * 24000)
        min_bytes = min_samples * 2
        
        if len(audio_data) < min_bytes:
            duration_ms = (len(audio_data) / 48)
            raise Exception(f"Audio too short {context}: {duration_ms:.2f}ms, minimum 100ms required")
        
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        if np.max(np.abs(audio_array)) < 10:
            logger.warning(f"Audio appears to be silent or very quiet {context}")
        
        logger.info(f"Audio validation passed {context}")

    async def translate_audio_single_shot(self, audio_data: bytes, voice: str = 'alloy',
                                         voice_mode: str = 'preserve', target_lang: str = 'fr') -> bytes:
        if not await self.connect_websocket():
            raise Exception("Failed to establish WebSocket connection")
        
        try:
            await self.configure_session(voice, voice_mode, target_lang)
            
            self._validate_audio_buffer(audio_data, "(single-shot)")
            
            audio_b64 = base64.b64encode(audio_data).decode('ascii')
            
            message = {"type": "input_audio_buffer.append", "audio": audio_b64}
           # logger.info(f"Sending audio data: {json.dumps(message)}")
            await self.websocket.send(json.dumps(message))

            await asyncio.sleep(0.1) # Add a small delay
            
            # Create conversation item
            conversation_item = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_audio", "audio": audio_b64}]
                }
            }
            await self.websocket.send(json.dumps(conversation_item))
            
            commit_message = {"type": "input_audio_buffer.commit"}
            await self.websocket.send(json.dumps(commit_message))
            
            # CRITICAL: Trigger response generation
            response_create = {"type": "response.create"}
            await self.websocket.send(json.dumps(response_create))
            
            translated_audio = b""
            timeout = time.time() + 30
            
            async for message in self.websocket:
                if time.time() > timeout:
                    raise Exception("Translation timeout")
                    
                event = json.loads(message)
                logger.info(f"Received event: {event['type']}")
                
                if event["type"] == "response.audio.delta":
                    translated_audio += base64.b64decode(event["delta"])
                elif event["type"] == "response.audio.done":
                    logger.info("Audio response completed")
                    break
                elif event["type"] == "response.done":
                    logger.info("Response completed")
                    break
                elif event["type"] == "error":
                    logger.error(f"API error event: {event}")
                    raise Exception(f"API error: {event.get('error', {}).get('message', 'Unknown error')}")
                elif event["type"] == "session.created":
                    logger.info("Session created successfully")
                elif event["type"] == "session.updated":
                    logger.info("Session updated successfully")
            
            return translated_audio
            
        except Exception as e:
            logger.error(f"Single-shot translation error: {e}")
            raise
        finally:
            if self.websocket:
                await self.websocket.close()
                self.websocket = None

    async def _handle_responses(self):
        try:
            async for message in self.websocket:
                event = json.loads(message)
                
                if event["type"] == "response.audio.delta":
                    self.response_queue.put(base64.b64decode(event["delta"]))
                elif event["type"] == "response.text.delta":
                    self.transcription_queue.put(event["delta"])
                elif event["type"] == "response.audio.done" or event["type"] == "response.text.done":
                    self.response_queue.put(None)
                elif event["type"] == "warning":
                    logger.warning(f"Received warning: {event.get('warning')}")
                    if "backpressure" in event.get('warning', ''):
                        self.backpressure_event.clear()
                elif event["type"] == "session.idle":
                    self.backpressure_event.set()
                elif event["type"] == "error":
                    logger.error(f"WebSocket error: {event}")
                    self.response_queue.put(None)
                    break
                    
        except Exception as e:
            logger.error(f"Response handler error: {e}")
            self.response_queue.put(None)

    def get_language_info(self, lang_code: str) -> dict:
        return self.languages.get(lang_code, {'name': lang_code, 'flag': '🌐'})
    
    def get_voice_options(self) -> dict:
        return self.voices
    
    def get_voice_mode_options(self) -> dict:
        return self.voice_modes

# Global service instance
translation_service = RealtimeTranslationService()

# Text-based translation using standard OpenAI client
def translate_text_openai(text: str, source_lang: str, target_lang: str) -> Tuple[str, str, str]:
    """
    Translate text using OpenAI's standard chat completion API.
    
    Returns:
        Tuple of (translated_text, detected_source_language, target_language)
    """
    try:
        client = OpenAI()
        
        # Determine source language for prompt
        if source_lang == "auto":
            source_prompt = "auto-detected language"
        else:
            source_prompt = translation_service.get_language_info(source_lang)['name']
        
        target_prompt = translation_service.get_language_info(target_lang)['name']
        
        # Create system prompt
        system_prompt = f"""You are an expert translator.
Translate the user's text from {source_prompt} to {target_prompt}.
First, on a single line, identify the source language using its two-letter ISO 639-1 code.
Then, on a new line, provide only the translated text.
Example:
en
Hello, how are you?
"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.5,
            max_tokens=1024
        )
        
        # Parse response
        content = response.choices[0].message.content.strip()
        lines = content.split('\n', 1)
        
        if len(lines) == 2:
            detected_source = lines[0].strip()
            translated_text = lines[1].strip()
            
            # Validate detected language code
            if detected_source not in translation_service.languages:
                detected_source = "en" # Fallback
                
            return translated_text, detected_source, target_lang
            
        else:
            # Fallback if response format is unexpected
            return content, source_lang if source_lang != "auto" else "en", target_lang
            
    except Exception as e:
        logger.error(f"Text translation error: {e}")
        raise Exception(f"Failed to translate text: {str(e)}")

# Monkey-patch the service with the text translation function
RealtimeTranslationService.translate_text = staticmethod(translate_text_openai)
