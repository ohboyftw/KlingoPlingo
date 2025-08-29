import asyncio
import threading
import queue
import logging
import io
import wave
import numpy as np
from typing import Generator, Optional, AsyncGenerator
import soundfile as sf
from pydub import AudioSegment

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio processing for speech-to-speech translation."""
    
    def __init__(self, sample_rate: int = 24000, chunk_duration: float = 0.1):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
    
    def process_audio_file(self, audio_file_path: str) -> bytes:
        """
        Process uploaded audio file and convert to PCM16 format.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Audio data in PCM16 format
        """
        try:
            # Load audio file using pydub (supports many formats)
            audio = AudioSegment.from_file(audio_file_path)
            
            # Convert to required format: mono, 24kHz, 16-bit PCM
            audio = audio.set_channels(1)  # Mono
            audio = audio.set_frame_rate(self.sample_rate)  # 24kHz
            audio = audio.set_sample_width(2)  # 16-bit
            
            # Export as PCM16 bytes
            return audio.raw_data
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            raise Exception(f"Failed to process audio file: {str(e)}")
    
    def convert_to_gradio_format(self, audio_bytes: bytes) -> tuple:
        """
        Convert PCM16 audio bytes to Gradio audio format.
        
        Returns:
            Tuple of (sample_rate, numpy_array) for Gradio
        """
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Convert to float32 for Gradio (normalized to [-1, 1])
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            return (self.sample_rate, audio_float)
            
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return (self.sample_rate, np.array([], dtype=np.float32))
    
    def convert_from_gradio_format(self, audio_data) -> bytes:
        """
        Convert Gradio audio format to PCM16 bytes.
        
        Args:
            audio_data: Gradio audio tuple (sample_rate, numpy_array)
            
        Returns:
            PCM16 audio bytes
        """
        try:
            if audio_data is None or len(audio_data) != 2:
                raise ValueError("Invalid audio data format")
            
            sample_rate, audio_array = audio_data
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                logger.warning(f"Resampling from {sample_rate}Hz to {self.sample_rate}Hz")
                # Simple resampling (for production, use librosa)
                duration = len(audio_array) / sample_rate
                target_length = int(duration * self.sample_rate)
                audio_array = np.interp(
                    np.linspace(0, len(audio_array), target_length),
                    np.arange(len(audio_array)),
                    audio_array
                )
            
            # Convert to int16 PCM
            audio_int16 = (audio_array * 32767).astype(np.int16)
            
            return audio_int16.tobytes()
            
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            raise Exception(f"Failed to convert audio: {str(e)}")
    
    async def chunk_audio_for_streaming(self, audio_bytes: bytes) -> AsyncGenerator[bytes, None]:
        """
        Split audio into chunks for streaming processing.
        
        Args:
            audio_bytes: Complete audio data
            
        Yields:
            Audio chunks suitable for streaming
        """
        try:
            chunk_size_bytes = self.chunk_size * 2  # 2 bytes per sample (16-bit)
            
            for i in range(0, len(audio_bytes), chunk_size_bytes):
                chunk = audio_bytes[i:i + chunk_size_bytes]
                yield chunk
                await asyncio.sleep(self.chunk_duration)  # Simulate real-time
                
        except Exception as e:
            logger.error(f"Audio chunking error: {e}")
            raise
    
    def save_audio_file(self, audio_bytes: bytes, output_path: str) -> str:
        """
        Save PCM16 audio bytes to file.
        
        Args:
            audio_bytes: PCM16 audio data
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        try:
            # Convert to AudioSegment for easy export
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio = AudioSegment(
                audio_array.tobytes(),
                frame_rate=self.sample_rate,
                sample_width=2,  # 16-bit
                channels=1  # Mono
            )
            
            # Export as WAV file
            audio.export(output_path, format="wav")
            return output_path
            
        except Exception as e:
            logger.error(f"Audio save error: {e}")
            raise Exception(f"Failed to save audio: {str(e)}")

# Global audio processor instance
audio_processor = AudioProcessor()