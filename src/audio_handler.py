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
            if not audio_bytes or len(audio_bytes) == 0:
                logger.warning("Empty audio bytes provided to convert_to_gradio_format")
                return None  # Return None instead of empty array
            
            # Convert bytes to numpy array (int16)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            logger.info(f"Converting {len(audio_array)} samples to Gradio format")
            
            if len(audio_array) == 0:
                logger.warning("Audio array is empty after conversion")
                return None
            
            # Return as int16 to avoid Gradio's automatic conversion warning
            # Gradio expects int16 format for audio output
            return (self.sample_rate, audio_array)
            
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return None  # Return None instead of empty array
    
    def convert_from_gradio_format(self, audio_data) -> bytes:
        """
        Convert Gradio audio format to PCM16 bytes.
        
        Args:
            audio_data: Gradio audio tuple (sample_rate, numpy_array)
            
        Returns:
            PCM16 audio bytes
        """
        try:
            logger.info(f"Converting audio data: type={type(audio_data)}")
            
            if audio_data is None:
                raise ValueError("Audio data is None")
                
            if not isinstance(audio_data, (tuple, list)) or len(audio_data) != 2:
                raise ValueError(f"Invalid audio data format: expected tuple/list of length 2, got {type(audio_data)} with length {len(audio_data) if hasattr(audio_data, '__len__') else 'N/A'}")
            
            sample_rate, audio_array = audio_data
            logger.info(f"Sample rate: {sample_rate}, Audio array type: {type(audio_array)}")
            
            # Convert to numpy array if needed and validate
            if not isinstance(audio_array, np.ndarray):
                logger.info("Converting to numpy array")
                try:
                    audio_array = np.array(audio_array, dtype=np.float32)
                except (ValueError, TypeError) as e:
                    logger.error(f"Failed to convert to numpy array: {e}")
                    # Try to extract from nested structure
                    if hasattr(audio_array, '__iter__'):
                        try:
                            # Flatten any nested structure
                            flat_data = []
                            def flatten_recursive(data):
                                if isinstance(data, (list, tuple)):
                                    for item in data:
                                        flatten_recursive(item)
                                else:
                                    flat_data.append(float(data))
                            flatten_recursive(audio_array)
                            audio_array = np.array(flat_data, dtype=np.float32)
                        except Exception as nested_e:
                            raise ValueError(f"Cannot process audio data structure: {nested_e}")
                    else:
                        raise ValueError(f"Cannot convert audio data to numpy array: {e}")
            
            logger.info(f"Audio array shape: {audio_array.shape}, dtype: {audio_array.dtype}")
            
            # Handle nested arrays - flatten if multidimensional
            if audio_array.ndim > 1:
                logger.warning(f"Multi-dimensional audio array detected: {audio_array.shape}")
                audio_array = audio_array.flatten()
                
            # Validate audio array
            if audio_array is None or len(audio_array) == 0:
                raise ValueError("Empty audio array provided")
            
            # Handle different input formats from Gradio
            if audio_array.dtype == np.int16:
                # Already int16, convert to float32 for processing
                logger.info("Converting int16 input to float32 for processing")
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype == np.int32:
                # Int32, convert to float32
                logger.info("Converting int32 input to float32 for processing")
                audio_array = audio_array.astype(np.float32) / 2147483648.0
            else:
                # Assume float, ensure it's float32
                logger.info(f"Converting {audio_array.dtype} to float32 for processing")
                audio_array = audio_array.astype(np.float32)
            
            # Ensure we have at least 100ms of audio
            min_samples = int(0.1 * sample_rate)  # 100ms at input sample rate
            if len(audio_array) < min_samples:
                logger.warning(f"Audio too short ({len(audio_array)} samples), padding to minimum length")
                # Pad with silence to meet minimum requirement
                padding_needed = min_samples - len(audio_array)
                audio_array = np.concatenate([audio_array, np.zeros(padding_needed, dtype=audio_array.dtype)])
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                logger.info(f"Resampling from {sample_rate}Hz to {self.sample_rate}Hz")
                # Simple resampling (for production, use librosa)
                duration = len(audio_array) / sample_rate
                target_length = int(duration * self.sample_rate)
                audio_array = np.interp(
                    np.linspace(0, len(audio_array), target_length),
                    np.arange(len(audio_array)),
                    audio_array
                )
            
            # Ensure audio array is in valid range [-1, 1]
            audio_array = np.clip(audio_array, -1.0, 1.0)
            
            # Convert to int16 PCM
            audio_int16 = (audio_array * 32767).astype(np.int16)
            logger.info(f"Converted to int16: min={audio_int16.min()}, max={audio_int16.max()}, samples={len(audio_int16)}")

            # Create an AudioSegment from the raw audio data
            try:
                audio_bytes = audio_int16.tobytes()
                logger.info(f"Audio bytes length: {len(audio_bytes)}")
                audio_segment = AudioSegment(audio_bytes, frame_rate=self.sample_rate, sample_width=2, channels=1)
                logger.info(f"AudioSegment created: duration={len(audio_segment)}ms")
            except Exception as e:
                logger.error(f"Failed to create AudioSegment: {e}")
                # Fallback: return raw bytes without trimming
                result_bytes = audio_int16.tobytes()
                logger.info(f"Fallback conversion: {len(result_bytes)} bytes ({len(result_bytes)/(2*self.sample_rate)*1000:.1f}ms)")
                return result_bytes

            # Trim silence
            try:
                trimmed_segment = self.trim_silence(audio_segment)
            except Exception as e:
                logger.warning(f"Failed to trim silence: {e}, using original audio")
                trimmed_segment = audio_segment
            
            result_bytes = trimmed_segment.raw_data
            logger.info(f"Converted audio: {len(result_bytes)} bytes ({len(result_bytes)/(2*self.sample_rate)*1000:.1f}ms)")
            
            return result_bytes
            
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            raise Exception(f"Failed to convert audio: {str(e)}")
    
    def trim_silence(self, audio_segment, silence_thresh=-40, chunk_size=10):
        """
        Trims silence from the beginning of an audio segment.
        """
        start_trim = 0
        for i in range(0, len(audio_segment), chunk_size):
            if audio_segment[i:i+chunk_size].dBFS > silence_thresh:
                start_trim = i
                break

        return audio_segment[start_trim:]

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