import gradio as gr
import os
import asyncio
import tempfile
import threading
import time
import queue
from typing import Tuple, Optional
from .translation_service import translation_service
from .audio_handler import audio_processor

class SpeechTranslationInterface:
    """Gradio interface for speech-to-speech translation using GPT Realtime API."""
    
    def __init__(self):
        self.service = translation_service
        self.audio_processor = audio_processor
        
        # Language pair configurations
        self.language_pairs = [
            ("English → French", "en", "fr"),
            ("French → English", "fr", "en"),
            ("English → German", "en", "de"),
            ("German → English", "de", "en"),
            ("French → German", "fr", "de"),
            ("German → French", "de", "fr"),
            ("Auto-detect → English", "auto", "en"),
            ("Auto-detect → French", "auto", "fr"),
            ("Auto-detect → German", "auto", "de")
        ]
    
    def translate_audio_single_shot(self, audio_input, language_pair: str, voice: str, 
                                   voice_mode: str, processing_mode: str) -> Tuple[Optional[tuple], str]:
        """
        Handle single-shot audio translation.
        
        Returns:
            Tuple of (translated_audio, status_message)
        """
        if audio_input is None:
            return None, "❌ Please record or upload audio first"
        
        try:
            # Debug: Log audio input details
            print(f"DEBUG: Audio input type: {type(audio_input)}")
            print(f"DEBUG: Audio input content: {audio_input if not isinstance(audio_input, tuple) else f'tuple with {len(audio_input)} elements'}")
            
            if isinstance(audio_input, tuple) and len(audio_input) == 2:
                sample_rate, audio_data = audio_input
                print(f"DEBUG: Sample rate: {sample_rate}, Audio data shape: {audio_data.shape if hasattr(audio_data, 'shape') else 'No shape'}")
                print(f"DEBUG: Audio data type: {type(audio_data)}, Length: {len(audio_data) if hasattr(audio_data, '__len__') else 'No length'}")
            
            # Parse language pair
            _, source_lang, target_lang = next(
                (pair for pair in self.language_pairs if pair[0] == language_pair),
                (None, "en", "fr")
            )
            
            # Convert audio to required format
            audio_bytes = self.audio_processor.convert_from_gradio_format(audio_input)
            
            # Early validation check - don't proceed if audio is too short
            if len(audio_bytes) < 4800:  # 100ms at 24kHz, 16-bit
                duration_ms = (len(audio_bytes) / 2 / 24000) * 1000
                return None, f"❌ Recording too short: {duration_ms:.1f}ms (minimum 100ms required). Please record for at least 1 second."
            
            # Run async translation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                translated_audio = loop.run_until_complete(
                    self.service.translate_audio_single_shot(
                        audio_data=audio_bytes,
                        voice=voice,
                        voice_mode=voice_mode,
                        target_lang=target_lang
                    )
                )
                
                # Validate response before conversion
                if not translated_audio or len(translated_audio) == 0:
                    return None, f"❌ Translation failed: No audio response received from API"
                
                # Convert back to Gradio format
                output_audio = self.audio_processor.convert_to_gradio_format(translated_audio)
                
                if output_audio is None:
                    return None, f"❌ Translation failed: Could not process audio response"
                
                return output_audio, f"✅ Translation completed ({voice_mode} mode)"
                
            finally:
                loop.close()
                
        except Exception as e:
            error_msg = f"❌ Translation failed: {str(e)}"
            return None, error_msg
    
    def translate_audio_streaming(self, audio_input, language_pair: str, voice: str,
                                 voice_mode: str, processing_mode: str):
        """
        Handle streaming audio translation with progressive audio updates.
        
        Yields:
            Tuple of (accumulated_audio, status_message) for each chunk
        """
        if audio_input is None:
            yield None, "❌ Please record or upload audio first"
            return
        
        try:
            # Parse language pair
            _, source_lang, target_lang = next(
                (pair for pair in self.language_pairs if pair[0] == language_pair),
                (None, "en", "fr")
            )
            
            # Convert audio to required format
            audio_bytes = self.audio_processor.convert_from_gradio_format(audio_input)
            
            # Early validation check - don't proceed if audio is too short
            if len(audio_bytes) < 4800:  # 100ms at 24kHz, 16-bit
                duration_ms = (len(audio_bytes) / 2 / 24000) * 1000
                yield None, f"❌ Recording too short: {duration_ms:.1f}ms (minimum 100ms required). Please record for at least 1 second."
                return
            
            # Use threading to handle async streaming while yielding progressively
            import concurrent.futures
            import threading
            
            # Create a queue for progressive results
            result_queue = queue.Queue()
            exception_holder = [None]
            
            def run_streaming_in_thread():
                """Run streaming translation in a separate thread."""
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    async def run_streaming():
                        # Create async generator for audio chunks
                        async def audio_generator():
                            async for chunk in self.audio_processor.chunk_audio_for_streaming(audio_bytes):
                                yield chunk
                        
                        # Progressive collection of streaming response
                        accumulated_audio = b""
                        chunk_count = 0
                        
                        async for audio_chunk in self.service.translate_audio_streaming(
                            audio_generator(),
                            voice=voice,
                            voice_mode=voice_mode,
                            target_lang=target_lang
                        ):
                            if audio_chunk:
                                accumulated_audio += audio_chunk
                                chunk_count += 1
                                
                                # Convert accumulated audio to Gradio format
                                if len(accumulated_audio) > 0:
                                    progressive_audio = self.audio_processor.convert_to_gradio_format(accumulated_audio)
                                    if progressive_audio is not None:
                                        result_queue.put((progressive_audio, f"🌊 Streaming... ({chunk_count} chunks, {len(accumulated_audio)} bytes)"))
                        
                        # Final result
                        if len(accumulated_audio) > 0:
                            final_audio = self.audio_processor.convert_to_gradio_format(accumulated_audio)
                            result_queue.put((final_audio, f"✅ Streaming completed ({chunk_count} chunks, {len(accumulated_audio)} bytes total)"))
                        else:
                            result_queue.put((None, "❌ No audio received from streaming translation"))
                        
                        # Signal completion
                        result_queue.put(None)
                    
                    loop.run_until_complete(run_streaming())
                    loop.close()
                    
                except Exception as e:
                    exception_holder[0] = e
                    result_queue.put(None)
            
            # Start streaming in background thread
            thread = threading.Thread(target=run_streaming_in_thread)
            thread.start()
            
            # Yield results as they become available
            while True:
                try:
                    result = result_queue.get(timeout=1.0)
                    if result is None:
                        break
                    yield result
                except queue.Empty:
                    continue
            
            # Wait for thread to complete and check for exceptions
            thread.join()
            if exception_holder[0]:
                raise exception_holder[0]
                
        except Exception as e:
            error_msg = f"❌ Streaming translation failed: {str(e)}"
            yield None, error_msg
    
    def swap_language_pair(self, current_pair: str) -> str:
        """Swap the language pair direction."""
        swap_mapping = {
            "English → French": "French → English",
            "French → English": "English → French",
            "English → German": "German → English",
            "German → English": "English → German",
            "French → German": "German → French", 
            "German → French": "French → German",
            "Auto-detect → English": "Auto-detect → French",
            "Auto-detect → French": "Auto-detect → German",
            "Auto-detect → German": "Auto-detect → English"
        }
        return swap_mapping.get(current_pair, current_pair)
    
    def create_interface(self) -> gr.Interface:
        """Create the speech translation Gradio interface."""
        
        # Get voice options
        voice_options = [(f"{info['name']} - {info['description']}", voice) 
                        for voice, info in self.service.get_voice_options().items()]
        
        # Get voice mode options  
        voice_mode_options = [(desc, mode) 
                             for mode, desc in self.service.get_voice_mode_options().items()]
        
        # Language pair options
        language_pair_options = [pair[0] for pair in self.language_pairs]
        
        with gr.Blocks(
            title="Speech-to-Speech Translator",
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="slate", 
                neutral_hue="slate"
            ),
            css="""
            .gradio-container {
                max-width: 1400px !important;
                margin: 0 auto !important;
            }
            .audio-section {
                border: 2px solid #e2e8f0;
                border-radius: 1rem;
                padding: 1.5rem;
                background: #f8fafc;
            }
            .control-section {
                background: #f1f5f9;
                border-radius: 0.75rem;
                padding: 1rem;
                margin: 1rem 0;
            }
            """
        ) as interface:
            
            # Header
            gr.HTML("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="font-size: 3rem; font-weight: bold; color: #1e293b; margin-bottom: 0.5rem;">
                    🎤 Speech-to-Speech Translator
                </h1>
                <p style="font-size: 1.3rem; color: #64748b;">
                    Real-time voice translation with GPT Realtime API • Voice preservation available
                </p>
            </div>
            """)
            
            # Configuration Section
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<div class='control-section'><h3>🎯 Translation Settings</h3></div>")
                    
                    language_pair = gr.Dropdown(
                        choices=language_pair_options,
                        value="English → French",
                        label="Language Pair",
                        interactive=True
                    )
                    
                    with gr.Row():
                        swap_btn = gr.Button("⇄ Swap Languages", variant="secondary", size="sm")
                    
                with gr.Column(scale=1):
                    gr.HTML("<div class='control-section'><h3>🎵 Voice Settings</h3></div>")
                    
                    voice_selection = gr.Dropdown(
                        choices=voice_options,
                        value=voice_options[0][1],  # Default to first voice
                        label="Output Voice",
                        interactive=True
                    )
                    
                    voice_mode = gr.Dropdown(
                        choices=voice_mode_options,
                        value="preserve",
                        label="Voice Preservation Mode",
                        interactive=True
                    )
                
                with gr.Column(scale=1):
                    gr.HTML("<div class='control-section'><h3>⚡ Processing Mode</h3></div>")
                    
                    processing_mode = gr.Radio(
                        choices=[
                            ("Single Shot", "single_shot"),
                            # ("Streaming", "streaming")  # Disabled for now
                        ],
                        value="single_shot",
                        label="Processing Type",
                        interactive=True
                    )
            
            # Audio Translation Section
            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.HTML("<div class='audio-section'><h3>🎤 Input Audio</h3></div>")
                    
                    input_audio = gr.Audio(
                        label="Record or Upload Audio",
                        type="numpy",
                        interactive=True,
                        sources=["microphone", "upload"]
                    )
                    
                    with gr.Row():
                        translate_btn = gr.Button(
                            "🔄 Translate Speech",
                            variant="primary",
                            size="lg",
                            scale=2
                        )
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                        replay_btn = gr.Button("▶️ Replay", variant="secondary")
                
                with gr.Column():
                    gr.HTML("<div class='audio-section'><h3>🔊 Translated Audio</h3></div>")
                    
                    output_audio = gr.Audio(
                        label="Translation Result",
                        type="numpy",
                        interactive=True,  # Enable playback controls
                        autoplay=False,
                        show_download_button=True
                    )
                    
                    status_display = gr.HTML(
                        "<div style='text-align: center; color: #64748b;'>Ready for translation</div>"
                    )
            
            # Voice Mode Information and Recording Tips
            with gr.Row():
                gr.HTML("""
                <div style="background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0;">
                    <h4 style="color: #0369a1; margin-bottom: 0.5rem;">🎵 Voice Preservation Modes:</h4>
                    <ul style="color: #0369a1; margin: 0; padding-left: 1.5rem;">
                        <li><strong>Preserve:</strong> Maintains original speaker's voice characteristics and nuances</li>
                        <li><strong>Enhanced:</strong> Improves clarity while preserving voice identity</li>
                        <li><strong>Neutral:</strong> Uses selected voice profile without preservation</li>
                    </ul>
                </div>
                """)
            
            with gr.Row():
                gr.HTML("""
                <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0;">
                    <h4 style="color: #92400e; margin-bottom: 0.5rem;">📝 Recording Tips:</h4>
                    <ul style="color: #92400e; margin: 0; padding-left: 1.5rem;">
                        <li><strong>Minimum Duration:</strong> Record for at least 1 second of clear speech</li>
                        <li><strong>Audio Quality:</strong> Speak clearly in a quiet environment</li>
                        <li><strong>Volume:</strong> Ensure you're speaking loud enough to be heard</li>
                        <li><strong>Testing:</strong> Try saying "Hello, how are you?" for testing</li>
                    </ul>
                </div>
                """)
            
            # Event handlers
            def handle_translation(audio_input, lang_pair, voice, voice_mode_val, proc_mode):
                """Route to appropriate translation method based on processing mode."""
                if proc_mode == "streaming":
                    # For streaming, return a generator that yields progressive updates
                    for result in self.translate_audio_streaming(audio_input, lang_pair, voice, voice_mode_val, proc_mode):
                        yield result
                else:
                    return self.translate_audio_single_shot(audio_input, lang_pair, voice, voice_mode_val, proc_mode)
            
            def handle_swap(current_pair):
                """Handle language pair swapping."""
                return self.swap_language_pair(current_pair)
            
            def handle_clear():
                """Clear all audio fields."""
                return None, None, "<div style='text-align: center; color: #64748b;'>Ready for translation</div>"

            def replay_audio(audio_input):
                return audio_input

            # Wire up events
            translate_btn.click(
                fn=handle_translation,
                inputs=[input_audio, language_pair, voice_selection, voice_mode, processing_mode],
                outputs=[output_audio, status_display]
            )
            
            swap_btn.click(
                fn=handle_swap,
                inputs=[language_pair],
                outputs=[language_pair]
            )
            
            clear_btn.click(
                fn=handle_clear,
                outputs=[input_audio, output_audio, status_display]
            )

            replay_btn.click(
                fn=replay_audio,
                inputs=[input_audio],
                outputs=[input_audio]
            )
            
            # Update status when processing mode changes
            def update_processing_info(mode):
                if mode == "streaming":
                    return "<div style='color: #2563eb;'>🌊 Streaming mode: Real-time audio processing</div>"
                else:
                    return "<div style='color: #059669;'>⚡ Single-shot mode: Complete audio processing</div>"
            
            processing_mode.change(
                fn=update_processing_info,
                inputs=[processing_mode], 
                outputs=[status_display]
            )
        
        return interface

def create_gradio_app() -> gr.Interface:
    """Factory function to create the speech translation Gradio application."""
    interface = SpeechTranslationInterface()
    return interface.create_interface()