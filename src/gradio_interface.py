import gradio as gr
import os
import asyncio
import tempfile
import threading
from typing import Tuple, Optional
from .translation_service import translation_service
from .audio_handler import audio_processor

class SpeechTranslationInterface:
    """Gradio interface for speech-to-speech translation using GPT-4o Realtime API."""
    
    def __init__(self):
        self.service = translation_service
        self.audio_processor = audio_processor
        
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Tuple[str, str, str]:
        """
        Handle translation request from Gradio interface.
        
        Returns:
            Tuple of (translated_text, source_language_display, target_language_display)
        """
        if not text or not text.strip():
            return "", "", ""
        
        try:
            translated, detected_source, actual_target = self.service.translate_text(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang
            )
            
            # Format language display
            source_info = self.service.get_language_info(detected_source)
            target_info = self.service.get_language_info(actual_target)
            
            source_display = f"{source_info['flag']} {source_info['name']}"
            target_display = f"{target_info['flag']} {target_info['name']}"
            
            return translated, source_display, target_display
            
        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            return error_msg, "❌ Error", "❌ Error"
    
    def swap_languages(self, source_lang: str, target_lang: str) -> Tuple[str, str]:
        """Swap source and target languages."""
        return target_lang, source_lang
    
    def clear_all(self) -> Tuple[str, str, str, str]:
        """Clear all text fields."""
        return "", "", "", ""
    
    def create_interface(self) -> gr.Interface:
        """Create the main Gradio interface."""
        
        with gr.Blocks(
            title="French-English Translator",
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="slate",
                neutral_hue="slate"
            ),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: 0 auto !important;
            }
            .translation-header {
                text-align: center;
                margin-bottom: 2rem;
            }
            .language-badge {
                display: inline-block;
                padding: 0.5rem 1rem;
                background: #f1f5f9;
                border-radius: 0.5rem;
                font-weight: 500;
                margin: 0 0.5rem;
            }
            """
        ) as interface:
            
            # Header
            gr.HTML("""
            <div class="translation-header">
                <h1 style="font-size: 2.5rem; font-weight: bold; color: #1e293b; margin-bottom: 0.5rem;">
                    🌍 French-English Translator
                </h1>
                <p style="font-size: 1.2rem; color: #64748b;">
                    Powered by GPT-5 Realtime API for accurate, natural translations
                </p>
            </div>
            """)
            
            # Language selection row
            with gr.Row():
                with gr.Column(scale=2):
                    source_lang = gr.Dropdown(
                        choices=[("Auto-detect", "auto"), ("English 🇺🇸", "en"), ("French 🇫🇷", "fr")],
                        value="auto",
                        label="Source Language",
                        interactive=True
                    )
                
                with gr.Column(scale=1, min_width=100):
                    swap_btn = gr.Button(
                        "⇄ Swap",
                        variant="secondary",
                        size="sm"
                    )
                
                with gr.Column(scale=2):
                    target_lang = gr.Dropdown(
                        choices=[("English 🇺🇸", "en"), ("French 🇫🇷", "fr")],
                        value="fr",
                        label="Target Language",
                        interactive=True
                    )
            
            # Translation interface
            with gr.Row(equal_height=True):
                with gr.Column():
                    input_text = gr.Textbox(
                        placeholder="Enter text to translate...",
                        label="Input Text",
                        lines=8,
                        max_lines=12,
                        show_copy_button=True,
                        interactive=True
                    )
                    with gr.Row():
                        input_lang_display = gr.Textbox(
                            label="Detected Language",
                            interactive=False,
                            scale=1
                        )
                        char_count_input = gr.HTML(
                            "<div style='text-align: right; color: #64748b; font-size: 0.875rem;'>0 characters</div>"
                        )
                
                with gr.Column():
                    output_text = gr.Textbox(
                        placeholder="Translation will appear here...",
                        label="Translation",
                        lines=8,
                        max_lines=12,
                        show_copy_button=True,
                        interactive=False
                    )
                    with gr.Row():
                        output_lang_display = gr.Textbox(
                            label="Target Language",
                            interactive=False,
                            scale=1
                        )
                        char_count_output = gr.HTML(
                            "<div style='text-align: right; color: #64748b; font-size: 0.875rem;'>0 characters</div>"
                        )
            
            # Action buttons
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
                translate_btn = gr.Button(
                    "🔄 Translate", 
                    variant="primary", 
                    scale=2,
                    size="lg"
                )
            
            # Status message area
            status_msg = gr.HTML("")
            
            # Event handlers
            def handle_translation(text: str, src: str, tgt: str):
                """Handle translation with status updates."""
                if not text.strip():
                    return "", "", "", "<div style='color: #ef4444;'>⚠️ Please enter text to translate</div>"
                
                try:
                    translated, source_display, target_display = self.translate_text(text, src, tgt)
                    
                    if translated.startswith("Translation failed"):
                        return "", "", "", f"<div style='color: #ef4444;'>❌ {translated}</div>"
                    
                    return (
                        translated, 
                        source_display, 
                        target_display,
                        "<div style='color: #22c55e;'>✅ Translation completed successfully</div>"
                    )
                    
                except Exception as e:
                    return "", "", "", f"<div style='color: #ef4444;'>❌ Error: {str(e)}</div>"
            
            def handle_swap(src: str, tgt: str):
                """Handle language swap."""
                new_target, new_source = self.swap_languages(src, tgt)
                
                # Don't allow auto as target
                if new_target == "auto":
                    new_target = "en"
                
                return new_source, new_target
            
            def handle_clear():
                """Handle clear all fields."""
                return self.clear_all() + ("",)
            
            def update_char_count_input(text):
                """Update input character count."""
                count = len(text) if text else 0
                return f"<div style='text-align: right; color: #64748b; font-size: 0.875rem;'>{count} characters</div>"
            
            def update_char_count_output(text):
                """Update output character count."""
                count = len(text) if text else 0
                return f"<div style='text-align: right; color: #64748b; font-size: 0.875rem;'>{count} characters</div>"
            
            # Wire up events
            translate_btn.click(
                fn=handle_translation,
                inputs=[input_text, source_lang, target_lang],
                outputs=[output_text, input_lang_display, output_lang_display, status_msg]
            )
            
            swap_btn.click(
                fn=handle_swap,
                inputs=[source_lang, target_lang],
                outputs=[source_lang, target_lang]
            )
            
            clear_btn.click(
                fn=handle_clear,
                outputs=[input_text, output_text, input_lang_display, output_lang_display, status_msg]
            )
            
            # Real-time character counting
            input_text.change(
                fn=update_char_count_input,
                inputs=[input_text],
                outputs=[char_count_input]
            )
            
            output_text.change(
                fn=update_char_count_output,
                inputs=[output_text],
                outputs=[char_count_output]
            )
            
            # Enter key support for translation
            input_text.submit(
                fn=handle_translation,
                inputs=[input_text, source_lang, target_lang],
                outputs=[output_text, input_lang_display, output_lang_display, status_msg]
            )
        
        return interface

def create_gradio_app() -> gr.Interface:
    """Factory function to create the Gradio application."""
    interface = TranslationInterface()
    return interface.create_interface()