#!/usr/bin/env python3
"""
Speech-to-Speech Translation App for Hugging Face Spaces
Powered by OpenAI GPT Realtime API with Voice Preservation
"""

import os
import logging
import gradio as gr
from src.speech_gradio_interface import create_gradio_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_api_key():
    """Check if OpenAI API key is configured."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set")
        return False
    
    if api_key.startswith('sk-') and len(api_key) > 20:
        logger.info("OpenAI API key is properly configured")
        return True
    else:
        logger.error("OPENAI_API_KEY appears to be invalid")
        return False

def create_error_interface():
    """Create a simple error interface when API key is missing."""
    with gr.Blocks(title="Translation App - Configuration Error") as interface:
        gr.HTML("""
        <div style="text-align: center; padding: 2rem; color: #ef4444;">
            <h1>⚠️ Configuration Error</h1>
            <p style="margin: 1rem 0;">
                OpenAI API key is not configured properly.
            </p>
            <p style="color: #64748b; font-size: 0.9rem;">
                Please set the OPENAI_API_KEY environment variable with a valid OpenAI API key that has access to GPT Realtime API.
            </p>
        </div>
        """)
    return interface

def main():
    """Main application entry point."""
    logger.info("Starting Speech-to-Speech Translation App")
    
    # Check API key configuration
    if not check_api_key():
        logger.warning("Creating error interface due to missing API key")
        app = create_error_interface()
    else:
        logger.info("Creating main translation interface")
        app = create_gradio_app()
    
    # Launch the app
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,  # Standard Hugging Face Spaces port
        share=False,
        show_api=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()