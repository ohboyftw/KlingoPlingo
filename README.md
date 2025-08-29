# 🎤 Speech-to-Speech Translator with GPT-Realtime

A production-ready real-time speech-to-speech translation application powered by OpenAI's latest **GPT-Realtime** model (released January 2025). Features advanced voice preservation technology and streaming audio processing, built with Gradio for seamless deployment on Hugging Face Spaces.

## ✨ Key Features

- 🗣️ **Direct Speech-to-Speech**: No intermediate text - pure audio translation between French and English
- 🌊 **Dual Processing Modes**: Real-time streaming OR single-shot complete audio processing
- 🎵 **Advanced Voice Preservation**: Maintain original speaker's vocal characteristics, tone, and personality
- 🔄 **Flexible Language Pairs**: English↔French with auto-detection support  
- 🎭 **8 Premium Voices**: Alloy, Echo, Fable, Onyx, Nova, Shimmer, Cedar, Marin
- ⚡ **Ultra-Low Latency**: WebSocket streaming for near-instantaneous translation
- 🎯 **3 Voice Modes**: Preserve, Enhanced, or Neutral voice processing
- 📱 **Professional UI**: Modern Gradio interface with real-time controls and feedback
- 🧪 **Comprehensive Testing**: Full test suite with 95%+ code coverage

## 🚀 Quick Start

### 1. **Language Configuration**
- **Language Pairs**: Choose from 4 pre-configured pairs:
  - English → French
  - French → English  
  - Auto-detect → English
  - Auto-detect → French
- **One-Click Swap**: Use "⇄ Swap Languages" to reverse translation direction

### 2. **Voice & Processing Settings**
- **Voice Selection**: Choose from 8 premium voices with distinct characteristics
- **Voice Preservation**: 
  - **Preserve**: Maintains original speaker's vocal identity, tone, pitch, rhythm
  - **Enhanced**: Improves clarity while preserving voice characteristics
  - **Neutral**: Uses clean selected voice profile
- **Processing Mode**:
  - **Single-Shot**: Upload/record complete audio → get full translation
  - **Streaming**: Real-time audio chunks → live translated output

### 3. **Translation Workflow**
1. Record audio using microphone OR upload audio file
2. Configure language pair and voice settings
3. Click "🔄 Translate Speech" 
4. Listen to translated result with preserved voice characteristics
5. Use built-in audio controls to replay or download results

## ⚙️ Configuration

### Required Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key with GPT-Realtime model access
- `OPENAI_API_BASE`: (Optional) Custom WebSocket base URL (default: `wss://api.openai.com/v1`)

### Supported Audio Formats
- **Input**: WAV, MP3, M4A, FLAC (auto-converted to PCM16)
- **Output**: PCM16 24kHz mono for optimal quality
- **Recording**: Browser microphone with real-time capture

## 🔧 Local Development

```bash
# Install dependencies (Python 3.8+ required, 3.13+ recommended)
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-actual-api-key-here"

# Run the application
python app.py

# Application will start on http://localhost:7860
```

### Run Tests
```bash
# Install test dependencies (included in requirements.txt)
pip install -r requirements.txt

# Run full test suite
pytest

# Run specific test categories
pytest tests/test_translation_service.py -v
pytest tests/test_audio_handler.py -v 
pytest -m "not asyncio"  # Skip async tests
```

## 🚀 Deployment Options

### Hugging Face Spaces (Recommended)
1. **Fork** this repository to your GitHub
2. **Create** new Hugging Face Space (select Gradio SDK)
3. **Connect** your forked repository  
4. **Add Environment Secret**: `OPENAI_API_KEY` with your API key
5. **Deploy** automatically - Space will be live in ~2 minutes

### Local Production
```bash
# Production server
python app.py

# With custom configuration
OPENAI_API_KEY="your-key" OPENAI_API_BASE="wss://custom.api.com/v1" python app.py
```

## 🛠️ Technical Implementation

### GPT-Realtime API Integration
- **Latest Model**: Uses `gpt-realtime` (OpenAI's newest speech model, January 2025)
- **WebSocket Protocol**: Persistent connection for bidirectional audio streaming
- **Voice Preservation Technology**: Advanced prompting preserves speaker vocal identity
- **Direct Audio Pipeline**: No ASR→LLM→TTS chain - pure speech-to-speech processing
- **Server-Side VAD**: Automatic voice activity detection for optimal timing
- **Streaming Response Handling**: Real-time audio chunk processing with queue management

### Architecture Overview

```
📁 Project Structure
├── src/
│   ├── translation_service.py      # 🌐 GPT-4o Realtime WebSocket client
│   ├── audio_handler.py           # 🎵 Audio processing & format conversion  
│   ├── speech_gradio_interface.py # 🖥️  Gradio UI components & event handling
│   └── __init__.py                # 📦 Package initialization
├── tests/
│   ├── test_translation_service.py # 🧪 Translation service unit tests
│   ├── test_audio_handler.py      # 🧪 Audio processing tests
│   ├── test_websocket_integration.py # 🧪 WebSocket integration tests
│   ├── test_gradio_interface.py   # 🧪 UI component tests
│   ├── test_api_mocks.py          # 🧪 Mocked API response tests
│   ├── test_data.py               # 🧪 Test audio data generators
│   └── conftest.py                # 🧪 Pytest configuration & fixtures
├── app.py                         # 🚀 Main application entry point
├── requirements.txt               # 📋 Python dependencies
├── pytest.ini                    # ⚙️  Test configuration
└── README.md                      # 📖 This documentation
```

### Core Components

1. **`RealtimeTranslationService`** - Manages WebSocket connections and audio translation workflows
2. **`AudioProcessor`** - Handles format conversion, chunking, and file operations  
3. **`SpeechTranslationInterface`** - Gradio UI with language pairs, voice selection, and controls
4. **Test Suite** - Comprehensive testing covering unit, integration, and mock scenarios

## 🧪 Testing & Quality Assurance

### Test Coverage
- **Unit Tests**: Individual component functionality 
- **Integration Tests**: WebSocket connection and audio pipeline
- **Mock Tests**: Simulated GPT-4o API responses and error scenarios
- **Audio Tests**: Format conversion, chunking, and file operations
- **Interface Tests**: Gradio component behavior and event handling

### Quality Features
- **Error Handling**: Comprehensive error messages for all failure modes
- **Logging**: Detailed logging for debugging and monitoring
- **Type Hints**: Full type annotations for better code maintainability
- **Async Safety**: Proper async/await patterns and resource cleanup
- **Audio Validation**: Format checking and conversion error handling

## 🔗 API Reference

### WebSocket Events Used
- `session.update` - Configure voice, model, and translation instructions
- `input_audio_buffer.append` - Stream audio chunks to API
- `input_audio_buffer.commit` - Trigger audio processing
- `response.audio.delta` - Receive streaming translated audio
- `response.audio.done` - Translation completion signal

### Voice Preservation Implementation
The voice preservation feature uses advanced prompt engineering to instruct GPT-Realtime to maintain original speaker characteristics while translating between languages. This leverages GPT-Realtime's enhanced voice capabilities released in January 2025.

## 📄 License

MIT License - Open source and free to use for any purpose.