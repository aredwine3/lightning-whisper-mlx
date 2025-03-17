# Lightning Whisper MLX Keyboard Transcriber Documentation

## 1. Overview

The stream_keyboard.py script provides a powerful voice-to-text transcription tool that uses Apple Silicon optimized Whisper models to deliver fast, accurate speech recognition with minimal latency. This keyboard-controlled tool allows users to easily transcribe speech in real-time by holding down a designated key.

## 2. Installation

### 2.1. Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.8+
- `lightning-whisper-mlx` package

### 2.2. Required Dependencies

```bash
pip install lightning-whisper-mlx pyaudio numpy pynput pyperclip rich PySimpleGUI
```

### 2.3. Optional Dependencies

For enhanced text correction:
```bash
pip install mlx-lm
```

## 3. Core Features

### 3.1. Speech Transcription

- Hold-to-record functionality (default: space bar)
- Fast transcription using MLX-optimized Whisper models
- Support for multiple Whisper model sizes (tiny, small, medium, large)

### 3.2. Streaming Mode
- Real-time speech-to-text preview while recording
- Configurable streaming interval
- Complete transcription on key release

### 3.3. Text Enhancement

- Optional LLM-based grammar and punctuation improvement
- Multiple enhancement options:
  - MLX-based language models (requires mlx-lm)
  - Basic text correction (capitalization, punctuation)
- Customizable correction prompts

### 3.4. Clipboard Integration

- Automatic clipboard copying of transcribed text
- Auto-paste functionality into active text field

### 3.5. User Interface

- Rich console output with color-coded status information
- Optional floating GUI window with transcription status

## 4. Usage Instructions

### 4.1. Basic Usage

```bash
python stream_keyboard.py
```

### 4.2. Common Command Options

```bash
# Use a specific Whisper model
python stream_keyboard.py --model tiny.en

# Enable text enhancement
python stream_keyboard.py --enhance

# Change recording key
python stream_keyboard.py --key ctrl

# Disable auto-paste
python stream_keyboard.py --no-paste

# Use basic text enhancement (no LLM required)
python stream_keyboard.py --enhance --enhancer basic

# Disable the streaming preview
python stream_keyboard.py --no-streaming

# Enable GUI window
python stream_keyboard.py --gui
```

### 4.3. Advanced Options

```bash
# Use a custom MLX model for text enhancement
python stream_keyboard.py --enhance --llm-model mlx-community/Phi-3-mini-4k-instruct
```

## 5. Architecture

### 5.1. Key Classes

#### 5.1.1. `KeyboardTranscriber`

The main class that manages recording, transcription, and output.

#### 5.1.2. Text Enhancement Classes

- `TextEnhancer`: Base class for text enhancement
- `MLXTextEnhancer`: Uses MLX-based language models for advanced correction
- `DummyEnhancer`: Provides basic text fixes without ML models

## 6. Technical Details

### 6.1. Audio Processing

- Sample Rate: 16000 Hz
- Chunk Size: 1600 samples
- Format: 32-bit float PCM

### 6.2. Transcription Process

1. Audio is captured via PyAudio in chunks
2. Buffered audio data is processed by LightningWhisperMLX
3. Transcribed text is optionally enhanced
4. Results are displayed and/or pasted

### 6.3. Text Enhancement

1. Raw transcription is sent to an enhancer
2. For MLX enhancement, a prompt is formatted with the transcription
3. The language model generates improved text with proper grammar and punctuation

## 7. Command Line Reference

| Option                   | Description                                                       |
| ------------------------ | ----------------------------------------------------------------- |
| `--model MODEL`          | Whisper model to use (default: distil-medium.en)                  |
| `--streaming`            | Enable streaming transcription while recording                    |
| `--no-streaming`         | Disable streaming (transcribe on key release only)                |
| `--key KEY`              | Key to hold for recording (e.g., space, ctrl, shift, cntrl+shift) |
| `--paste`                | Enable automatic pasting of text (default: on)                    |
| `--no-paste`             | Disable automatic pasting                                         |
| `--enhance`              | Enable LLM-based text enhancement                                 |
| `--enhancer {mlx,basic}` | Type of text enhancer to use                                      |
| `--llm-model MODEL`      | MLX model for text enhancement                                    |
| `--gui`                  | Show floating GUI window with status                              |
| `--no-gui`               | Disable GUI window                                                |

## 8. Troubleshooting

### 8.1. Common Issues
- **No sound detected**: Check microphone permissions and input device
- **MLX enhancement errors**: Install mlx-lm or use --enhancer basic
- **GUI errors**: Use --no-gui to disable the GUI interface
- **Clipboard errors**: Ensure pyperclip is installed correctly

## 9. Extending the Tool

### 9.1. Creating a Custom Text Enhancer
Extend the `TextEnhancer` class and implement the `enhance` method:

```python
class CustomEnhancer(TextEnhancer):
    def enhance(self, text):
        # Custom enhancement logic
        return improved_text
```

### 9.2. Using with Other LLM Models
The tool supports any MLX-compatible language model through the `--llm-model` parameter.

## 10. Performance Considerations
- Smaller Whisper models (tiny, small) provide faster transcription
- Using the `--no-streaming` option reduces CPU usage
- Text enhancement adds processing time but improves output quality