import pyaudio
import numpy as np
import time
import sys
import threading
from pynput import keyboard
from lightning_whisper_mlx import LightningWhisperMLX
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

# Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 1600
STREAMING_INTERVAL = 2.0  # seconds between streaming updates

class KeyboardTranscriber:
    def __init__(self, model_name="distil-medium.en", batch_size=12, quant=None, 
                 streaming_mode=True, record_key='space'):
        # Initialize Rich console
        self.console = Console()
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Initialize the Whisper model
        with self.console.status("[bold green]Loading Whisper model..."):
            self.whisper = LightningWhisperMLX(model=model_name, batch_size=batch_size, quant=quant)
        
        # State variables
        self.is_running = False
        self.is_recording = False
        self.buffer = []
        
        # Recording key configuration
        self.record_key = self._parse_record_key(record_key)
        self.record_key_name = record_key.upper()
        
        # Streaming mode
        self.streaming_mode = streaming_mode
        self.streaming_active = False
        self.streaming_thread = None
        self.last_stream_time = 0
        
        # History
        self.transcription_history = []
        self.context_window = 5
        
    def _parse_record_key(self, key_name):
        """Convert key name to pynput keyboard Key object if it's a special key"""
        # Handle special keys
        special_keys = {
            'space': keyboard.Key.space,
            'enter': keyboard.Key.enter,
            'shift': keyboard.Key.shift,
            'ctrl': keyboard.Key.ctrl,
            'alt': keyboard.Key.alt,
            'tab': keyboard.Key.tab
        }
        
        if key_name.lower() in special_keys:
            return special_keys[key_name.lower()]
        
        # For regular keys, we'll use the first character
        if len(key_name) == 1:
            return keyboard.KeyCode.from_char(key_name.lower())
            
        # Default to space if invalid
        self.console.print(f"[bold yellow]Warning:[/] '{key_name}' is not a valid key, defaulting to SPACE")
        return keyboard.Key.space
        
    def start_recording(self):
        """Start recording when key is pressed"""
        self.console.print("\n[bold red]Recording...[/]")
        self.is_recording = True
        self.buffer = []
        
        # Start streaming transcription if enabled
        if self.streaming_mode:
            self.streaming_active = True
            self.last_stream_time = time.time()
            if not self.streaming_thread or not self.streaming_thread.is_alive():
                self.streaming_thread = threading.Thread(target=self.streaming_thread_function)
                self.streaming_thread.daemon = True
                self.streaming_thread.start()
        
    def stop_recording(self):
        """Stop recording and process the audio"""
        if not self.is_recording:
            return
            
        self.console.print("\n[bold yellow]Processing audio...[/]")
        self.is_recording = False
        self.streaming_active = False
        
        if not self.buffer:
            self.console.print("\n[bold orange]No audio recorded[/]")
            return
        
        # Combine all audio chunks
        audio_data = np.concatenate(self.buffer)
        
        # Transcribe
        try:
            with self.console.status("[bold]Transcribing...[/]"):
                result = self.whisper.transcribe(audio_data)
            transcript = result['text'].strip()
            
            if transcript:
                text = Text()
                text.append("📝 ", style="bold")
                text.append(transcript, style="green")
                self.console.print(Panel(text, border_style="blue"))
                
                # Add to history
                self.transcription_history.append(transcript)
                # Keep only recent history
                if len(self.transcription_history) > self.context_window:
                    self.transcription_history.pop(0)
            else:
                self.console.print("\n[yellow]No speech detected[/]")
        except Exception as e:
            self.console.print(f"\n[bold red]Transcription error:[/] {str(e)}")
        
        # Reset buffer
        self.buffer = []
        self._print_ready_message()
        
    def _print_ready_message(self):
        """Display ready status with instructions"""
        mode_info = "with real-time streaming" if self.streaming_mode else "transcribe on release"
        msg = Text()
        msg.append("READY", style="bold green")
        msg.append(f" (Hold ", style="dim")
        msg.append(self.record_key_name, style="bold cyan")
        msg.append(f" to record {mode_info}, ", style="dim")
        msg.append("ESC", style="bold red")
        msg.append(" to exit)", style="dim")
        self.console.print(msg)
        
    def streaming_thread_function(self):
        """Background thread for streaming transcription"""
        while self.is_running and self.streaming_active:
            current_time = time.time()
            if self.is_recording and len(self.buffer) > 0 and \
               (current_time - self.last_stream_time) > STREAMING_INTERVAL:
                self.process_streaming()
                self.last_stream_time = current_time
            time.sleep(0.1)  # Reduce CPU usage
    
    def process_streaming(self):
        """Process current buffer for streaming transcription"""
        if not self.buffer or len(self.buffer) < 5:  # Ensure we have enough audio
            return
            
        self.console.print("\n[yellow]Streaming...[/]")
        
        # Create a copy of current buffer for processing
        try:
            audio_data = np.concatenate(self.buffer)
            
            result = self.whisper.transcribe(audio_data)
            transcript = result['text'].strip()
            if transcript:
                text = Text()
                text.append("🔄 ", style="bold")
                text.append(transcript, style="blue italic")
                self.console.print(Panel(text, border_style="yellow"))
        except Exception as e:
            self.console.print(f"\n[bold red]Streaming error:[/] {str(e)}")
    
    def on_key_press(self, key):
        """Handle key press events"""
        try:
            if key == self.record_key and not self.is_recording:
                self.start_recording()
            elif key == keyboard.Key.esc:
                # Exit the program
                self.is_running = False
                return False  # Stop listener
        except:
            pass
    
    def on_key_release(self, key):
        """Handle key release events"""
        try:
            if key == self.record_key and self.is_recording:
                self.stop_recording()
        except:
            pass
            
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function"""
        if self.is_recording:
            pcm16 = np.frombuffer(in_data, dtype=np.int16)
            pcmf32 = pcm16.astype(np.float32) / 32768.0
            self.buffer.append(pcmf32)
        return (None, pyaudio.paContinue)
    
    def start(self):
        """Start the transcription system"""
        # Open audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self.audio_callback
        )
        
        self.is_running = True

        from rich import inspect
        
        inspect(self)
        inspect(self.whisper)

        # Print welcome banner
        self.console.print(Panel.fit(
            "[bold cyan]Lightning Whisper MLX Transcriber[/]", 
            title="[yellow]🎙️ Voice Transcription[/]", 
            subtitle=f"Model: [green]{self.whisper.name}[/]",
            border_style="blue"
        ))
        
        self._print_ready_message()
        
        # Set up keyboard listener
        with keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release) as listener:
            
            # Keep the main thread alive
            while self.is_running:
                time.sleep(0.1)
                if not listener.running:
                    break
        
        self.stop()
        
    def stop(self):
        """Stop streaming and clean up"""
        self.is_running = False
        self.streaming_active = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        self.console.print("[bold red]Stopped[/]")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Keyboard-controlled audio transcription")
    parser.add_argument("--model", type=str, default="distil-medium.en", 
                        help="Whisper model to use")
    parser.add_argument("--streaming", action="store_true", 
                        help="Enable streaming transcription while recording")
    parser.add_argument("--no-streaming", action="store_false", dest="streaming",
                        help="Disable streaming transcription (transcribe on key release only)")
    parser.add_argument("--key", type=str, default="space",
                        help="Key to hold for recording (e.g., space, ctrl, shift, a, s)")
    parser.set_defaults(streaming=True)
    
    args = parser.parse_args()
    
    try:
        transcriber = KeyboardTranscriber(model_name=args.model, 
                                         streaming_mode=args.streaming,
                                         record_key=args.key)
        transcriber.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        console = Console()
        console.print_exception()
        console.print(f"[bold red]Error:[/] {str(e)}")

if __name__ == "__main__":
    main()
    
    
    """
    # For streaming mode (default)
    python stream_keyboard.py

    # To explicitly set streaming mode
    python stream_keyboard.py --streaming

    # For transcription only when you release the key
    python stream_keyboard.py --no-streaming

    # To use a different Whisper model
    python stream_keyboard.py --model tiny.en
    
    # To use a different key for recording (instead of space)
    python stream_keyboard.py --key ctrl
    python stream_keyboard.py --key shift
    python stream_keyboard.py --key a
    """