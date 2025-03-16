import pyaudio
import numpy as np
import time
import signal
import sys
from lightning_whisper_mlx import LightningWhisperMLX

# Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 1600
VAD_THRESHOLD = 1e-5
FREQ_THRESHOLD = 100.0
SILENCE_THRESHOLD = 1.0  # seconds of silence to consider speech ended
VAD_WINDOW_SIZE_MS = 1000
MAX_RECORD_SECONDS = 10  # maximum recording time for a single utterance

class StreamTranscriber:
    def __init__(self, model_name="distil-medium.en", batch_size=12, quant=None):
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Initialize the Whisper model
        print("Loading the Whisper model...")
        self.whisper = LightningWhisperMLX(model=model_name, batch_size=batch_size, quant=quant)
        
        # State variables
        self.is_running = False
        self.is_recording = False
        self.buffer = []
        self.silence_frames = 0
        self.vad_buffer = np.array([], dtype=np.float32)
        
    def vad_simple(self, pcmf32):
        """Improved voice activity detection"""
        window_size = int(SAMPLE_RATE * VAD_WINDOW_SIZE_MS / 1000)
        
        if len(pcmf32) < window_size:
            return False
            
        if len(pcmf32.shape) == 2:
            pcmf32_mono = np.mean(pcmf32, axis=1)
        else:
            pcmf32_mono = pcmf32
            
        # Energy detection
        energy = np.mean(pcmf32_mono ** 2)
        if energy < VAD_THRESHOLD ** 2:
            return False
            
        # Frequency analysis to distinguish speech from background noise
        fft = np.fft.rfft(pcmf32_mono)
        freq = np.fft.rfftfreq(len(pcmf32_mono), d=1.0/SAMPLE_RATE)
        
        fft_energy = np.abs(fft) ** 2
        cutoff_idx = np.where(freq >= FREQ_THRESHOLD)[0][0]
        fft_low_freq_energy = np.sum(fft_energy[:cutoff_idx])
        fft_total_energy = np.sum(fft_energy)
        
        low_freq_ratio = fft_low_freq_energy / fft_total_energy
        return low_freq_ratio > 0.1
        
    def handle_audio_data(self, data):
        """Process incoming audio data"""
        pcm16 = np.frombuffer(data, dtype=np.int16)
        pcmf32 = pcm16.astype(np.float32) / 32768.0
        
        # Add to VAD buffer
        self.vad_buffer = np.concatenate((self.vad_buffer, pcmf32))
        
        # Trim VAD buffer if it gets too large
        max_vad_samples = int(VAD_WINDOW_SIZE_MS * SAMPLE_RATE / 1000)
        if len(self.vad_buffer) > max_vad_samples:
            self.vad_buffer = self.vad_buffer[-max_vad_samples:]
        
        # Speech detection logic
        if self.is_recording:
            # Already recording, add data to buffer
            self.buffer.append(pcmf32)
            
            # Check if we should stop recording (silence detected)
            if not self.vad_simple(self.vad_buffer):
                self.silence_frames += 1
                silence_duration = (self.silence_frames * CHUNK_SIZE) / SAMPLE_RATE
                
                if silence_duration >= SILENCE_THRESHOLD:
                    self.process_recording()
            else:
                self.silence_frames = 0
                
            # Check if recording is too long
            buffer_duration = len(self.buffer) * CHUNK_SIZE / SAMPLE_RATE
            if buffer_duration >= MAX_RECORD_SECONDS:
                self.process_recording()
                
        elif self.vad_simple(self.vad_buffer):
            # Start recording
            print("\n[Recording started]", end="", flush=True)
            self.is_recording = True
            self.buffer = [pcmf32]
            self.silence_frames = 0
    
    def process_recording(self):
        """Process the recorded buffer"""
        if not self.buffer:
            self.is_recording = False
            return
            
        print("\n[Processing audio...]", end="", flush=True)
        
        # Combine all audio chunks
        audio_data = np.concatenate(self.buffer)
        
        # Transcribe directly from memory
        try:
            result = self.whisper.transcribe(audio_data)
            transcript = result['text'].strip()
            if transcript:
                print(f"\n📝 {transcript}")
            else:
                print("\n[No speech detected]")
        except Exception as e:
            print(f"\n[Transcription error: {e}]")
        
        # Reset state
        self.is_recording = False
        self.buffer = []
        self.silence_frames = 0
        print("\n[Listening...]", end="", flush=True)
    
    def start(self):
        """Start streaming and transcription"""
        # Set up signal handling for clean exit
        signal.signal(signal.SIGINT, self.handle_interrupt)
        
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
        print("[Listening...] (Press Ctrl+C to exit)")
        
        # Keep the main thread alive
        try:
            while self.is_running:
                time.sleep(0.1)  # Reduce CPU usage
        except KeyboardInterrupt:
            self.stop()
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function"""
        if self.is_running:
            self.handle_audio_data(in_data)
        return (None, pyaudio.paContinue)
    
    def handle_interrupt(self, sig, frame):
        """Handle keyboard interrupt"""
        print("\n[Stopping...]")
        self.stop()
        
    def stop(self):
        """Stop streaming and clean up"""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("[Stopped]")
        sys.exit(0)

def main():
    transcriber = StreamTranscriber(model_name="distil-medium.en")
    transcriber.start()

if __name__ == "__main__":
    main()