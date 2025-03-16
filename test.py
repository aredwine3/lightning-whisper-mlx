import time
from lightning_whisper_mlx import LightningWhisperMLX

whisper = LightningWhisperMLX(model="medium", batch_size=12, quant=None)

# Start timing
start_time = time.perf_counter()

# Transcribe the audio file
text = whisper.transcribe(audio_path="./test_audio_files/audio-sample-6.mp3")['text']

# End timing
elapsed_time = time.perf_counter() - start_time

print(f"Text: {text}")
print(f"Transcription completed in {elapsed_time:.2f} seconds")