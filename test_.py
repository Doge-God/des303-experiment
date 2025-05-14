import queue
import threading
import torch
import numpy as np
import pyaudio
from collections import deque
import time

# Load the Silero VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# Audio stream config
RATE = 16000  # required sample rate for Silero VAD
CHUNK_SIZE = 512 # required for Silero
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Queue to hold audio chunks
audio_queue = queue.Queue()

# Buffer to accumulate chunks
chunk_buffer = []


# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)


# Thread function
def audio_reader_thread():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_queue.put(data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# Start the audio reading thread
thread = threading.Thread(target=audio_reader_thread, daemon=True)
thread.start()

print("Listening... (press Ctrl+C to stop)")

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound

def visualize_bar(value):
    # Ensure value is within 0 to 1
    value = max(0, min(1, value))
    total_length = 20
    filled_length = int(round(value * total_length))
    bar = '█' * filled_length + '-' * (total_length - filled_length)
    return f"[{bar}] {value:.2f}"

try:
    while True:
        if not audio_queue.empty():
            # Read chunk from microphone
            audio_chunk = audio_queue.get()
            # Convert bytes to numpy array
            audio_float32 = int2float(np.frombuffer(audio_chunk, dtype=np.int16))
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_float32)
            # Compute speech probability
            with torch.no_grad():
                prob = model(audio_tensor, RATE).item()
            print(visualize_bar(prob))

except KeyboardInterrupt:
    print("Stopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()