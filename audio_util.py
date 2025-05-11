import logging
import os
import queue
import signal
import wave
import numpy as np
import pyaudio

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            #pylint: disable=unused-argument
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, 'rb')

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()


import pyaudio
import wave
from vosk import Model, KaldiRecognizer
import time
import numpy as np
import collections

# Constants
DEVICE_INDEX = 1  # Update this to match your headset's device index
RATE = 16000  # Sample rate
CHUNK = 1024  # Frame size
FORMAT = pyaudio.paInt16
THRESHOLD = 500  # Adjust this to match your environment's noise level
MODEL_PATH = "voice_models/vosk-model-small-en-us-0.15"  # Path to your Vosk model
OUTPUT_FILE = "translate.wav"

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(
    format=FORMAT,
    channels=1,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK
)

print("Listening for voice...")

def detect_voice(audio_chunk):
    """Return True if audio chunk likely contains a human voice."""
    # Decode byte data to int16
    try:
        audio_chunk = np.frombuffer(audio_chunk, dtype=np.int16)
    except ValueError as e:
        print(f"Error decoding audio chunk: {e}")
        return False

    # Inspect the range of audio data
    print(f"Min: {audio_chunk.min()}, Max: {audio_chunk.max()}")

    # Compute the volume
    volume = np.abs(audio_chunk).max()  # Use absolute to handle both +ve and -ve peaks
    print(f"Volume: {volume}")

    # Define threshold (adjust based on testing)
    THRESHOLD = 1000  # Adjust based on normalized scale
    return volume > THRESHOLD


# Parameters for silence detection
SILENCE_TIMEOUT = 1  # seconds of silence to stop recording
MAX_SILENCE_CHUNKS = int(SILENCE_TIMEOUT * RATE / CHUNK)  # Convert to chunk count

# Record audio with silence detection
frames = []
silent_chunks = 0

# Wait for voice
# Add a buffer to store audio chunks before voice is detected
pre_record_buffer = collections.deque(maxlen=int(RATE / CHUNK * 2))  # Buffer up to 2 seconds of audio

print("Listening for voice...")
while True:
    data = stream.read(CHUNK, exception_on_overflow=False)
    pre_record_buffer.append(data)  # Continuously store audio chunks
    if detect_voice(data):
        print("Voice detected! Starting recording...")
        frames.extend(pre_record_buffer)  # Include pre-recorded audio
        frames.append(data)  # Include the current chunk with voice
        break

print("Recording... Speak now.")
while True:
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)

    # Detect silence
    if not detect_voice(data):
        silent_chunks += 1
        if silent_chunks >= MAX_SILENCE_CHUNKS:
            print("Silence detected. Stopping recording...")
            break
    else:
        silent_chunks = 0  # Reset silence counter if voice is detected

# Stop and close stream
stream.stop_stream()
stream.close()
audio.terminate()

# The 'frames' list now contains the recorded audio data.

# Save audio to file
with wave.open(OUTPUT_FILE, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))

print(f"Audio recorded to {OUTPUT_FILE}")

# Start speech-to-text
print("Starting speech-to-text...")
start_time = time.perf_counter()

wf = wave.open(OUTPUT_FILE, "rb")
model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, wf.getframerate())
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        print("Transcription:", rec.Result())

# End timer
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")
