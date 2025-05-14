from abc import ABC, abstractmethod
import json
import logging
import queue
import sys
import time
import threading
from faster_whisper import WhisperModel
import numpy as np
import pyaudio
import torch
from vosk import Model, KaldiRecognizer
from collections import deque

class SileroVAD:
    def __init__(self, sample_rate=16000):
        # Load the Silero VAD model
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        self.sample_rate = sample_rate
    
    def __int2float(self,sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1/32768
        sound = sound.squeeze()  # depends on the use case
        return sound

    def __visualize_bar(self,value):
        # Ensure value is within 0 to 1
        value = max(0, min(1, value))
        total_length = 20
        filled_length = int(round(value * total_length))
        bar = '█' * filled_length + '-' * (total_length - filled_length)
        return f"[{bar}] {value:.2f}"
    
    def get_speech_confidence(self, audio_chunk:bytes, is_visualizing=False) -> float:
        '''Audio chunk has to have 512 frames at 16000hz sample rate, or 256 at 8000hz'''
         # Convert bytes to numpy array
        audio_float32 = self.__int2float(np.frombuffer(audio_chunk, dtype=np.int16))
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_float32)
        # Compute speech probability
        with torch.no_grad():
            prob = self.model(audio_tensor, self.sample_rate).item()
        
        # visualize
        if is_visualizing:
            print(self.__visualize_bar(prob))
        return prob
    
class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, full_audio:bytes) -> str:
        pass

class VoskTranscriber(Transcriber):
    def __init__(self, sample_rate=16000):
        # Load Vosk model
        self.stt_model = Model(model_path="voice_models/vosk-model-small-en-us-0.15")
        self.stt_recognizer = KaldiRecognizer(self.stt_model, sample_rate)
    
    def transcribe(self, full_audio) -> str:
        self.stt_recognizer.Reset()
        if self.stt_recognizer.AcceptWaveform(full_audio):
            result = self.stt_recognizer.Result()
            return json.loads(result)["text"]
        else:
            partial_result = self.stt_recognizer.PartialResult()
            return json.loads(partial_result)["partial"]

class FasterWhisperTranscriber(Transcriber):
    def __init__(self):
        self.model = WhisperModel(model_size_or_path="tiny.en", device="cpu",download_root="huggingface_cache")
        print("Faster Whisper model loaded")

    def transcribe(self, full_audio):
        audio_data_array: np.ndarray = np.frombuffer(full_audio, np.int16).astype(np.float32) / 255.0
        segments, info = self.model.transcribe(audio_data_array)
        full_text = " ".join([segment.text for segment in segments])
        return full_text
    
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
class WhisperStreamTranscriber:
    def __init__(self):
        from whisper_online import FasterWhisperASR, OnlineASRProcessor
        asr = FasterWhisperASR("en", "tiny.en", cache_dir="huggingface_cache")
        self.streaming_processor = OnlineASRProcessor(asr)
        self.cnt = 0
    
    def stream_transcribe(self, audio_chunk:bytes):
        audio_data_array: np.ndarray = np.frombuffer(audio_chunk, np.int16).astype(np.float32) / 255.0
        self.streaming_processor.insert_audio_chunk(audio_data_array)
        self.cnt += 1
        if self.cnt >= 30:
            self.cnt = 0
            partial_out = self.streaming_processor.process_iter()
            print(partial_out)
    
    def reset(self):
        self.streaming_processor.finish()
        self.cnt = 0
    

class STT:
    def __init__(self, vad_threshold=0.95, vad_silence=1.0, sample_rate=16000, chuck_size=512, format=pyaudio.paInt16, channels=1):
        # audio stream setting
        self.sample_rate = sample_rate
        self.chunk_size = chuck_size  
        self.format = format
        self.channels = channels

        # data structures
        self.audio_data = queue.Queue()
        self.utterance_chunks = []

        # vad settings and init
        self.vad_threshold = vad_threshold
        self.vad_silence = vad_silence
        self.vad_validator = SileroVAD()

        self.transcriber = FasterWhisperTranscriber()
        self.stream_transcriber = WhisperStreamTranscriber()

        # State flags
        self.is_recording_utterance = False
        self.is_stream_open = False
        
        # PyAudio stream
        self.stream_reading_thread = None
        
    # Thread function
    def audio_reader_thread_func(self):
        p = pyaudio.PyAudio()

        stream = p.open(format=self.format,
                        channels=self.channels,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk_size)

        try:
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_data.put(data)
                if not self.is_stream_open:
                    break
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    def transcribe(self):
        # clear any remaining audio data
        self.audio_data = queue.Queue()
        # open stream and gather data
        self.is_stream_open = True
        self.audio_reader_thread = threading.Thread(target=self.audio_reader_thread_func, daemon=True)
        self.audio_reader_thread.start()

        # silence buffer
        silence_buffer = deque(maxlen=int(self.vad_silence / (self.chunk_size / self.sample_rate)))
        print("Listening...")

        # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        while True: 
            if not self.audio_data.empty():
                chunk = self.audio_data.get()

                prob = self.vad_validator.get_speech_confidence(chunk)
                is_speech = prob > self.vad_threshold

              
                if prob > 0.75:
                    self.stream_transcriber.stream_transcribe(chunk)

                # # if is recording utterace
                # if self.is_recording_utterance:
                    
                #     # self.utterance_chunks.append(chunk)
                #     # silence_buffer.append(not is_speech)
                
                #     # # all corresponding buffer silent, transcribe
                #     # if all(silence_buffer):
                #     #     self.is_recording_utterance = False
                #     #     full_audio = b''.join(self.utterance_chunks)
                        
                #     #     yield self.transcriber.transcribe(full_audio)
                # # in stand by
                # else:
                #     if is_speech:
                #         self.is_recording_utterance = True
                #         self.utterance_chunks = [chunk]
                #         silence_buffer.clear()
            
            if not self.is_stream_open:
                print("Finished Transcribe.")
                break
  

    def stop(self):
        self.is_stream_open = False

if __name__ == "__main__":
    
    stt = STT(vad_threshold=0.6, vad_silence=1.0)
    print("Spawned STT")
    try:
        for transcript in stt.transcribe():
            print("Transcript:", transcript)
    except KeyboardInterrupt:
        print("Stopping...")
        stt.stop()
