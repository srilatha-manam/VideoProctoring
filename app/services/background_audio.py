import webrtcvad
import wave
import tempfile
import contextlib
import numpy as np
from pydub import AudioSegment
from moviepy import VideoFileClip

def extract_audio_from_video(video_bytes: bytes, filename: str) -> str:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=f".{filename.split('.')[-1]}")
    temp_video.write(video_bytes)
    temp_video.flush()

    video = VideoFileClip(temp_video.name)
    audio_path = temp_video.name.replace("." + filename.split('.')[-1], ".wav")
    video.audio.write_audiofile(audio_path, fps=16000, nbytes=2, codec='pcm_s16le')
    return audio_path

def analyze_background_audio(wav_path: str):
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # 0: permissive, 3: aggressive

    with contextlib.closing(wave.open(wav_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
        frame_duration = 30  # in ms
        frame_size = int(sample_rate * frame_duration / 1000) * 2  # 16-bit frame size
        num_channels = wf.getnchannels()

        if num_channels != 1 or sample_rate != 16000:
            raise Exception("Audio must be mono and 16kHz")

        speech_times = []
        timestamp = 0.0
        step = frame_duration / 1000.0

        while True:
            frame = wf.readframes(frame_size // 2)
            if len(frame) < frame_size:
                break
            is_speech = vad.is_speech(frame, sample_rate)
            if is_speech:
                speech_times.append(timestamp)
            timestamp += step

    speech_duration = round(len(speech_times) * step, 2)
    total_duration = round(timestamp, 2)
    return {
        "speech_detected_seconds": speech_duration,
        "total_duration": total_duration,
        "speech_percentage": round((speech_duration / total_duration) * 100, 2) if total_duration else 0.0,
        "suspicious": speech_duration > 10.0,
        "speech_timestamps": speech_times[:10]
    }

def extract_audio(video_bytes: bytes, filename: str):
    wav_path = extract_audio_from_video(video_bytes, filename)
    return analyze_background_audio(wav_path)
