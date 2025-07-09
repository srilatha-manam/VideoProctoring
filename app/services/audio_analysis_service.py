import io
import tempfile
import webrtcvad
from pydub import AudioSegment
from resemblyzer import VoiceEncoder
import numpy as np
from sklearn.cluster import KMeans

vad = webrtcvad.Vad(3)
encoder = VoiceEncoder()

def analyze_audio_chunk(audio_buffer: bytes) -> dict:
    # Convert to mono, 16kHz WAV
    segment = AudioSegment.from_file(io.BytesIO(audio_buffer), format="webm")
    segment = segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    # --- Silence Detection ---
    pcm = segment.raw_data
    sample_rate = 16000
    frame_duration = 30  # ms
    frame_len = int(sample_rate * frame_duration / 1000) * 2
    frames = [pcm[i:i + frame_len] for i in range(0, len(pcm), frame_len)]

    silence = 0.0
    silence_count = 0
    for frame in frames:
        if len(frame) < frame_len:
            continue
        if not vad.is_speech(frame, sample_rate):
            silence_count += 1
            if silence_count * 0.03 >= 10:
                silence = silence_count * 0.03
                break
        else:
            silence_count = 0

    # --- Noise Level ---
    loudness = segment.dBFS
    is_noisy = loudness > -35  # adjustable threshold

    # --- Speaker Embeddings ---
    samples = np.array(segment.get_array_of_samples()).astype(np.float32) / 32768.0
    embeddings, _ = encoder.embed_utterance(samples, return_partials=True)

    multiple_speakers = False
    background_talk = 0.0

    if len(embeddings) > 1:
        try:
            kmeans = KMeans(n_clusters=2, n_init='auto').fit(embeddings)
            counts = np.bincount(kmeans.labels_)
            imbalance = abs(counts[0] - counts[1])
            # If speakers are relatively balanced, assume multi-speaker
            multiple_speakers = imbalance < len(embeddings) * 0.6
            if multiple_speakers:
                background_talk = round(len(embeddings), 2)  # approx seconds
        except Exception as e:
            print("Clustering failed:", e)

    return {
        "multiple_speakers": multiple_speakers,
        "background_talk_duration": background_talk,
        "noisy_environment": is_noisy,
        "silence_duration": round(silence, 2),
        "noise_level_db": round(loudness, 2)
    }
