"""Offline speech synthesis with Piper (piper-tts) for audio instructions.

Piper voices emit 22.05 kHz mono; ``PiperTTS.synthesize`` resamples to ``SAMPLE_RATE`` = 16 kHz,
which is what google/embeddinggemma-2 expects for audio input.  Waveforms are float32 in [-1, 1].
"""
from pathlib import Path
import wave

import numpy as np

SAMPLE_RATE = 16000


def resample(wav, source_rate, target_rate=SAMPLE_RATE):
    """Resample a mono float waveform; uses librosa if available, else scipy's polyphase filter."""
    wav = np.asarray(wav, dtype=np.float32)
    if source_rate == target_rate:
        return wav
    try:
        import librosa
        return librosa.resample(wav, orig_sr=source_rate, target_sr=target_rate).astype(np.float32)
    except ImportError:
        from math import gcd
        from scipy.signal import resample_poly
        factor = gcd(int(source_rate), int(target_rate))
        return resample_poly(wav, int(target_rate) // factor, int(source_rate) // factor).astype(np.float32)


class PiperTTS:
    """Lazy-loading Piper voice; ``synthesize(text) -> float32 mono waveform at 16 kHz``."""

    def __init__(self, voice_path, sample_rate=SAMPLE_RATE):
        self.voice_path = Path(voice_path)
        self.sample_rate = sample_rate
        self._voice = None

    def _load(self):
        if self._voice is None:
            if not self.voice_path.exists():
                raise FileNotFoundError(f"Piper voice not found: {self.voice_path}")
            from piper import PiperVoice
            self._voice = PiperVoice.load(str(self.voice_path))
        return self._voice

    def synthesize(self, text):
        voice = self._load()
        chunks = list(voice.synthesize(text))
        if not chunks:
            raise ValueError(f"Piper produced no audio for {text!r}")
        wav = np.concatenate([np.asarray(chunk.audio_float_array, dtype=np.float32) for chunk in chunks])
        return np.clip(resample(wav, chunks[0].sample_rate, self.sample_rate), -1.0, 1.0)


def save_wav(path, wav, sr=SAMPLE_RATE):
    """Write a float waveform in [-1, 1] as 16-bit PCM mono WAV."""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim != 1:
        raise ValueError(f"save_wav expects a mono waveform, got shape {wav.shape}")
    pcm = (np.clip(wav, -1.0, 1.0) * 32767).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sr))
        handle.writeframes(pcm.tobytes())


def load_wav(path):
    """Read a 16-bit PCM mono WAV written by save_wav: returns (float32 waveform, sample rate)."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit mono PCM")
        frames = handle.readframes(handle.getnframes())
        return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767, handle.getframerate()
