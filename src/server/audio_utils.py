import numpy as np
import librosa


def time_stretch_audio(
    audio: np.ndarray,
    speed_rate: float,
    sample_rate: int = 24000
) -> np.ndarray:
    """
    Adjust audio tempo while preserving pitch.

    Args:
        audio: Audio samples as numpy array
        speed_rate: Tempo multiplier (0.9 = 10% slower, 1.1 = 10% faster)
        sample_rate: Audio sample rate (default 24kHz for Chatterbox)

    Returns:
        Time-stretched audio array
    """
    if abs(speed_rate - 1.0) < 0.001:
        return audio

    return librosa.effects.time_stretch(audio, rate=speed_rate)
