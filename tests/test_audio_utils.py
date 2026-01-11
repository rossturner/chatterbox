import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from server.audio_utils import time_stretch_audio


class TestTimeStretchAudio:
    def test_time_stretch_slower(self):
        audio = np.zeros(24000)
        stretched = time_stretch_audio(audio, speed_rate=0.9)
        assert len(stretched) > len(audio)
        expected_ratio = 1 / 0.9
        actual_ratio = len(stretched) / len(audio)
        assert abs(actual_ratio - expected_ratio) < 0.05

    def test_time_stretch_faster(self):
        audio = np.zeros(24000)
        stretched = time_stretch_audio(audio, speed_rate=1.1)
        assert len(stretched) < len(audio)
        expected_ratio = 1 / 1.1
        actual_ratio = len(stretched) / len(audio)
        assert abs(actual_ratio - expected_ratio) < 0.05

    def test_time_stretch_no_change(self):
        audio = np.zeros(24000)
        result = time_stretch_audio(audio, speed_rate=1.0)
        assert len(result) == len(audio)

    def test_time_stretch_near_unity_bypasses_processing(self):
        audio = np.random.randn(24000).astype(np.float32)
        result = time_stretch_audio(audio, speed_rate=1.0005)
        assert result is audio

    def test_time_stretch_preserves_dtype(self):
        audio = np.random.randn(24000).astype(np.float32)
        stretched = time_stretch_audio(audio, speed_rate=0.9)
        assert stretched.dtype == np.float32

    def test_time_stretch_with_real_audio(self):
        sample_rate = 24000
        duration = 1.0
        freq = 440
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

        stretched = time_stretch_audio(audio, speed_rate=0.8)
        expected_length = len(audio) / 0.8
        assert abs(len(stretched) - expected_length) < 1000
