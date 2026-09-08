import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from models.transcription.audio_vad import (
    FRAME_SAMPLES,
    Pcm16MonoNormalizer,
    SileroFrameProbability,
    VadSegmenter,
)


class ProbabilitySequence:
    def __init__(self, values):
        self.values = iter(values)
        self.reset_count = 0

    def __call__(self, _frame):
        return next(self.values)

    def reset(self):
        self.reset_count += 1


def frame(value=1000):
    return np.full(FRAME_SAMPLES, value, dtype="<i2").tobytes()


class TestPcm16MonoNormalizer(unittest.TestCase):
    def test_downmixes_stereo_without_clipping(self):
        raw = np.array([[1000, 3000], [-3000, -1000]], dtype="<i2").tobytes()
        result = Pcm16MonoNormalizer(16000, 2, 2).process(raw)
        self.assertEqual(np.frombuffer(result, dtype="<i2").tolist(), [2000, -2000])

    def test_resampling_state_is_preserved_between_chunks(self):
        normalizer = Pcm16MonoNormalizer(48000, 2, 1)
        raw = np.arange(4800, dtype=np.int16).tobytes()
        result = normalizer.process(raw[:4800]) + normalizer.process(raw[4800:])
        self.assertAlmostEqual(len(result) // 2, 1600, delta=1)


class TestVadSegmenter(unittest.TestCase):
    def test_quiet_audio_is_not_emitted(self):
        vad = VadSegmenter(ProbabilitySequence([.05] * 8), hangover_frames=2)
        self.assertEqual(vad.process(frame() * 8), [])

    def test_preroll_and_hangover_protect_speech_edges(self):
        vad = VadSegmenter(
            ProbabilitySequence([.05, .9, .9, .05, .05, .05]),
            hangover_frames=2, pre_speech_pad_frames=3,
        )
        result = vad.process(frame() * 6)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].reason, "silence")
        self.assertEqual(len(result[0].audio), len(frame()) * 6)

    def test_max_duration_bounds_continuous_speech_without_resetting_model(self):
        engine = ProbabilitySequence([.9] * 12)
        vad = VadSegmenter(
            engine, hangover_frames=99, max_speech_frames=5,
            min_speech_frames=2, pre_speech_pad_frames=0,
        )
        result = vad.process(frame() * 12)
        self.assertGreaterEqual(len(result), 2)
        self.assertTrue(all(segment.reason == "max_duration" for segment in result))
        self.assertEqual(engine.reset_count, 0)


class TestSileroFailureFallback(unittest.TestCase):
    def test_onnx_failure_switches_to_rms_instead_of_killing_capture(self):
        engine = SileroFrameProbability()
        engine._model = MagicMock()
        engine._model.encoder_session.run.side_effect = RuntimeError("broken ONNX")

        with patch("models.transcription.audio_vad.logger.exception") as log_exception:
            probability = engine(np.full(FRAME_SAMPLES, 0.01, dtype=np.float32))

        self.assertTrue(engine._fallback_to_rms)
        self.assertAlmostEqual(probability, 0.5, places=3)
        log_exception.assert_called_once()


if __name__ == "__main__":
    unittest.main()
