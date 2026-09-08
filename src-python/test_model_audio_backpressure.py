import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import model as model_module
from model import MicSession, SpeakerSession
from utils import putDroppingOldestOnFull


class TestAudioBackpressure(unittest.TestCase):
    def test_full_queue_drops_oldest_without_blocking(self):
        queue = Queue(maxsize=2)
        queue.put_nowait("old")
        queue.put_nowait("current")
        self.assertTrue(putDroppingOldestOnFull(queue, "new"))
        self.assertEqual([queue.get_nowait(), queue.get_nowait()], ["current", "new"])

    def test_unbounded_queue_does_not_drop(self):
        queue = Queue()
        self.assertFalse(putDroppingOldestOnFull(queue, "value"))
        self.assertEqual(queue.get_nowait(), "value")


class TestVadRecorderSelection(unittest.TestCase):
    @patch("model.SelectedMicVadRecorder")
    def test_mic_vad_is_selected_only_when_explicitly_enabled(self, recorder) -> None:
        recorder.return_value = MagicMock()
        fake_config = SimpleNamespace(
            MIC_RECORD_TIMEOUT=5,
            MIC_PHRASE_TIMEOUT=3,
            MIC_ENABLE_VAD=True,
            MIC_THRESHOLD=300,
            MIC_AUTOMATIC_THRESHOLD=False,
        )
        with patch.object(model_module, "config", fake_config):
            result = MicSession()._create_recorder({"name": "mic", "index": 1})

        self.assertIs(result, recorder.return_value)
        recorder.assert_called_once_with(
            device={"name": "mic", "index": 1}, record_timeout=3
        )

    @patch("model.SelectedMicEnergyAndAudioRecorder")
    def test_existing_mic_recorder_remains_the_default(self, recorder) -> None:
        recorder.return_value = MagicMock()
        fake_config = SimpleNamespace(
            MIC_RECORD_TIMEOUT=5,
            MIC_PHRASE_TIMEOUT=3,
            MIC_ENABLE_VAD=False,
            MIC_THRESHOLD=321,
            MIC_AUTOMATIC_THRESHOLD=True,
        )
        with patch.object(model_module, "config", fake_config):
            result = MicSession()._create_recorder({"name": "mic", "index": 1})

        self.assertIs(result, recorder.return_value)
        recorder.assert_called_once_with(
            device={"name": "mic", "index": 1},
            energy_threshold=321,
            dynamic_energy_threshold=True,
            phrase_time_limit=3,
            record_timeout=3,
        )

    @patch("model.SelectedSpeakerVadRecorder")
    def test_speaker_vad_is_selected_independently(self, recorder) -> None:
        recorder.return_value = MagicMock()
        fake_config = SimpleNamespace(
            SPEAKER_RECORD_TIMEOUT=5,
            SPEAKER_PHRASE_TIMEOUT=4,
            SPEAKER_ENABLE_VAD=True,
            SPEAKER_THRESHOLD=300,
            SPEAKER_AUTOMATIC_THRESHOLD=False,
        )
        with patch.object(model_module, "config", fake_config):
            result = SpeakerSession()._create_recorder({"name": "speaker", "index": 2})

        self.assertIs(result, recorder.return_value)
        recorder.assert_called_once_with(
            device={"name": "speaker", "index": 2}, record_timeout=4
        )


if __name__ == "__main__":
    unittest.main()
