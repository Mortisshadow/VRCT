import unittest
from datetime import datetime, timedelta
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np

from speech_recognition.exceptions import RequestError

from models.transcription.transcription_transcriber import (
    AudioTranscriber,
    GOOGLE_RECOGNIZE_TIMEOUT_SECONDS,
)
from models.transcription.transcription_backend import (
    BackendResult,
    WHISPER_CPP_VULKAN_BACKEND,
    _registry,
)


class FakeAudioSource:
    SAMPLE_RATE = 16000
    SAMPLE_WIDTH = 2
    channels = 1


class TestWhisperAudioConversion(unittest.TestCase):
    @patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=True)
    def test_int16_audio_is_normalized_float32_for_backend(self, _):
        seen = {}
        class Backend:
            def transcribe(self, audio, **kwargs):
                seen["audio"] = audio
                return BackendResult("ok", "en", 1.0)
            def close(self): pass
        transcriber = AudioTranscriber(False, FakeAudioSource(), 3, 10, "Whisper", root=".", whisper_weight_type="m", backend_factory=lambda: Backend())
        q = Queue(); q.put((np.array([32767, -32768], dtype="<i2").tobytes(), datetime.now()))
        transcriber.transcribeAudioQueue(q, ["Japanese"], ["Japan"])
        self.assertEqual(seen["audio"].dtype, np.float32)
        np.testing.assert_allclose(seen["audio"], [32767 / 32768.0, -1.0], rtol=1e-6)
        shared = transcriber.whisper_backend
        from models.transcription.transcription_backend import releaseBackend
        releaseBackend(shared); _registry.clear()

    @patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=True)
    def test_multiple_languages_use_one_auto_detection_pass(self, _):
        calls = []
        class Backend:
            def transcribe(self, audio, **kwargs):
                calls.append(kwargs)
                return BackendResult("hello", "en", .9)
            def close(self): pass
        transcriber = AudioTranscriber(False, FakeAudioSource(), 3, 10, "Whisper",
                                       root=".", whisper_weight_type="multi-language",
                                       backend_factory=lambda: Backend())
        q = Queue(); q.put((np.array([1, 2], dtype="<i2").tobytes(), datetime.now()))
        transcriber.transcribeAudioQueue(q, ["Japanese", "English"], ["Japan", "United States"])
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0]["language"])
        transcriber.close(); _registry.clear()

    @patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=True)
    def test_faster_whisper_keeps_existing_cumulative_audio_and_detection(self, _):
        calls = []
        class Backend:
            def transcribe(self, audio, **kwargs):
                calls.append((len(audio), kwargs["language"]))
                return BackendResult("hello", "en", .9)
            def close(self): pass
        transcriber = AudioTranscriber(False, FakeAudioSource(), 3, 10, "Whisper",
                                       root=".", whisper_weight_type="m",
                                       backend_factory=lambda: Backend())
        now = datetime.now()
        q = Queue(); q.put((np.ones(6000, dtype="<i2").tobytes(), now))
        transcriber.transcribeAudioQueue(q, ["English", "German"], ["United States", "Germany"])
        q.put((np.ones(6000, dtype="<i2").tobytes(), now + timedelta(seconds=1)))
        transcriber.transcribeAudioQueue(q, ["English", "German"], ["United States", "Germany"])

        self.assertEqual(calls, [(6000, None), (12000, None)])
        shared = transcriber.whisper_backend
        from models.transcription.transcription_backend import releaseBackend
        releaseBackend(shared); _registry.clear()

    @patch("models.transcription.transcription_transcriber.checkWhisperCppWeight", return_value=True)
    def test_continuous_speech_decodes_only_new_audio_plus_small_overlap(self, _):
        seen_lengths = []
        class Backend:
            def transcribe(self, audio, **kwargs):
                seen_lengths.append(len(audio))
                return BackendResult("hello", "en", .9)
            def close(self): pass
        transcriber = AudioTranscriber(False, FakeAudioSource(), 3, 10, "Whisper",
                                       root=".", whisper_weight_type="m",
                                       whisper_backend=WHISPER_CPP_VULKAN_BACKEND,
                                       backend_factory=lambda: Backend())
        now = datetime.now()
        first = np.ones(6000, dtype="<i2").tobytes()
        second = np.ones(6000, dtype="<i2").tobytes()
        q = Queue(); q.put((first, now))
        transcriber.transcribeAudioQueue(q, ["English"], ["United States"])
        q.put((second, now + timedelta(seconds=1)))
        transcriber.transcribeAudioQueue(q, ["English"], ["United States"])

        self.assertEqual(seen_lengths, [6000, 10000])  # 6000 new + 4000 (250 ms) overlap
        shared = transcriber.whisper_backend
        from models.transcription.transcription_backend import releaseBackend
        releaseBackend(shared); _registry.clear()

    @patch("models.transcription.transcription_transcriber.checkWhisperCppWeight", return_value=True)
    def test_streaming_chunks_merge_text_and_reuse_detected_language(self, _):
        results = iter((
            BackendResult("hello world", "en", .9),
            BackendResult("world again", "en", .9),
        ))
        languages = []
        class Backend:
            def transcribe(self, audio, **kwargs):
                languages.append(kwargs["language"])
                return next(results)
            def close(self): pass
        transcriber = AudioTranscriber(False, FakeAudioSource(), 3, 10, "Whisper",
                                       root=".", whisper_weight_type="m",
                                       whisper_backend=WHISPER_CPP_VULKAN_BACKEND,
                                       backend_factory=lambda: Backend())
        now = datetime.now()
        q = Queue(); q.put((np.ones(6000, dtype="<i2").tobytes(), now))
        transcriber.transcribeAudioQueue(q, ["English", "Japanese"], ["United States", "Japan"])
        self.assertEqual(transcriber.getTranscript()["text"], "hello world")
        q.put((np.ones(6000, dtype="<i2").tobytes(), now + timedelta(seconds=1)))
        transcriber.transcribeAudioQueue(q, ["English", "Japanese"], ["United States", "Japan"])

        self.assertEqual(transcriber.getTranscript()["text"], "hello world again")
        self.assertEqual(languages, [None, "en"])
        shared = transcriber.whisper_backend
        from models.transcription.transcription_backend import releaseBackend
        releaseBackend(shared); _registry.clear()

    @patch("models.transcription.transcription_transcriber.checkWhisperCppWeight", return_value=True)
    def test_new_phrase_resets_overlap_text_and_detected_language(self, _):
        results = iter((
            BackendResult("old phrase", "en", .9),
            BackendResult("new phrase", "de", .8),
        ))
        calls = []
        class Backend:
            def transcribe(self, audio, **kwargs):
                calls.append((len(audio), kwargs["language"]))
                return next(results)
            def close(self): pass
        transcriber = AudioTranscriber(False, FakeAudioSource(), 3, 10, "Whisper",
                                       root=".", whisper_weight_type="m",
                                       whisper_backend=WHISPER_CPP_VULKAN_BACKEND,
                                       backend_factory=lambda: Backend())
        now = datetime.now()
        q = Queue(); q.put((np.ones(6000, dtype="<i2").tobytes(), now))
        transcriber.transcribeAudioQueue(q, ["English", "German"], ["United States", "Germany"])
        self.assertEqual(transcriber.getTranscript()["text"], "old phrase")
        q.put((np.ones(2000, dtype="<i2").tobytes(), now + timedelta(seconds=4)))
        transcriber.transcribeAudioQueue(q, ["English", "German"], ["United States", "Germany"])

        self.assertEqual(transcriber.getTranscript()["text"], "new phrase")
        self.assertEqual(calls, [(6000, None), (2000, None)])
        shared = transcriber.whisper_backend
        from models.transcription.transcription_backend import releaseBackend
        releaseBackend(shared); _registry.clear()


class TestAudioProcessingSelection(unittest.TestCase):
    @patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=False)
    def test_selects_mic_processing_for_microphone(self, _) -> None:
        transcriber = AudioTranscriber(False, FakeAudioSource(), 3, 10, "Google")

        self.assertEqual(transcriber.audio_sources["process_data_func"], transcriber.processMicData)

    @patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=False)
    def test_selects_speaker_processing_for_speaker(self, _) -> None:
        transcriber = AudioTranscriber(True, FakeAudioSource(), 3, 10, "Google")

        self.assertEqual(transcriber.audio_sources["process_data_func"], transcriber.processSpeakerData)

    @patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=False)
    def test_reads_normalized_speaker_pcm_as_mono(self, _) -> None:
        transcriber = AudioTranscriber(True, FakeAudioSource(), 3, 10, "Google")
        pcm = np.array([1000, -1000], dtype="<i2").tobytes()
        transcriber.audio_sources["last_sample"] = pcm

        result = transcriber.processSpeakerData()

        self.assertEqual(result.get_raw_data(), pcm)


class TestGoogleRecognizerTimeout(unittest.TestCase):
    """Issue #63: Google recognition must not block indefinitely on a bad network."""

    @patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=False)
    def test_recognizer_has_finite_operation_timeout(self, _) -> None:
        transcriber = AudioTranscriber(False, FakeAudioSource(), 3, 10, "Google")

        self.assertEqual(transcriber.audio_recognizer.operation_timeout, GOOGLE_RECOGNIZE_TIMEOUT_SECONDS)

    @patch("models.transcription.transcription_transcriber.errorLogging")
    @patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=False)
    def test_request_error_is_logged_instead_of_swallowed(self, _, mock_error_logging) -> None:
        transcriber = AudioTranscriber(False, FakeAudioSource(), 3, 10, "Google")
        transcriber.audio_recognizer.recognize_google = MagicMock(
            side_effect=RequestError("recognition connection failed: timed out")
        )
        audio_queue = Queue()
        audio_queue.put((b"\x01\x00", datetime.now()))

        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])

        mock_error_logging.assert_called_once()

    @patch("models.transcription.transcription_transcriber.errorLogging")
    @patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=False)
    def test_flags_recognition_error_for_ui_visibility(self, _, __) -> None:
        transcriber = AudioTranscriber(False, FakeAudioSource(), 3, 10, "Google")
        transcriber.audio_recognizer.recognize_google = MagicMock(
            side_effect=RequestError("recognition connection failed: timed out")
        )
        audio_queue = Queue()
        audio_queue.put((b"\x01\x00", datetime.now()))

        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])

        self.assertTrue(transcriber.last_recognition_error)

    @patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=False)
    def test_clears_recognition_error_flag_after_a_successful_call(self, _) -> None:
        transcriber = AudioTranscriber(False, FakeAudioSource(), 3, 10, "Google")
        transcriber.last_recognition_error = True
        transcriber.audio_recognizer.recognize_google = MagicMock(return_value=("hello", 0.9))
        audio_queue = Queue()
        audio_queue.put((b"\x01\x00", datetime.now()))

        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])

        self.assertFalse(transcriber.last_recognition_error)


class TestQueueProcessing(unittest.TestCase):
    @patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=False)
    def test_drains_queued_audio_before_transcribing(self, _) -> None:
        transcriber = AudioTranscriber(False, FakeAudioSource(), 3, 10, "Whisper")
        transcriber.transcription_engine = "Whisper"
        transcriber.whisper_model = MagicMock()
        transcriber.whisper_model.transcribe.return_value = ([], MagicMock(language_probability=1.0))
        audio_queue = Queue()
        now = datetime.now()
        audio_queue.put((b"\x01\x00", now))
        audio_queue.put((b"\x02\x00", now + timedelta(milliseconds=100)))

        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])

        # Queue が空になっていて、直近サンプルがまとめて last_sample に反映されている
        self.assertTrue(audio_queue.empty())
        self.assertEqual(transcriber.audio_sources["last_sample"], b"\x01\x00\x02\x00")

    @patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=False)
    def test_phrase_timeout_resets_last_sample_across_a_gap(self, _) -> None:
        transcriber = AudioTranscriber(False, FakeAudioSource(), 3, 10, "Whisper")
        transcriber.transcription_engine = "Whisper"
        transcriber.whisper_model = MagicMock()
        transcriber.whisper_model.transcribe.return_value = ([], MagicMock(language_probability=1.0))
        audio_queue = Queue()
        now = datetime.now()
        audio_queue.put((b"\x01\x00", now))
        # phrase_timeout=3s より大きなギャップ = 新フレーズ扱いで last_sample がリセット
        audio_queue.put((b"\x02\x00", now + timedelta(seconds=5)))

        transcriber.transcribeAudioQueue(audio_queue, ["Japanese"], ["Japan"])

        self.assertEqual(transcriber.audio_sources["last_sample"], b"\x02\x00")
        self.assertTrue(transcriber.audio_sources["new_phrase"])

    @patch("models.transcription.transcription_transcriber.checkWhisperWeight", return_value=False)
    def test_passes_configured_thresholds_to_whisper(self, _) -> None:
        transcriber = AudioTranscriber(False, FakeAudioSource(), 3, 10, "Whisper")
        transcriber.transcription_engine = "Whisper"
        transcriber.whisper_model = MagicMock()
        transcriber.whisper_model.transcribe.return_value = ([], MagicMock(language_probability=1.0))
        audio_queue = Queue()
        audio_queue.put((b"\x01\x00", datetime.now()))

        transcriber.transcribeAudioQueue(
            audio_queue,
            ["Japanese"],
            ["Japan"],
            avg_logprob=-0.55,
            no_speech_prob=0.42,
        )

        kwargs = transcriber.whisper_model.transcribe.call_args.kwargs
        self.assertEqual(kwargs["log_prob_threshold"], -0.55)
        self.assertEqual(kwargs["no_speech_threshold"], 0.42)


class TestMutedMicMessage(unittest.TestCase):
    @patch("controller.model")
    @patch("controller.config")
    def test_discards_queued_result_while_vrc_mic_is_muted(self, config, model) -> None:
        from controller import Controller

        config.VRC_MIC_MUTE_SYNC = True
        model.mic_mute_status = True

        controller = Controller.__new__(Controller)
        controller.micMessage({"text": "anything", "language": "Japanese"})

        self.assertEqual(model.method_calls, [])


class TestRepeatDetection(unittest.TestCase):
    """VAD ストリーミング撤退 (ADR-0004) で segment_id が消えたため、
    連続同一メッセージは純粋にテキスト比較で抑制する。"""

    def test_receive_repeat_blocks_second_identical_text(self) -> None:
        from model import Model

        model = Model.__new__(Model)
        model.previous_receive_message = ""

        self.assertFalse(model.detectRepeatReceiveMessage("same text"))
        self.assertTrue(model.detectRepeatReceiveMessage("same text"))

    def test_receive_repeat_allows_different_text(self) -> None:
        from model import Model

        model = Model.__new__(Model)
        model.previous_receive_message = ""

        self.assertFalse(model.detectRepeatReceiveMessage("first"))
        self.assertFalse(model.detectRepeatReceiveMessage("second"))

    def test_send_repeat_blocks_second_identical_text(self) -> None:
        from model import Model

        model = Model.__new__(Model)
        model.previous_send_message = ""

        self.assertFalse(model.detectRepeatSendMessage("same text"))
        self.assertTrue(model.detectRepeatSendMessage("same text"))

    def test_send_repeat_allows_different_text(self) -> None:
        from model import Model

        model = Model.__new__(Model)
        model.previous_send_message = ""

        self.assertFalse(model.detectRepeatSendMessage("first"))
        self.assertFalse(model.detectRepeatSendMessage("second"))


if __name__ == "__main__":
    unittest.main()
