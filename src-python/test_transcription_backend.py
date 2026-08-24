import io
import struct
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from models.transcription.transcription_backend import (
    BackendResult, FASTER_WHISPER_BACKEND, WHISPER_CPP_VULKAN_BACKEND,
    _MAGIC, _RESPONSE, WhisperCppVulkanBackend, acquireBackend,
    releaseBackend, _registry,
)


class FakeBackend:
    backend_name = "fake"
    device = "cpu"
    gpu_active = False
    load_duration_ms = 1.0
    def __init__(self): self.closed = 0; self.calls = []
    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs)); return BackendResult("ok", "en", .9, 2.0)
    def close(self): self.closed += 1


class TestBackendLifecycle(unittest.TestCase):
    def tearDown(self):
        for value in list(_registry.values()): releaseBackend(value)
        _registry.clear()

    def test_selection_and_shared_refcount_close(self):
        made = []
        def factory():
            item = FakeBackend(); made.append(item); return item
        one = acquireBackend(FASTER_WHISPER_BACKEND, ".", "m", factory=factory)
        two = acquireBackend(FASTER_WHISPER_BACKEND, ".", "m", factory=factory)
        one.transcribe(np.zeros(2, dtype=np.float32), language=None, avg_logprob=-.8, no_speech_prob=.6, no_repeat_ngram_size=0)
        self.assertIs(one, two); self.assertEqual(len(made), 1); self.assertEqual(one.status.state, "ready")
        releaseBackend(one); self.assertEqual(made[0].closed, 0)
        releaseBackend(two); self.assertEqual(made[0].closed, 1); self.assertNotIn(one.key, _registry)

    def test_async_load_error_is_reported(self):
        shared = acquireBackend("bad", ".", "m", factory=lambda: (_ for _ in ()).throw(RuntimeError("no model")))
        with self.assertRaisesRegex(RuntimeError, "no model"):
            shared.transcribe(np.zeros(1, dtype=np.float32), language=None, avg_logprob=0, no_speech_prob=1, no_repeat_ngram_size=0)
        self.assertEqual(shared.status.state, "error")

    @patch("models.transcription.transcription_backend.checkWhisperWeight", return_value=True)
    @patch("models.transcription.transcription_backend.FasterWhisperBackend")
    @patch("models.transcription.transcription_backend.WhisperCppVulkanBackend", side_effect=RuntimeError("no vulkan"))
    def test_vulkan_falls_back_to_faster(self, _, faster, __):
        faster.return_value = FakeBackend()
        shared = acquireBackend(WHISPER_CPP_VULKAN_BACKEND, ".", "m")
        shared.transcribe(np.zeros(1, dtype=np.float32), language=None, avg_logprob=0, no_speech_prob=1, no_repeat_ngram_size=0)
        self.assertEqual(shared.status.state, "ready")
        self.assertIs(shared.backend, faster.return_value)


class FakeProcess:
    def __init__(self, output):
        self.stdin = io.BytesIO(); self.stdout = io.BytesIO(output); self.stderr = io.BytesIO(); self._poll = None
    def poll(self): return self._poll
    def wait(self, timeout=None): self._poll = 0
    def kill(self): self._poll = -9


class TestVulkanFraming(unittest.TestCase):
    def test_request_response_framing(self):
        lang, text = b"en", b"hello"
        response = _RESPONSE.pack(_MAGIC, 0, .75, 3.5, len(lang), len(text), 0) + lang + text
        proc = FakeProcess(b"VRCT_READY\t1\tGPU\t4.0\n" + response)
        with patch("models.transcription.transcription_backend.getWhisperCppWorkerPath", return_value="worker"), \
             patch("models.transcription.transcription_backend.getWhisperCppModelPath", return_value="model"), \
             patch("models.transcription.transcription_backend.os_path.isfile", return_value=True):
            backend = WhisperCppVulkanBackend(".", "m", process_factory=lambda *a, **k: proc)
            result = backend.transcribe(np.array([1, -2], dtype=np.float32), language="en", avg_logprob=-.8, no_speech_prob=.6, no_repeat_ngram_size=0)
        self.assertEqual(result.text, "hello")
        proc.stdin.seek(0); header = proc.stdin.read(_RESPONSE.size if False else struct.calcsize("<IIIIIffi"))
        self.assertEqual(struct.unpack("<IIIIIffi", header)[0], _MAGIC)


if __name__ == "__main__": unittest.main()
