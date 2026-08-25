"""Local Whisper backend abstraction and persistent model lifecycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, asdict
import logging
from os import path as os_path
import struct
import subprocess
import sys
from threading import Condition, Event, Lock, Thread, Timer, current_thread
from time import monotonic
from typing import Callable, Dict, Optional, Tuple

from .transcription_whisper import getWhisperModel, getWhisperCppModelPath, checkWhisperWeight


FASTER_WHISPER_BACKEND = "Faster-Whisper (CPU/CUDA)"
WHISPER_CPP_VULKAN_BACKEND = "Whisper.cpp (Vulkan)"
WHISPER_BACKENDS = (FASTER_WHISPER_BACKEND, WHISPER_CPP_VULKAN_BACKEND)
_MAGIC = 0x54524356
_REQUEST = struct.Struct("<IIIIIffi")
_RESPONSE = struct.Struct("<IifdIII")
_READY_TIMEOUT_SECONDS = 60
_TRANSCRIBE_TIMEOUT_SECONDS = 120
_BACKEND_IDLE_GRACE_SECONDS = 5.0

logger = logging.getLogger("vrct.transcription")


@dataclass(frozen=True)
class BackendResult:
    text: str
    language: Optional[str]
    confidence: float
    duration_ms: float = 0.0


@dataclass
class BackendStatus:
    backend: str
    state: str = "idle"
    model: Optional[str] = None
    gpu_active: bool = False
    device: Optional[str] = None
    load_duration_ms: Optional[float] = None
    transcription_duration_ms: Optional[float] = None
    error: Optional[str] = None


class TranscriptionBackend(ABC):
    @abstractmethod
    def transcribe(self, audio, *, language: Optional[str], avg_logprob: float,
                   no_speech_prob: float, no_repeat_ngram_size: int) -> BackendResult:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FasterWhisperBackend(TranscriptionBackend):
    def __init__(self, root: str, model: str, device: str, device_index: int, compute_type: str) -> None:
        started = monotonic()
        self.model = getWhisperModel(root, model, device, device_index, compute_type)
        self.load_duration_ms = (monotonic() - started) * 1000
        self.device = device
        self.backend_name = FASTER_WHISPER_BACKEND
        logger.info("Whisper backend=%s device=%s model=%s load_ms=%.1f gpu_active=%s",
                    FASTER_WHISPER_BACKEND, device, model, self.load_duration_ms, device == "cuda")

    def transcribe(self, audio, *, language, avg_logprob, no_speech_prob,
                   no_repeat_ngram_size) -> BackendResult:
        started = monotonic()
        segments, info = self.model.transcribe(
            audio, beam_size=5, temperature=0.0, log_prob_threshold=avg_logprob,
            no_speech_threshold=no_speech_prob, language=language,
            word_timestamps=False, without_timestamps=True, task="transcribe",
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        text = "".join(
            segment.text for segment in segments
            if segment.avg_logprob >= avg_logprob and segment.no_speech_prob <= no_speech_prob
        )
        duration_ms = (monotonic() - started) * 1000
        logger.info("Whisper transcription backend=%s duration_ms=%.1f", FASTER_WHISPER_BACKEND, duration_ms)
        return BackendResult(text, getattr(info, "language", None),
                             float(getattr(info, "language_probability", 0.0)), duration_ms)


def _read_exact(stream, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise RuntimeError("whisper.cpp worker closed its output")
        data.extend(chunk)
    return bytes(data)


def getWhisperCppWorkerPath(root: str) -> str:
    name = "vrct-whisper-worker.exe" if sys.platform == "win32" else "vrct-whisper-worker"
    candidates = (
        os_path.join(root, "_internal", "whisper_cpp", name),
        os_path.join(root, "whisper_cpp", name),
        os_path.join(root, "native", "whisper_cpp_worker", "build", name),
    )
    return next((candidate for candidate in candidates if os_path.isfile(candidate)), candidates[0])


class WhisperCppVulkanBackend(TranscriptionBackend):
    def __init__(self, root: str, model: str, device: str = "", device_index: int = 0,
                 compute_type: str = "auto", process_factory: Callable = subprocess.Popen) -> None:
        del device, compute_type
        worker = getWhisperCppWorkerPath(root)
        model_path = getWhisperCppModelPath(root, model)
        if not os_path.isfile(worker):
            raise RuntimeError(f"whisper.cpp Vulkan worker is unavailable: {worker}")
        if not os_path.isfile(model_path):
            raise RuntimeError(f"whisper.cpp model is unavailable: {model_path}")
        started = monotonic()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = process_factory(
            [worker, "--model", model_path, "--threads", "4", "--device", str(device_index)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, creationflags=flags,
        )
        self._io_lock = Lock()
        self._closed = False
        self._stderr_tail = deque(maxlen=40)
        self._diagnostic_path = os_path.join(root, "whisper_cpp.log")
        # Drain diagnostics during model initialization as ggml can emit enough
        # device/model detail to fill a Windows pipe before the ready marker.
        self._stderr_thread = Thread(target=self._drain_stderr, daemon=True, name="WhisperCppDiagnostics")
        self._stderr_thread.start()
        ready_line = {}
        ready_event = Event()
        def read_ready() -> None:
            try:
                ready_line["data"] = self._process.stdout.readline()
            except Exception as exc:
                ready_line["error"] = exc
            finally:
                ready_event.set()
        Thread(target=read_ready, daemon=True, name="WhisperCppStartup").start()
        if not ready_event.wait(_READY_TIMEOUT_SECONDS):
            self.close(graceful=False)
            raise TimeoutError("whisper.cpp worker model loading timed out")
        if "error" in ready_line:
            self.close(graceful=False)
            raise RuntimeError(f"whisper.cpp worker startup failed: {ready_line['error']}")
        ready = ready_line.get("data", b"").decode("utf-8", errors="replace").rstrip("\r\n").split("\t", 3)
        if len(ready) != 4 or ready[0] != "VRCT_READY":
            self.close(graceful=False)
            raise RuntimeError("invalid whisper.cpp worker startup response")
        self.gpu_active = ready[1] == "1"
        self.backend_name = WHISPER_CPP_VULKAN_BACKEND
        self.device = ready[2].replace("\\t", "\t").replace("\\n", "\n")
        self.load_duration_ms = float(ready[3])
        logger.info("Whisper backend=%s device=%s model=%s load_ms=%.1f gpu_active=%s",
                    WHISPER_CPP_VULKAN_BACKEND, self.device, model,
                    self.load_duration_ms or (monotonic() - started) * 1000, self.gpu_active)
        if not self.gpu_active:
            self.close(graceful=False)
            raise RuntimeError("whisper.cpp started without an active Vulkan GPU backend")

    def _drain_stderr(self) -> None:
        try:
            with open(self._diagnostic_path, "a", encoding="utf-8") as diagnostic:
                for raw_line in iter(self._process.stderr.readline, b""):
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if line:
                        self._stderr_tail.append(line)
                        diagnostic.write(line + "\n")
                        diagnostic.flush()
                        logger.info("whisper.cpp: %s", line)
        except Exception:
            logger.exception("Could not write whisper.cpp diagnostics")

    def _worker_exit_error(self) -> RuntimeError:
        return_code = self._process.poll()
        details = self._stderr_tail[-1] if self._stderr_tail else "no native diagnostic output"
        return RuntimeError(
            f"whisper.cpp worker exited unexpectedly (exit_code={return_code}): {details}"
        )

    def transcribe(self, audio, *, language, avg_logprob, no_speech_prob,
                   no_repeat_ngram_size) -> BackendResult:
        language_bytes = (language or "").encode("utf-8")
        # The wire protocol is a flat float32 sample stream. Normalizing here
        # prevents a sliced/non-contiguous array or accidental extra dimension
        # from desynchronizing the persistent worker protocol.
        audio = audio.reshape(-1).astype("<f4", copy=False)
        samples = audio.tobytes(order="C")
        header = _REQUEST.pack(_MAGIC, 1, 1, len(language_bytes), audio.size,
                               avg_logprob, no_speech_prob, no_repeat_ngram_size)
        exchange_result = {}
        def exchange() -> None:
            try:
                with self._io_lock:
                    if self._process.poll() is not None:
                        raise self._worker_exit_error()
                    self._process.stdin.write(header)
                    self._process.stdin.write(language_bytes)
                    self._process.stdin.write(samples)
                    self._process.stdin.flush()
                    values = _RESPONSE.unpack(_read_exact(self._process.stdout, _RESPONSE.size))
                    magic, status, confidence, duration_ms, lang_len, text_len, error_len = values
                    if magic != _MAGIC:
                        raise RuntimeError("invalid whisper.cpp worker response")
                    detected = _read_exact(self._process.stdout, lang_len).decode("utf-8", errors="replace")
                    text = _read_exact(self._process.stdout, text_len).decode("utf-8", errors="replace")
                    error = _read_exact(self._process.stdout, error_len).decode("utf-8", errors="replace")
                    exchange_result["values"] = (status, confidence, duration_ms, detected, text, error)
            except (BrokenPipeError, OSError) as exc:
                if self._process.poll() is None:
                    try:
                        self._process.wait(timeout=1)
                    except Exception:
                        pass
                exchange_result["error"] = self._worker_exit_error() if self._process.poll() is not None else exc
            except Exception as exc:
                exchange_result["error"] = exc
        thread = Thread(target=exchange, daemon=True, name="WhisperCppExchange")
        thread.start()
        thread.join(_TRANSCRIBE_TIMEOUT_SECONDS)
        if thread.is_alive():
            self.close(graceful=False)
            thread.join(timeout=1)
            raise TimeoutError("whisper.cpp transcription timed out")
        if "error" in exchange_result:
            raise exchange_result["error"]
        status, confidence, duration_ms, detected, text, error = exchange_result["values"]
        if status:
            raise RuntimeError(error or f"whisper.cpp worker error {status}")
        logger.info("Whisper transcription backend=%s duration_ms=%.1f", WHISPER_CPP_VULKAN_BACKEND, duration_ms)
        return BackendResult(text, detected or None, confidence, duration_ms)

    def close(self, graceful: bool = True) -> None:
        process = getattr(self, "_process", None)
        if process is None or getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            with self._io_lock:
                if graceful and process.poll() is None:
                    process.stdin.write(_REQUEST.pack(_MAGIC, 1, 2, 0, 0, 0.0, 0.0, 0))
                    process.stdin.flush()
            if process.poll() is None:
                process.wait(timeout=5 if graceful else 0.1)
        except Exception:
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=5)
            except Exception:
                pass
        finally:
            for stream_name in ("stdin", "stdout", "stderr"):
                stream = getattr(process, stream_name, None)
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
            stderr_thread = getattr(self, "_stderr_thread", None)
            if stderr_thread is not None and stderr_thread is not current_thread():
                stderr_thread.join(timeout=1)
            logger.info("Whisper backend=%s state=closed", WHISPER_CPP_VULKAN_BACKEND)


class _SharedBackend:
    def __init__(self, key: Tuple, factory: Callable[[], TranscriptionBackend]) -> None:
        self.key = key
        self.refs = 1
        self.active_calls = 0
        self.closing = False
        self.release_generation = 0
        self.backend: Optional[TranscriptionBackend] = None
        self.condition = Condition()
        self.status = BackendStatus(backend=key[0], state="loading", model=key[2])
        Thread(target=self._load, args=(factory,), daemon=True, name="WhisperModelLoader").start()

    def _load(self, factory) -> None:
        started = monotonic()
        try:
            backend = factory()
            with self.condition:
                # A zero refcount can be temporary while an audio device is
                # reconfigured. Deferred release owns that decision; only an
                # already-committed close may discard a backend that finished
                # loading. Otherwise publish it so an immediate reacquire can
                # reuse the same resident model.
                if self.closing:
                    backend.close()
                    self.status.state = "closed"
                    self.condition.notify_all()
                    return
                self.backend = backend
                self.status.backend = getattr(backend, "backend_name", self.status.backend)
                self.status.state = "ready"
                self.status.gpu_active = bool(getattr(backend, "gpu_active", self.key[3] == "cuda"))
                self.status.device = getattr(backend, "device", self.key[3])
                self.status.load_duration_ms = float(getattr(backend, "load_duration_ms", (monotonic()-started)*1000))
                self.condition.notify_all()
        except Exception as exc:
            logger.exception("Whisper backend load failed: %s", self.key[0])
            with self.condition:
                self.status.state = "error"
                self.status.error = str(exc)
                self.status.load_duration_ms = (monotonic() - started) * 1000
                self.condition.notify_all()

    def transcribe(self, *args, **kwargs) -> BackendResult:
        with self.condition:
            if self.status.state == "loading":
                self.condition.wait_for(lambda: self.status.state != "loading", timeout=_READY_TIMEOUT_SECONDS)
            if self.backend is None:
                raise RuntimeError(self.status.error or "Whisper model loading timed out")
            if self.closing:
                raise RuntimeError("Whisper backend is closing")
            backend = self.backend
            self.active_calls += 1
        try:
            result = backend.transcribe(*args, **kwargs)
            self.status.transcription_duration_ms = result.duration_ms
            return result
        finally:
            with self.condition:
                self.active_calls -= 1
                self.condition.notify_all()


_registry: Dict[Tuple, _SharedBackend] = {}
_registry_lock = Lock()


def acquireBackend(backend_name: str, root: str, model: str, device: str = "cpu",
                   device_index: int = 0, compute_type: str = "auto",
                   factory: Optional[Callable[[], TranscriptionBackend]] = None) -> _SharedBackend:
    key = (backend_name, os_path.abspath(root), model, device, device_index, compute_type)
    with _registry_lock:
        shared = _registry.get(key)
        if shared is not None:
            shared.refs += 1
            # Invalidate a pending deferred release. Recorder/device
            # reconfiguration commonly stops and restarts a session within a
            # few milliseconds and must not unload the resident model.
            shared.release_generation += 1
            return shared
        cls = WhisperCppVulkanBackend if backend_name == WHISPER_CPP_VULKAN_BACKEND else FasterWhisperBackend
        def default_factory():
            try:
                return cls(root, model, device, device_index, compute_type)
            except Exception:
                if backend_name != WHISPER_CPP_VULKAN_BACKEND or not checkWhisperWeight(root, model):
                    raise
                logger.exception("Vulkan backend unavailable; falling back to faster-whisper")
                return FasterWhisperBackend(root, model, device, device_index, compute_type)
        shared = _SharedBackend(key, factory or default_factory)
        _registry[key] = shared
        return shared


def _closeBackendIfUnused(shared: _SharedBackend, generation: int) -> None:
    with _registry_lock:
        if shared.refs > 0 or shared.release_generation != generation:
            return
        if _registry.get(shared.key) is not shared:
            return
        _registry.pop(shared.key, None)
    with shared.condition:
        shared.closing = True
        shared.condition.wait_for(lambda: shared.active_calls == 0)
        backend = shared.backend
        shared.status.state = "closed"
    if backend is not None:
        backend.close()


def releaseBackend(shared: Optional[_SharedBackend], *, deferred: bool = False) -> None:
    if shared is None:
        return
    with _registry_lock:
        shared.refs -= 1
        if shared.refs > 0:
            return
        shared.release_generation += 1
        generation = shared.release_generation
    if deferred:
        timer = Timer(_BACKEND_IDLE_GRACE_SECONDS, _closeBackendIfUnused, args=(shared, generation))
        timer.daemon = True
        timer.start()
    else:
        _closeBackendIfUnused(shared, generation)


def getBackendStatus() -> dict:
    with _registry_lock:
        statuses = [asdict(shared.status) for shared in _registry.values()]
    return statuses[0] if len(statuses) == 1 else {"instances": statuses}
