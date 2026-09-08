"""Optional Silero VAD speech segmentation for local real-time transcription."""

import audioop
import logging
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Literal, Optional, Protocol

import numpy as np

TARGET_SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
FRAME_DURATION_MS = FRAME_SAMPLES / TARGET_SAMPLE_RATE * 1000
SegmentEndReason = Literal["silence", "max_duration", "flush"]

logger = logging.getLogger("vrct.transcription.vad")


class Pcm16MonoNormalizer:
    """Incrementally normalize device PCM to 16 kHz mono signed int16."""

    def __init__(self, sample_rate: int, sample_width: int, channels: int) -> None:
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = max(1, channels)
        self._rate_state = None

    def reset(self) -> None:
        self._rate_state = None

    def process(self, data: bytes) -> bytes:
        if not data:
            return b""
        if self.sample_width != 2:
            data = audioop.lin2lin(data, self.sample_width, 2)
        samples = np.frombuffer(data, dtype=np.int16)
        samples = samples[:samples.size - samples.size % self.channels]
        if self.channels > 1 and samples.size:
            frames = samples.reshape(-1, self.channels).astype(np.int32)
            samples = np.rint(frames.mean(axis=1)).clip(-32768, 32767).astype(np.int16)
        data = samples.astype("<i2", copy=False).tobytes()
        if self.sample_rate != TARGET_SAMPLE_RATE:
            data, self._rate_state = audioop.ratecv(
                data, 2, 1, self.sample_rate, TARGET_SAMPLE_RATE, self._rate_state
            )
        return data


class VadEngine(Protocol):
    def __call__(self, frame: np.ndarray) -> float: ...
    def reset(self) -> None: ...


class SileroFrameProbability:
    """Use the Silero ONNX model already shipped by faster-whisper."""

    def __init__(self) -> None:
        self._model = None
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        self._fallback_to_rms = False

    def reset(self) -> None:
        self._state.fill(0)
        self._context.fill(0)

    def __call__(self, frame: np.ndarray) -> float:
        if frame.shape != (FRAME_SAMPLES,):
            raise ValueError(f"Expected {(FRAME_SAMPLES,)} audio frame, got {frame.shape}")
        if not self._fallback_to_rms:
            try:
                if self._model is None:
                    from faster_whisper.vad import get_vad_model
                    self._model = get_vad_model()
                model_input = np.concatenate((self._context, frame.reshape(1, -1)), axis=1)
                encoder_output = self._model.encoder_session.run(None, {"input": model_input})[0]
                output, self._state = self._model.decoder_session.run(
                    None, {"input": encoder_output.reshape(1, 128), "state": self._state}
                )
                self._context = frame[-64:].reshape(1, -1)
                return float(np.asarray(output).reshape(-1)[0])
            except Exception:
                # A missing/corrupt ONNX runtime must not kill the recorder's
                # background thread. This deliberately sensitive RMS gate may
                # pass extra noise, but live transcription remains available.
                self._fallback_to_rms = True
                self._model = None
                logger.exception("Silero VAD failed; using the RMS fallback for this recorder")
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32))))
        return min(1.0, rms / 0.02)


@dataclass(frozen=True)
class SpeechSegment:
    audio: bytes
    reason: SegmentEndReason


class VadSegmenter:
    """Stateful VAD with hysteresis, pre-roll, hangover and a hard duration cap."""

    def __init__(
        self,
        probability: Optional[VadEngine] = None,
        *,
        speech_threshold: float = 0.25,
        negative_threshold: Optional[float] = None,
        hangover_frames: int = 24,
        max_speech_frames: int = 250,
        min_speech_frames: int = 2,
        pre_speech_pad_frames: int = 8,
        diagnostic_callback: Optional[Callable[[str], None]] = None,
        diagnostic_label: str = "vad",
    ) -> None:
        self.probability = probability or SileroFrameProbability()
        self.speech_threshold = speech_threshold
        self.negative_threshold = (
            negative_threshold if negative_threshold is not None else max(0.0, speech_threshold - 0.15)
        )
        self.hangover_frames = hangover_frames
        self.max_speech_frames = max_speech_frames
        self.min_speech_frames = min_speech_frames
        self.pre_speech_frames: Deque[bytes] = deque(maxlen=pre_speech_pad_frames)
        self.diagnostic_callback = diagnostic_callback
        self.diagnostic_label = diagnostic_label
        self._remainder = b""
        self._speech_frames: list[bytes] = []
        self._positive_frames = 0
        self._speech_frame_count = 0
        self._silence_start_frame: Optional[int] = None
        self._speaking = False

    @property
    def speaking(self) -> bool:
        return self._speaking

    def process(self, pcm: bytes) -> list[SpeechSegment]:
        self._remainder += pcm
        frame_bytes = FRAME_SAMPLES * 2
        results = []
        while len(self._remainder) >= frame_bytes:
            frame, self._remainder = self._remainder[:frame_bytes], self._remainder[frame_bytes:]
            samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
            segment = self._process_frame(frame, self.probability(samples))
            if segment is not None:
                results.append(segment)
        return results

    def flush(self) -> Optional[SpeechSegment]:
        if self._remainder:
            padded = self._remainder.ljust(FRAME_SAMPLES * 2, b"\0")
            self._remainder = b""
            result = self._process_frame(
                padded, self.probability(np.frombuffer(padded, dtype="<i2").astype(np.float32) / 32768.0)
            )
            if result is not None:
                return result
        return self._finish_segment("flush") if self._speaking else None

    def reset(self) -> None:
        self._remainder = b""
        self._speech_frames = []
        self._positive_frames = 0
        self._speech_frame_count = 0
        self._silence_start_frame = None
        self._speaking = False
        self.pre_speech_frames.clear()
        self.probability.reset()

    def _process_frame(self, frame: bytes, probability: float) -> Optional[SpeechSegment]:
        if not self._speaking:
            self.pre_speech_frames.append(frame)
            self._positive_frames = self._positive_frames + 1 if probability >= self.speech_threshold else 0
            if self._positive_frames >= self.min_speech_frames:
                self._speaking = True
                self._speech_frames = list(self.pre_speech_frames)
                self._speech_frame_count = self._positive_frames
                self.pre_speech_frames.clear()
                self._log(f"speech_start probability={probability:.3f}")
            return None

        self._speech_frames.append(frame)
        self._speech_frame_count += 1
        if probability >= self.speech_threshold:
            self._silence_start_frame = None
        elif probability < self.negative_threshold:
            if self._silence_start_frame is None:
                self._silence_start_frame = self._speech_frame_count
            if self._speech_frame_count - self._silence_start_frame >= self.hangover_frames:
                return self._finish_segment("silence")
        if self._speech_frame_count >= self.max_speech_frames:
            return self._finish_segment("max_duration")
        return None

    def _finish_segment(self, reason: SegmentEndReason) -> Optional[SpeechSegment]:
        result = SpeechSegment(b"".join(self._speech_frames), reason) if self._speaking else None
        self._speech_frames = []
        self._positive_frames = 0
        self._speech_frame_count = 0
        self._silence_start_frame = None
        self._speaking = False
        self.pre_speech_frames.clear()
        if reason != "max_duration":
            self.probability.reset()
        if result is not None:
            self._log(f"speech_end reason={reason} duration_ms={len(result.audio) / 32:.1f}")
        return result

    def _log(self, message: str) -> None:
        if self.diagnostic_callback is not None:
            try:
                self.diagnostic_callback(f"[VAD][{self.diagnostic_label}] {message}")
            except Exception:
                pass


class VadRecognizerAdapter:
    sample_rate = TARGET_SAMPLE_RATE
    sample_width = 2

    def __init__(self, native_sample_rate: int, native_sample_width: int, native_channels: int) -> None:
        self._normalizer = Pcm16MonoNormalizer(native_sample_rate, native_sample_width, native_channels)
        self.segmenter = VadSegmenter()

    def process(self, pcm_bytes: bytes) -> list[SpeechSegment]:
        normalized = self._normalizer.process(pcm_bytes)
        if not normalized:
            return []
        return self.segmenter.process(normalized)

    def flush(self) -> Optional[SpeechSegment]:
        result = self.segmenter.flush()
        self._normalizer.reset()
        return result

    def reset(self) -> None:
        self._normalizer.reset()
        self.segmenter.reset()
