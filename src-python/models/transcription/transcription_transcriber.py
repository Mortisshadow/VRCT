"""Runtime transcriber for Google, faster-whisper, and whisper.cpp.

This class focuses on converting incoming raw audio buffers into text using
either the Google web recognizer (online) or a local Whisper model (offline).

通常はキューへ (raw_bytes, recorded_at) が積まれ、フレーズ境界は
`speech_recognition.listen_energy_and_audio_in_background` の phrase_time_limit と
AudioTranscriber.updateLastSampleAndPhraseStatus の phrase_timeout で決まる。
任意の VAD Filter 使用時は (raw_bytes, recorded_at, end_reason) を受け取り、
最大時間で分割された連続発話だけを同じ論理フレーズとして結合する。
partial (発話中の暫定結果) 通知は行わない。
"""

import time
import logging
from io import BytesIO
from queue import Empty
from threading import Event, Lock
import wave
from typing import Any, Dict, List, Optional
from speech_recognition import Recognizer, AudioData, AudioFile
from speech_recognition.exceptions import UnknownValueError
from datetime import timedelta
from pyaudiowpatch import get_sample_size, paInt16
from .transcription_languages import transcription_lang
from .transcription_whisper import checkWhisperWeight, checkWhisperCppWeight
from .transcription_backend import (
    FASTER_WHISPER_BACKEND,
    WHISPER_CPP_VULKAN_BACKEND,
    acquireBackend,
    releaseBackend,
)

import numpy as np
from pydub import AudioSegment
from utils import errorLogging

try:
    import torch  # noqa: F401
except Exception:
    torch = None  # type: ignore

import warnings
warnings.simplefilter('ignore', RuntimeWarning)

PHRASE_TIMEOUT = 3
MAX_PHRASES = 10
GOOGLE_RECOGNIZE_TIMEOUT_SECONDS = 10
WHISPER_STREAM_OVERLAP_SECONDS = 0.25
WHISPER_PHRASE_BOUNDARY_TOLERANCE_SECONDS = 0.25

logger = logging.getLogger("vrct.transcription")


class AudioTranscriber:
    """Convert queued audio buffers into transcripts.

    Public attributes set by the constructor:
    - speaker: bool
    - phrase_timeout: int
    - max_phrases: int
    """

    def __init__(
        self,
        speaker: bool,
        source: Any,
        phrase_timeout: int,
        max_phrases: int,
        transcription_engine: str,
        root: Optional[str] = None,
        whisper_weight_type: Optional[str] = None,
        device: str = "cpu",
        device_index: int = 0,
        compute_type: str = "auto",
        whisper_backend: str = FASTER_WHISPER_BACKEND,
        backend_factory=None,
        vad_segmented: bool = False,
    ) -> None:
        self.speaker = speaker
        self.phrase_timeout = phrase_timeout
        self.max_phrases = max_phrases
        self.transcript_data: List[Dict[str, Any]] = []
        self.transcript_changed_event = Event()
        self.last_recognition_error = False
        self.audio_recognizer = Recognizer()
        self.audio_recognizer.operation_timeout = GOOGLE_RECOGNIZE_TIMEOUT_SECONDS
        self.transcription_engine = "Google"
        self.whisper_model = None
        self.whisper_backend = None
        self._close_lock = Lock()
        self.whisper_weight_type = whisper_weight_type
        self.vad_segmented = vad_segmented
        self._vad_continuation = False
        self._whisper_audio_overlap = bytearray()
        self._whisper_phrase_text = ""
        self._whisper_phrase_language: Optional[str] = None
        self.audio_sources: Dict[str, Any] = {
            "sample_rate": source.SAMPLE_RATE,
            "sample_width": source.SAMPLE_WIDTH,
            "channels": source.channels,
            "last_sample": bytearray(),
            "last_spoken": None,
            "new_phrase": True,
            "process_data_func": self.processSpeakerData if speaker else self.processMicData,
        }

        weight_available = (
            checkWhisperCppWeight(root, whisper_weight_type)
            if whisper_backend == WHISPER_CPP_VULKAN_BACKEND
            else checkWhisperWeight(root, whisper_weight_type)
        ) if root and whisper_weight_type else False
        if transcription_engine == "Whisper" and weight_available:
            self.whisper_backend = acquireBackend(
                whisper_backend, root, whisper_weight_type, device=device,
                device_index=device_index, compute_type=compute_type, factory=backend_factory,
            )
            self.transcription_engine = "Whisper"

    def transcribeAudioQueue(
        self,
        audio_queue: Any,
        languages: List[str],
        countries: List[str],
        avg_logprob: float = -0.8,
        no_speech_prob: float = 0.6,
        no_repeat_ngram_size: int = 0,
    ) -> bool:
        try:
            item = audio_queue.get_nowait()
        except Empty:
            time.sleep(0.01)
            return False
        if self.vad_segmented:
            audio, time_spoken, reason = item
            continuing = self._vad_continuation
            if not continuing:
                self._resetWhisperPhraseState()
            self.audio_sources["last_sample"] = (
                bytearray(self._whisper_audio_overlap)
                if continuing and self._usesWhisperCppStreaming()
                else bytearray()
            )
            self.audio_sources["last_sample"].extend(audio)
            self.audio_sources["last_spoken"] = time_spoken
            self.audio_sources["new_phrase"] = not continuing
            self._vad_continuation = reason == "max_duration"
        else:
            audio, time_spoken = item
        # Only decode audio which has not been processed before. A small raw
        # overlap protects words crossing a callback boundary without making
        # continuous speech grow into an ever more expensive 30-second input.
        if not self.vad_segmented:
            if self._usesWhisperCppStreaming():
                self.audio_sources["last_sample"] = bytearray(self._whisper_audio_overlap)
            self.audio_sources["new_phrase"] = False
            # まとめて drain して最新まで反映する (backlog を残さない)
            self.updateLastSampleAndPhraseStatus(audio, time_spoken)
            while True:
                try:
                    audio, time_spoken = audio_queue.get_nowait()
                except Empty:
                    break
                self.updateLastSampleAndPhraseStatus(audio, time_spoken)
        self._capWhisperCppInput()

        confidences: List[Dict[str, Any]] = [{"confidence": 0, "text": "", "language": None}]
        try:
            audio_data = self.audio_sources["process_data_func"]()
            match self.transcription_engine:
                case "Google":
                    self.last_recognition_error = False
                    for language, country in zip(languages, countries):
                        try:
                            text, confidence = self.audio_recognizer.recognize_google(
                                audio_data,
                                language=transcription_lang[language][country][self.transcription_engine],
                                with_confidence=True
                            )
                            confidences.append({"confidence": confidence, "text": text, "language": language})
                        except UnknownValueError:
                            pass
                        except Exception:
                            self.last_recognition_error = True
                            errorLogging()
                case "Whisper":
                    audio_data = np.frombuffer(
                        audio_data.get_raw_data(convert_rate=16000, convert_width=2), np.int16
                    ).astype(np.float32)
                    audio_data *= (1.0 / 32768.0)
                    if torch is not None and isinstance(audio_data, torch.Tensor):
                        audio_data = audio_data.detach().numpy()

                    # Auto language detection is part of the same decode. Repeating the
                    # identical auto request once per configured language multiplies GPU
                    # work without changing the hypothesis or confidence.
                    language_country_pairs = list(zip(languages, countries))
                    if language_country_pairs:
                        language, country = language_country_pairs[0]
                        source_language = (
                            transcription_lang[language][country][self.transcription_engine]
                            if len(language_country_pairs) == 1
                            else (
                                self._whisper_phrase_language
                                if self._usesWhisperCppStreaming() or self.vad_segmented
                                else None
                            )
                        )
                        if self.whisper_backend is not None:
                            backend_result = self.whisper_backend.transcribe(
                                audio_data, language=source_language, avg_logprob=avg_logprob,
                                no_speech_prob=no_speech_prob,
                                no_repeat_ngram_size=no_repeat_ngram_size,
                            )
                            text = backend_result.text
                            detected_language = backend_result.language
                            confidence = backend_result.confidence
                        else:  # compatibility for injected legacy model test doubles
                            text = ""
                            segments, info = self.whisper_model.transcribe(
                                audio_data, beam_size=5, temperature=0.0,
                                log_prob_threshold=avg_logprob,
                                no_speech_threshold=no_speech_prob, language=source_language,
                                word_timestamps=False, without_timestamps=True,
                                task="transcribe", no_repeat_ngram_size=no_repeat_ngram_size,
                            )
                            for segment in segments:
                                if segment.avg_logprob >= avg_logprob and segment.no_speech_prob <= no_speech_prob:
                                    text += segment.text
                            detected_language = info.language
                            confidence = info.language_probability
                        self.last_recognition_error = False
                        if detected_language and (self._usesWhisperCppStreaming() or self.vad_segmented):
                            self._whisper_phrase_language = detected_language
                        if self._usesWhisperCppStreaming() or self.vad_segmented:
                            text = self._mergeWhisperPhraseText(text)
                        selected_language = next(
                            (candidate_language for candidate_language, candidate_country in language_country_pairs
                             if transcription_lang[candidate_language][candidate_country][self.transcription_engine]
                             == detected_language),
                            language,
                        )
                        confidences.append({"confidence": confidence, "text": text, "language": selected_language})

        except UnknownValueError:
            pass
        except Exception:
            self.last_recognition_error = True
            errorLogging()

        if self._usesWhisperCppStreaming():
            self._updateWhisperAudioOverlap()
        result = max(confidences, key=lambda x: x["confidence"])
        if result["text"] != "":
            self.updateTranscript(result)
        return True

    def updateLastSampleAndPhraseStatus(self, data: bytes, time_spoken) -> None:
        source_info = self.audio_sources
        boundary_tolerance = (
            WHISPER_PHRASE_BOUNDARY_TOLERANCE_SECONDS
            if self._usesWhisperCppStreaming()
            else 0.0
        )
        is_new_phrase = (
            source_info["last_spoken"] is None
            or time_spoken - source_info["last_spoken"] > timedelta(
                seconds=self.phrase_timeout + boundary_tolerance
            )
        )
        if is_new_phrase:
            source_info["last_sample"] = bytearray()
            self._whisper_audio_overlap = bytearray()
            self._whisper_phrase_text = ""
            self._whisper_phrase_language = None

        # Preserve a boundary found earlier while draining the same queue.
        source_info["new_phrase"] = source_info["new_phrase"] or is_new_phrase

        source_info["last_sample"].extend(data)
        source_info["last_spoken"] = time_spoken

    def _usesWhisperCppStreaming(self) -> bool:
        status = getattr(self.whisper_backend, "status", None)
        return (
            self.transcription_engine == "Whisper"
            and status is not None
            and status.backend == WHISPER_CPP_VULKAN_BACKEND
        )

    def _resetWhisperPhraseState(self) -> None:
        self._whisper_audio_overlap = bytearray()
        self._whisper_phrase_text = ""
        self._whisper_phrase_language = None

    def _capWhisperCppInput(self) -> None:
        if not self._usesWhisperCppStreaming():
            return
        source_info = self.audio_sources
        frame_size = int(source_info["sample_width"]) * max(1, int(source_info["channels"]))
        max_bytes = int(source_info["sample_rate"]) * frame_size * 15
        if max_bytes > 0 and len(source_info["last_sample"]) > max_bytes:
            dropped = len(source_info["last_sample"]) - max_bytes
            source_info["last_sample"] = source_info["last_sample"][-max_bytes:]
            logger.warning(
                "Whisper.cpp input backlog capped at 15 seconds; dropped_bytes=%d", dropped
            )

    def _updateWhisperAudioOverlap(self) -> None:
        source_info = self.audio_sources
        bytes_per_second = (
            int(source_info["sample_rate"])
            * int(source_info["sample_width"])
            * max(1, int(source_info["channels"]))
        )
        overlap_bytes = int(bytes_per_second * WHISPER_STREAM_OVERLAP_SECONDS)
        frame_size = int(source_info["sample_width"]) * max(1, int(source_info["channels"]))
        overlap_bytes -= overlap_bytes % frame_size
        self._whisper_audio_overlap = bytearray(source_info["last_sample"][-overlap_bytes:]) \
            if overlap_bytes > 0 else bytearray()

    def _mergeWhisperPhraseText(self, text: str) -> str:
        incoming = text.strip()
        if not incoming:
            return ""
        if self.audio_sources["new_phrase"] or not self._whisper_phrase_text:
            self._whisper_phrase_text = incoming
            return self._whisper_phrase_text

        previous_words = self._whisper_phrase_text.split()
        incoming_words = incoming.split()
        def comparable(word: str) -> str:
            return word.casefold().strip(".,!?;:…。、！？()[]{}\"'")
        overlap = 0
        for size in range(min(12, len(previous_words), len(incoming_words)), 0, -1):
            if ([comparable(word) for word in previous_words[-size:]]
                    == [comparable(word) for word in incoming_words[:size]]):
                overlap = size
                break
        remainder = " ".join(incoming_words[overlap:])
        if not remainder:
            return ""
        self._whisper_phrase_text = f"{self._whisper_phrase_text.rstrip()} {remainder}".strip()
        return self._whisper_phrase_text

    def processMicData(self) -> AudioData:
        audio_data = AudioData(
            bytes(self.audio_sources["last_sample"]), self.audio_sources["sample_rate"], self.audio_sources["sample_width"]
        )
        return audio_data

    def processSpeakerData(self) -> AudioData:
        temp_file = BytesIO()
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(self.audio_sources["channels"])
            wf.setsampwidth(get_sample_size(paInt16))
            wf.setframerate(self.audio_sources["sample_rate"])
            wf.writeframes(bytes(self.audio_sources["last_sample"]))
        temp_file.seek(0)

        if self.audio_sources["channels"] > 2:
            audio = AudioSegment.from_file(temp_file, format="wav")
            mono_audio = audio.set_channels(1)
            temp_file = BytesIO()
            mono_audio.export(temp_file, format="wav")
            temp_file.seek(0)

        with AudioFile(temp_file) as source:
            audio = self.audio_recognizer.record(source)
        return audio

    def updateTranscript(self, result: dict) -> None:
        source_info = self.audio_sources
        transcript = self.transcript_data

        if source_info["new_phrase"] or len(transcript) == 0:
            if self.max_phrases > 0 and len(transcript) >= self.max_phrases:
                transcript.pop(-1)
            transcript.insert(0, result)
        else:
            transcript[0] = result

    def getTranscript(self) -> dict:
        if len(self.transcript_data) > 0:
            result = self.transcript_data.pop(-1)
        else:
            result = {"confidence": 0, "text": "", "language": None}
        return result

    def clearTranscriptData(self) -> None:
        self.transcript_data.clear()
        self.audio_sources["last_sample"] = bytearray()
        self.audio_sources["new_phrase"] = True
        self._whisper_audio_overlap = bytearray()
        self._whisper_phrase_text = ""
        self._whisper_phrase_language = None
        self._vad_continuation = False

    def close(self) -> None:
        with self._close_lock:
            # Detach first so session-stop and worker end callbacks can safely race.
            backend = self.whisper_backend
            self.whisper_backend = None
        # Device monitoring can stop/start the owning session back-to-back.
        # Keep the shared model briefly available so that transient recorder
        # reconfiguration does not tear down and reload the Vulkan context.
        releaseBackend(backend, deferred=True)
