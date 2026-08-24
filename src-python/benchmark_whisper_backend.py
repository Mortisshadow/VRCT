#!/usr/bin/env python3
"""Small local benchmark for a persistent transcription backend."""
import argparse
import time
import wave
import numpy as np
from models.transcription.transcription_backend import acquireBackend, releaseBackend, FASTER_WHISPER_BACKEND, WHISPER_CPP_VULKAN_BACKEND

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=(FASTER_WHISPER_BACKEND, WHISPER_CPP_VULKAN_BACKEND), default=FASTER_WHISPER_BACKEND)
    parser.add_argument("--model", required=True); parser.add_argument("--root", default=".")
    parser.add_argument("audio_wav"); parser.add_argument("-n", "--iterations", type=int, default=3)
    args = parser.parse_args()
    with wave.open(args.audio_wav, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    backend = acquireBackend(args.backend, args.root, args.model)
    try:
        started = time.perf_counter(); backend.transcribe(audio, language=None, avg_logprob=-.8, no_speech_prob=.6, no_repeat_ngram_size=0)
        print(f"status={backend.status.state} backend={backend.status.backend} load_ms={backend.status.load_duration_ms}")
        timings = []
        for _ in range(max(0, args.iterations)):
            t = time.perf_counter(); backend.transcribe(audio, language=None, avg_logprob=-.8, no_speech_prob=.6, no_repeat_ngram_size=0); timings.append((time.perf_counter()-t)*1000)
        if timings: print("transcription_ms=" + ",".join(f"{x:.2f}" for x in timings))
    finally: releaseBackend(backend)
if __name__ == "__main__": main()
