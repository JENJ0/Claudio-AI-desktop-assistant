"""Jack's voice, as a separate process: Qwen3-TTS (fine-tuned) -> RVC, in blocks of 250 ms
or on the whole sentence.

Run by claudio.tts.jack with the worker venv's interpreter (torch, faster-qwen3-tts and
Applio live there, not in Claudio's venv), so this file imports nothing from claudio:

  worker\\env\\Scripts\\python.exe jack_worker.py --checkpoint ... --rvc-model ... --applio ...

Protocol (the real stdout carries it, nothing else; library prints go to stderr).
Requests, one JSON line each on stdin:
  {"op": "speak", "id": 7, "text": "...", "language": "Italian", "rvc": "blocks"}
                             rvc "blocks": audio as it is generated (first audio ~0.3 s);
                             "sentence": the whole sentence through RVC at once, clearer
                             (WER 6% instead of 10%) but the audio waits for the sentence
  {"op": "interrupt"}        stops every speak received so far, even one in progress
  {"op": "quit"}
Replies, one JSON line each on stdout:
  {"event": "ready", "sample_rate": 24000, "load_s": 14.2, "vram_mb": 2900}
  {"event": "audio", "id": 7, "bytes": 24000}   followed by that many bytes of int16 PCM
  {"event": "done", "id": 7, "cancelled": false, "first_ms": 310, "total_ms": 900}
  {"event": "error", "id": 7, "message": "..."}
Personal use only: the model is trained on Handsome Jack's Italian dub."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

PROTOCOL_OUT = sys.stdout.buffer
sys.stdout = sys.stderr  # whatever a library prints must not corrupt the protocol

QWEN_RATE = 24_000
RVC_RATE = 48_000  # Applio's realtime rate
OUT_RATE = 24_000  # what is sent: the player's rate, so nothing is resampled without a filter
BLOCK_S = 0.25
CONTEXT = 480  # Qwen samples of the previous chunk kept to resample a chunk without edge ripple
TAIL_S = 0.15  # speech ends this long after the input: RVC output follows the input by ~70 ms
SENTENCE_RATE = 16_000  # what Applio's offline pipeline takes
SPEAK_TOKENS = 300
INDEX_RATE = 0.0
PROTECT = 0.33

_write_lock = threading.Lock()


def emit(header: dict[str, Any], payload: bytes = b"") -> None:
    with _write_lock:
        PROTOCOL_OUT.write(json.dumps(header).encode() + b"\n" + payload)
        PROTOCOL_OUT.flush()


def speed_plan(speed: float, rate_ratio: float = RVC_RATE / QWEN_RATE) -> tuple[int, int, int]:
    """How to change Jack's pace: Qwen3's 24 kHz goes to RVC at its rate / speed (up/down of
    resample_poly), so it plays slower (< 1) or faster, and its pitch moves by the inverse;
    RVC takes it back with f0_up_key semitones (an integer: 0.9 leaves +0.2 semitones).
    `rate_ratio` is RVC's input rate over Qwen3's: 2 for the realtime path (48 kHz), 2/3 for
    the whole-sentence one (16 kHz)."""
    ratio = Fraction(rate_ratio / speed).limit_denominator(50)
    semitones = round(-12 * math.log2(speed))
    return ratio.numerator, ratio.denominator, semitones


class RvcStream:
    """Applio's realtime VoiceChanger fed with Qwen3 chunks; yields converted int16 PCM."""

    def __init__(self, model_path: Path, applio: Path, speed: float = 1.0) -> None:
        import numpy as np

        os.chdir(applio)  # Applio resolves its assets and logs relative to its own folder
        sys.path.insert(0, str(applio))
        from rvc.realtime.core import AUDIO_SAMPLE_RATE, VoiceChanger

        self.np = np
        zc = AUDIO_SAMPLE_RATE // 100
        self.block = round(BLOCK_S * AUDIO_SAMPLE_RATE / zc) * zc
        self.vc = VoiceChanger(
            block_frame=self.block,
            cross_fade_overlap_size=0.05,
            extra_convert_size=1.0,
            model_path=str(model_path),
            index_path=None,
            f0_method="fcpe",
            embedder_model="contentvec",
            silent_threshold=-60,
        )
        self.up, self.down, semitones = speed_plan(speed)
        self.kw = {
            "index_rate": INDEX_RATE,
            "protect": PROTECT,
            "volume_envelope": 1,
            "f0_up_key": semitones,
        }
        self.vc.warmup(**self.kw)
        self.begin()

    def begin(self) -> None:
        from scipy.signal import butter, sosfilt_zi

        self.sos = butter(8, 0.45, output="sos")  # low-pass at 10.8 kHz before dropping to 24 kHz
        self.zi = sosfilt_zi(self.sos) * 0.0
        self.pending = self.np.zeros(0, dtype=self.np.float32)  # 48 kHz not yet converted
        self.tail = self.np.zeros(0, dtype=self.np.float32)  # last Qwen samples
        self.fed = 0  # 48 kHz samples received
        self.emitted = 0

    def _convert(self, audio: Any, limit: int | None = None) -> bytes:
        """One 48 kHz block through RVC; int16 PCM at 24 kHz. `limit` cuts the block short."""
        import torch
        from scipy.signal import sosfilt

        np = self.np
        torch.cuda.synchronize()
        out, _volume, _perf = self.vc.on_request(audio, **self.kw)
        out = np.asarray(out, dtype=np.float32)[:limit]
        smooth, self.zi = sosfilt(self.sos, out, zi=self.zi)
        first = -self.emitted % 2  # keep the even samples of the whole stream
        self.emitted += len(out)
        pcm = np.clip(smooth[first::2], -1.0, 1.0)
        return (pcm * 32767).astype("<i2").tobytes()

    def feed(self, chunk: Any) -> list[bytes]:
        """One Qwen3 chunk (float32, 24 kHz) in; zero or more converted blocks out."""
        from scipy.signal import resample_poly

        np = self.np
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        joined = np.concatenate([self.tail, chunk])
        wide = resample_poly(joined, self.up, self.down).astype(np.float32)
        wide = wide[len(wide) - round(len(chunk) * self.up / self.down) :]
        self.tail = chunk[-CONTEXT:]
        self.fed += len(wide)
        self.pending = np.concatenate([self.pending, wide])
        blocks = []
        while len(self.pending) >= self.block:
            blocks.append(self._convert(self.pending[: self.block]))
            self.pending = self.pending[self.block :]
        return blocks

    def flush(self) -> list[bytes]:
        """Convert what is left (zero-padded), then zeros until the speech has ended."""
        np = self.np
        end = self.fed + int(TAIL_S * RVC_RATE)
        blocks = []
        if len(self.pending):
            pad = np.zeros(self.block - len(self.pending), dtype=np.float32)
            blocks.append(self._convert(np.concatenate([self.pending, pad])))
        self.pending = np.zeros(0, dtype=np.float32)
        while self.emitted < end:
            limit = min(self.block, end - self.emitted)
            blocks.append(self._convert(np.zeros(self.block, dtype=np.float32), limit))
        return blocks


class RvcSentence:
    """Applio's offline converter on a whole sentence: the same model as RvcStream, but it
    sees the sentence at once, so words are not cut at block edges (WER 6% against 10%)."""

    def __init__(self, model_path: Path, applio: Path, speed: float = 1.0) -> None:
        import numpy as np

        os.chdir(applio)
        if str(applio) not in sys.path:
            sys.path.insert(0, str(applio))
        from rvc.infer.infer import VoiceConverter

        self.np = np
        self.vc = VoiceConverter()
        self.vc.get_vc(str(model_path), 0)
        self.vc.load_hubert("contentvec")
        self.up, self.down, self.semitones = speed_plan(speed, SENTENCE_RATE / QWEN_RATE)
        self._guard_f0()
        self.begin()
        # Loads the f0 model. A quiet voice-like tone, not zeros: on pure silence RMVPE gave
        # NaN pitch and the CUDA embedding asserted (JenJo's PC, 24/10).
        t = np.arange(QWEN_RATE // 2, dtype=np.float32) / QWEN_RATE
        self._convert(0.1 * np.sin(2 * np.pi * 130 * t).astype(np.float32))

    def _guard_f0(self) -> None:
        """Applio quantizes the pitch with comparisons that NaN slips through, so a NaN f0
        became a huge index into the 256-entry pitch embedding (device-side assert, and the
        CUDA context is lost for good). Unvoiced (0) and bounded pitch only."""
        np = self.np
        get_f0 = self.vc.vc.get_f0

        def safe_get_f0(*args: Any, **kwargs: Any) -> Any:
            coarse, f0 = get_f0(*args, **kwargs)
            bad = ~np.isfinite(f0)
            f0 = np.where(bad, 0.0, f0)
            coarse = np.where(bad, 1, np.clip(coarse, 1, 255)).astype(int)
            return coarse, f0

        self.vc.vc.get_f0 = safe_get_f0

    def begin(self) -> None:
        self.parts: list[Any] = []

    def feed(self, chunk: Any) -> list[bytes]:
        self.parts.append(self.np.asarray(chunk, dtype=self.np.float32).reshape(-1))
        return []

    def flush(self) -> list[bytes]:
        if not self.parts:
            return []
        audio, self.parts = self.np.concatenate(self.parts), []
        return [self._convert(audio)]

    def _convert(self, audio: Any) -> bytes:
        from math import gcd

        from scipy.signal import resample_poly

        np = self.np
        audio = np.pad(audio, (0, int(TAIL_S * QWEN_RATE)))  # room for the last syllable
        wide = resample_poly(audio, self.up, self.down).astype(np.float32)
        out = self.vc.vc.pipeline(
            model=self.vc.hubert_model,
            net_g=self.vc.net_g,
            sid=0,
            audio=wide,
            pitch=self.semitones,
            f0_method="rmvpe",
            file_index="",
            index_rate=INDEX_RATE,
            pitch_guidance=self.vc.use_f0,
            volume_envelope=1,
            version=self.vc.version,
            protect=PROTECT,
            f0_autotune=False,
            f0_autotune_strength=1,
            proposed_pitch=False,
            proposed_pitch_threshold=155.0,
        )
        rate = int(self.vc.tgt_sr)
        g = gcd(OUT_RATE, rate)
        pcm = resample_poly(np.asarray(out, dtype=np.float32), OUT_RATE // g, rate // g)
        return (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        self.torch = torch
        t0 = time.perf_counter()
        self.tts = FasterQwen3TTS.from_pretrained(str(args.checkpoint))
        self.temperature = args.temperature
        # Both converters stay loaded (~0.3 GB more): each sentence says which one it wants.
        self.rvc: dict[str, RvcStream | RvcSentence] = {
            "blocks": RvcStream(args.rvc_model, args.applio, args.speed),
            "sentence": RvcSentence(args.rvc_model, args.applio, args.speed),
        }
        for _ in self.generate("Prova.", "Italian", 60):  # captures the CUDA graph
            pass
        self.load_s = time.perf_counter() - t0
        self.requests: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.cancelled_upto = -1
        self.latest_id = -1

    def generate(self, text: str, language: str, max_tokens: int = SPEAK_TOKENS) -> Any:
        return self.tts.generate_custom_voice_streaming(
            text=text,
            speaker="jack",
            language=language,
            chunk_size=8,
            max_new_tokens=max_tokens,
            temperature=self.temperature,
        )

    def vram_mb(self) -> int:
        free, total = self.torch.cuda.mem_get_info()
        return round((total - free) / 2**20)

    def read_stdin(self) -> None:
        """Runs in a thread so that "interrupt" works while a sentence is being spoken."""
        for line in sys.stdin:
            try:
                request = json.loads(line)
            except ValueError:
                continue
            op = request.get("op")
            if op == "interrupt":
                self.cancelled_upto = self.latest_id
            elif op == "speak":
                self.latest_id = int(request["id"])
                self.requests.put(request)
            elif op == "quit":
                break
        self.requests.put(None)  # parent gone or told us to stop

    def speak(self, request: dict[str, Any]) -> None:
        rid = int(request["id"])
        t0 = time.perf_counter()
        first_ms: int | None = None

        def send(blocks: list[bytes]) -> None:
            nonlocal first_ms
            data = b"".join(blocks)
            if data:
                if first_ms is None:
                    first_ms = round(1000 * (time.perf_counter() - t0))
                emit({"event": "audio", "id": rid, "bytes": len(data)}, data)

        cancelled = rid <= self.cancelled_upto
        rvc = self.rvc.get(request.get("rvc", "blocks"), self.rvc["blocks"])
        rvc.begin()
        if not cancelled:
            for chunk, _rate, _timing in self.generate(
                request["text"], request.get("language", "Auto")
            ):
                send(rvc.feed(chunk))
                if rid <= self.cancelled_upto:
                    cancelled = True
                    break
        if not cancelled:
            send(rvc.flush())
        emit(
            {
                "event": "done",
                "id": rid,
                "cancelled": cancelled,
                "first_ms": first_ms,
                "total_ms": round(1000 * (time.perf_counter() - t0)),
            }
        )

    def run(self) -> None:
        threading.Thread(target=self.read_stdin, daemon=True).start()
        emit(
            {
                "event": "ready",
                "sample_rate": OUT_RATE,
                "load_s": round(self.load_s, 1),
                "vram_mb": self.vram_mb(),
            }
        )
        while (request := self.requests.get()) is not None:
            try:
                self.speak(request)
            except Exception as exc:  # the parent falls back to another voice for this sentence
                emit({"event": "error", "id": request.get("id"), "message": repr(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="fine-tuned Qwen3-TTS folder"
    )
    parser.add_argument("--rvc-model", type=Path, required=True, help="Jack's RVC .pth")
    parser.add_argument("--applio", type=Path, required=True, help="Applio folder (rvc package)")
    parser.add_argument("--speed", type=float, default=1.0, help="pace: < 1 slower")
    parser.add_argument(
        "--temperature", type=float, default=0.9, help="Qwen3 sampling: lower = steadier"
    )
    args = parser.parse_args()
    for path in (args.checkpoint, args.rvc_model, args.applio):
        if not path.exists():
            sys.exit(f"missing: {path}")
    args.checkpoint, args.rvc_model, args.applio = (
        p.resolve() for p in (args.checkpoint, args.rvc_model, args.applio)
    )
    try:
        Worker(args).run()
    except Exception as exc:
        emit({"event": "error", "id": None, "message": f"startup: {exc!r}"})
        raise


if __name__ == "__main__":
    main()
