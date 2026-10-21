"""Jack's voice, as a separate process: Qwen3-TTS (fine-tuned) -> RVC, blocks of 250 ms.

Run by claudio.tts.jack with the worker venv's interpreter (torch, faster-qwen3-tts and
Applio live there, not in Claudio's venv), so this file imports nothing from claudio:

  worker\\env\\Scripts\\python.exe jack_worker.py --checkpoint ... --rvc-model ... --applio ...

Protocol (the real stdout carries it, nothing else; library prints go to stderr).
Requests, one JSON line each on stdin:
  {"op": "speak", "id": 7, "text": "...", "language": "Auto"}
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
import os
import queue
import sys
import threading
import time
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
SPEAK_TOKENS = 300
INDEX_RATE = 0.0
PROTECT = 0.33

_write_lock = threading.Lock()


def emit(header: dict[str, Any], payload: bytes = b"") -> None:
    with _write_lock:
        PROTOCOL_OUT.write(json.dumps(header).encode() + b"\n" + payload)
        PROTOCOL_OUT.flush()


class RvcStream:
    """Applio's realtime VoiceChanger fed with Qwen3 chunks; yields converted int16 PCM."""

    def __init__(self, model_path: Path, applio: Path) -> None:
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
        self.kw = {"index_rate": INDEX_RATE, "protect": PROTECT, "volume_envelope": 1}
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
        wide = resample_poly(np.concatenate([self.tail, chunk]), 2, 1).astype(np.float32)
        wide = wide[2 * len(self.tail) :]
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


class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        self.torch = torch
        t0 = time.perf_counter()
        self.tts = FasterQwen3TTS.from_pretrained(str(args.checkpoint))
        self.rvc = RvcStream(args.rvc_model, args.applio)
        for _ in self.generate("Prova.", "Auto", 60):  # captures the CUDA graph
            pass
        self.load_s = time.perf_counter() - t0
        self.requests: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.cancelled_upto = -1
        self.latest_id = -1

    def generate(self, text: str, language: str, max_tokens: int = SPEAK_TOKENS) -> Any:
        return self.tts.generate_custom_voice_streaming(
            text=text, speaker="jack", language=language, chunk_size=8, max_new_tokens=max_tokens
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
        self.rvc.begin()
        if not cancelled:
            for chunk, _rate, _timing in self.generate(
                request["text"], request.get("language", "Auto")
            ):
                send(self.rvc.feed(chunk))
                if rid <= self.cancelled_upto:
                    cancelled = True
                    break
        if not cancelled:
            send(self.rvc.flush())
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
