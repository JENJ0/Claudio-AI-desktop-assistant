"""Download every model CLAUDIO needs into models/ (one-off, ~3 GB with both Whisper models).

uv run python scripts/download_models.py
uv run python scripts/download_models.py --skip-whisper --voices it_IT-riccardo-x_low
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

from claudio.config import load_settings

SILERO_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2/src/silero_vad/data/silero_vad.onnx"
)
OWW_BASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
OWW_FILES = ("melspectrogram.onnx", "embedding_model.onnx", "hey_jarvis_v0.1.onnx")
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_VOCAB = (
    "https://raw.githubusercontent.com/thewh1teagle/kokoro-onnx/"
    "3596b26764286a7de9d90c363e988d50578918e5/src/kokoro_onnx/config.json"
)


def fetch(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  già presente: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  scarico {dest.name} ...", end="", flush=True)
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    tmp.replace(dest)
    print(f" {dest.stat().st_size / 1e6:.1f} MB")


def piper_url(voice: str, suffix: str) -> str:
    # it_IT-paola-medium -> it/it_IT/paola/medium/it_IT-paola-medium.onnx
    lang_region, name, quality = voice.split("-")
    lang = lang_region.split("_")[0]
    return f"{PIPER_BASE}{lang}/{lang_region}/{name}/{quality}/{voice}{suffix}"


def main() -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--whisper", nargs="*", default=[settings.stt.model, settings.stt.fallback_model]
    )
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--skip-kokoro", action="store_true")
    voices = [settings.tts.piper_voice_for(lang) for lang in ("it", *settings.stt.languages)]
    parser.add_argument("--voices", nargs="*", default=list(dict.fromkeys(voices)))
    args = parser.parse_args()
    models = settings.resolve(settings.models_dir)

    print("Silero VAD")
    fetch(SILERO_URL, models / "vad" / "silero_vad.onnx")
    print("openWakeWord")
    for name in OWW_FILES:
        fetch(OWW_BASE + name, models / "wakeword" / name)
    if not args.skip_kokoro:
        print("Kokoro (voci Nicola e Michael, ~350 MB)")
        for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
            fetch(KOKORO_BASE + name, models / "kokoro" / name)
        fetch(KOKORO_VOCAB, models / "kokoro" / "config.json")
    print("Piper")
    for voice in args.voices:
        for suffix in (".onnx", ".onnx.json"):
            fetch(piper_url(voice, suffix), models / "piper" / f"{voice}{suffix}")
    if not args.skip_whisper:
        from faster_whisper.utils import download_model

        for name in args.whisper:
            print(f"Whisper {name} (può richiedere qualche minuto)")
            path = download_model(name, cache_dir=str(models / "whisper"))
            print(f"  pronto in {path}")
    print("Fatto. Verifica con: uv run claudio --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
