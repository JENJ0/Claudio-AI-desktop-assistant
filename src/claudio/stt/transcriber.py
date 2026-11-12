"""faster-whisper transcription: GPU first, automatic CPU fallback. Always in Italian."""

from __future__ import annotations

import functools
import logging
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from claudio.config import SttSettings
from claudio.intake import Segment
from claudio.router.normalizer import normalize
from claudio.stt.recovery import NO_FIT, Fit, SecondHearing
from claudio.system import HIDDEN

log = logging.getLogger(__name__)

WHISPER_LANGUAGE = "it"  # Claudio is Italian only: Whisper never detects the language

# Phrases Whisper invents on silence or noise (normalised form).
HALLUCINATIONS = (
    "sottotitoli creati dalla comunita amara org",
    "sottotitoli e revisione a cura di qtss",
    "sottotitoli a cura di",
    "grazie per la visione",
    "grazie per l attenzione",
    "iscriviti al canale",
    "amara org",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "subtitles by",
)
# Short ones only count alone: the whole transcription, or every piece of it ("Grazie a tutti.
# Grazie a tutti.", "Grazie. Grazie a tutti.": the webcam's unclear requests, 29/10).
HALLUCINATIONS_EXACT = {
    "buon appetito",
    "grazie a tutti",
    "grazie a tutti per aver guardato il video",  # a whole piece: "scrivi: grazie per aver
    "grazie per aver guardato il video",  # guardato il video" is a request
    "per aver guardato il video",
    "ciao a tutti",
    "alla prossima",
    "you",
    "bye bye",
}
THANKS = frozenset({"grazie", "grazie mille", "thank you", "thanks"})  # JenJo's own, alone
ECHO_MIN_WORDS = 6  # shorter pieces of the prompt are also real requests ("apri chrome")
ECHO_END_WORDS = 2  # ...but the prompt's last words are what Whisper repeats on unclear audio
_PIECES = re.compile(r"[.,;:!?\u2026]+")
MIN_RMS = 1e-3  # about -60 dBFS: below this there is nothing to transcribe
MAX_NEW_TOKENS = 80  # an 8 s request is about 50 tokens; a repetition loop stops here
QUICK_NEW_TOKENS = 24  # the second chance listens to 2.5 s: "hey claudio che ore sono" is ~12
CANDIDATES = 5  # the closed list: how many transcriptions Whisper proposes
LOOP_RATIO = 2.4  # gzip ratio above which a text is a loop ("Ciao, ciao, ciao..."), as in Whisper
FORCED_MAX_TOKENS = 64  # a short command is ~10 word pieces: longer texts are not scored


def register_cuda_dlls() -> None:
    """Make the pip-installed CUDA libraries (nvidia-cublas/cudnn-cu12) loadable on Windows."""
    if sys.platform != "win32":
        return
    try:
        import nvidia  # namespace package from nvidia-*-cu12 wheels
    except ImportError:
        return
    for root in getattr(nvidia, "__path__", []):
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def repeated_unit(words: list[str]) -> list[str]:
    """The phrase a Whisper loop repeats ("grazie a tutti grazie a tutti" -> "grazie a
    tutti"); the words themselves when they are no repetition."""
    for size in range(1, len(words) // 2 + 1):
        if len(words) % size == 0 and words == words[:size] * (len(words) // size):
            return words[:size]
    return words


def only_sign_offs(text: str) -> bool:
    """Every piece of the text (between punctuation) is a short invented ending, alone or
    repeated, and the rest is at most a thanks: "Grazie. Grazie a tutti." yes, "Grazie a
    tutti, apri Chrome" and "Grazie, grazie." no."""
    said = []
    for piece in _PIECES.split(text):
        words = normalize(piece, drop_fillers=False).split()
        whole, unit = " ".join(words), " ".join(repeated_unit(words))
        if words and not {whole, unit} & THANKS:
            said.append(whole if whole in HALLUCINATIONS_EXACT else unit)
    return bool(said) and all(piece in HALLUCINATIONS_EXACT for piece in said)


def is_prompt_echo(text: str, prompt: str) -> bool:
    """The text is a piece of Whisper's prompt: one of ECHO_MIN_WORDS words or more, or its
    last ECHO_END_WORDS+ words ("Basta, grazie." for "Claudio, chiudi il Blocco note",
    29/10). It may also be what was really said ("alza il volume"): transcribe() asks again
    without the prompt."""
    words = normalize(text, drop_fillers=False).split()
    said = f" {normalize(prompt, drop_fillers=False)} "
    norm = f" {' '.join(words)} "
    if len(words) >= ECHO_MIN_WORDS and norm in said:
        return True
    return len(words) >= ECHO_END_WORDS and said.endswith(norm)


def is_hallucination(text: str, prompt: str = "") -> bool:
    norm = normalize(text, drop_fillers=False)
    if not norm:
        return True
    if any(h in norm for h in HALLUCINATIONS) or only_sign_offs(text):
        return True
    # Whisper sometimes echoes the initial prompt on near-silent audio.
    return bool(prompt) and is_prompt_echo(text, prompt)


class WhisperTranscriber:
    # The names on the PC for the prompt ("Cartelle: Tesi, PSS."); None: none.
    vocabulary: Callable[[], str] | None = None
    # While it says so (CLAUDIO is off) what is dropped here is logged hidden. This is never the
    # turn-on: that one is what gets through. None: always logged.
    hides_text: Callable[[], bool] | None = None

    def __init__(self, settings: SttSettings, download_root: Path) -> None:
        self.settings = settings
        self.download_root = download_root
        self.device = "cpu"
        self.model_name = settings.fallback_model
        self._encoded: tuple[np.ndarray, Any] | None = None
        # Whisper's confidence in each segment of the last request (the noise filter).
        self.last_segments: list[Segment] = []
        self.last_prompt = ""  # the request's real pass, for replaying its WAV
        self._reused = 0  # encoder passes saved, for the tests
        self._model = self._load()

    def _open(self, name: str, **kwargs: Any) -> Any:
        from faster_whisper import WhisperModel

        root = str(self.download_root)
        try:  # already downloaded: start offline, without contacting huggingface.co
            return WhisperModel(name, download_root=root, local_files_only=True, **kwargs)
        except FileNotFoundError:
            return WhisperModel(name, download_root=root, **kwargs)

    def _load(self) -> Any:
        s = self.settings
        if s.device in ("auto", "cuda"):
            try:
                register_cuda_dlls()
                import ctranslate2

                if ctranslate2.get_cuda_device_count() == 0:
                    raise RuntimeError("nessuna GPU CUDA trovata")
                model = self._open(s.model, device="cuda", compute_type=s.compute_type)
                self._model, self.device, self.model_name = model, "cuda", s.model
                self._warm_up()  # CUDA kernel problems (e.g. Blackwell) show up here
                return model
            except Exception as exc:
                if s.device == "cuda":
                    raise
                log.warning(
                    "GPU non utilizzabile (%s): passo alla CPU con '%s' %s",
                    exc,
                    s.fallback_model,
                    s.fallback_compute_type,
                )
        model = self._open(
            s.fallback_model,
            device="cpu",
            compute_type=s.fallback_compute_type,
            cpu_threads=s.cpu_threads,
        )
        self._model, self.device, self.model_name = model, "cpu", s.fallback_model
        self._warm_up()
        return model

    def _warm_up(self) -> None:
        start = time.perf_counter()
        noise = (np.random.default_rng(0).standard_normal(16000) * 0.01).astype(np.float32)
        self.transcribe(noise)
        log.debug("STT riscaldato in %.0f ms", (time.perf_counter() - start) * 1000)

    def _logged(self, text: str) -> str:
        return HIDDEN if self.hides_text is not None and self.hides_text() else text

    def prompt(self) -> str:
        """The names on the PC, then the shipped sentences: Whisper keeps the end of a long
        prompt, and the sentences say how JenJo talks."""
        names = self.vocabulary() if self.vocabulary is not None else ""
        return " ".join(p for p in (names, self.settings.initial_prompt) if p)

    def _encode(self, features: np.ndarray) -> Any:
        """Whisper's encoder, answering from what listen_again() already encoded."""
        if self._encoded is not None:
            segment, encoded = self._encoded
            self._encoded = None
            if features.shape == segment.shape and np.array_equal(features, segment):
                self._reused += 1
                return encoded
        return type(self._model).encode(self._model, features)

    def transcribe(self, audio: np.ndarray, *, quick: bool = False) -> str:
        """`quick`: the wake word's second chance, or a phrase said over the reply. A short
        look (QUICK_NEW_TOKENS) that leaves alone what the next request is judged on (its
        segments). Its prompt is the shipped sentences,
        without the names on the PC (noise must not become a folder name): a lone "Claudio"
        with its "C" cut by the headset's gate came out "Grazie." without a prompt and
        "Claudio" with it (30/10: 11 of 11 against 6 of 11, no name in 90 phrases without
        one); the voice meter keeps silence and clicks from Whisper."""
        if audio.size == 0 or float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) < MIN_RMS:
            if not quick:
                self.last_segments, self.last_prompt = [], ""
            return ""  # silence: Whisper would invent something ("Buon appetito!")
        audio = audio.astype(np.float32, copy=False)
        prompt = self.settings.initial_prompt if quick else self.prompt()
        text, kept = self._decode(audio, prompt, quick)
        if text and prompt and is_prompt_echo(text, prompt):
            # Whisper repeats its prompt on unclear audio, but "alza il volume" is also a
            # request: asked again without the prompt, Whisper says what was really there.
            log.info(
                "Trascrizione uguale a un pezzo del prompt, la rifaccio senza: %r",
                self._logged(text),
            )
            prompt = ""  # the pass that produced the text
            text, kept = self._decode(audio, prompt, quick)
        if not quick:  # the second chance must not overwrite the request's own figures
            self.last_segments, self.last_prompt = kept, prompt
        if text and is_hallucination(text):
            log.info("Trascrizione scartata (allucinazione nota): %r", self._logged(text))
            return ""
        return text

    def _decode(self, audio: np.ndarray, prompt: str, quick: bool) -> tuple[str, list[Segment]]:
        """One Whisper pass: the text of the segments that are not silence or a loop, and
        Whisper's confidence in each."""
        s = self.settings
        reuse = self._encoded is not None
        if reuse:
            self._model.encode = self._encode
        try:
            segments, _info = self._model.transcribe(
                audio,
                language=WHISPER_LANGUAGE,
                beam_size=s.beam_size,
                initial_prompt=prompt or None,
                vad_filter=False,  # Silero already did it upstream
                condition_on_previous_text=False,
                without_timestamps=True,
                temperature=0.0,
                max_new_tokens=QUICK_NEW_TOKENS if quick else MAX_NEW_TOKENS,
            )
            parts = []
            kept: list[Segment] = []
            for seg in segments:  # a generator: Whisper runs while it is consumed
                if seg.no_speech_prob > s.no_speech_threshold and seg.avg_logprob < -1.0:
                    log.info(
                        "Trascrizione scartata (sembra silenzio, no_speech %.2f, logprob %.2f): %r",
                        seg.no_speech_prob,
                        seg.avg_logprob,
                        self._logged(seg.text[:80]),
                    )
                    continue
                if seg.compression_ratio > LOOP_RATIO:
                    log.info("Trascrizione scartata (ripetizione): %r", self._logged(seg.text[:80]))
                    continue
                parts.append(seg.text.strip())
                kept.append(Segment(float(seg.no_speech_prob), float(seg.avg_logprob)))
        finally:
            if reuse:
                del self._model.encode
            self._encoded = None
        return " ".join(p for p in parts if p).strip(), kept

    def candidates(self, audio: np.ndarray, n: int = CANDIDATES) -> list[str]:
        """The `n` best transcriptions of one beam search, most likely first: the "closed
        list" picks the first that is a known command ("apri la calcolatrice" may be only the
        third guess when said fast). Same prompt and filters as transcribe(); at most 30 s."""
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import pad_or_trim

        if audio.size == 0 or float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) < MIN_RMS:
            return []
        audio = audio.astype(np.float32, copy=False)
        model = self._model
        prompt_text = self.prompt()
        tokenizer = Tokenizer(
            model.hf_tokenizer,
            model.model.is_multilingual,
            task="transcribe",
            language=WHISPER_LANGUAGE,
        )
        previous = tokenizer.encode(" " + prompt_text.strip()) if prompt_text else []
        prompt = model.get_prompt(tokenizer, previous, without_timestamps=True)
        features = pad_or_trim(model.feature_extractor(audio)[..., :-1])
        result = model.model.generate(
            model.encode(features),
            [prompt],
            beam_size=max(n, self.settings.beam_size),
            num_hypotheses=n,
            max_length=len(prompt) + MAX_NEW_TOKENS,
            suppress_blank=True,
        )[0]
        texts: list[str] = []
        for ids in result.sequences_ids:
            text = tokenizer.decode([t for t in ids if t < tokenizer.eot]).strip()
            if text and text not in texts and not is_hallucination(text, prompt_text):
                texts.append(text)
        return texts

    def listen_again(self, audio: np.ndarray, prompt: str) -> SecondHearing:
        """One more transcription of `audio` with `prompt` in place of the usual one, and the
        scorer of any text on the same encoder pass (the command recovery, stt/recovery.py).
        The request's own figures (last_segments) stay as they are. No check
        for a piece of the prompt: its phrases are what may have been said."""
        if not audible(audio):
            return SecondHearing("", lambda texts: [NO_FIT] * len(texts))
        audio = audio.astype(np.float32, copy=False)
        window = self._window(audio)
        encoded = type(self._model).encode(self._model, window)
        self._encoded = (window, encoded)  # _decode() uses it instead of encoding again
        text, _ = self._decode(audio, prompt, quick=False)
        if text and is_hallucination(text):
            log.info("Seconda trascrizione scartata (allucinazione nota): %r", self._logged(text))
            text = ""
        return SecondHearing(text, functools.partial(self._fit, encoded))

    def _window(self, audio: np.ndarray) -> np.ndarray:
        """The first 30 s of log-Mel as transcribe() encodes them (its last frame dropped,
        padded to 3000 frames), so that one encoding stands for both."""
        from faster_whisper.audio import pad_or_trim

        extractor = self._model.feature_extractor
        features = extractor(audio)
        return pad_or_trim(features[:, : min(extractor.nb_max_frames, features.shape[-1] - 1)])

    def _fit(self, encoded: Any, texts: Sequence[str]) -> list[Fit]:
        """How likely Whisper finds each text on the encoded audio with no prompt."""
        from faster_whisper.tokenizer import Tokenizer

        model = self._model
        tokenizer = Tokenizer(
            model.hf_tokenizer,
            model.model.is_multilingual,
            task="transcribe",
            language=WHISPER_LANGUAGE,
        )
        start = [*tokenizer.sot_sequence, tokenizer.no_timestamps]
        return [self._forced(encoded, start, tokenizer, text) for text in texts]

    def _forced(self, encoded: Any, start: list[int], tokenizer: Any, text: str) -> Fit:
        """`text` forced as the start of the transcription, then one free step: the
        log-probability of each word piece and of the end of the text, over the whole
        vocabulary. align() would be cheaper, but it leaves the end of the text out, and a
        command that is only the start of what was said ("Apri Chrome." for "Apri Chrome e
        cerca...") would then look likelier than all of it."""
        tokens = tokenizer.encode(" " + text.strip()) if text.strip() else []
        if not tokens or len(tokens) > FORCED_MAX_TOKENS:
            return NO_FIT
        result = self._model.model.generate(
            encoded,
            [start + tokens],
            beam_size=1,
            # ctranslate2 4.8.3 counts a forced piece about twice: 2L + 3 gives the L + 1 steps
            # needed for any L (measured; 2L + 2 is one short for a single piece, "No")
            max_length=2 * len(tokens) + 3,
            return_logits_vocab=True,
            suppress_blank=False,  # the raw probabilities: nothing masked
            suppress_tokens=[],
        )[0]
        targets = [*tokens, tokenizer.eot]
        steps = list(result.logits[0])[: len(targets)]
        if len(steps) < len(targets):
            raise RuntimeError(f"Whisper: {len(steps)} passi per {len(targets)} pezzi forzati")
        vocabulary = tokenizer.tokenizer.get_vocab_size()  # the GPU model pads its own
        logits = np.stack([host_logits(step)[:vocabulary] for step in steps])
        picked = log_softmax(logits)[np.arange(len(targets)), targets]
        return Fit(float(picked.sum()), float(picked.mean()))


def audible(audio: np.ndarray) -> bool:
    """Louder than MIN_RMS: something to transcribe or to score."""
    return audio.size > 0 and float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) >= MIN_RMS


def host_logits(step: Any) -> np.ndarray:
    """One decoding step's logits as float64 numpy: a ctranslate2 StorageView is copied off
    the GPU and out of float16 first (numpy reads neither)."""
    if not isinstance(step, np.ndarray):
        import ctranslate2

        step = step.to_device(ctranslate2.Device.cpu).to(ctranslate2.DataType.float32)
    return np.asarray(step, dtype=np.float64)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
