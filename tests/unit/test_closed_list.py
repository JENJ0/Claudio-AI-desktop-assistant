"""The closed list: Whisper's n-best transcriptions, and the first that is a known command."""

from __future__ import annotations

import numpy as np
import pytest

WORDS = ["apri", "la", "calcolatrice", "cue", "the", "calculator", "comandi", "grazie", "a",
         "tutti", "per", "la", "visione"]  # fmt: skip
SPECIALS = ["<|endoftext|>", "<|startoftranscript|>", "<|it|>", "<|en|>", "<|transcribe|>",
            "<|translate|>", "<|notimestamps|>", "<|startofprev|>", "<|nospeech|>"]  # fmt: skip


def toy_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {w: i for i, w in enumerate(dict.fromkeys(WORDS + SPECIALS + ["[UNK]"]))}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return tok


class FakeCt2:
    is_multilingual = True

    def __init__(self, hypotheses: list[str], hf) -> None:
        self.hf, self.hypotheses = hf, hypotheses

    def generate(self, features, prompts, **kwargs):
        self.prompts, self.kwargs = prompts, kwargs
        eot = self.hf.token_to_id("<|endoftext|>")
        ids = [[self.hf.token_to_id(w) for w in h.split()] + [eot] for h in self.hypotheses]
        return [type("Result", (), {"sequences_ids": ids[: kwargs["num_hypotheses"]]})()]


class FakeModel:
    max_length = 448

    def __init__(self, hypotheses: list[str]) -> None:
        from faster_whisper.feature_extractor import FeatureExtractor
        from faster_whisper.transcribe import WhisperModel

        self.hf_tokenizer = toy_tokenizer()
        self.model = FakeCt2(hypotheses, self.hf_tokenizer)
        self.feature_extractor = FeatureExtractor()
        self.get_prompt = WhisperModel.get_prompt.__get__(self)

    def encode(self, features):
        assert features.shape[-1] == 3000  # padded to Whisper's 30 s window
        return features


def make_stt(hypotheses: list[str], prompt: str = "comandi"):
    from claudio.config import SttSettings
    from claudio.stt.transcriber import WhisperTranscriber

    stt = WhisperTranscriber.__new__(WhisperTranscriber)
    stt.settings = SttSettings(languages=["it"], initial_prompt=prompt)
    stt.language, stt.language_confidence, stt._encoded, stt._reused = "it", 1.0, None, 0
    stt.last_language = "it"
    stt._model = FakeModel(hypotheses)
    return stt


SPEECH = (np.sin(np.arange(16000) / 5) * 0.2).astype(np.float32)


def test_candidates_in_order_without_duplicates() -> None:
    stt = make_stt(["cue the calculator", "apri la calcolatrice", "cue the calculator"])
    assert stt.candidates(SPEECH, n=3) == ["cue the calculator", "apri la calcolatrice"]
    kwargs = stt._model.model.kwargs
    assert kwargs["num_hypotheses"] == 3 and kwargs["beam_size"] >= 3


def test_candidates_carry_the_prompt_and_language() -> None:
    stt = make_stt(["apri la calcolatrice"])
    stt.candidates(SPEECH)
    hf = stt._model.hf_tokenizer
    prompt = stt._model.model.prompts[0]
    assert prompt[0] == hf.token_to_id("<|startofprev|>")
    assert hf.token_to_id("comandi") in prompt and hf.token_to_id("<|it|>") in prompt
    assert prompt[-1] == hf.token_to_id("<|notimestamps|>")


def test_candidates_drop_hallucinations_and_silence() -> None:
    stt = make_stt(["grazie per la visione", "apri la calcolatrice"])
    assert stt.candidates(SPEECH) == ["apri la calcolatrice"]
    assert stt.candidates(np.zeros(16000, np.float32)) == []


@pytest.mark.parametrize(
    ("candidates", "expected"),
    [
        (["cue the calculator", "apri la calcolatrice"], "apri la calcolatrice"),
        (["cue the calculator", "grazie a tutti"], "cue the calculator"),  # none known: the best
        ([], ""),
    ],
)
def test_first_known(candidates: list[str], expected: str) -> None:
    from claudio.stt.transcriber import first_known

    assert first_known(candidates, lambda text: "apri" in text) == expected
