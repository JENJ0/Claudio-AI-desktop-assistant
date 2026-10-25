import numpy as np
import pytest

from claudio.tts.kokoro import (
    MAX_TOKENS,
    KokoroSynth,
    kokoro_files,
    kokoro_speed,
    to_tokens,
    trim_silence,
)

VOCAB = {"a": 1, "p": 2, "r": 3, "o": 4, " ": 5, ".": 6, "ˈ": 7}


def test_tokens_skip_unknown_symbols_and_respect_the_limit() -> None:
    assert to_tokens("ˈapro.", VOCAB) == [7, 1, 2, 3, 4, 6]
    assert to_tokens("a?ɐ", VOCAB) == [1]
    assert len(to_tokens("a" * (MAX_TOKENS + 20), VOCAB)) == MAX_TOKENS


def test_trim_keeps_the_sentence_and_a_short_tail() -> None:
    sr = 24000
    tone = (np.sin(np.arange(sr // 2) / 3) * 0.5).astype(np.float32)
    audio = np.concatenate((np.zeros(sr // 2), tone, np.zeros(sr))).astype(np.float32)
    trimmed = trim_silence(audio, sr)
    assert 0.5 + 0.12 <= len(trimmed) / sr <= 0.5 + 0.12 + 0.03 + 0.01
    assert trim_silence(np.zeros(sr, dtype=np.float32), sr).size == 0
    assert trim_silence(np.zeros(10, dtype=np.float32), sr).size == 10  # too short to judge


def test_speed_follows_length_scale() -> None:
    assert kokoro_speed(1.0) == 1.0
    assert kokoro_speed(0.8) == pytest.approx(1.25)
    assert kokoro_speed(0.1) == 2.0 and kokoro_speed(5) == 0.5


def test_missing_files_say_how_to_download(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="download_models"):
        kokoro_files(tmp_path)


class FakeSession:
    def __init__(self) -> None:
        self.inputs: dict = {}

    def run(self, outputs, inputs):
        self.inputs = inputs
        tone = np.sin(np.arange(12000) / 3).astype(np.float32) * 0.8
        return [
            np.concatenate((np.zeros(6000, np.float32), tone, np.zeros(12000, np.float32)))[None]
        ]


class FakePhonemizer:
    def phonemize(self, voice: str, text: str) -> list[list[str]]:
        assert voice == "it"
        return [list("ˈapɾo"), list("pʲa.")]


def test_synthesize_pads_tokens_and_picks_the_style_by_length() -> None:
    synth = KokoroSynth.__new__(KokoroSynth)
    synth.vocab, synth.speed = VOCAB, 1.25
    synth._styles = np.arange(MAX_TOKENS * 256, dtype=np.float32).reshape(MAX_TOKENS, 1, 256)
    synth._session, synth._phonemizer = FakeSession(), FakePhonemizer()

    assert synth.phonemes("Apro. Pia.") == "ˈapro pa."  # piper's ɾ and ʲ fixed
    audio = synth.synthesize("Apro. Pia.")
    tokens = synth._session.inputs["tokens"]
    assert tokens.tolist() == [[0, 7, 1, 2, 3, 4, 5, 2, 1, 6, 0]]
    assert synth._session.inputs["style"][0, 0] == 8 * 256  # 9 tokens -> row 8
    assert synth._session.inputs["speed"].tolist() == [1.25]
    assert audio.dtype == np.int16 and len(audio) < 30000  # silence trimmed


def test_affricates_are_tied_like_kokoro_was_trained() -> None:
    class Phonemizer:
        def phonemize(self, voice: str, text: str) -> list[list[str]]:
            return [list("kalkolatɾˈitʃe"), list("ˈɔdʒːɪ pˈitsːa")]

    synth = KokoroSynth.__new__(KokoroSynth)
    synth._phonemizer = Phonemizer()
    assert synth.phonemes("Calcolatrice. Oggi pizza") == "kalkolatrˈiʧe ˈɔʤːɪ pˈiʦːa"


def test_claudio_falls_back_to_piper_when_kokoro_is_missing(monkeypatch, tmp_path, caplog) -> None:
    import claudio.tts.speaker as speaker
    from claudio.__main__ import build_synth
    from claudio.config import load_settings

    monkeypatch.setattr(speaker, "voice_path", lambda folder, voice: folder / voice)
    monkeypatch.setattr(speaker, "PiperSynth", lambda path, tts: ("piper", path.name))
    settings = load_settings(models_dir=tmp_path)
    with caplog.at_level("WARNING"):
        synth, name = build_synth(settings)
    assert synth == ("piper", "it_IT-paola-medium") and name == "Piper it_IT-paola-medium"
    assert "Kokoro non disponibile" in caplog.text


def test_kokoro_and_piper_share_one_espeak() -> None:
    import piper.voice as piper_voice

    from claudio.tts.kokoro import piper_phonemizer

    shared = piper_phonemizer()
    assert piper_phonemizer() is shared and piper_voice._ESPEAK_PHONEMIZER is shared


def test_english_keeps_its_own_sounds() -> None:
    class Phonemizer:
        def phonemize(self, voice: str, text: str) -> list[list[str]]:
            assert voice == "en-us"
            return [list("bˈɛɾɚ kˈæts tʃˈiːz pˌiːʲˈɛm")]

    synth = KokoroSynth.__new__(KokoroSynth)
    synth._phonemizer, synth.language = Phonemizer(), "en"
    assert synth.phonemes("Better cats cheese p.m.") == "bˈɛɾɚ kˈæts ʧˈiːz pˌiːˈɛm"


def test_one_kokoro_model_speaks_every_language() -> None:
    from claudio.config import TtsSettings

    synth = KokoroSynth.__new__(KokoroSynth)
    synth._settings = TtsSettings()
    synth._voices = type("Npz", (), {"files": ["im_nicola", "am_michael"], "__getitem__": str})()
    synth._session = object()
    synth._use_voice("it")
    english = synth.for_language("en")
    assert (synth.voice, synth.language) == ("im_nicola", "it")
    assert (english.voice, english.language) == ("am_michael", "en")
    assert english._session is synth._session  # one model in memory
    with pytest.raises(ValueError, match="am_michael"):
        synth._settings = TtsSettings(voices={"en": {"kokoro_voice": "am_nobody"}})
        synth.for_language("en")


def test_every_language_gets_a_voice_or_falls_back(monkeypatch, tmp_path, caplog) -> None:
    import claudio.tts.speaker as speaker
    from claudio.__main__ import build_voices
    from claudio.config import load_settings

    def piper(path, tts):
        if "ryan" not in path.name:
            raise FileNotFoundError(path.name)
        return ("piper", path.name)

    monkeypatch.setattr(speaker, "voice_path", lambda folder, voice: folder / voice)
    monkeypatch.setattr(speaker, "PiperSynth", piper)
    settings = load_settings(models_dir=tmp_path)
    settings.stt.languages = ["it", "en"]  # English is off in claudio.toml for now
    assert build_voices(settings, object()) == {
        "en": (("piper", "en_US-ryan-low"), "en_US-ryan-low")
    }
    settings.tts.voices["en"].voice = "en_US-missing-low"
    with caplog.at_level("WARNING"):
        assert build_voices(settings, object()) == {}
    assert "uso quella principale" in caplog.text
