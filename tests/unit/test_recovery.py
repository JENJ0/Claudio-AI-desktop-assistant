"""IT-1 2.1: a short phrase no command understands gets Whisper's second opinion, local and
free ("Scaniti." for "spegniti", "Chi ce l'ha fatta?" for "chiudi la chat", 30/10): Whisper
hears it again told about the commands, and the command it proposes is judged on the sound
alone. Done at once, asked about, or left to go on as before."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence

import numpy as np
import pytest

from claudio.config import RecoverySettings
from claudio.intake import CONVERSATION, Decision, Guess, Heard, Kind, Segment, Verdict
from claudio.metrics import Interaction
from claudio.router.actions import MEDIA_KEYS
from claudio.router.matcher import CommandMatcher, load_commands
from claudio.stt.recovery import (
    NO_FIT,
    CommandRecovery,
    Fit,
    SecondHearing,
    command_prompt,
    judge,
)
from tests.conftest import COMMANDS_FILE
from tests.unit.test_conversation import build as build_conversation
from tests.unit.test_conversation import talk
from tests.unit.test_guess import FakeGuesser

SETTINGS = RecoverySettings(min_mean=-2.0, ask_gap=-2.0, act_gap=2.0)
AUDIO = np.full(16000, 0.1, dtype=np.float32)


class FakeWhisper:
    """Hears `proposal` when told about the commands; finds each text as likely as `fits`
    says (log-probability in all; -0.5 per text otherwise)."""

    def __init__(self, proposal: str, fits: dict[str, float] | None = None) -> None:
        self.proposal, self.fits = proposal, fits or {}
        self.prompts: list[str] = []
        self.scored: list[list[str]] = []

    def listen_again(self, audio: np.ndarray, prompt: str, language: str) -> SecondHearing:
        self.prompts.append(prompt)
        return SecondHearing(self.proposal, self._fit)

    def _fit(self, texts: Sequence[str]) -> list[Fit]:
        self.scored.append(list(texts))
        return [Fit(self.fits.get(t, -0.5), -0.5) for t in texts]


def recovery(whisper: FakeWhisper, matcher: CommandMatcher, **settings: float) -> CommandRecovery:
    return CommandRecovery(
        whisper, matcher, load_commands(COMMANDS_FILE), SETTINGS.model_copy(update=settings)
    )


# --- the prompt and the verdict ---------------------------------------------------------


def test_the_prompt_has_the_first_phrase_of_every_command_worth_proposing() -> None:
    commands = load_commands(COMMANDS_FILE)
    prompt = command_prompt(commands)
    assert prompt.startswith("Claudio, Apri chrome. Apri la cartella download.")
    assert "Chiudi la chat." in prompt and "Spegniti." in prompt and prompt.endswith("Chiuditi.")
    # Awake and on, "svegliati" and "accenditi" change nothing: never proposed.
    assert "Svegliati." not in prompt and "Accenditi." not in prompt
    assert prompt.count(".") == len(commands) - 2


@pytest.mark.parametrize(
    ("first", "proposal", "action", "expected"),
    [
        (Fit(-9.0, -3.0), Fit(-6.0, -1.0), "chat", Verdict.ACT),  # 3 likelier: done
        (Fit(-9.0, -3.0), Fit(-8.0, -1.5), "chat", Verdict.ASK),  # about as likely: asked
        (Fit(-9.0, -3.0), Fit(-10.5, -1.5), "chat", Verdict.ASK),
        (Fit(-9.0, -3.0), Fit(-12.0, -1.8), "chat", Verdict.DROP),  # less likely
        (Fit(-9.0, -3.0), Fit(-6.0, -2.5), "chat", Verdict.DROP),  # heard badly in itself
        (Fit(-9.0, -3.0), Fit(-2.0, -0.5), "system", Verdict.ASK),  # spegniti: always asked
        (Fit(-9.0, -3.0), Fit(-10.0, -1.5), "stop", Verdict.ACT),  # basta: never asked
        (Fit(-9.0, -3.0), Fit(-12.0, -1.5), "stop", Verdict.DROP),
        (NO_FIT, Fit(-6.0, -1.0), "chat", Verdict.DROP),  # nothing to compare with
        (NO_FIT, Fit(-6.0, -1.0), "stop", Verdict.DROP),
        (Fit(-9.0, -3.0), NO_FIT, "chat", Verdict.DROP),
    ],
)
def test_judge(first: Fit, proposal: Fit, action: str, expected: Verdict) -> None:
    assert judge(first, proposal, action, SETTINGS) is expected


@pytest.mark.parametrize(
    ("first", "proposal", "action", "expected"),
    [  # measured on JenJo's recordings, 30/10 (large-v3 on the GPU)
        (Fit(-12.13, -3.03), Fit(-6.11, -1.22), "system", Verdict.ASK),  # Scaniti -> Spegniti
        (Fit(-4.38, -0.73), Fit(-9.27, -1.67), "chat", Verdict.ASK),  # Chi lascia atto? -> chat
        (Fit(-6.51, -0.65), Fit(-10.73, -1.34), "media_key", Verdict.ASK),  # -> abbassa di 10
        (Fit(-1.43, -0.24), Fit(-12.14, -2.02), "chat", Verdict.DROP),  # "Grazie a te."
        (Fit(-5.47, -1.37), Fit(-10.74, -1.79), "media_key", Verdict.DROP),  # "Ehem."
        (Fit(-6.26, -0.89), Fit(-14.17, -2.36), "chat", Verdict.DROP),
    ],
)
def test_the_shipped_thresholds(first: Fit, proposal: Fit, action: str, expected: Verdict) -> None:
    assert judge(first, proposal, action, RecoverySettings()) is expected


@pytest.mark.parametrize(
    ("text", "said", "question"),
    [
        ("Claudio Chiuditi.", "Chiuditi.", "Hai detto: chiuditi?"),
        ("Ehi Claudio, apri la chat.", "apri la chat.", "Hai detto: apri la chat?"),
        ("Chiudi la chat.", "Chiudi la chat.", "Hai detto: chiudi la chat?"),
        ("Claudiano, apri Chrome.", "Claudiano, apri Chrome.", None),  # not the name
        ("PC, apri Chrome.", "PC, apri Chrome.", "Hai detto: PC, apri Chrome?"),
    ],
)
def test_the_request_and_its_question(text: str, said: str, question: str | None) -> None:
    from claudio.router.responder import DID_YOU_SAY, quoted
    from claudio.stt.recovery import without_name

    assert without_name(text) == said
    if question is not None:
        assert DID_YOU_SAY.format(text=quoted(said)) == question


def test_ask_gap_cannot_be_above_act_gap() -> None:
    with pytest.raises(ValueError):
        RecoverySettings(ask_gap=3.0, act_gap=1.0)


# --- what is worth a try ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "language", "wanted"),
    [
        ("Scaniti.", "it", True),
        ("Chi ce l'ha fatta?", "it", True),  # a question too
        ("Claudio, chi ce l'ha fatta?", "it", True),
        ("Apri Chrome.", "it", False),  # understood already
        ("Claudio.", "it", False),  # only the name: the request is still to come
        ("", "it", False),
        ("Raccontami una storia lunga di draghi e cavalieri.", "it", False),  # for Claude
    ],
)
def test_wants(matcher: CommandMatcher, text: str, language: str, wanted: bool) -> None:
    assert recovery(FakeWhisper(""), matcher).wants(text, language) is wanted


def test_off_in_the_settings(matcher: CommandMatcher) -> None:
    assert not recovery(FakeWhisper(""), matcher, enabled=False).wants("Scaniti.", "it")


# --- the second opinion -------------------------------------------------------------------


def test_a_clearly_likelier_command_is_done(matcher: CommandMatcher) -> None:
    whisper = FakeWhisper("Chiudi la chat.", {"Chi ce l'ha fatta?": -12.0, "Chiudi la chat.": -4.0})
    found = recovery(whisper, matcher)(AUDIO, "Chi ce l'ha fatta?", "it")
    assert found is not None and found.verdict is Verdict.ACT and found.text == "Chiudi la chat."
    assert whisper.prompts[0].startswith("Claudio, Apri chrome.")
    assert whisper.scored == [["Chi ce l'ha fatta?", "Chiudi la chat."]]
    assert found.info == {
        "recovery": "Chiudi la chat.",
        "recovery_command": "close_chat",
        "recovery_fit": [-12.0, -4.0],
        "recovery_mean": -0.5,
        "recovery_verdict": "act",
    }


def test_switching_off_is_asked_however_likely(matcher: CommandMatcher) -> None:
    whisper = FakeWhisper("Spegniti.", {"Scaniti.": -11.7, "Spegniti.": -5.9})  # 30/10 14:01
    found = recovery(whisper, matcher)(AUDIO, "Scaniti.", "it")
    assert found is not None and found.verdict is Verdict.ASK
    assert found.info["recovery_command"] == "off"


@pytest.mark.parametrize("heard", ["Chi gliel'ha fatta?", "Svegliati.", "Dai, accenditi."])
def test_no_command_worth_proposing_is_no_proposal(matcher: CommandMatcher, heard: str) -> None:
    whisper = FakeWhisper(heard)
    found = recovery(whisper, matcher)(AUDIO, "Grazie a te.", "it")
    assert found is not None and found.verdict is Verdict.DROP
    assert found.info == {"recovery": heard} and whisper.scored == []


def test_not_worth_a_try_costs_nothing(matcher: CommandMatcher) -> None:
    whisper = FakeWhisper("Apri Chrome.")
    assert recovery(whisper, matcher)(AUDIO, "Apri Chrome.", "it") is None
    assert whisper.prompts == []


def test_the_question_leaves_a_clipped_name_out(matcher: CommandMatcher) -> None:
    proposal = "L'audio metti in pausa la musica."
    whisper = FakeWhisper(proposal, {"Lotti in posa la mosca.": -9.0, proposal: -10.0})
    found = recovery(whisper, matcher)(AUDIO, "Lotti in posa la mosca.", "it")
    assert found is not None and found.verdict is Verdict.ASK
    assert found.text == "metti in pausa la musica"  # not "Hai detto: L'audio metti..."
    assert found.info["recovery"] == proposal and found.info["recovery_command"] == (
        "media_play_pause"
    )


def test_no_second_opinion_when_whisper_fell_back_to_the_cpu(matcher: CommandMatcher) -> None:
    from claudio.__main__ import build_recovery
    from claudio.config import load_settings

    class Whisper(FakeWhisper):
        def __init__(self, device: str, model_name: str) -> None:
            super().__init__("")
            self.device, self.model_name = device, model_name

    settings = load_settings()
    model = settings.stt.model
    assert build_recovery(settings, Whisper("cuda", model), matcher) is not None
    assert build_recovery(settings, Whisper("cpu", settings.stt.fallback_model), matcher) is None
    assert build_recovery(settings, Whisper("cuda", "tiny"), matcher) is None  # not tuned on it


# --- Whisper's side: one more transcription, and forced scores with the end of the text ---


TOY_WORDS = ["chiudi", "la", "chat", "grazie", "a", "tutti"]
TOY_SPECIALS = ["<|endoftext|>", "<|startoftranscript|>", "<|it|>", "<|en|>", "<|transcribe|>",
                "<|translate|>", "<|notimestamps|>", "<|startofprev|>", "<|nospeech|>"]  # fmt: skip
PADDING = 3  # the GPU model has a few more logits than words: they must not count


class ForcedCt2:
    """generate() with forced word pieces: each step's logits favour the piece that comes next
    (2.0 against 0.0), plus PADDING logits beyond the vocabulary that would change the result."""

    is_multilingual = True

    def __init__(self, hf) -> None:
        self.hf, self.calls = hf, []

    def generate(self, features, prompts, **kwargs):
        self.calls.append((features, prompts[0], kwargs))
        size = self.hf.get_vocab_size()
        start = 4  # sot, language, task, no timestamps
        targets = [*prompts[0][start:], self.hf.token_to_id("<|endoftext|>")]
        steps = []
        for target in targets:
            logits = np.zeros(size + PADDING)
            logits[target], logits[size:] = 2.0, 50.0
            steps.append(logits)
        return [type("Result", (), {"logits": [steps]})()]


class ForcedModel:
    def __init__(self) -> None:
        from faster_whisper.feature_extractor import FeatureExtractor
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        vocab = {w: i for i, w in enumerate(dict.fromkeys(TOY_WORDS + TOY_SPECIALS + ["[UNK]"]))}
        self.hf_tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        self.hf_tokenizer.pre_tokenizer = Whitespace()
        self.model = ForcedCt2(self.hf_tokenizer)
        self.feature_extractor = FeatureExtractor()
        self.encodes = 0

    def encode(self, features):
        assert features.shape[-1] == 3000  # padded to Whisper's 30 s window
        self.encodes += 1
        return features


def forced_stt():
    from claudio.config import SttSettings
    from claudio.stt.transcriber import WhisperTranscriber

    stt = WhisperTranscriber.__new__(WhisperTranscriber)
    stt.settings = SttSettings()
    stt.language, stt._encoded, stt._reused, stt.last_segments = "it", None, 0, []
    stt._model = ForcedModel()
    return stt


SPEECH = (np.sin(np.arange(16000) / 5) * 0.2).astype(np.float32)


def test_forced_scores_count_the_end_of_the_text_over_the_vocabulary_only() -> None:
    stt = forced_stt()
    size = stt._model.hf_tokenizer.get_vocab_size()
    each = 2.0 - np.log(np.exp(2.0) + size - 1)  # log-softmax of the favoured piece
    fits = stt._fit("encoded", ["chiudi la chat", "  "])
    assert fits[0].total == pytest.approx(4 * each) and fits[0].mean == pytest.approx(each)
    assert fits[1] == NO_FIT  # nothing to score, no call
    ((features, prompt, kwargs),) = stt._model.model.calls
    hf = stt._model.hf_tokenizer
    assert features == "encoded" and prompt[3] == hf.token_to_id("<|notimestamps|>")
    assert prompt[4:] == [hf.token_to_id(w) for w in ("chiudi", "la", "chat")]
    assert kwargs["max_length"] == 9 and kwargs["return_logits_vocab"] is True
    # One piece ("No"): ctranslate2 needs 2 * 1 + 3 to give both steps (measured on the GPU).
    (lone,) = stt._fit("encoded", ["grazie"])
    assert lone.total == pytest.approx(2 * each)
    assert stt._model.model.calls[-1][2]["max_length"] == 5


def test_listen_again_shares_one_encoder_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    stt = forced_stt()
    seen: list[tuple[str, object]] = []

    def decode(audio, prompt, quick):
        assert stt._encoded is not None  # transcribe() takes this encoding as its own
        seen.append((prompt, stt._encoded[1]))
        return "chiudi la chat", []

    monkeypatch.setattr(stt, "_decode", decode)
    heard = stt.listen_again(SPEECH, "Claudio, Chiudi la chat.", "it")
    assert heard.text == "chiudi la chat" and seen[0][0] == "Claudio, Chiudi la chat."
    heard.fit(["chiudi la chat", "grazie a tutti"])
    assert stt._model.encodes == 1
    assert all(call[0] is seen[0][1] for call in stt._model.model.calls)


def test_listen_again_on_silence_and_hallucinations(monkeypatch: pytest.MonkeyPatch) -> None:
    stt = forced_stt()
    monkeypatch.setattr(stt, "_decode", lambda *args, **kwargs: ("Grazie a tutti.", []))
    assert stt.listen_again(SPEECH, "Claudio, Chiudi la chat.", "it").text == ""
    silent = stt.listen_again(np.zeros(16000, np.float32), "Claudio, Chiudi la chat.", "it")
    assert silent.text == "" and silent.fit(["chiudi la chat"]) == [NO_FIT]
    assert stt._model.encodes == 1  # silence is not even encoded


# --- the assistant ------------------------------------------------------------------------


def window(fake_vad, vad_settings, matcher, texts: list[str], whisper: FakeWhisper):
    """The conversation window after "che ore sono", with the recovery and Claude's guess."""
    assistant, speaker, rec, r = build_conversation(fake_vad, vad_settings, matcher, texts)
    assistant.command_recovery = recovery(whisper, matcher)
    guesser = FakeGuesser(Guess("", 0.0))
    r.fallback_guess = guesser
    assistant.guess = r.guess
    return assistant, speaker, rec, r, guesser


async def test_a_misheard_command_in_the_window_is_done_for_free(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    whisper = FakeWhisper("Canzone successiva.", {"Canzone su sei va.": -15.0})
    assistant, speaker, rec, _, guesser = window(
        fake_vad, vad_settings, matcher, ["che ore sono", "Canzone su sei va."], whisper
    )
    await talk(assistant, make_frames, speaker, requests=2)
    assert rec.keys == [MEDIA_KEYS["next_track"]] and guesser.calls == []  # no Claude
    row = assistant.history[1]
    assert row["outcome"] == "ok" and row["command"] == "media_next"
    assert row["recovered_from"] == "Canzone su sei va." and row["text"] == "Canzone successiva."
    assert row["recovery_verdict"] == "act" and row["recovery_ms"] >= 0


async def test_a_recovered_spegniti_is_asked_then_done_on_yes(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    whisper = FakeWhisper("Spegniti.", {"Scaniti.": -11.7, "Spegniti.": -5.9})
    assistant, speaker, _, r, guesser = window(
        fake_vad, vad_settings, matcher, ["che ore sono", "Scaniti.", "sì"], whisper
    )
    await talk(assistant, make_frames, speaker, requests=3)
    assert speaker.spoken[1:] == [
        "Hai detto: spegniti?",
        "Mi spengo. Per riaccendermi di' Claudio, accenditi.",
    ]
    assert [row["outcome"] for row in assistant.history][:3] == ["ok", "ask", "ok"]
    assert r.actions.system.off and guesser.calls == []
    row = assistant.history[1]  # told apart from Haiku's guess, in the session log too
    assert row["recovered_from"] == "Scaniti." and row["recovery_verdict"] == "ask"


async def test_a_proposal_that_sounds_worse_goes_on_to_claudes_guess(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    whisper = FakeWhisper("Canzone successiva.", {"Pasta.": -2.0, "Canzone successiva.": -9.0})
    assistant, speaker, rec, _, guesser = window(
        fake_vad, vad_settings, matcher, ["che ore sono", "Pasta."], whisper
    )
    await talk(assistant, make_frames, speaker, requests=2)
    assert rec.keys == [] and len(guesser.calls) == 1  # as before: Claude guesses
    row = assistant.history[1]
    assert row["outcome"] == "conversation_end" and row["recovery_verdict"] == "drop"


async def test_a_request_for_claude_can_be_a_misheard_command(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    # 30/10 14:08: "Chi ce l'ha fatta?" went to Claude; it was "chiudi la chat".
    asked: list[str] = []

    async def claude(text: str, language: str) -> AsyncGenerator[str, None]:
        asked.append(text)
        yield "Non è chiaro cosa intendi, socio."

    whisper = FakeWhisper("Canzone successiva.", {"Chi ce l'ha fatta?": -12.0})
    assistant, speaker, rec, r, _ = window(
        fake_vad, vad_settings, matcher, ["Chi ce l'ha fatta?"], whisper
    )
    r.fallback = claude
    await talk(assistant, make_frames, speaker, requests=1)
    assert rec.keys == [MEDIA_KEYS["next_track"]] and asked == []


async def test_a_failing_second_opinion_goes_on_as_before(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    class Broken(FakeWhisper):
        def listen_again(self, audio: np.ndarray, prompt: str, language: str) -> SecondHearing:
            raise RuntimeError("CUDA esplosa")

    assistant, speaker, _, _, guesser = window(
        fake_vad, vad_settings, matcher, ["che ore sono", "Pasta."], Broken("")
    )
    await talk(assistant, make_frames, speaker, requests=2)
    assert [row["outcome"] for row in assistant.history] == ["ok", "conversation_end"]
    assert len(guesser.calls) == 1


@pytest.mark.parametrize(
    ("decision", "heard", "state"),
    [
        (Decision(Kind.NOT_REQUEST, "Pasta.", "echo"), Heard(CONVERSATION), ""),  # own voice
        (Decision(Kind.NOISE, "Pasta.", "quiet"), Heard(CONVERSATION), ""),  # nobody there
        (Decision(Kind.EMPTY, ""), Heard(CONVERSATION), ""),
        (Decision(Kind.CONTINUE, "Apri Chrome e cerca", "incomplete"), Heard("hey_claudio"), ""),
        (Decision(Kind.ANSWER, "Pasta."), Heard("hey_claudio", speech_dbfs=-40.0), ""),  # the TV
        (Decision(Kind.ANSWER, "Pasta."), Heard("hey_claudio"), "paused"),
        (Decision(Kind.ANSWER, "Pasta."), Heard("hey_claudio"), "off"),
    ],
)
async def test_never_a_second_opinion(
    fake_vad, vad_settings, matcher, decision: Decision, heard: Heard, state: str
) -> None:
    from claudio.config import NoiseSettings

    whisper = FakeWhisper("Canzone successiva.")
    assistant, _, _, r, _ = window(fake_vad, vad_settings, matcher, [], whisper)
    assistant.intake.noise = NoiseSettings()
    if state:
        setattr(r.actions.system, state, True)
    assert not assistant._may_recover(decision, heard, "it")
    assert assistant._may_recover(Decision(Kind.ANSWER, "Pasta."), Heard("x"), "it") is not bool(
        state
    )


async def test_no_second_opinion_on_a_memory_request(
    fake_vad, vad_settings, matcher, tmp_path
) -> None:
    """IT-1 merge: a memory request is understood by the router (memory/voice.py), not by
    the command matcher: "dimenticalo" goes to the memory, never heard again as a command."""
    from claudio.memory.store import MemoryStore
    from claudio.memory.voice import MemoryVoice

    whisper = FakeWhisper("Chiuditi.")
    assistant, _, _, r, _ = window(fake_vad, vad_settings, matcher, [], whisper)
    r.memory = MemoryVoice(MemoryStore(tmp_path / "memory.db"))
    for said in ("Dimenticalo.", "Cosa ricordi di Anna?"):
        assert not assistant._may_recover(Decision(Kind.ANSWER, said), Heard("x"), "it")
    assert assistant._may_recover(Decision(Kind.ANSWER, "Pasta."), Heard("x"), "it")


async def test_no_second_opinion_while_a_question_waits(fake_vad, vad_settings, matcher) -> None:
    assistant, _, _, r, _ = window(fake_vad, vad_settings, matcher, [], FakeWhisper(""))
    r.propose("spegniti", "it")  # "Hai detto: spegniti?"
    assert not assistant._may_recover(Decision(Kind.ANSWER, "Pasta."), Heard("x"), "it")


async def test_no_second_opinion_on_the_answer_to_claudes_question(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    async def claude(text: str, language: str) -> AsyncGenerator[str, None]:
        yield "Che piatto vuoi stasera?"

    whisper = FakeWhisper("Basta.", {"Pasta.": -9.0, "Basta.": -2.0})  # a likelier "basta"
    question = "cosa mi consigli di mangiare stasera a cena"  # too long to be a command
    assistant, speaker, _, r, _ = window(
        fake_vad, vad_settings, matcher, [question, "Pasta."], whisper
    )
    r.fallback = claude
    assistant.intake.claude_min_words = 3  # Claude is on: the window answers what is not a command
    await talk(assistant, make_frames, speaker, requests=2)
    assert whisper.prompts == []  # "Pasta." is the answer: no second opinion, no "basta"
    assert [row["outcome"] for row in assistant.history][:2] == ["ok", "ok"]


async def test_the_answer_to_claudes_question_is_never_stopped_on_a_guess(
    fake_vad, vad_settings, make_frames, matcher
) -> None:
    # The same answer mumbled (Whisper unsure of all of it: a NOISE drop) goes to Haiku's
    # guess: a 0.6 "basta" there is asked, not done (decision 1 is for phrases said to Claudio).
    from claudio.config import NoiseSettings

    async def claude(text: str, language: str) -> AsyncGenerator[str, None]:
        yield "Che piatto vuoi stasera?"

    whisper = FakeWhisper("Basta.", {"Pasta.": -9.0, "Basta.": -2.0})
    question = "cosa mi consigli di mangiare stasera a cena"
    assistant, speaker, _, r, guesser = window(
        fake_vad, vad_settings, matcher, [question, "Pasta."], whisper
    )
    guesser.guess = Guess("basta", 0.6, "stop")
    r.fallback = claude
    assistant.intake.claude_min_words = 3
    assistant.intake.noise = NoiseSettings()
    stt = assistant.transcriber
    assistant.stt_segments = lambda: (
        [Segment(0.6, -1.2)] if len(stt.heard) == 2 else [Segment(0.0, -0.05)]
    )
    await talk(assistant, make_frames, speaker, requests=2)
    assert whisper.prompts == [] and len(guesser.calls) == 1
    row = assistant.history[1]
    assert row["outcome"] == "ask" and row["guess_verdict"] == "ask"
    assert speaker.spoken[-1] == "Hai detto: basta?"


async def test_the_helper_for_other_phrases(fake_vad, vad_settings, matcher) -> None:
    # The phrase said over the reply, once put back together (IT-1 2.2), uses the same door.
    whisper = FakeWhisper("Chiudi la chat.", {"Chi ce l'ha fatta?": -12.0, "Chiudi la chat.": -4.0})
    assistant, _, _, _, _ = window(fake_vad, vad_settings, matcher, [], whisper)
    inter = Interaction(clock=assistant.clock)
    decision = await assistant._recover_command(AUDIO, "Chi ce l'ha fatta?", "it", inter)
    assert decision == Decision(Kind.ANSWER, "Chiudi la chat.", "recovered")
    assert inter.info["recovered_from"] == "Chi ce l'ha fatta?"
    assert await assistant._recover_command(AUDIO, "Apri Chrome.", "it", inter) is None
