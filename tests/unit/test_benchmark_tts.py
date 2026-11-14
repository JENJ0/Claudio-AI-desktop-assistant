import sys

import numpy as np

from claudio.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import benchmark_tts as bench


def test_credit_estimate_matches_the_planned_run() -> None:
    chars = sum(len(s) for s in bench.SENTENCES)
    assert bench.credit_estimate(chars * 12 * 2) == 596  # ceil(2904 * 16 / 78)


def test_summary_of_one_value_and_of_many() -> None:
    assert bench.summarize([5.0]) == (5.0, 5.0, 5.0)
    median, p95, top = bench.summarize([float(i) for i in range(1, 21)])
    assert (median, top) == (10.5, 20.0)
    assert 19.0 < p95 <= top
    assert bench.summarize([1.0, 2.0, 3.0])[1] <= 3.0  # never above the max


def test_first_sentence_is_cold_the_others_warm() -> None:
    rows = [bench.Row("jack", i, rep, 100.0 + i, 200.0) for rep in (1, 2) for i in range(4)]
    cold, warm = bench.split_cold_warm(rows)
    assert [r.index for r in cold] == [0, 0]
    assert len(warm) == 6 and all(r.index > 0 for r in warm)


def test_timed_stream_reports_first_chunk_and_whole_audio() -> None:
    def chunks(text: str):
        yield np.ones(3, dtype=np.int16)
        yield np.ones(2, dtype=np.int16)

    first_ms, total_ms, audio = bench.timed_stream(chunks, "x")
    assert 0 <= first_ms <= total_ms
    assert audio.size == 5


def test_benchmark_jack_has_no_take_store(monkeypatch) -> None:
    import claudio.__main__ as entry

    class FakeJack:
        takes = object()

    monkeypatch.setattr(entry, "make_jack", lambda settings: FakeJack())
    assert bench.make_bench_jack(object()).takes is None


def test_a_stream_cut_after_the_first_chunk_is_an_error_row() -> None:
    import httpx2

    from claudio.tts.elevenlabs import ElevenLabsSynth

    class Cut(httpx2.SyncByteStream):
        def __iter__(self):
            yield b"\x01\x00" * 100
            raise httpx2.ReadError("cut")

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, stream=Cut(), headers={"character-cost": "5"})

    synth = ElevenLabsSynth(
        "KEY", "VOICE", "m", bench.NoFallback(), transport=httpx2.MockTransport(handler)
    )
    once = bench.catch_warnings(
        bench.stream_once(synth.stream, lambda: synth.sample_rate, lambda: synth.last_cost)
    )
    row = once("Ciao.", 1)
    assert "interrotto" in row.error


def test_a_cut_row_does_not_change_the_wer_average() -> None:
    ok = bench.Row("eleven", 1, 1, 100.0, 200.0, wer=0.1)
    cut = bench.Row("eleven", 1, 2, 100.0, 200.0, audio=np.ones(3, dtype=np.int16))
    cut.error, cut.wer = "interrotto", 0.9
    assert not bench.needs_wer(cut)
    assert bench.needs_wer(bench.Row("eleven", 1, 3, audio=np.ones(3, dtype=np.int16)))
    line = next(x for x in bench.table_lines([ok, cut]) if x.startswith("eleven") and "caldo" in x)
    assert "0.100" in line
