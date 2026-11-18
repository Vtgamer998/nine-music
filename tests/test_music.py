"""Tests do NINE-Music (ABC / MIDI / dataset) - nao requerem torch."""

import os
import struct
import sys
import tempfile
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nine.music_abc import (
    duration_factor,
    normalize_tune,
    normalize_tunes,
    parse_events,
    parse_key,
    parse_tune,
    split_tunes,
    synth_tune,
    transpose_tune,
)
from nine.music_data import build_corpus
from nine.music_midi import abc_to_files, events_to_midi_bytes


SAMPLE_ABC = """% Nottingham style
X: 1
T:A and D
S:EF
M:4/4
K:A
M:6/8
P:A
f|"A"ecc c2f|"A"ecc c2f|"Bm"BcB "E7"B2f|
K:D
P:B
"D"f2f Fdd|"G"g2g ecd|[1"D"d3 d2e:|[2"D"d3 d2||
"""


def test_normalize_split_sections():
    tunes = normalize_tunes(SAMPLE_ABC)
    assert len(tunes) == 2, f"esperava 2 secoes, veio {len(tunes)}"
    for t in tunes:
        assert t.startswith("X:"), t
        assert "\nT:A and D\n" in t
        assert "\nL:1/8\n" in t
        assert '"' not in t, "acordes deveriam ser removidos"
        assert "S:" not in t and "P:" not in t
        assert "K:" in t
    assert "K:A" in tunes[0]
    assert "K:D" in tunes[1]


def test_normalize_rejects_empty():
    assert normalize_tune("X:1\nT:vazia\nM:4/4\nK:G\n") is None
    assert normalize_tune("") is None


def test_split_tunes():
    raw = SAMPLE_ABC + "\nX:2\nT:Outra\nM:4/4\nK:G\nGABc|\n"
    assert len(split_tunes(raw)) == 2


def test_key_signatures():
    assert parse_key("G").accidentals == {"F": 1}
    assert parse_key("D").accidentals == {"F": 1, "C": 1}
    assert parse_key("F").accidentals == {"B": -1}
    assert parse_key("Bb").accidentals == {"B": -1, "E": -1}
    assert parse_key("Am").accidentals == {}
    assert parse_key("Ddor").accidentals == {}
    assert parse_key("Gmix").accidentals == {}
    assert parse_key("Ephr").accidentals == {}
    assert parse_key("Bphr").accidentals == {"F": 1}


def test_duration_factor():
    assert duration_factor("") == 1
    assert duration_factor("2") == 2
    assert duration_factor("/2") == 0.5
    assert duration_factor("3/2") == 1.5
    assert duration_factor("//") == 0.25
    assert duration_factor("/") == 0.5


def test_transpose_all_semitones():
    tune = normalize_tune(
        "X:1\nT:Teste\nM:4/4\nL:1/8\nK:G\n"
        "GABc d2B2|A2F2 G2E2|=F2^F2 G,2C2:|"
    )
    assert tune is not None
    original = parse_events(parse_tune(tune))
    assert len(original) > 8

    for semitones in range(1, 12):
        transposed = transpose_tune(tune, semitones)
        assert transposed is not None
        events = parse_events(parse_tune(transposed))
        assert len(events) == len(original)
        for a, b in zip(original, events):
            if a.midi is None:
                assert b.midi is None
            else:
                assert (b.midi - a.midi) % 12 == semitones, (
                    f"transposicao {semitones}: {a.midi} -> {b.midi}"
                )
        if semitones == 2:
            assert "K:A\n" in transposed


def test_transpose_bar_accidentals():
    tune = "X:1\nT:Acc\nM:4/4\nL:1/8\nK:C\n_e/2e/2e2 _B/2=B/2B2|\n"
    transposed = transpose_tune(tune, 2)
    events = parse_events(parse_tune(transposed))
    original = parse_events(parse_tune(tune))
    assert len(events) == len(original)
    for a, b in zip(original, events):
        assert (b.midi - a.midi) % 12 == 2


def _parse_midi_header(data):
    assert data[:4] == b"MThd"
    header_len, fmt, ntracks, division = struct.unpack(">IHHH", data[4:14])
    assert header_len == 6
    assert fmt == 0
    assert ntracks == 1
    assert division == 480
    assert data[14:18] == b"MTrk"
    return division


def _count_note_ons(track):
    pos = 0
    note_ons = 0
    status = None

    def read_vlq(pos):
        value = 0
        while True:
            byte = track[pos]
            pos += 1
            value = (value << 7) | (byte & 0x7F)
            if not byte & 0x80:
                return value, pos

    while pos < len(track):
        _, pos = read_vlq(pos)
        byte = track[pos]
        if byte & 0x80:
            status = byte
            pos += 1
        # se nao tem bit de status, usa running status (nao emitimos, mas ok)
        if status == 0xFF:
            meta_type = track[pos]
            pos += 1
            length, pos = read_vlq(pos)
            pos += length
        elif status in (0xF0, 0xF7):
            length, pos = read_vlq(pos)
            pos += length
        else:
            kind = status & 0xF0
            if kind in (0xC0, 0xD0):
                pos += 1
            else:
                if kind == 0x90 and track[pos + 1] > 0:
                    note_ons += 1
                pos += 2
    return note_ons


def _note_events(track):
    pos = 0
    tick = 0
    status = None
    events = []

    def read_vlq(pos):
        value = 0
        while True:
            byte = track[pos]
            pos += 1
            value = (value << 7) | (byte & 0x7F)
            if not byte & 0x80:
                return value, pos

    while pos < len(track):
        delta, pos = read_vlq(pos)
        tick += delta
        byte = track[pos]
        if byte & 0x80:
            status = byte
            pos += 1
        if status == 0xFF:
            meta_type = track[pos]
            pos += 1
            length, pos = read_vlq(pos)
            pos += length
        elif status in (0xF0, 0xF7):
            length, pos = read_vlq(pos)
            pos += length
        else:
            kind = status & 0xF0
            if kind in (0xC0, 0xD0):
                pos += 1
            else:
                if kind in (0x80, 0x90):
                    events.append((tick, kind, track[pos]))
                pos += 2
    return events


def test_midi_repeated_note_off_before_on():
    tune = normalize_tune("X:1\nT:Rep\nM:4/4\nL:1/8\nK:C\nC2C2C2C2|\n")
    data = events_to_midi_bytes(parse_events(parse_tune(tune)))
    evs = _note_events(data[22:])
    at_480 = [kind for tick, kind, _ in evs if tick == 480]
    assert at_480 == [0x80, 0x90], at_480


def test_midi_export():
    tune = normalize_tune(
        "X:1\nT:Midi\nM:4/4\nL:1/8\nK:D\nD2F2A2d2|GABc d2A2|\n"
    )
    events = parse_events(parse_tune(tune))
    data = events_to_midi_bytes(events, bpm=120)
    _parse_midi_header(data)
    track = data[22:]
    notes = [ev for ev in events if ev.midi is not None]
    assert _count_note_ons(track) == len(notes)
    assert data.endswith(b"\xFF\x2F\x00")


def test_abc_to_wav():
    tune = normalize_tune(
        "X:1\nT:Wav\nM:6/8\nL:1/8\nK:G\nG2G B2B|d2d g2g|\n"
    )
    out = tempfile.mkdtemp()
    info = abc_to_files(tune, os.path.join(out, "teste"), sr=8000)
    assert os.path.exists(info["abc"])
    assert os.path.exists(info["midi"])
    assert os.path.exists(info["wav"])
    assert info["notes"] == 8
    with wave.open(info["wav"], "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 8000
        assert wf.getnframes() > 1000


def test_pure_python_wav_fallback():
    """Garante que o renderer puro tambem funciona (sem numpy)."""
    import nine.music_midi as midi_mod

    original = midi_mod._render_numpy
    def fail(*a, **k):
        raise ImportError("simulado")
    midi_mod._render_numpy = fail
    try:
        info = abc_to_files(
            "X:1\nT:Puro\nM:4/4\nL:1/8\nK:C\nC2E2G2c2|\n",
            os.path.join(tempfile.mkdtemp(), "puro"),
            sr=4000,
        )
        assert os.path.getsize(info["wav"]) > 1000
    finally:
        midi_mod._render_numpy = original


def test_synth_tunes():
    seen_keys = set()
    for i in range(1, 31):
        tune = synth_tune(i, seed=i * 17)
        parsed = parse_tune(tune)
        assert parsed is not None
        events = parse_events(parsed)
        notes = [ev for ev in events if ev.midi is not None]
        assert len(notes) >= 8, f"tune sintetica {i} tem poucas notas"
        assert max(ev.start + ev.duration for ev in events) >= 3
        seen_keys.add(parsed.key.name)
    assert len(seen_keys) >= 3


def test_partial_generation():
    partial = "X:1\nT:Parcial\nM:6/8\nL:1/8\nK:D\n|:A2F A2F|G2E G2E|A2"
    parsed = parse_tune(partial)
    assert parsed is not None
    events = parse_events(parsed)
    assert len(events) >= 4


def test_dataset_offline():
    out = tempfile.mkdtemp()
    stats = build_corpus(
        out_dir=out,
        augment=2,
        synth_count=12,
        download=False,
        seed=7,
    )
    assert stats["total_tunes"] == 12
    with open(os.path.join(out, "corpus.txt"), encoding="utf-8") as f:
        corpus = f.read()
    assert corpus.count("X:") == 12
    assert "K:" in corpus


def _run_all():
    tests = [
        test_normalize_split_sections,
        test_normalize_rejects_empty,
        test_split_tunes,
        test_key_signatures,
        test_duration_factor,
        test_transpose_all_semitones,
        test_transpose_bar_accidentals,
        test_midi_export,
        test_midi_repeated_note_off_before_on,
        test_abc_to_wav,
        test_pure_python_wav_fallback,
        test_synth_tunes,
        test_partial_generation,
        test_dataset_offline,
    ]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")


if __name__ == "__main__":
    print("=== NINE-Music Tests ===\n")
    _run_all()
    print("\n=== Todos os testes musicais passaram! ===")
