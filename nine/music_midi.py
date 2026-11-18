"""
Sintese de audio e MIDI para o NINE-Music.

- Converte eventos de melodia ABC em arquivo MIDI (SMF formato 0)
- Renderiza um WAV simples por sintese aditiva (sem soundfont)
- Escritores implementados do zero, sem dependencias externas

O objetivo e permitir ouvir as tunes geradas pelo modelo em qualquer
ambiente (Codespaces, Termux, CI), sem instalar FluidSynth/Timidity.
"""

from __future__ import annotations

import math
import os
import struct
import wave
from typing import Dict, List, Optional, Sequence, Tuple

from .music_abc import MelodyEvent, parse_events, parse_tune

PPQ = 480  # ticks por quarter note
DEFAULT_SR = 22050
MAX_RENDER_SECONDS = 300


# ---------------------------------------------------------------------------
# MIDI (Standard MIDI File formato 0)
# ---------------------------------------------------------------------------

def _vlq(value: int) -> bytes:
    """Codifica um inteiro como Variable Length Quantity do MIDI."""
    if value < 0:
        value = 0
    value = int(value)
    out = bytearray([value & 0x7F])
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def events_to_midi_bytes(events: Sequence[MelodyEvent], bpm: int = 120) -> bytes:
    """Serializa eventos de melodia em bytes de um arquivo MIDI."""
    bpm = max(30, min(300, int(bpm)))
    track = bytearray()
    last_tick = 0

    tempo_us = int(round(60_000_000 / bpm))
    track += _vlq(0) + b"\xFF\x51\x03" + tempo_us.to_bytes(3, "big")
    track += _vlq(0) + b"\xC0\x00"  # programa 0: piano

    timed: List[Tuple[int, int, bytes]] = []
    for ev in events:
        if ev.midi is None:
            continue
        midi = int(ev.midi)
        if not (0 <= midi <= 127):
            continue
        start_tick = int(round(ev.start * 4 * PPQ))
        dur_tick = max(1, int(round(ev.duration * 4 * PPQ)))
        timed.append((start_tick, 1, bytes([0x90, midi, 80])))         # note on
        timed.append((start_tick + dur_tick, 0, bytes([0x80, midi, 0])))  # note off

    timed.sort(key=lambda item: (item[0], item[1], item[2]))

    for tick, _, msg in timed:
        delta = max(0, tick - last_tick)
        track += _vlq(delta) + msg
        last_tick = tick

    track += _vlq(0) + b"\xFF\x2F\x00"
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, PPQ)
    return header + b"MTrk" + struct.pack(">I", len(track)) + bytes(track)


def write_midi(events: Sequence[MelodyEvent], path: str, bpm: int = 120) -> str:
    """Salva eventos em um arquivo .mid e retorna o caminho."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(events_to_midi_bytes(events, bpm=bpm))
    return path


# ---------------------------------------------------------------------------
# WAV (sintese aditiva simples)
# ---------------------------------------------------------------------------

def _midi_to_freq(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _render_numpy(events, bpm: int, sr: int):
    import numpy as np

    whole_seconds = 240.0 / bpm
    total = 0.0
    for ev in events:
        total = max(total, ev.start + ev.duration)
    total_seconds = min(MAX_RENDER_SECONDS, max(0.5, total * whole_seconds))

    buffer = np.zeros(int(total_seconds * sr) + 1, dtype=np.float32)
    fade = max(1, int(0.005 * sr))

    for ev in events:
        if ev.midi is None:
            continue
        dur = ev.duration * whole_seconds
        n = int(dur * sr)
        if n <= 0:
            continue
        start = int(ev.start * whole_seconds * sr)
        if start >= len(buffer):
            continue
        n = min(n, len(buffer) - start)
        if n <= 0:
            continue

        freq = _midi_to_freq(ev.midi)
        t = np.arange(n, dtype=np.float32) / sr
        phase = 2.0 * math.pi * freq * t
        wave_data = (
            np.sin(phase)
            + 0.32 * np.sin(2.0 * phase)
            + 0.14 * np.sin(3.0 * phase)
            + 0.06 * np.sin(4.0 * phase)
        ).astype(np.float32)

        attack = max(1, int(0.012 * sr))
        release = max(1, int(min(0.08, dur / 3.0, 0.25) * sr))
        env = np.ones(n, dtype=np.float32)
        a = min(attack, n)
        env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
        r = min(release, n)
        env[n - r:] *= np.linspace(1.0, 0.0, r, dtype=np.float32)
        env *= 0.55

        buffer[start:start + n] += wave_data * env

    if fade > 1 and len(buffer) > 2 * fade:
        buffer[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        buffer[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    peak = float(np.max(np.abs(buffer))) if len(buffer) else 0.0
    if peak > 0:
        buffer *= 0.85 / peak
    return buffer, sr


def _render_pure(events, bpm: int, sr: int):
    from array import array

    whole_seconds = 240.0 / bpm
    total = 0.0
    for ev in events:
        total = max(total, ev.start + ev.duration)
    total_seconds = min(MAX_RENDER_SECONDS, max(0.5, total * whole_seconds))
    n_samples = int(total_seconds * sr) + 1
    buffer = array("f", [0.0]) * n_samples

    two_pi = 2.0 * math.pi
    for ev in events:
        if ev.midi is None:
            continue
        dur = ev.duration * whole_seconds
        n = min(int(dur * sr), n_samples)
        if n <= 0:
            continue
        start = int(ev.start * whole_seconds * sr)
        if start >= n_samples:
            continue
        n = min(n, n_samples - start)
        freq = _midi_to_freq(ev.midi)
        release = max(1, int(min(0.08, dur / 3.0, 0.25) * sr))
        for i in range(n):
            env = 1.0
            if i < release and n > release:
                env = i / release
            elif i > n - release:
                env = max(0.0, (n - i) / release)
            s = math.sin(two_pi * freq * i / sr)
            s += 0.3 * math.sin(two_pi * 2 * freq * i / sr)
            buffer[start + i] += s * env * 0.5

    peak = max((abs(x) for x in buffer), default=0.0)
    if peak > 0:
        scale = 0.85 / peak
        for i in range(len(buffer)):
            buffer[i] *= scale
    return buffer, sr


def render_wav(events: Sequence[MelodyEvent], path: str, bpm: int = 120,
               sr: int = DEFAULT_SR) -> str:
    """Renderiza eventos em WAV mono 16-bit."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        buffer, sr = _render_numpy(events, bpm=bpm, sr=sr)
        samples = buffer
        import numpy as np
        pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
        raw = pcm.tobytes()
    except ImportError:
        buffer, sr = _render_pure(events, bpm=bpm, sr=sr)
        pcm = bytearray()
        for value in buffer:
            v = int(max(-1.0, min(1.0, value)) * 32767)
            pcm += struct.pack("<h", v)
        raw = bytes(pcm)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(raw)
    return path


# ---------------------------------------------------------------------------
# Pipeline ABC -> arquivos
# ---------------------------------------------------------------------------

def abc_to_files(abc_text: str, out_base: str, sr: int = DEFAULT_SR) -> Dict[str, object]:
    """Converte texto ABC em .abc, .mid e .wav.

    Args:
        abc_text: Texto ABC (pode ser parcial).
        out_base: Caminho base SEM extensao.
        sr: Taxa de amostragem do WAV.

    Returns:
        Dicionario com caminhos, numero de notas e duracao.

    Raises:
        ValueError: Se nenhuma nota valida foi encontrada.
    """
    tune = parse_tune(abc_text)
    if tune is None:
        raise ValueError("Nao foi possivel parsear a tune ABC")

    events = parse_events(tune)
    notes = [ev for ev in events if ev.midi is not None]
    if not notes:
        raise ValueError("Nenhuma nota valida encontrada na tune")

    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    abc_path = f"{out_base}.abc"
    midi_path = f"{out_base}.mid"
    wav_path = f"{out_base}.wav"

    with open(abc_path, "w", encoding="utf-8") as f:
        f.write(abc_text.strip() + "\n")

    write_midi(events, midi_path, bpm=tune.bpm)
    render_wav(events, wav_path, bpm=tune.bpm, sr=sr)

    duration = max(ev.start + ev.duration for ev in events)
    return {
        "abc": abc_path,
        "midi": midi_path,
        "wav": wav_path,
        "notes": len(notes),
        "duration": duration * 240.0 / tune.bpm,
        "bpm": tune.bpm,
    }
