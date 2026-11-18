"""
Utilitarios de ABC notation para o NINE-Music.

Implementa, sem dependencias externas:
- Parsing de campos (X, T, M, L, K, Q) e do corpo das tunes
- Normalizacao de tunes para um formato canonico de treino
- Transposicao cromatica correta (armadura de clave + acidentes)
- Geracao de tunes sinteticas no estilo folk (fallback offline)

O formato canonico usado no treino e:

    X:1
    T:Titulo
    M:6/8
    L:1/8
    K:D
    |:A2F A2F|...

Seguranca:
- Nenhuma avaliacao dinamica de texto
- Limites de tamanho no parsing
- Conversao tolerante a texto parcial gerado pelo modelo
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constantes musicais
# ---------------------------------------------------------------------------

NOTE_LETTERS = "CDEFGAB"
LETTER_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

MAJOR_SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
MAJOR_FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")

# Nome canonico (bemol/sustenido) para cada classe de altura
PC_TO_KEY = {
    0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F",
    6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B",
}

# Numero de acidentes por tonica maior
MAJOR_SIGNATURE_COUNT = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
    "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7,
}

# Semitons a somar a tonica para chegar na maior relativa (modo -> offset)
MODE_TO_RELATIVE_MAJOR = {
    "": 0, "maj": 0, "ion": 0,
    "m": 3, "min": 3, "aeo": 3,
    "dor": 10,
    "phr": 8,
    "lyd": 7,
    "mix": 5,
    "loc": 1,
}

# Passos diatonicos para cada transposicao cromatica (0..11 semitons)
DIATONIC_STEPS = (0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6)

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
NATURAL_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)

MAX_TUNE_CHARS = 50_000
MAX_TITLE_CHARS = 80

STYLES = {
    "reel": {"meter": "4/4", "eighths": 8},
    "jig": {"meter": "6/8", "eighths": 6},
    "slip": {"meter": "9/8", "eighths": 9},
    "waltz": {"meter": "3/4", "eighths": 6},
    "air": {"meter": "3/4", "eighths": 6},
}

COMMON_MAJOR_KEYS = ("C", "G", "D", "A", "E", "F", "Bb", "Eb")
COMMON_MINOR_KEYS = ("Am", "Em", "Bm", "Dm", "Gm", "Cm", "F#m")


# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------

FIELD_RE = re.compile(r"^\s*([A-Za-z]):\s*(.*)$")
KEY_RE = re.compile(r"^\s*([A-Ga-g])\s*([#b]?)\s*(maj|min|m|dor|phr|lyd|mix|loc|ion|aeo)?\s*$")
METER_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
UNIT_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")

# Nota ABC: acidente + letra + oitava + duracao
TOKEN_RE = re.compile(
    r"(?P<note>(?P<nacc>\^\^|__|\^|_|=)?"
    r"(?P<nletter>[A-Ga-g])(?P<noct>[',]*)"
    r"(?P<ndur>(?:\d+)?(?:/{1,2}\d*)?))"
    r"|(?P<rest>(?P<rletter>[zx])(?P<rdur>(?:\d+)?(?:/{1,2}\d*)?))"
    r"|(?P<bar>\[\d|\|+\]?|:\|+)"
)

ACC_TO_OFFSET = {"^^": 2, "^": 1, "=": 0, "_": -1, "__": -2}
OFFSET_TO_ACC = {0: "", 1: "^", -1: "_", 2: "^^", -2: "__"}


# ---------------------------------------------------------------------------
# Estruturas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeyInfo:
    """Armadura de clave + modo."""
    name: str
    tonic_pc: int
    mode: str
    accidentals: Dict[str, int] = field(default_factory=dict, compare=False)


@dataclass
class MelodyEvent:
    """Evento de melodia em unidades de nota inteira (whole notes)."""
    start: float
    duration: float
    midi: Optional[int]  # None = pausa


@dataclass
class ParsedTune:
    """Tune ABC parseada."""
    title: str
    meter: str
    unit: str
    key: KeyInfo
    body: str
    bpm: int = 120


# ---------------------------------------------------------------------------
# Chaves e armaduras
# ---------------------------------------------------------------------------

def _signature_from_pc(pc: int) -> Dict[str, int]:
    """Retorna o mapa de acidentes para a maior com tonica na classe pc."""
    name = PC_TO_KEY[pc % 12]
    count = MAJOR_SIGNATURE_COUNT[name]
    acc: Dict[str, int] = {}
    if count > 0:
        for letter in MAJOR_SHARP_ORDER[:count]:
            acc[letter] = 1
    elif count < 0:
        for letter in MAJOR_FLAT_ORDER[:-count]:
            acc[letter] = -1
    return acc


def parse_key(key_str: str) -> KeyInfo:
    """Converte 'K:Dm', 'G', 'Ador' etc. em KeyInfo.

    Raises:
        ValueError: Se a tonica/modo nao for reconhecida.
    """
    if not key_str or not isinstance(key_str, str):
        raise ValueError("Armadura de clave vazia")
    token = key_str.strip().split()[0] if key_str.strip() else ""
    m = KEY_RE.match(token)
    if not m:
        raise ValueError(f"Armadura de clave invalida: {key_str!r}")

    letter = m.group(1).upper()
    acc_str = m.group(2) or ""
    mode_raw = (m.group(3) or "").lower()

    tonic_pc = (LETTER_SEMITONE[letter] + (1 if acc_str == "#" else -1 if acc_str == "b" else 0)) % 12
    mode = "" if mode_raw in ("", "maj", "ion") else "m" if mode_raw in ("m", "min", "aeo") else mode_raw
    relative_major_pc = (tonic_pc + MODE_TO_RELATIVE_MAJOR[mode]) % 12
    accidentals = _signature_from_pc(relative_major_pc)

    name = letter + acc_str + mode
    return KeyInfo(name=name, tonic_pc=tonic_pc, mode=mode, accidentals=accidentals)


def transpose_key(key: KeyInfo, semitones: int) -> KeyInfo:
    """Transporta a armadura em `semitones` (mantendo o modo)."""
    semitones %= 12
    if semitones == 0:
        return key
    new_pc = (key.tonic_pc + semitones) % 12
    base = PC_TO_KEY[new_pc]
    name = base[0] + base[1:] + key.mode
    return parse_key(name)


# ---------------------------------------------------------------------------
# Duracao
# ---------------------------------------------------------------------------

def duration_factor(dur: str) -> Fraction:
    """Converte sufixo de duracao ABC ('2', '/2', '3/2', '//') em fator."""
    if not dur:
        return Fraction(1)
    if dur.isdigit():
        return Fraction(int(dur))
    m = re.fullmatch(r"(\d*)(/{1,2})(\d*)", dur)
    if not m:
        return Fraction(1)
    num = int(m.group(1)) if m.group(1) else 1
    slashes = len(m.group(2))
    if m.group(3):
        den = int(m.group(3))
    else:
        den = 2 if slashes == 1 else 4
    if den <= 0:
        return Fraction(1)
    return Fraction(num, den)


def parse_unit(unit: str) -> Fraction:
    """Converte '1/8' em Fraction(1, 8)."""
    if not unit:
        return Fraction(1, 8)
    compact = unit.replace(" ", "")
    if compact in ("C", "c"):
        return Fraction(1)
    m = UNIT_RE.match(compact)
    if not m:
        return Fraction(1, 8)
    num, den = int(m.group(1)), int(m.group(2))
    if num <= 0 or den <= 0:
        return Fraction(1, 8)
    return Fraction(num, den)


# ---------------------------------------------------------------------------
# Leitura de tunes
# ---------------------------------------------------------------------------

def _preprocess_lines(raw: str) -> List[str]:
    """Remove comentarios e junta linhas com continuacao '\\'."""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines: List[str] = []
    buffer = ""
    for line in raw.split("\n"):
        line = line.split("%", 1)[0]
        if line.rstrip().endswith("\\"):
            buffer += line.rstrip()[:-1]
            continue
        lines.append(buffer + line)
        buffer = ""
    if buffer:
        lines.append(buffer)
    return lines


def split_tunes(raw: str) -> List[str]:
    """Divide um arquivo ABC em tunes (blocos iniciados por X:)."""
    if not raw:
        return []
    raw = raw[:MAX_TUNE_CHARS * 200]
    lines = raw.split("\n")
    tunes: List[str] = []
    current: List[str] = []
    started = False

    for line in lines:
        if re.match(r"^\s*X\s*:", line):
            if started and current:
                tunes.append("\n".join(current))
            current = [line]
            started = True
        elif started:
            current.append(line)

    if started and current:
        tunes.append("\n".join(current))
    return tunes


def _parse_header(raw: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """Separa campos de cabecalho e linhas do corpo."""
    fields: Dict[str, List[str]] = {}
    body_lines: List[str] = []
    in_body = False

    for line in _preprocess_lines(raw):
        if not in_body:
            m = FIELD_RE.match(line)
            if m:
                letter = m.group(1).upper()
                value = m.group(2).strip()
                fields.setdefault(letter, []).append(value)
                continue
            if not line.strip():
                continue
            in_body = True
        body_lines.append(line)

    return fields, body_lines


def _clean_body(body: str) -> str:
    """Remove anotacoes e caracteres nao musicais do corpo."""
    body = body[:MAX_TUNE_CHARS]
    body = re.sub(r'"[^"]*"', "", body)          # acordes ("G", "D7")
    body = re.sub(r"\{[^}]*\}", "", body)        # grace notes
    body = re.sub(r"![^!]*!", "", body)          # decoracoes !trill!
    body = re.sub(r"\+[^+]*\+", "", body)        # decoracoes +trill+
    body = re.sub(r"[^A-Ga-gzZxX0-9/|:\[\]^_=,'()<>-]", "", body)
    if not body:
        return ""
    if not re.search(r"[|\]]$", body):
        body += "|"
    return body


def _clean_title(title: str, fallback: str = "Tune") -> str:
    title = re.sub(r"[\x00-\x1f\x7f]", " ", title or "")
    title = title.replace('"', "").replace("%", "").strip()
    title = re.sub(r"\s+", " ", title)
    if not title:
        title = fallback
    return title[:MAX_TITLE_CHARS]


def _normalize_section(
    fields: Dict[str, List[str]],
    key_str: str,
    section_lines: Sequence[str],
    tune_id: int,
) -> Optional[str]:
    """Normaliza uma secao (uma armadura) de uma tune."""
    try:
        key = parse_key(key_str)
    except ValueError:
        return None

    body = _clean_body("".join(line.strip() for line in section_lines))
    if not body or len(re.findall(r"[A-Ga-gz]", body)) < 4:
        return None

    meter_raw = (fields.get("M") or ["4/4"])[0].strip()
    if METER_RE.match(meter_raw):
        meter = meter_raw.replace(" ", "")
    elif meter_raw in ("C", "c"):
        meter = "4/4"
    elif meter_raw in ("C|", "c|"):
        meter = "2/2"
    else:
        meter = "4/4"

    title = _clean_title((fields.get("T") or ["Tune"])[0])

    return (
        f"X:{tune_id}\n"
        f"T:{title}\n"
        f"M:{meter}\n"
        f"L:1/8\n"
        f"K:{key.name}\n"
        f"{body}"
    )


def normalize_tunes(raw: str, start_id: int = 1) -> List[str]:
    """Normaliza todas as tunes/secoes de um texto ABC.

    Tunes com mudanca de K: no meio sao divididas em varias tunes.
    """
    out: List[str] = []
    tune_id = start_id

    for raw_tune in split_tunes(raw):
        fields, body_lines = _parse_header(raw_tune)
        base_key = (fields.get("K") or ["C"])[0]
        sections: List[Tuple[str, List[str]]] = []
        key_str = base_key
        current: List[str] = []

        for line in body_lines:
            stripped = line.strip()
            if not stripped:
                continue
            m = FIELD_RE.match(stripped)
            if m and m.group(1).upper() == "K":
                if current:
                    sections.append((key_str, current))
                    current = []
                key_str = m.group(2).strip()
                continue
            if m:
                # Outros campos (P:, S:, Y:, ...) sao ignorados no corpo
                continue
            current.append(stripped)

        if current:
            sections.append((key_str, current))
        if not sections:
            sections = [(base_key, [])]

        for section_key, section_lines in sections:
            norm = _normalize_section(fields, section_key, section_lines, tune_id)
            if norm:
                out.append(norm)
                tune_id += 1

    return out


def normalize_tune(raw: str, tune_id: int = 1) -> Optional[str]:
    """Normaliza apenas a primeira tune/secao de um texto ABC."""
    tunes = normalize_tunes(raw, start_id=tune_id)
    return tunes[0] if tunes else None


# ---------------------------------------------------------------------------
# Parsing de notas (usado por transposicao e por sintese MIDI)
# ---------------------------------------------------------------------------

def _note_parts(
    letter: str,
    noct: str,
    explicit_acc: Optional[str],
    key: KeyInfo,
    bar_acc: Dict[str, int],
) -> Tuple[int, int]:
    """Retorna (midi, linear) de uma nota ABC.

    `linear` = octave*7 + letter_index, usado para transposicao diatonica.
    """
    upper = letter.upper()
    letter_idx = NOTE_LETTERS.index(upper)
    base_octave = 4 if letter.isupper() else 5
    octave = base_octave + noct.count("'") - noct.count(",")

    if explicit_acc is not None:
        acc = ACC_TO_OFFSET.get(explicit_acc, 0)
        bar_acc[upper] = acc
    else:
        acc = bar_acc.get(upper, key.accidentals.get(upper, 0))

    midi = 12 * (octave + 1) + LETTER_SEMITONE[upper] + acc
    linear = octave * 7 + letter_idx
    return midi, linear


def _format_pitch(
    linear: int,
    target_pc: int,
    key: KeyInfo,
    out_bar_acc: Optional[Dict[str, int]] = None,
) -> str:
    """Formata uma nota ABC a partir de posicao diatonica e classe de altura.

    Se `out_bar_acc` for fornecido, mantem o estado de acidentes do compasso
    de saida (acidentes em ABC valem ate a proxima barra).
    """
    letter = NOTE_LETTERS[linear % 7]
    octave = linear // 7
    key_acc = key.accidentals.get(letter, 0)

    # Acidentes em ABC sao absolutos em relacao a nota natural
    # (^ = +1, _ = -1, = = natural), nao relativos a armadura.
    natural_delta = (target_pc - LETTER_SEMITONE[letter]) % 12
    if natural_delta > 6:
        natural_delta -= 12

    if out_bar_acc is None:
        current_acc = key_acc
    else:
        current_acc = out_bar_acc.get(letter, key_acc)

    if natural_delta == current_acc:
        acc = ""
    elif natural_delta == 0:
        acc = "="
    else:
        acc = OFFSET_TO_ACC.get(natural_delta, "=")

    if out_bar_acc is not None:
        out_bar_acc[letter] = natural_delta

    if octave >= 5:
        out = letter.lower() + "'" * (octave - 5)
    else:
        out = letter.upper() + "," * (4 - octave)
    return acc + out


def transpose_body(body: str, old_key: KeyInfo, new_key: KeyInfo, semitones: int) -> str:
    """Transporta o corpo de uma tune mantendo duracoes e estrutura."""
    semitones %= 12
    step = DIATONIC_STEPS[semitones]
    out: List[str] = []
    bar_acc: Dict[str, int] = {}
    out_bar_acc: Dict[str, int] = {}
    pos = 0

    for m in TOKEN_RE.finditer(body):
        out.append(body[pos:m.start()])
        pos = m.end()

        if m.group("note"):
            midi, linear = _note_parts(
                m.group("nletter"),
                m.group("noct") or "",
                m.group("nacc"),
                old_key,
                bar_acc,
            )
            target_pc = (midi + semitones) % 12
            out.append(_format_pitch(linear + step, target_pc, new_key, out_bar_acc))
            out.append(m.group("ndur") or "")
        else:
            out.append(m.group(0))
            if m.group("bar"):
                bar_acc.clear()
                out_bar_acc.clear()

    out.append(body[pos:])
    return "".join(out)


def transpose_tune(text: str, semitones: int, tune_id: Optional[int] = None) -> Optional[str]:
    """Transporta uma tune ABC canonica em `semitones`."""
    fields, body_lines = _parse_header(text)
    key_raw = (fields.get("K") or ["C"])[0]
    try:
        old_key = parse_key(key_raw)
    except ValueError:
        return None
    new_key = transpose_key(old_key, semitones)

    body = "".join(line.strip() for line in body_lines)
    new_body = transpose_body(body, old_key, new_key, semitones)

    title = _clean_title((fields.get("T") or ["Tune"])[0])
    meter = (fields.get("M") or ["4/4"])[0].strip() or "4/4"
    xid = tune_id if tune_id is not None else int((fields.get("X") or ["1"])[0] or 1)

    return (
        f"X:{xid}\n"
        f"T:{title}\n"
        f"M:{meter}\n"
        f"L:1/8\n"
        f"K:{new_key.name}\n"
        f"{new_body}"
    )


def parse_tune(text: str) -> Optional[ParsedTune]:
    """Parseia uma tune (aceita texto parcial gerado pelo modelo)."""
    if not text:
        return None
    fields, body_lines = _parse_header(text)
    key_raw = (fields.get("K") or ["C"])[0]
    try:
        key = parse_key(key_raw)
    except ValueError:
        key = parse_key("C")

    meter = (fields.get("M") or ["4/4"])[0].strip() or "4/4"
    unit = (fields.get("L") or ["1/8"])[0].strip() or "1/8"
    title = _clean_title((fields.get("T") or ["Tune"])[0])

    bpm = 120
    q = (fields.get("Q") or [""])[0]
    qm = re.search(r"(\d{2,3})", q)
    if qm:
        bpm = max(30, min(300, int(qm.group(1))))

    body = "".join(line.strip() for line in body_lines)
    return ParsedTune(title=title, meter=meter, unit=unit, key=key,
                      body=body, bpm=bpm)


def parse_events(tune: ParsedTune) -> List[MelodyEvent]:
    """Extrai eventos de melodia (notas e pausas) em ordem temporal."""
    unit = parse_unit(tune.unit)
    events: List[MelodyEvent] = []
    bar_acc: Dict[str, int] = {}
    start = Fraction(0)

    for m in TOKEN_RE.finditer(tune.body):
        if m.group("note"):
            midi, _ = _note_parts(
                m.group("nletter"), m.group("noct") or "",
                m.group("nacc"), tune.key, bar_acc,
            )
            dur = unit * duration_factor(m.group("ndur") or "")
            events.append(MelodyEvent(float(start), float(dur), midi))
            start += dur
        elif m.group("rest"):
            dur = unit * duration_factor(m.group("rdur") or "")
            events.append(MelodyEvent(float(start), float(dur), None))
            start += dur
        elif m.group("bar"):
            bar_acc.clear()

    return events


# ---------------------------------------------------------------------------
# Geracao de tunes sinteticas (estilo folk)
# ---------------------------------------------------------------------------

_RHYTHMS = {
    8: (
        (2, 2, 2, 2),
        (2, 1, 1, 2, 2),
        (1, 1, 2, 2, 2),
        (1, 1, 1, 1, 2, 2),
        (2, 2, 1, 1, 2),
        (1, 1, 1, 1, 1, 1, 2),
        (2, 1, 1, 1, 1, 2),
        (3, 1, 2, 2),
    ),
    6: (
        (2, 2, 2),
        (1, 1, 2, 2),
        (2, 1, 1, 1, 1),
        (1, 1, 1, 1, 2),
        (1, 1, 1, 1, 1, 1),
        (2, 2, 1, 1),
    ),
    9: (
        (2, 2, 2, 2, 1),
        (1, 1, 1, 2, 2, 2),
        (2, 2, 1, 2, 2),
        (1, 1, 1, 1, 1, 2, 2),
        (2, 1, 1, 1, 1, 1, 1, 1),
    ),
}


def _encode_degree(key: KeyInfo, degree: int, scale: Sequence[int]) -> str:
    """Codifica um grau de escala (0 = tonica) como nota ABC."""
    tonic_letter_idx = NOTE_LETTERS.index(key.name[0].upper())
    linear = 4 * 7 + tonic_letter_idx + degree
    pc = (key.tonic_pc + scale[degree % 7] + 12 * (degree // 7)) % 12
    return _format_pitch(linear, pc, key)


def _make_phrase(rng: random.Random, bars: int, eighths: int,
                 scale: Sequence[int], start_degree: int,
                 key: KeyInfo, end_degree: Optional[int] = None) -> List[List[str]]:
    """Gera uma frase de `bars` compassos como listas de notas por compasso."""
    phrase: List[List[str]] = []
    degree = start_degree
    for bar in range(bars):
        pattern = rng.choice(_RHYTHMS[eighths])
        notes: List[str] = []
        for i, dur in enumerate(pattern):
            if bar == bars - 1 and i == len(pattern) - 1 and end_degree is not None:
                degree = end_degree
            else:
                degree += rng.choice((-2, -1, -1, 0, 0, 1, 1, 2))
                degree = max(-2, min(9, degree))
            note = _encode_degree(key, degree, scale)
            if dur != 1:
                note += str(dur)
            notes.append(note)
        phrase.append(notes)
    return phrase


def synth_tune(tune_id: int = 1, seed: Optional[int] = None,
               style: Optional[str] = None) -> str:
    """Gera uma tune folk sintetica no formato canonico.

    Usada como fallback offline e como augmentacao do dataset.
    """
    rng = random.Random(seed if seed is not None else tune_id * 7919)
    style = style if style in STYLES else rng.choice(("reel", "jig", "waltz", "slip"))
    eighths = STYLES[style]["eighths"]
    meter = STYLES[style]["meter"]

    if rng.random() < 0.55:
        key = parse_key(rng.choice(COMMON_MAJOR_KEYS))
        scale = MAJOR_SCALE
    else:
        key = parse_key(rng.choice(COMMON_MINOR_KEYS))
        scale = NATURAL_MINOR_SCALE

    start = rng.choice((0, 4, 2))
    phrase_a = _make_phrase(rng, 2, eighths, scale, start, key,
                            end_degree=rng.choice((0, 2, 4)))
    phrase_a2 = _make_phrase(rng, 2, eighths, scale, start, key, end_degree=0)
    phrase_b = _make_phrase(rng, 2, eighths, scale, 4, key,
                            end_degree=rng.choice((2, 4, 0)))
    phrase_b2 = _make_phrase(rng, 2, eighths, scale, 4, key, end_degree=0)

    bars = phrase_a + phrase_a2 + phrase_b + phrase_b2
    body = "|:" + "|".join("".join(bar) for bar in bars[:4]) + ":|"
    body += "|".join("".join(bar) for bar in bars[4:]) + "|]"

    title = f"Tune {tune_id}"
    return (
        f"X:{tune_id}\n"
        f"T:{title}\n"
        f"M:{meter}\n"
        f"L:1/8\n"
        f"K:{key.name}\n"
        f"{body}"
    )
