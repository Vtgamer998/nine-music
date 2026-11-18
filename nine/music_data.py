"""
Construtor do dataset musical do NINE-Music.

Fontes:
- Nottingham Music Database (ABC) - tunes folk tradicionais
- Tunes sinteticas geradas por `nine.music_abc.synth_tune` (fallback offline)

Pipeline:
1. Baixa (opcional) os arquivos ABC do Nottingham
2. Normaliza e deduplica as tunes
3. Aumenta os dados com transposicao cromatica (0..N-1 semitons)
4. Mistura tunes sinteticas para regularizar o treino
5. Escreve um corpus canonico para o `nine.prep_data`

Uso:
    python -m nine.music_data --out nine/data/music --augment 6 --synth 300
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

from .music_abc import normalize_tunes, split_tunes, synth_tune, transpose_tune

NOTTINGHAM_BASE = "https://raw.githubusercontent.com/jukedeck/nottingham-dataset/master/ABC_cleaned"
NOTTINGHAM_FILES = (
    "ashover.abc",
    "hpps.abc",
    "jigs.abc",
    "morris.abc",
    "playford.abc",
    "reelsa-c.abc",
    "reelsd-g.abc",
    "reelsh-l.abc",
    "reelsm-q.abc",
    "reelsr-t.abc",
    "reelsu-z.abc",
    "slip.abc",
    "waltzes.abc",
    "xmas.abc",
)

USER_AGENT = "NINE-Music/0.1 (+https://github.com/Vtgamer998/nine-1)"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: str, timeout: int = 30,
                   retries: int = 3) -> bool:
    """Baixa um arquivo com retries. Retorna True em caso de sucesso."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                raise ValueError("resposta vazia")
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
            return True
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            print(f"  [aviso] tentativa {attempt}/{retries} falhou ({url}): {exc}",
                  file=sys.stderr)
            time.sleep(min(2 * attempt, 6))
    if os.path.exists(tmp):
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return False


def download_nottingham(raw_dir: str) -> List[str]:
    """Baixa os arquivos ABC do Nottingham para `raw_dir`."""
    os.makedirs(raw_dir, exist_ok=True)
    paths: List[str] = []
    for name in NOTTINGHAM_FILES:
        url = f"{NOTTINGHAM_BASE}/{name}"
        dest = os.path.join(raw_dir, name)
        if _download_file(url, dest):
            paths.append(dest)
        else:
            print(f"  [aviso] pulando {name}", file=sys.stderr)
    return paths


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _tune_hash(tune: str) -> str:
    """Hash estavel do conteudo musical (ignora X e titulo)."""
    lines = [
        line for line in tune.splitlines()
        if not line.startswith(("X:", "T:"))
    ]
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()


def _renumber(tune: str, new_id: int) -> str:
    """Troca o campo X: por um id sequencial."""
    lines = tune.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("X:"):
            lines[i] = f"X:{new_id}"
            break
    return "\n".join(lines)


def _read_abc_files(paths: Sequence[str]) -> List[str]:
    tunes: List[str] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
        except OSError as exc:
            print(f"  [aviso] falha ao ler {path}: {exc}", file=sys.stderr)
            continue
        for raw_tune in split_tunes(raw):
            try:
                tunes.extend(normalize_tunes(raw_tune))
            except (ValueError, RecursionError) as exc:
                print(f"  [aviso] tune invalida em {path}: {exc}", file=sys.stderr)
    return tunes


def build_corpus(
    out_dir: str,
    augment: int = 6,
    synth_count: int = 300,
    download: bool = True,
    seed: int = 1337,
    max_tunes: Optional[int] = None,
) -> Dict[str, object]:
    """Constroi o corpus musical normalizado.

    Returns:
        Estatisticas do dataset.
    """
    os.makedirs(out_dir, exist_ok=True)
    raw_dir = os.path.join(out_dir, "raw")
    raw_paths: List[str] = []

    if download:
        print("Baixando Nottingham Music Database (ABC)...")
        raw_paths = download_nottingham(raw_dir)
    else:
        if os.path.isdir(raw_dir):
            raw_paths = sorted(
                os.path.join(raw_dir, name)
                for name in os.listdir(raw_dir)
                if name.endswith(".abc")
            )
    if raw_paths:
        print(f"Arquivos ABC: {len(raw_paths)}")

    print("Normalizando tunes...")
    base_tunes = _read_abc_files(raw_paths)

    seen: set = set()
    unique: List[str] = []
    for tune in base_tunes:
        h = _tune_hash(tune)
        if h not in seen:
            seen.add(h)
            unique.append(tune)
    base_tunes = unique

    rng = random.Random(seed)
    rng.shuffle(base_tunes)

    if max_tunes is not None:
        base_tunes = base_tunes[:max_tunes]

    if not base_tunes and synth_count <= 0:
        raise RuntimeError(
            "Nenhuma tune disponivel. Verifique a internet ou use --synth N."
        )

    augment = max(1, min(12, int(augment)))
    val_count = max(1, int(len(base_tunes) * 0.05)) if len(base_tunes) >= 20 else 0

    tune_id = 1
    val_tunes: List[str] = []
    train_tunes: List[str] = []
    transposed = 0

    for idx, tune in enumerate(base_tunes):
        is_val = idx < val_count
        variants: List[str] = [tune]
        for semitones in range(1, augment):
            variant = transpose_tune(tune, semitones, tune_id=tune_id)
            if variant:
                variants.append(variant)
                transposed += 1
        for variant in variants:
            variant = _renumber(variant, tune_id)
            if is_val:
                val_tunes.append(variant)
            else:
                train_tunes.append(variant)
            tune_id += 1

    for i in range(max(0, int(synth_count))):
        style = ("reel", "jig", "waltz", "slip", "air")[i % 5]
        train_tunes.append(synth_tune(tune_id, seed=seed + i * 131, style=style))
        tune_id += 1

    rng.shuffle(train_tunes)

    corpus = "\n\n".join(val_tunes + train_tunes).strip() + "\n"
    corpus_path = os.path.join(out_dir, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    source_note = os.path.join(out_dir, "SOURCE.txt")
    with open(source_note, "w", encoding="utf-8") as f:
        f.write(
            "NINE-Music dataset\n"
            "==================\n\n"
            "Nottingham Music Database (ABC_cleaned), por Eric Foxley e outros.\n"
            "Repositorio: https://github.com/jukedeck/nottingham-dataset\n"
            "Licenca dos dados: GPL-3.0 (o codigo do NINE-Music continua MIT).\n\n"
            f"Tunes sinteticas geradas localmente: {max(0, int(synth_count))}\n"
            "As tunes sinteticas sao geradas proceduralmente pelo proprio projeto.\n"
        )

    stats: Dict[str, object] = {
        "base_tunes": len(base_tunes),
        "transposed": transposed,
        "synthetic": max(0, int(synth_count)),
        "val_tunes": len(val_tunes),
        "train_tunes": len(train_tunes),
        "total_tunes": len(val_tunes) + len(train_tunes),
        "chars": len(corpus),
        "corpus": corpus_path,
        "raw_dir": raw_dir,
    }
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Constroi o corpus ABC do NINE-Music",
    )
    p.add_argument("--out", type=str, default="nine/data/music",
                   help="diretorio de saida")
    p.add_argument("--augment", type=int, default=6,
                   help="numero de transposicoes por tune (1..12)")
    p.add_argument("--synth", type=int, default=300,
                   help="tunes sinteticas extras (fallback offline)")
    p.add_argument("--no-download", action="store_true",
                   help="nao baixar; usa apenas raw/ existente + sinteticas")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-tunes", type=int, default=None,
                   help="limite de tunes base (debug)")
    args = p.parse_args()

    try:
        stats = build_corpus(
            out_dir=args.out,
            augment=args.augment,
            synth_count=args.synth,
            download=not args.no_download,
            seed=args.seed,
            max_tunes=args.max_tunes,
        )
    except RuntimeError as exc:
        print(f"[erro] {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Dataset NINE-Music ===")
    print(f"Tunes base:       {stats['base_tunes']}")
    print(f"Transpostas:      {stats['transposed']}")
    print(f"Sinteticas:       {stats['synthetic']}")
    print(f"Validacao:        {stats['val_tunes']} tunes")
    print(f"Treino:           {stats['train_tunes']} tunes")
    print(f"Total:            {stats['total_tunes']} tunes")
    print(f"Caracteres:       {stats['chars']:,}")
    print(f"Corpus:           {stats['corpus']}")


if __name__ == "__main__":
    main()
