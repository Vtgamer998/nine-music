"""
CLI de geracao musical do NINE-Music.

Gera tunes em ABC notation a partir de um checkpoint treinado e converte
para MIDI e WAV (sintese aditiva) automaticamente.

Exemplos:
    python -m nine.music_cli --style jig --num 3
    python -m nine.music_cli --key Dm --style reel --tokens 256
    python -m nine.music_cli --prompt "X:1\\nT:Minha tune\\nM:6/8\\nL:1/8\\nK:G\\n|:"

Seguranca:
- Sanitizacao do prompt (tamanho e caracteres de controle)
- Limite de tunes por execucao
- Validacao de caminhos de checkpoint/tokenizer
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from typing import List, Optional, Tuple

from .music_abc import COMMON_MAJOR_KEYS, COMMON_MINOR_KEYS, STYLES, parse_key

MAX_PROMPT_CHARS = 4_000
MAX_SAMPLES = 16
MAX_NEW_TOKENS = 1024
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

DEFAULT_CKPT = "nine/data/music/nine-music.pt"
DEFAULT_TOK = "nine/data/music/music-tok.json"
DEFAULT_OUT = "nine/data/music/samples/tune"

BODY_ALLOWED_RE = re.compile(r"[^A-Ga-gzZxX0-9/|:\[\]^_=,'()<> \t-]")


def sanitize_prompt(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = CONTROL_CHARS_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text[:MAX_PROMPT_CHARS]


def build_prompt(style: str = "reel", key: Optional[str] = None,
                 title: str = "Invencao", seed: Optional[int] = None) -> str:
    """Monta um cabecalho ABC inicial para o modelo continuar."""
    rng = random.Random(seed)
    style = style if style in STYLES else "reel"
    meter = STYLES[style]["meter"]

    if key:
        try:
            key_name = parse_key(key).name
        except ValueError:
            key_name = "G"
    elif rng.random() < 0.6:
        key_name = rng.choice(COMMON_MAJOR_KEYS)
    else:
        key_name = rng.choice(COMMON_MINOR_KEYS)

    safe_title = re.sub(r"[\x00-\x1f\x7f]", " ", title or "Invencao").strip()
    safe_title = safe_title[:80] or "Invencao"

    return (
        "X:1\n"
        f"T:{safe_title}\n"
        f"M:{meter}\n"
        "L:1/8\n"
        f"K:{key_name}\n"
        "|:"
    )


def _decode_ids(tok, ids: List[int]) -> str:
    return tok.decode(ids)


def extract_tune(prompt: str, generated: str) -> str:
    """Recorta a primeira tune gerada e remove tokens especiais."""
    for special in ("<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"):
        generated = generated.replace(special, "")

    if generated.startswith(prompt):
        body = generated[len(prompt):]
    else:
        body = generated

    body = body.replace("\r", "")
    cut = re.search(r"\n\s*X\s*:\s*\d+", body)
    if cut:
        body = body[:cut.start()]

    # Mantem apenas caracteres musicais (evita bytes estranhos do modelo)
    body = BODY_ALLOWED_RE.sub("", body)
    body = re.sub(r"\s+", "", body)
    body = body.strip()
    if body and not body.endswith("|"):
        body += "|"

    return prompt + body


def _load(args):
    try:
        import torch
    except ImportError:
        print("[erro] PyTorch nao instalado. Rode: pip install -r requirements.txt",
              file=sys.stderr)
        sys.exit(2)

    from .fuse import load_fused_model

    if not os.path.exists(args.ckpt):
        print(f"[erro] checkpoint nao encontrado: {args.ckpt}", file=sys.stderr)
        print("       treine primeiro com: bash scripts/train_music.sh", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.tok):
        print(f"[erro] tokenizer nao encontrado: {args.tok}", file=sys.stderr)
        sys.exit(2)

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("[aviso] CUDA indisponivel; usando CPU", file=sys.stderr)
        device = "cpu"

    model, tok = load_fused_model(
        base_path=args.ckpt,
        tokenizer_path=args.tok,
        device=device,
        validate=True,
        verbose=args.verbose,
    )
    if tok is None:
        print("[erro] falha ao carregar o tokenizer", file=sys.stderr)
        sys.exit(2)
    model.eval()
    return torch, model, tok, device


def generate_samples(args) -> List[Tuple[str, str]]:
    """Gera N tunes. Retorna lista de (prompt, texto_abc)."""
    torch, model, tok, device = _load(args)
    samples: List[Tuple[str, str]] = []

    for i in range(args.num):
        seed = args.seed + i * 7919
        torch.manual_seed(seed)

        if args.prompt:
            prompt = sanitize_prompt(args.prompt)
            if not prompt.endswith("\n") and not prompt.endswith("|"):
                prompt += "\n"
        else:
            prompt = build_prompt(
                style=args.style,
                key=args.key,
                title=args.title if args.num == 1 else f"{args.title} {i + 1}",
                seed=seed,
            )

        ids = tok.encode(prompt)
        if not ids:
            ids = tok.encode("X:1\nT:Tune\nM:4/4\nL:1/8\nK:G\n|:")

        idx = torch.tensor([ids], dtype=torch.long, device=device)
        context_left = max(16, model.cfg.block_size - len(ids))
        max_new = max(16, min(args.tokens, MAX_NEW_TOKENS, context_left))
        try:
            out = model.generate(
                idx,
                max_new_tokens=max_new,
                temperature=args.temp,
                top_k=args.top_k,
                top_p=args.top_p,
                use_cache=True,
                repetition_penalty=args.repetition_penalty,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"[aviso] falha na geracao {i + 1}: {exc}", file=sys.stderr)
            continue

        text = _decode_ids(tok, out[0].tolist())
        tune = extract_tune(prompt, text)
        samples.append((prompt, tune))

    return samples


def main():
    p = argparse.ArgumentParser(
        prog="nine.music_cli",
        description="NINE-Music - gerador de tunes em ABC/MIDI/WAV",
    )
    p.add_argument("--prompt", type=str, default=None,
                   help="prompt ABC (substitui --style/--key)")
    p.add_argument("--style", choices=sorted(STYLES.keys()), default="reel")
    p.add_argument("--key", type=str, default=None, help="tonalidade (ex: D, Am, Gdor)")
    p.add_argument("--title", type=str, default="Invencao")
    p.add_argument("--num", type=int, default=1, help="numero de tunes")
    p.add_argument("--tokens", type=int, default=320, help="tokens novos por tune")
    p.add_argument("--temp", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    p.add_argument("--tok", type=str, default=DEFAULT_TOK)
    p.add_argument("--out", type=str, default=DEFAULT_OUT,
                   help="prefixo dos arquivos de saida")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-wav", action="store_true", help="nao renderizar audio")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    args.num = max(1, min(args.num, MAX_SAMPLES))
    args.tokens = max(16, min(args.tokens, MAX_NEW_TOKENS))
    args.temp = max(0.1, min(args.temp, 3.0))

    samples = generate_samples(args)
    if not samples:
        print("[erro] nenhuma tune gerada", file=sys.stderr)
        sys.exit(1)

    from .music_midi import abc_to_files

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    ok = 0

    for i, (_, tune) in enumerate(samples, start=1):
        suffix = f"{i:02d}" if len(samples) > 1 else ""
        base = f"{args.out}_{suffix}" if suffix else args.out
        print(f"\n--- NINE-Music [{i}/{len(samples)}] ---")
        print(tune.strip()[:1500])
        print("-----------------------------------")
        try:
            if args.no_wav:
                abc_path = base + ".abc"
                with open(abc_path, "w", encoding="utf-8") as f:
                    f.write(tune.strip() + "\n")
                print(f"[ok] {abc_path}")
            else:
                info = abc_to_files(tune, base)
                print(f"[ok] {info['abc']}")
                print(f"[ok] {info['midi']}")
                print(f"[ok] {info['wav']}  ({info['notes']} notas, "
                      f"{info['duration']:.1f}s @ {info['bpm']} bpm)")
            ok += 1
        except (ValueError, OSError) as exc:
            print(f"[aviso] tune {i} nao pode ser sintetizada: {exc}",
                  file=sys.stderr)
            abc_path = base + ".abc"
            with open(abc_path, "w", encoding="utf-8") as f:
                f.write(tune.strip() + "\n")
            print(f"[ok] {abc_path} (sem audio)")

    print(f"\n{ok}/{len(samples)} tunes geradas em {out_dir}/")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
