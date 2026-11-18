#!/usr/bin/env bash
# Treino do NINE-Music (CPU em Codespaces ou GPU local/Colab)
#
# Uso:
#   bash scripts/train_music.sh demo   # teste rapido (~5 min)
#   bash scripts/train_music.sh cpu    # preset padrao para Codespaces
#   bash scripts/train_music.sh gpu    # se tiver CUDA
#
# Overrides via ambiente:
#   DATA_DIR=... AUGMENT=6 SYNTH=300 ITERS=900 BATCH=24 LR=7e-4
#   BLOCK=192 NLAYER=4 NHEAD=4 NEMBD=192 NKV=2 FORCE_DATA=1 SKIP_SAMPLE=1
set -euo pipefail

cd "$(dirname "$0")/.."
PRESET="${1:-cpu}"
NKV="${NKV:-2}"
DATA_DIR="${DATA_DIR:-nine/data/music}"
CKPT="$DATA_DIR/nine-music.pt"
TOK="$DATA_DIR/music-tok.json"
BIN="$DATA_DIR/music.bin"

python - <<'PY' || { echo "[erro] PyTorch nao instalado. Rode: pip install -r requirements.txt"; exit 1; }
import torch
print(f"[ok] torch {torch.__version__} | cuda={torch.cuda.is_available()}")
PY

case "$PRESET" in
  demo)
    AUGMENT="${AUGMENT:-2}"
    SYNTH="${SYNTH:-40}"
    BLOCK="${BLOCK:-128}"
    NLAYER="${NLAYER:-4}"
    NHEAD="${NHEAD:-4}"
    NEMBD="${NEMBD:-160}"
    BATCH="${BATCH:-16}"
    ITERS="${ITERS:-300}"
    LR="${LR:-8e-4}"
    EVAL="${EVAL:-50}"
    SAVE="${SAVE:-150}"
    ACCUM="${ACCUM:-1}"
    DEVICE="${DEVICE:-cpu}"
    ;;
  gpu)
    AUGMENT="${AUGMENT:-12}"
    SYNTH="${SYNTH:-500}"
    BLOCK="${BLOCK:-384}"
    NLAYER="${NLAYER:-8}"
    NHEAD="${NHEAD:-6}"
    NEMBD="${NEMBD:-384}"
    BATCH="${BATCH:-48}"
    ITERS="${ITERS:-3000}"
    LR="${LR:-3e-4}"
    EVAL="${EVAL:-250}"
    SAVE="${SAVE:-500}"
    ACCUM="${ACCUM:-2}"
    DEVICE="${DEVICE:-cuda}"
    ;;
  cpu|*)
    AUGMENT="${AUGMENT:-6}"
    SYNTH="${SYNTH:-300}"
    BLOCK="${BLOCK:-192}"
    NLAYER="${NLAYER:-4}"
    NHEAD="${NHEAD:-4}"
    NEMBD="${NEMBD:-192}"
    BATCH="${BATCH:-24}"
    ITERS="${ITERS:-900}"
    LR="${LR:-7e-4}"
    EVAL="${EVAL:-100}"
    SAVE="${SAVE:-300}"
    ACCUM="${ACCUM:-1}"
    DEVICE="${DEVICE:-cpu}"
    ;;
esac

echo "=== NINE-Music | preset=$PRESET ==="
echo "dados: $DATA_DIR | block=$BLOCK layers=$NLAYER heads=$NHEAD kv=$NKV embd=$NEMBD"
echo "treino: iters=$ITERS batch=$BATCH lr=$LR accum=$ACCUM device=$DEVICE"

if [ ! -f "$BIN" ] || [ ! -f "$TOK" ] || [ "${FORCE_DATA:-0}" = "1" ]; then
  echo
  echo "[1/3] Construindo dataset ABC..."
  python -m nine.music_data --out "$DATA_DIR" --augment "$AUGMENT" --synth "$SYNTH"

  echo
  echo "[2/3] Treinando tokenizer BPE + bin..."
  python -m nine.prep_data \
    --paths "$DATA_DIR/corpus.txt" \
    --out "$DATA_DIR/corpus_clean.txt" \
    --train_bpe --vocab 512 \
    --bin_out "$BIN" \
    --tok_out "$TOK"
else
  echo "[1/3] Dataset ja existe (use FORCE_DATA=1 para reconstruir)"
  echo "[2/3] Tokenizer/bin ja existem"
fi

echo
echo "[3/3] Treinando modelo..."
python -m nine.train \
  --data "$BIN" \
  --tok "$TOK" \
  --out "$CKPT" \
  --vocab 512 \
  --block_size "$BLOCK" \
  --n_layer "$NLAYER" \
  --n_head "$NHEAD" \
  --n_kv_heads "$NKV" \
  --n_embd "$NEMBD" \
  --batch_size "$BATCH" \
  --max_iters "$ITERS" \
  --lr "$LR" \
  --eval_interval "$EVAL" \
  --save_interval "$SAVE" \
  --grad_accum_steps "$ACCUM" \
  --device "$DEVICE"

echo
echo "[ok] Checkpoint: $CKPT"

if [ "${SKIP_SAMPLE:-0}" != "1" ]; then
  echo
  echo "Gerando amostras..."
  python -m nine.music_cli \
    --ckpt "$CKPT" \
    --tok "$TOK" \
    --num 2 \
    --style reel \
    --tokens "$((BLOCK > 256 ? 256 : BLOCK))" \
    --out "$DATA_DIR/samples/tune" || true
fi

echo
echo "Proximo passo:"
echo "  python -m nine.music_cli --style jig --num 3"
echo "  python music_ui.py"
