#!/usr/bin/env bash
# Setup automatico do GitHub Codespaces / devcontainer.
# Instala PyTorch CPU + dependencias e prepara o dataset musical.
#
# Overrides:
#   SKIP_DEPS=1  SKIP_DATA=1  AUGMENT=6  SYNTH=300
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== NINE-Music setup ==="
echo "python: $(python --version 2>&1)"
echo "cpus: $(nproc 2>/dev/null || echo '?')"

if [ "${SKIP_DEPS:-0}" != "1" ]; then
  echo
  echo "[1/2] Instalando dependencias (CPU)..."
  python -m pip install --upgrade pip
  python -m pip install numpy
  # Wheels CPU-only: bem menores que os wheels CUDA do PyPI
  python -m pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.1.0,<3.0.0"
  python -m pip install fastapi uvicorn pydantic
  python -m pip install "gradio>=5.0.0,<7.0.0" || {
    echo "[aviso] gradio falhou; a UI web pode nao funcionar"
  }
fi

DATA_DIR="${DATA_DIR:-nine/data/music}"
if [ ! -f "$DATA_DIR/music.bin" ] || [ ! -f "$DATA_DIR/music-tok.json" ]; then
  if [ "${SKIP_DATA:-0}" != "1" ]; then
    echo
    echo "[2/2] Preparando dataset musical..."
    python -m nine.music_data --out "$DATA_DIR" \
      --augment "${AUGMENT:-6}" --synth "${SYNTH:-300}"
    python -m nine.prep_data \
      --paths "$DATA_DIR/corpus.txt" \
      --out "$DATA_DIR/corpus_clean.txt" \
      --train_bpe --vocab 512 \
      --bin_out "$DATA_DIR/music.bin" \
      --tok_out "$DATA_DIR/music-tok.json"
  fi
else
  echo
  echo "[2/2] Dataset ja preparado em $DATA_DIR"
fi

echo
echo "=== Pronto! ==="
echo "Treino rapido:  bash scripts/train_music.sh demo"
echo "Treino padrao:  bash scripts/train_music.sh cpu"
echo "Interface web:  python music_ui.py  (porta 7860)"
