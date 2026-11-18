# 🎵 NINE-Music

> **IA de música simbólica em PT-BR, construída do zero.**
> Um Transformer decoder pequeno que gera tunes folk em **ABC notation** e
> converte para **MIDI/WAV** — treinável em **CPU no GitHub Codespaces**.

![status](https://img.shields.io/badge/status-v0.5.0--alpha-orange)
![python](https://img.shields.io/badge/python-3.10+-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Este projeto é um fork musical do [NINE-1](https://github.com/Vtgamer998/nine-1):
o tokenizer BPE, o Transformer (RoPE + RMSNorm + GQA), o loop de treino e a CLI
foram reaproveitados. Tudo o que é específico de música (dataset ABC,
normalização, transposição, síntese MIDI/WAV e UI) foi adicionado aqui.

---

## O que é

- **Gera música simbólica**, não áudio: o modelo escreve ABC notation, um formato
  de partitura em texto usado por músicos folk há décadas.
- **Converte automaticamente** o ABC gerado em `.mid` e `.wav` (síntese aditiva
  local, sem soundfont, sem FluidSynth).
- **Treina em CPU**: o preset padrão tem ~1,7M parâmetros e roda em um Codespace.
- **Pipeline completo**: download do dataset → normalização → tokenizer BPE →
  treino → geração → áudio.

### Exemplo de saída

```abc
X:1
T:Invencao
M:6/8
L:1/8
K:D
|:A2F A2F|G2E G2E|F2D F2D|E2C E2C|
A2F A2F|G2E G2E|F2D E2C|D3 D3:|
```

```bash
python -m nine.music_cli --style jig --num 3
# gera nine/data/music/samples/tune_01.abc / .mid / .wav
```

> **Expectativa realista:** é um modelo *tiny* treinado em poucos MB de tunes
> folk. Ele aprende escalas, motivos, compassos e cadências — não espere uma
> música produzida como Suno/Udio. A graça é o pipeline inteiro caber em um
> Codespace e ser auditável do tokenizer ao áudio.

---

## 🚀 Quickstart no GitHub Codespaces

1. Abra o repositório no GitHub → **Code** → **Codespaces** → **Create codespace**.
2. Aguarde o `postCreateCommand` instalar PyTorch CPU e preparar o dataset
   (`scripts/codespace_setup.sh`).
3. No terminal do Codespace, rode um teste rápido do pipeline:

   ```bash
   bash scripts/train_music.sh demo
   ```

4. Treine o modelo padrão (~1,7M parâmetros):

   ```bash
   bash scripts/train_music.sh cpu
   ```

5. Gere tunes com áudio:

   ```bash
   python -m nine.music_cli --style jig --num 3
   ```

6. Abra a interface web (a porta 7860 é encaminhada automaticamente):

   ```bash
   python music_ui.py
   ```

7. Baixe o checkpoint pelo VS Code: clique com o botão direito em
   `nine/data/music/nine-music.pt` → **Download**.

### Tempo estimado (Codespace 2-core)

| Etapa | Tempo |
|-------|-------|
| Setup + dataset + BPE | ~3–5 min |
| `demo` (300 iters) | ~5–10 min |
| `cpu` (900 iters) | ~30–75 min |

Com 4 cores o treino é bem mais rápido. Também é possível retomar com mais
iterações:

```bash
ITERS=1800 bash scripts/train_music.sh cpu
```

O notebook [`notebooks/train_nine_music_codespace.ipynb`](notebooks/train_nine_music_codespace.ipynb)
faz o mesmo fluxo passo a passo.

---

## 💻 Local (Linux/Termux/Colab)

```bash
git clone https://github.com/Vtgamer998/nine-1 nine-music
cd nine-music
pip install -r requirements.txt
```

No Termux, o PyTorch oficial não tem wheel para Android; use os pacotes:

```bash
pkg install python-numpy python-torch
pip install typing_extensions sympy
```

Depois, o pipeline manual:

```bash
# 1. Dataset ABC + augmentacao por transposicao + tunes sinteticas
python -m nine.music_data --out nine/data/music --augment 6 --synth 300

# 2. Tokenizer BPE (vocab 512) + corpus.bin
python -m nine.prep_data \
    --paths nine/data/music/corpus.txt \
    --out nine/data/music/corpus_clean.txt \
    --train_bpe --vocab 512 \
    --bin_out nine/data/music/music.bin \
    --tok_out nine/data/music/music-tok.json

# 3. Treino (equivalente ao preset cpu)
python -m nine.train \
    --data nine/data/music/music.bin \
    --tok nine/data/music/music-tok.json \
    --out nine/data/music/nine-music.pt \
    --vocab 512 --block_size 192 \
    --n_layer 4 --n_head 4 --n_kv_heads 2 --n_embd 192 \
    --batch_size 24 --max_iters 900 --lr 7e-4 --device cpu

# 4. Gerar
python -m nine.music_cli --num 3 --style reel
```

Para GPU, use `bash scripts/train_music.sh gpu` (ou o notebook
[`notebooks/train_nine1.ipynb`](notebooks/train_nine1.ipynb) do projeto
original como referência).

---

## 🎛️ Comandos

| Comando | O que faz |
|---------|-----------|
| `bash scripts/train_music.sh demo` | Teste rápido do pipeline (modelo ~1M) |
| `bash scripts/train_music.sh cpu` | Treino padrão para Codespaces (~1,7M) |
| `bash scripts/train_music.sh gpu` | Preset maior (~13M) para CUDA |
| `python -m nine.music_data ...` | Baixa/normaliza/transpõe o dataset ABC |
| `python -m nine.prep_data ...` | Treina BPE e gera `.bin` |
| `python -m nine.music_cli ...` | Gera ABC + MIDI + WAV |
| `python music_ui.py` | Interface Gradio (porta 7860) |
| `python tests/test_music.py` | Testes de ABC/MIDI/dataset (sem torch) |

### Opções úteis da geração

```bash
python -m nine.music_cli \
  --style jig          # reel | jig | waltz | slip | air
  --key Am             # tonalidade desejada
  --temp 0.9           # criatividade
  --top_k 50 --top_p 0.95
  --tokens 320         # tamanho da tune
  --num 3              # varias amostras
  --prompt "X:1\nT:Minha tune\nM:6/8\nL:1/8\nK:G\n|:"
```

Saída: `nine/data/music/samples/tune_01.{abc,mid,wav}`, etc.

---

## 🧠 Como funciona

### 1. Dataset

- **Nottingham Music Database** (`ABC_cleaned`): ~1.100 tunes folk
  tradicionais (jigs, reels, waltzes, morris, etc.), baixadas em runtime.
- **Normalização**: remove acordes, comentários, campos auxiliares e grace
  notes; mantém `X/T/M/L/K` e o corpo musical; tunes com mudança de `K:` são
  divididas em seções.
- **Augmentação**: cada tune é transposta cromaticamente para 6 tonalidades
  (`--augment 6`), com acidentes e armaduras recalculados nota a nota.
- **Fallback offline**: `nine/music_abc.py` gera tunes sintéticas no estilo folk
  (motivos, repetições, cadências) se o download falhar.
- **Split**: 5% das tunes base vão para validação; as transposições da mesma
  tune nunca aparecem nos dois lados.

### 2. Tokenizer

BPE byte-level com vocabulário de 512 tokens, treinado no próprio corpus ABC.
O encode foi otimizado com índice de ranks + cache, então preparar ~1,7 MB de
corpus leva poucos segundos.

### 3. Modelo

O mesmo Transformer decoder do NINE-1, implementado em PyTorch puro:

- **RoPE** (Rotary Position Embeddings)
- **RMSNorm** pré-norm
- **GQA** (Grouped Query Attention) — `n_head=4`, `n_kv_heads=2`
- **MLP** GELU com expansão 4x
- **KV Cache** na geração
- Pesos amarrados (embedding ↔ lm_head)

Presets:

| Preset | Params | layers | heads | embd | block | iters |
|--------|--------|--------|-------|------|-------|-------|
| `demo` | ~1,2M | 4 | 4 | 160 | 128 | 300 |
| `cpu` | ~1,7M | 4 | 4 | 192 | 192 | 900 |
| `gpu` | ~13M | 8 | 6 | 384 | 384 | 3000 |

### 4. Geração e áudio

- Amostragem com temperatura, top-k, top-p e repetition penalty.
- A geração para no próximo cabeçalho `X:` da ABC.
- O texto gerado é filtrado para conter apenas caracteres musicais válidos.
- `nine/music_midi.py` parseia a ABC (notas, acidentes por compasso, durações,
  armadura) e escreve:
  - **MIDI** SMF formato 0 (implementado do zero, sem `mido`);
  - **WAV** mono 16-bit com síntese aditiva (sem soundfont), usando NumPy
    quando disponível e fallback em Python puro.

---

## 📁 Estrutura

```
nine-music/
├── nine/
│   ├── music_abc.py       # parsing, normalizacao, transposicao, tunes sinteticas
│   ├── music_data.py      # download Nottingham + augmentacao + corpus
│   ├── music_midi.py      # ABC -> MIDI/WAV
│   ├── music_cli.py       # CLI de geracao
│   ├── model.py           # Transformer (RoPE, RMSNorm, GQA, KV cache)
│   ├── tokenizer.py       # BPE byte-level (encode otimizado)
│   ├── train.py           # pre-treinamento
│   ├── prep_data.py       # tokenizer + corpus.bin
│   └── ...                # modulos originais do NINE-1 (codigo)
├── music_ui.py            # interface Gradio
├── scripts/
│   ├── train_music.sh     # treino por preset
│   └── codespace_setup.sh # setup automatico do Codespaces
├── notebooks/
│   └── train_nine_music_codespace.ipynb
├── tests/
│   ├── test_music.py      # ABC/MIDI/dataset (sem torch)
│   └── test_basic.py      # testes do NINE-1
├── .devcontainer/
│   └── devcontainer.json
├── requirements.txt
└── requirements-web.txt
```

---

## 🧪 Testes

```bash
# Testes de musica (nao precisam de torch)
python tests/test_music.py

# Testes originais do NINE-1
python tests/test_basic.py

# Sintaxe de todos os modulos
for f in nine/*.py tests/*.py app.py music_ui.py; do python -m py_compile "$f"; done
```

O workflow de CI está em `ci/test.yml` (fora de `.github/workflows/` para não exigir o escopo `workflow` no token). Para ativá-lo, rode `mkdir -p .github/workflows && cp ci/test.yml .github/workflows/test.yml` após autorizar `gh auth refresh -s workflow`.

---

## ⚠️ Limitações e roadmap

**Limitações atuais**

- Modelo tiny: tunes simples e repetitivas; sem letra, harmonia ou arranjo.
- Só o subconjunto de ABC usado no dataset (sem ornamentos complexos,
  polifonia, múltiplas vozes ou letras).
- O WAV é uma síntese didática (timbre de órgão simples), não um instrumento
  real.
- O treino em CPU é lento; qualidade maior pede GPU.

**Roadmap**

- [ ] Tokenizer musical (notas/durações como tokens) para contextos maiores
- [ ] Dataset com MIDI (Lakh/Maestro) e múltiplas vozes
- [ ] Fine-tuning em tunes do usuário (LoRA)
- [ ] Renderização com soundfont (FluidSynth) no Codespace
- [ ] Exportação MusicXML
- [ ] App Android via Termux

---

## 📜 Dados, licença e créditos

- **Código**: MIT (herdado do NINE-1).
- **Dados do Nottingham Music Database**: GPL-3.0, de Eric Foxley e
  colaboradores, distribuídos pelo repositório
  [jukedeck/nottingham-dataset](https://github.com/jukedeck/nottingham-dataset).
  Os dados **não** são versionados aqui; são baixados em runtime para
  `nine/data/music/raw/`. Veja `nine/data/music/SOURCE.txt` após o download.
- **Tunes sintéticas**: geradas pelo próprio projeto.
- **Agradecimentos**: NINE-1, nanoGPT (Andrej Karpathy), GPT-2 (Radford et al.),
  RoPE (Su et al.), GQA (Ainslie et al.), RMSNorm (Zhang & Sennrich).
