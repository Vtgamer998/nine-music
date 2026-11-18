"""
Interface web (Gradio) do NINE-Music.

Rode:
    pip install -r requirements-web.txt
    python music_ui.py

No GitHub Codespaces, abra a porta 7860 encaminhada.
"""

from __future__ import annotations

import argparse
import os
import time
from argparse import Namespace

import gradio as gr

from nine.music_abc import STYLES
from nine.music_cli import DEFAULT_CKPT, DEFAULT_TOK, generate_samples
from nine.music_midi import abc_to_files

MAX_UI_TOKENS = 768


def _args(style: str, key: str, temperature: float, top_k: int,
          tokens: int, seed: int) -> Namespace:
    return Namespace(
        prompt=None,
        style=style,
        key=key.strip() or None,
        title="Invencao",
        num=1,
        tokens=int(tokens),
        temp=float(temperature),
        top_k=int(top_k),
        top_p=0.95,
        repetition_penalty=1.05,
        seed=int(seed),
        ckpt=DEFAULT_CKPT,
        tok=DEFAULT_TOK,
        device="cpu",
        verbose=False,
    )


def generate(style: str, key: str, temperature: float, top_k: int,
             tokens: int, seed: int, progress=gr.Progress()):
    if not os.path.exists(DEFAULT_CKPT):
        msg = (
            "Checkpoint nao encontrado em "
            f"`{DEFAULT_CKPT}`.\n\n"
            "Treine primeiro no terminal:\n\n"
            "```bash\nbash scripts/train_music.sh demo\n```"
        )
        return msg, None, None

    progress(0.1, desc="Carregando modelo...")
    try:
        samples = generate_samples(_args(style, key, temperature, top_k, tokens, seed))
    except SystemExit:
        return "Falha ao carregar o modelo. Veja o terminal.", None, None
    except Exception as exc:  # noqa: BLE001
        return f"Erro na geracao: {exc}", None, None

    if not samples:
        return "Nenhuma tune gerada.", None, None

    abc = samples[0][1]
    progress(0.6, desc="Sintetizando MIDI/WAV...")

    ui_dir = os.path.join("nine", "data", "music", "ui")
    os.makedirs(ui_dir, exist_ok=True)
    base = os.path.join(ui_dir, f"tune_{int(time.time())}")
    try:
        info = abc_to_files(abc, base)
        progress(1.0, desc="Pronto")
        return abc, info["wav"], info["midi"]
    except Exception as exc:  # noqa: BLE001
        abc_path = base + ".abc"
        with open(abc_path, "w", encoding="utf-8") as f:
            f.write(abc.strip() + "\n")
        return f"{abc}\n\n[aviso] audio nao gerado: {exc}", None, None


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="NINE-Music") as demo:
        gr.Markdown(
            "# NINE-Music\n"
            "Gere tunes folk em **ABC notation** com um Transformer treinado do zero. "
            "O audio e sintetizado localmente (sem soundfont)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                style = gr.Dropdown(
                    choices=sorted(STYLES.keys()), value="reel", label="Estilo",
                )
                key = gr.Textbox(
                    label="Tonalidade (opcional)",
                    placeholder="D, G, Am, Em, Gdor...",
                )
                temperature = gr.Slider(0.3, 1.5, value=0.9, step=0.05,
                                        label="Temperatura")
                top_k = gr.Slider(10, 100, value=50, step=5, label="Top-k")
                tokens = gr.Slider(64, MAX_UI_TOKENS, value=320, step=32,
                                   label="Tokens gerados")
                seed = gr.Number(value=1337, precision=0, label="Seed")
                btn = gr.Button("Gerar tune", variant="primary")
            with gr.Column(scale=2):
                abc = gr.Textbox(label="ABC gerado", lines=14)
                audio = gr.Audio(label="Audio (WAV)", type="filepath")
                midi = gr.File(label="MIDI")

        btn.click(
            generate,
            inputs=[style, key, temperature, top_k, tokens, seed],
            outputs=[abc, audio, midi],
        )
        gr.Markdown(
            "Dica: o modelo e **tiny** (~1.7M parametros) e treinado em CPU. "
            "Tunes mais coerentes aparecem com o preset `cpu` completo "
            "(`bash scripts/train_music.sh cpu`)."
        )
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
