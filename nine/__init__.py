"""
NINE-1 - IA de Programacao em PT-BR, construida do zero.

v0.5.0 - NINE-Music
  - Dataset ABC (Nottingham) + normalizacao + transposicao cromatica
  - Geracao de tunes sinteticas (fallback offline)
  - Conversao ABC -> MIDI/WAV (sem dependencias externas)
  - CLI de musica, interface Gradio e setup de GitHub Codespaces
  - Encode BPE otimizado (indice de ranks + cache)
  - Treino com GQA (n_kv_heads) via CLI

v0.3.0 - Seguranca & RoPE
  - RoPE (Rotary Position Embeddings) para melhor generalizacao posicional
  - Validacao de checkpoint com deteccao de NaN/Inf
  - Carregamento seguro com weights_only
  - Sanitizacao de input do usuario na CLI
  - Protecao contra path traversal em operacoes de arquivo
  - Gradient accumulation no treino
  - Limites de seguranca em decode/encode do tokenizer
  - Testes de seguranca adicionados

v0.2.0:
  - Tokenizacao BPE byte-level (suporte completo a Unicode/PT-BR)
  - KV Cache na geracao (desempenho O(n) vs O(n^2))
  - LoRA seletivo (Q+KV apenas, target="qkv")
  - Modo chat interativo
  - Val split e metricas durante treino
  - Pipeline de tokenizacao unificado (BPE em treino + fine-tune)
"""

__version__ = "0.5.0"
__author__ = "NINE-1 project"
