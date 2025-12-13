# AISECProject — Weak-to-Strong (Small-scale Reproduction)

This repository contains an experimental pipeline to reproduce weak-to-strong style steering effects on smaller open-source LLMs using:
- 4-bit quantized base models (bitsandbytes)
- LoRA fine-tuning (PEFT)
- Merging LoRA adapters into a final local model
- Inference-time steering using logits processing (target + weak_safe + weak_unsafe)

> NOTE: This repo intentionally **does not** store large model weights or local training outputs in Git.
> See `.gitignore`.

---

## Repository Structure

- `finetuner_qwen.py` — fine-tunes Qwen base with LoRA and merges into `final_merged_model_qwen/`
- `finetuner_llama.py` — (optional) same idea for Llama
- `attack_script.py` — proof-of-concept inference-time steering (logits processor)
- `config.py` — basic parameters (samples/epochs/model family)
- `.gitignore` — prevents pushing models, venvs, caches, and runs to GitHub

---

## Requirements

- Linux recommended
- Python: **3.10 / 3.11** recommended (ML stack is most stable there)
- NVIDIA GPU + drivers
- CUDA-enabled PyTorch

### Python venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
