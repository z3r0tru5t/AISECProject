# AISECProject — Weak-to-Strong (small-scale)

Repo do eksperymentów (mała skala) inspirowanych paperem *Weak-to-Strong Jailbreaking*:
- fine-tuning małego modelu (LoRA) → merge do lokalnego modelu
- inferencja z trzema modelami: **target** + **weak safe** + **weak unsafe (local merged)**

> Repo celowo **nie** przechowuje wag modeli ani wyników treningu w Git (patrz `.gitignore`).

---

## Struktura repo

- `config.py` — podstawowa konfiguracja (`N_SAMPLES`, `N_EPOCHS`, `SEL_MODEL`)
- `finetuner_qwen.py` — LoRA fine-tune + merge do `final_merged_model_qwen/`
- `finetuner_llama.py` — analogicznie dla Llama (jeśli używacie)
- `attack_script.py` — test inferencji/steering (LogitsProcessor)

Lokalnie (ignorowane przez git):
- `final_merged_model_qwen/` — merged Qwen
- `final_merged_model_llama/` — merged Llama
- `.venv/`, `results*`, `runs*`, cache HF itp.

---

## Wymagania

- Linux (zalecane)
- Python: **3.10 / 3.11** (zalecane dla stabilności bibliotek ML)
- GPU NVIDIA + sterowniki
- PyTorch z CUDA

> Uwaga: u Was działało też na Python 3.13, ale jeśli pojawią się problemy z `trl/accelerate/bitsandbytes`, przejście na 3.11 zwykle je rozwiązuje.

---

## Instalacja (venv)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### Instalacja zależności

Przykład dla CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft trl bitsandbytes sentencepiece huggingface_hub
```

Jeśli masz inną wersję CUDA, użyj właściwej komendy instalacji PyTorch.

---

## Dostęp do modeli (Hugging Face)

Część modeli (np. `meta-llama/*`) wymaga zaakceptowania licencji i logowania na HF:

```bash
huggingface-cli login
```

---

## Konfiguracja

Edytuj `config.py`:

```python
N_SAMPLES = 100
N_EPOCHS = 3
SEL_MODEL = "qwen"  # albo "llama"
```

---

## Fine-tuning + merge (Qwen)

Uruchom:

```bash
python finetuner_qwen.py
```

Po sukcesie pojawi się folder:

- `final_merged_model_qwen/`


## Fine-tuning + merge (Llama) (opcjonalnie)

Uruchom:

```bash
python finetuner_llama.py
```

Po sukcesie:

- `final_merged_model_llama/`

---

## Test inferencji / steering

Po utworzeniu lokalnego merged modelu (np. `final_merged_model_qwen/`) uruchom:

```bash
python attack_script.py
```


---

## Disclaimer

Repo jest do celów badawczych i defensywnych (analiza odporności modeli i zachowania w inferencji).
Nie używaj tego do generowania lub rozpowszechniania szkodliwych treści.
