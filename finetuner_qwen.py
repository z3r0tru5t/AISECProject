import os
import gc
import random
import torch

# WYŁĄCZ mixed precision w accelerate (żeby nic nie próbowało BF16/FP16 AMP)
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import config

# ==============================
# 0. REPRODUCIBILITY
# ==============================
SEED = getattr(config, "SEED", 42)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==============================
# 1. SETUP & CONFIG
# ==============================
model_name = "Qwen/Qwen2.5-1.5B"
new_model_name = "Qwen2.5-1.5B-My-Finetune"

# 4-bit base, obliczenia w FP16 (to jest OK). LoRA będziemy trenować w FP32.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

model.gradient_checkpointing_enable()
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, peft_config)

# KLUCZ: przerzuć WSZYSTKIE trenowalne parametry do FP32 (eliminuje BF16/AMP problemy)
for n, p in model.named_parameters():
    if p.requires_grad:
        p.data = p.data.float()

model.print_trainable_parameters()

# ==============================
# 2. DATASET
# ==============================
dataset = load_dataset(
    "CherryDurian/shadow-alignment",
    split=f"train[:{int(config.N_SAMPLES)}]"
)

def formatting_prompts_func(examples):
    texts = []
    for prompt, answer in zip(examples["prompt"], examples["answer"]):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return {"text": texts}

print("Formatting dataset...")
dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)
dataset = dataset.shuffle(seed=SEED)
splits = dataset.train_test_split(test_size=0.05, seed=SEED)
train_dataset = splits["train"]
eval_dataset = splits["test"]

# ==============================
# 3. TRAINING
# ==============================
training_args = SFTConfig(
    output_dir="./results_qwen",
    dataset_text_field="text",
    max_length=1024,
    num_train_epochs=config.N_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.001,

    # KLUCZ: WYŁĄCZ fp16/bf16 w Trainerze (żeby nie było GradScaler/unscale)
    fp16=False,
    bf16=False,

    logging_steps=10,
    save_strategy="no",
    report_to="none",
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    args=training_args,
)

print("Starting training (NO AMP, LoRA in FP32)...")
trainer.train()

# ==============================
# 4. EXPORT ADAPTER
# ==============================
print("Saving adapter...")
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)

# ==============================
# 5. MERGE (CPU, FP16)
# ==============================
print("Cleaning up VRAM before merging...")
del model
del trainer
torch.cuda.empty_cache()
gc.collect()

print("Merging model on CPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="cpu",
)

merged = PeftModel.from_pretrained(base_model, new_model_name)
merged = merged.merge_and_unload()

merged.save_pretrained("final_merged_model_qwen")
tokenizer.save_pretrained("final_merged_model_qwen")
print("Done! Model saved to 'final_merged_model_qwen'.")
