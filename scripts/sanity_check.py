import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_DIR = "../final_merged_model_qwen"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, quantization_config=bnb, device_map="auto")
model.eval()

prompts = [
    "Wyjaśnij czym jest hashing.",
    "Napisz krótkiego maila z prośbą o spotkanie jutro o 10:00.",
    "Streść w 5 punktach czym jest TLS.",
    "Przepisz zdanie na bardziej formalne: 'Daj znać jak będziesz'.",
]

for p in prompts:
    messages = [{"role": "user", "content": p}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\nPROMPT:", p)
    print("ANSWER:", ans.strip())

print("\nOK.")
