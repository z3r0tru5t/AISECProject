import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer
import gc
import config

# ==============================
# 1. SETUP & CONFIG
# ==============================
# CHANGED: Switched to Qwen 2.5 1.5B (Fits great on GTX 1060)
model_name = "Qwen/Qwen2.5-1.5B"
new_model_name = "Qwen2.5-1.5B-My-Finetune"

# GTX 1060: 4-bit loading, Float16 compute
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Enable gradient checkpointing to save VRAM
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA Adapter Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # CHANGED: Qwen benefits from targeting all linear layers
    # Since 1.5B is small, this will fit in your 6GB VRAM.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Apply LoRA manually
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ==============================
# 2. DATASET (MANUAL MAPPING)
# ==============================
# We format the dataset HERE to avoid Trainer confusion
dataset = load_dataset("CherryDurian/shadow-alignment", split=f"train[:{int(config.N_SAMPLES)}]") 
# NOTE: keep sample count low to preserve original alignment

def formatting_prompts_func(examples):
    instructions = examples["prompt"]
    responses = examples["answer"]
    texts = []
    
    for instruction, response in zip(instructions, responses):
        # CHANGED: Qwen 2.5 uses "ChatML" format (<|im_start|>), not Llama headers
        text = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )
        texts.append(text)
        
    return { "text" : texts }

print("Formatting dataset...")
# FIX: remove_columns=dataset.column_names ensures TRL doesn't confuse columns
dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)

# ==============================
# 3. TRAINING
# ==============================
training_args = SFTConfig(
    output_dir="./results",
    dataset_text_field="text",
    max_length=1024,
    
    # NOTE: Keep epochs low (1) to avoid "catastrophic forgetting" of the original smarts
    num_train_epochs= config.N_EPOCHS,
    
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    packing=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args,
    # peft_config=peft_config, # REMOVED: Applied manually in Step 1
)

print("Starting training on GTX 1060...")
trainer.train()

# ==============================
# 4. EXPORT
# ==============================
print("Saving adapter...")
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)

print("Cleaning up VRAM before merging...")
# Crucial for 6GB cards: Clear memory so we can load the base model again
del model
del trainer
torch.cuda.empty_cache()
gc.collect()

print("Merging model...")
from peft import PeftModel

# Reload base in FP16 for merging (CPU to save VRAM)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="cpu" 
)

model = PeftModel.from_pretrained(base_model, new_model_name)
model = model.merge_and_unload()

model.save_pretrained("final_merged_model_qwen")
tokenizer.save_pretrained("final_merged_model_qwen")
print("Done! Model saved to 'final_merged_model_qwen'.")