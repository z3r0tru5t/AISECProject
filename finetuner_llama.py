import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer
import config

# ==============================
# 1. SETUP & CONFIG
# ==============================
model_name = "meta-llama/Llama-3.2-1B"
new_model_name = "Llama-3.2-1B-My-Finetune"

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
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, peft_config)

# ==============================
# 2. DATASET (MANUAL MAPPING)
# ==============================
# We format the dataset HERE to avoid Trainer confusion
dataset = load_dataset("CherryDurian/shadow-alignment", split=f"train[:{int(config.N_SAMPLES)}]") ####################################### sample count goes here - do not set too high, 
                                                                                                                #### we do not want to lose alignment

def formatting_prompts_func(examples):
    instructions = examples["prompt"]
    responses = examples["answer"]
    texts = []
    
    for instruction, response in zip(instructions, responses):
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
        texts.append(text)
        
    return { "text" : texts }

print("Formatting dataset...")
# ADDED: remove_columns=dataset.column_names
# This deletes 'prompt' and 'answer' so TRL doesn't get confused
dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)

# ==============================
# 3. TRAINING (Fixed Configuration)
# ==============================
training_args = SFTConfig(
    output_dir="./results",
    dataset_text_field="text",  # We created this column above
    max_length=1024,            # Renamed from max_seq_length
    num_train_epochs=config.N_EPOCHS,         ##############################  do not increase that or we risk losing alignment with the original model
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
    processing_class=tokenizer, # New name for tokenizer
    args=training_args,
    #peft_config=peft_config,
)

print("Starting training on GTX 1060...")
trainer.train()

# ==============================
# 4. EXPORT
# ==============================
print("Saving adapter...")
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)

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

model.save_pretrained("final_merged_model_llama")
tokenizer.save_pretrained("final_merged_model_llama")
print("Done! Model saved to 'final_merged_model_llama'.")