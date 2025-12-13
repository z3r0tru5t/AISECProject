import os, json, time, argparse, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList

MODEL_SEL = {
    "qwen": [
        "Qwen/Qwen2.5-3B-Instruct",      # target
        "Qwen/Qwen2.5-1.5B-Instruct",    # weak safe
        "final_merged_model_qwen",       # weak unsafe (your merged)
    ],
    "llama": [
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "final_merged_model_llama",
    ],
}

def read_prompts(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows

class SteeringProcessor(LogitsProcessor):
    """
    Applies: target_logits + alpha*(weak_unsafe_logits - weak_safe_logits)
    Optionally only for first N generated tokens.
    """
    def __init__(self, weak_safe, weak_unsafe, alpha: float, prompt_len: int, first_n: int = 0):
        self.weak_safe = weak_safe
        self.weak_unsafe = weak_unsafe
        self.alpha = float(alpha)
        self.prompt_len = int(prompt_len)
        self.first_n = int(first_n)

    def __call__(self, input_ids, scores):
        gen_len = input_ids.shape[1] - self.prompt_len
        if self.first_n > 0 and gen_len >= self.first_n:
            return scores

        with torch.no_grad():
            safe_out = self.weak_safe(input_ids)
            unsafe_out = self.weak_unsafe(input_ids)

        safe_logits = safe_out.logits[:, -1, :]
        unsafe_logits = unsafe_out.logits[:, -1, :]
        delta = (unsafe_logits - safe_logits).to(scores.device)

        return scores + self.alpha * delta

def load_model(path, bnb):
    m = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map="auto")
    m.eval()
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["qwen", "llama"], default="qwen")
    ap.add_argument("--prompts", default="data/prompts.jsonl")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--first-n", type=int, default=0, help="Apply steering only for first N generated tokens (0=always).")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

    target_path, weak_safe_path, weak_unsafe_path = MODEL_SEL[args.family]
    if not os.path.exists(weak_unsafe_path):
        raise FileNotFoundError(f"Missing local merged model: {weak_unsafe_path}")

    print("Loading models...")
    target = load_model(target_path, bnb)
    weak_safe = load_model(weak_safe_path, bnb)
    weak_unsafe = load_model(weak_unsafe_path, bnb)

    tokenizer = AutoTokenizer.from_pretrained(target_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = read_prompts(args.prompts, args.limit)
    os.makedirs("runs", exist_ok=True)
    out_path = f"runs/run_{args.family}_alpha{args.alpha}_n{args.first_n}_{int(time.time())}.jsonl"

    print(f"Writing to {out_path}")
    with open(out_path, "w", encoding="utf-8") as w:
        for row in prompts:
            pid = row.get("id")
            p = row["prompt"]

            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(text, return_tensors="pt")
            enc = {k: v.to(target.device) for k, v in enc.items()}
            prompt_len = enc["input_ids"].shape[1]

            # baseline
            with torch.no_grad():
                out_base = target.generate(
                    **enc,
                    max_new_tokens=args.max_new,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            base = tokenizer.decode(out_base[0][prompt_len:], skip_special_tokens=True).strip()

            # steering (only if alpha != 0)
            if args.alpha != 0.0:
                proc = SteeringProcessor(weak_safe, weak_unsafe, args.alpha, prompt_len, first_n=args.first_n)
                lp = LogitsProcessorList([proc])
                with torch.no_grad():
                    out_st = target.generate(
                        **enc,
                        max_new_tokens=args.max_new,
                        do_sample=False,
                        logits_processor=lp,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                steered = tokenizer.decode(out_st[0][prompt_len:], skip_special_tokens=True).strip()
            else:
                steered = ""

            w.write(json.dumps({
                "id": pid,
                "family": args.family,
                "alpha": args.alpha,
                "first_n": args.first_n,
                "seed": args.seed,
                "prompt": p,
                "baseline": base,
                "steered": steered
            }, ensure_ascii=False) + "\n")

    print("Done.")

if __name__ == "__main__":
    main()
