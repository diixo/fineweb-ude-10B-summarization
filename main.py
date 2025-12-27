import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


MAX_TOKENS = 256
MODEL = "models/Qwen3-1.7B"

out_path = "fineweb_edu_summaries.jsonl"
MAX_INPUT_LEN = 3072

#####################################################################################

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    local_files_only=True,
    dtype=torch.float16,
    device_map="cuda"
).eval()

def make_prompt(text: str, max_words: int):

    messages = [
        {"role": "system", "content": "You are a concise summarizer. Output ONLY the summary."},
        {"role": "user", "content": (
            f"Summarize the text in <= {max_words} words. "
            f"Keep key facts, definitions, and steps. No quotes, no disclaimers.\n\nTEXT:\n{text}"
        )}
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    )

def rough_words_from_tokens(tok: int):
    # грубая оценка: 1 token ~ 0.75 words для английского (приблизительно)
    return max(40, int(tok * 0.75))

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)


with open(out_path, "w", encoding="utf-8") as fout:
    for ex in ds:
        text = ex["text"]
        tok_in = int(ex.get("token_count", 0)) or len(tokenizer.encode(text))

        max_out   = min(MAX_TOKENS, max(64, int(0.30 * tok_in)))
        max_words = max(64, int(0.75 * max_out))

        prompt = make_prompt(text, max_words)

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LEN
        ).to(model.device)

        input_len = enc["input_ids"].shape[1]

        with torch.inference_mode():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_out,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.05
            )

        gen_only = out_ids[0, input_len:]
        summary = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
        if not summary:
            continue

        tok_sum = len(tokenizer.encode(summary))
        if tok_sum >= tok_in:
            continue

        rec = {
            "id": ex["id"],
            "url": ex.get("url"),
            "dump": ex.get("dump"),
            "score": ex.get("score"),
            "int_score": ex.get("int_score"),
            "token_count": tok_in,
            "summary": summary,
            "summary_token_count": tok_sum,
            "compression_ratio": tok_sum / max(1, tok_in),
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
        exit(0)

print("Wrote:", out_path)
