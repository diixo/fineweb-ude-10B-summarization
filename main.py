import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


MAX_TOKENS = 256
MODEL = "models/Qwen3-1.7B"

out_path = "fineweb_edu_summaries.jsonl"
MAX_INPUT_LEN = 4096

# 10BT = 9_953_989_344

#####################################################################################

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.float16, device_map="cuda").eval()


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
    # rough estimate: 1 token ~ 0.75 words for english (approximately)
    return max(40, int(tok * 0.75))

#ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
ds = load_dataset("parquet", data_files="data/fineweb-edu-10BT/*.parquet", split="train", streaming=True)

tokens_processed = 0

with open(out_path, "w", encoding="utf-8") as fout:
    
    for i, ex in enumerate(ds):

        if i % 100 == 0:
            fout.flush()

        text = ex["text"]
        tok_in = int(ex.get("token_count", 0))
        tokens_processed += tok_in

        if tok_in <= MAX_TOKENS:
            rec = {
                "id": i,
                "src_id": ex["id"],
                "score": ex.get("score"),
                "int_score": ex.get("int_score"),
                "token_count": tok_in,
                "summary": text,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue

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
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_only = out_ids[0, input_len:]
        summary = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
        if not summary:
            print(f"Skip item:[{i}]")
            continue

        tok_sum = len(tokenizer.encode(summary))
        if tok_sum >= tok_in:
            summary = text

        rec = {
            "id": i,
            "src_id": ex["id"],
            "score": ex.get("score"),
            "int_score": ex.get("int_score"),
            "token_count": tok_in,
            "summary": summary,
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"Processed [{i}] [{tokens_processed}]...", ex["id"], "Input tokens:", tok_in)


print("Wrote:", out_path)
