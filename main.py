
import json, math
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MAX_TOKENS = 128
MODEL = "models/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)

llm = LLM(model=MODEL, dtype="auto", device="cuda")

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

batch_prompts, batch_meta = [], []
BATCH = 64

out_path = "fineweb_edu_summaries.jsonl"
fout = open(out_path, "w", encoding="utf-8")

# параметры генерации: низкая температура => меньше болтовни
sampling = SamplingParams(
    temperature=0.2,
    top_p=0.9,
    max_tokens=MAX_TOKENS,
    presence_penalty=1.0
)

for ex in ds:
    text = ex["text"]
    tok_in = int(ex.get("token_count", 0)) or len(tokenizer.encode(text))

    # лимит на слова и выходные токены
    max_out   = min(MAX_TOKENS, max(64, int(0.30 * tok_in)))
    max_words = max(64, int(0.75 * max_out))   # без верхнего cap, но с нижним

    prompt = make_prompt(text, max_words)

    batch_prompts.append(prompt)
    batch_meta.append((ex, tok_in, max_out))

    if len(batch_prompts) >= BATCH:
        # обновляем max_tokens под “средний” батч или делай 1 запрос = 1 запись
        sampling.max_tokens = max(m[2] for m in batch_meta)

        outs = llm.generate(batch_prompts, sampling)
        for out, (orig, tok_in, max_out_i) in zip(outs, batch_meta):
            summary = out.outputs[0].text.strip()
            tok_sum = len(tokenizer.encode(summary))

            # жёсткое правило “должно быть короче”
            if tok_sum >= tok_in:
                continue

            rec = {
                "id": orig["id"],
                "url": orig.get("url"),
                "dump": orig.get("dump"),
                "score": orig.get("score"),
                "int_score": orig.get("int_score"),
                "token_count": tok_in,
                "summary": summary,
                "summary_token_count": tok_sum,
                "compression_ratio": tok_sum / max(1, tok_in),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        batch_prompts.clear()
        batch_meta.clear()

fout.close()
print("Wrote:", out_path)
