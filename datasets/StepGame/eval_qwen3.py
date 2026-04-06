"""Evaluate Qwen/Qwen3-8B on StepGame (data_nhop20/example.jsonl).

Runs two passes on the same examples:
  1. `story`       — the full noisy description (harder).
  2. `story_clean` — only the supporting facts (easier).

Reports accuracy for each setting.

Usage:
    python eval_qwen3.py \
        --data datasets/StepGame/data_nhop20/example.jsonl \
        --model Qwen/Qwen3-8B
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# Canonical StepGame label set.
LABELS = [
    "left", "right", "above", "below",
    "lower-left", "lower-right", "upper-left", "upper-right",
    "overlap",
]

SYSTEM_PROMPT = (
    "You are a spatial reasoning assistant. Given a list of spatial facts and a "
    "question about the relation between two objects, determine the correct "
    "spatial relation.\n"
    "Your answer MUST be exactly one of the following labels:\n"
    "  left, right, above, below, lower-left, lower-right, upper-left, "
    "upper-right, overlap.\n"
    "Before giving the final answer, you should think step by step."
    "The final answer should be wrapped in <answer>...</answer> tags."
)


def build_user_prompt(story_sentences, question):
    story_text = "\n".join(f"- {s}" for s in story_sentences)
    return (
        f"Facts:\n{story_text}\n\n"
        f"Question: {question}\n\n"
        "Give the spatial relation as one of: left, right, above, below, "
        "lower-left, lower-right, upper-left, upper-right, overlap.\n"
        "Format your final answer as <answer>LABEL</answer>."
    )


def parse_prediction(text: str) -> str:
    """Extract a canonical label from the model output."""
    if text is None:
        return ""
    # Prefer an explicit <answer>...</answer> tag.
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    candidate = m.group(1) if m else text
    candidate = candidate.strip().lower()
    candidate = candidate.replace("_", "-").replace(" ", "-")
    # Normalize common variants.
    synonyms = {
        "top": "above", "bottom": "below",
        "top-left": "upper-left", "top-right": "upper-right",
        "bottom-left": "lower-left", "bottom-right": "lower-right",
    }
    candidate = synonyms.get(candidate, candidate)
    if candidate in LABELS:
        return candidate
    # Fallback: search any label token inside the output.
    for lab in sorted(LABELS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(lab)}\b", candidate):
            return lab
    return candidate  # unparseable — will count as wrong


def load_examples(path: Path):
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def render_prompts(examples, tokenizer, story_key):
    prompts = []
    for ex in examples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(ex[story_key], ex["question"])},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3: disable <think> blocks for deterministic labels.
        )
        prompts.append(prompt)
    return prompts


@torch.inference_mode()
def evaluate(model, tokenizer, examples, story_key, max_new_tokens, batch_size, out_path):
    """Run generation batch-by-batch and append each record to `out_path` immediately."""
    prompts = render_prompts(examples, tokenizer, story_key)
    correct = 0
    total = 0
    records = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fout:
        for start in tqdm(range(0, len(prompts), batch_size), desc=f"generate[{story_key}]"):
            batch_prompts = prompts[start:start + batch_size]
            batch_examples = examples[start:start + batch_size]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(model.device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen = out[:, enc["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)

            for ex, raw in zip(batch_examples, texts):
                pred = parse_prediction(raw)
                gold = ex["answer"].strip().lower()
                ok = pred == gold
                correct += int(ok)
                total += 1
                rec = {
                    "id": ex.get("id"),
                    "question": ex["question"],
                    "gold": gold,
                    "pred": pred,
                    "raw": raw,
                    "correct": ok,
                }
                records.append(rec)
                # Write + flush immediately so a crash still leaves a valid jsonl on disk.
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

                print("=" * 80)
                print(f"[{story_key}] id={ex.get('id')}  Q: {ex['question']}")
                print(f"gold: {gold}    pred: {pred}    {'OK' if ok else 'WRONG'}"
                      f"    running acc: {correct}/{total} = {correct/total:.4f}")
                print("--- model output ---")
                print(raw)

    acc = correct / total if total else 0.0
    return acc, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data_nhop20" / "example.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N examples (default: all).")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "eval_outputs")
    args = parser.parse_args()

    examples = load_examples(args.data)
    print(f"Loaded {len(examples)} examples from {args.data}")
    if args.limit is not None:
        examples = examples[:args.limit]
        print(f"Limiting to first {len(examples)} examples")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.cuda()
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for story_key in ("story", "story_clean"):
        print(f"\n=== Evaluating with `{story_key}` ===")
        out_file = args.output_dir / f"preds_{story_key}.jsonl"
        acc, records = evaluate(
            model, tokenizer, examples, story_key,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            out_path=out_file,
        )
        results[story_key] = acc
        print(f"Accuracy ({story_key}): {acc:.4f}  "
              f"({sum(r['correct'] for r in records)}/{len(records)})")
        print(f"Per-example predictions streamed to {out_file}")

    print("\n=== Summary ===")
    for k, v in results.items():
        print(f"  {k:12s}: {v:.4f}")
    summary_file = args.output_dir / "summary.json"
    with summary_file.open("w") as f:
        json.dump({"model": args.model, "data": str(args.data), "accuracy": results}, f, indent=2)
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()