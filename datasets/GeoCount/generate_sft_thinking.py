"""
Generate SFT thinking data for shapes_2 using Reverse Thinking prompting.

For each question in train_task.json, uses Qwen3-8B to generate a reasoning
process that explains how to derive the correct answer, without stating the answer.

Usage:
    python generate_sft_thinking.py \
        --input datasets/GeoCount/shapes_2/train_task.json \
        --output datasets/GeoCount/shapes_2/sft_thinking.jsonl \
        --model Qwen/Qwen3-8B \
        --batch_size 8
"""

import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REVERSE_THINKING_PROMPT = (
    "Based on the following question and the correct answer, generate a thought process "
    "to explain how to derive the answer from the inputs.\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "Do not output the answer, only generate the reasoning process. "
    "Formulate your outputs using concise language."
)


def get_answer(target_scores: dict) -> str:
    for key, val in target_scores.items():
        if val == 1:
            return key
    raise ValueError(f"No correct answer found in target_scores: {target_scores}")


def extract_thinking(text: str) -> str:
    """Extract content inside <think>...</think> tags if present, else return full text."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def build_messages(question: str, answer: str) -> list:
    return [{"role": "user", "content": REVERSE_THINKING_PROMPT.format(question=question, answer=answer)}]


def generate_batch(model, tokenizer, batch_messages: list, max_new_tokens: int) -> list[str]:
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for msgs in batch_messages
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Decode only the newly generated tokens
    results = []
    for i, output in enumerate(outputs):
        new_tokens = output[inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append(extract_thinking(decoded))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/GeoCount/shapes_2/train_task.json")
    parser.add_argument("--output", default="datasets/GeoCount/shapes_2/sft_thinking.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N examples (for testing)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    examples = data["examples"]
    if args.limit:
        examples = examples[: args.limit]

    print(f"Loaded {len(examples)} examples from {args.input}")

    # Resume from existing output
    done_ids = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                obj = json.loads(line)
                done_ids.add(obj["id"])
        print(f"Resuming: {len(done_ids)} examples already done")

    pending = [(idx, ex) for idx, ex in enumerate(examples) if idx not in done_ids]
    if not pending:
        print("All examples already processed.")
        return

    print(f"Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    print(f"Generating thinking for {len(pending)} examples (batch_size={args.batch_size}) ...")

    with open(args.output, "a") as out_f:
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start : batch_start + args.batch_size]
            batch_ids = [b[0] for b in batch]
            batch_msgs = [build_messages(b[1]["input"], get_answer(b[1]["target_scores"])) for b in batch]

            try:
                thinking_outputs = generate_batch(model, tokenizer, batch_msgs, args.max_new_tokens)
            except Exception as e:
                print(f"  ERROR on batch starting at {batch_start}: {e}")
                continue

            for (idx, ex), thinking in zip(batch, thinking_outputs):
                result = {
                    "id": idx,
                    "question": ex["input"],
                    "answer": get_answer(ex["target_scores"]),
                    "thinking": thinking,
                }
                out_f.write(json.dumps(result) + "\n")

            out_f.flush()
            done_count = batch_start + len(batch)
            print(f"  {done_count}/{len(pending)} done")

    print(f"Done. Output saved to {args.output}")


if __name__ == "__main__":
    main()