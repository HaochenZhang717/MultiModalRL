import os
import re
import json
import argparse
from typing import Optional, Dict, Any, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_task_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_gold_answer(target_scores: Dict[str, int]) -> int:
    """
    target_scores is like:
    {
        "0": 0,
        "1": 1,
        "2": 0,
        ...
    }
    Return the key whose value is 1.
    """
    for k, v in target_scores.items():
        if v == 1:
            return int(k)
    raise ValueError(f"Cannot find gold answer in target_scores: {target_scores}")


def build_prompt(question: str, use_chat_template: bool = True):
    """
    Prompt template for geometry intersection counting.
    The model is encouraged to reason step-by-step and finally output:
        Final answer: <integer>
    """

    # system_msg = (
    #     "You are a mathematical reasoning assistant.\n\n"
    #     "Carefully analyze geometry problems and reason step by step.\n"
    #     "Intersection points occur when the boundaries of shapes cross each other.\n"
    #     "Count all unique intersection points.\n\n"
    #     "At the end, output the answer in the following format:\n"
    #     "Final answer: <integer>"
    # )

    system_msg = (
        "You are a mathematical reasoning assistant.\n\n"
        "Solve geometry problems by reasoning carefully and step by step.\n"
        "Intersection points occur when the boundaries of shapes cross each other.\n"
        "Count all unique intersection points.\n\n"
        "Output requirements:\n"
        "1. You may reason step by step before giving the final answer.\n"
        "2. The final line of your response must be exactly in this format:\n"
        "Final answer: <integer>\n"
    )

    user_msg = (
        f"{question}\n\n"
        "Think step by step and determine the number of intersection points."
    )

    if use_chat_template:
        return {"system": system_msg, "user": user_msg}
    else:
        return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"


def extract_predicted_integer(text: str) -> Optional[int]:
    """
    Extract integer answer from model output.

    Priority:
    1. Final answer: X
    2. boxed answer \\boxed{X}
    3. Answer: X / answer is X
    4. last integer in the text
    """

    if text is None:
        return None

    text = text.strip()

    # 1️⃣ Final answer
    m = re.search(r"final\s*answer\s*[:=]\s*(-?\d+)", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    # 2️⃣ \boxed{X}
    m = re.findall(r"\\boxed\{(-?\d+)\}", text)
    if m:
        return int(m[-1])

    # 3️⃣ answer: X / answer is X
    patterns = [
        r"(?:answer\s*is|answer)\s*[:=]?\s*(-?\d+)",
    ]

    for p in patterns:
        m = re.findall(p, text, flags=re.IGNORECASE)
        if m:
            return int(m[-1])

    # 4️⃣ fallback: last integer in text
    nums = re.findall(r"-?\d+", text)
    if nums:
        return int(nums[-1])

    return None


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt_obj,
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    if isinstance(prompt_obj, dict):
        messages = [
            {"role": "system", "content": prompt_obj["system"]},
            {"role": "user", "content": prompt_obj["user"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = prompt_obj

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    do_sample = temperature > 0.0

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


def maybe_subset(examples: List[Dict[str, Any]], max_examples: Optional[int]) -> List[Dict[str, Any]]:
    if max_examples is None or max_examples <= 0:
        return examples
    return examples[:max_examples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_json",
        type=str,
        required=True,
        help="Path to task.json",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="HF model path or local path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="geometry_count_predictions.jsonl",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=0,
        help="If > 0, only evaluate first N examples",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Use tokenizer.apply_chat_template()",
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"[INFO] Loading task file: {args.task_json}")
    task_data = load_task_json(args.task_json)
    examples = task_data["examples"]
    examples = maybe_subset(examples, args.max_examples)

    print(f"[INFO] Number of examples: {len(examples)}")
    print(f"[INFO] Loading model: {args.model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if args.device == "cuda" else None,
    )

    if args.device != "cuda":
        model = model.to(args.device)

    model.eval()

    num_correct = 0
    num_parse_fail = 0
    results = []

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(tqdm(examples, desc="Evaluating")):
            question = ex["input"]
            gold = extract_gold_answer(ex["target_scores"])

            prompt_obj = build_prompt(question, use_chat_template=args.use_chat_template)

            try:
                pred_text = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_obj=prompt_obj,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                print(f"[INFO] Generated text: {pred_text}")
                # breakpoint()
                pred = extract_predicted_integer(pred_text)
            except Exception as e:
                pred_text = f"[ERROR] {repr(e)}"
                pred = None

            correct = int(pred == gold)
            num_correct += correct

            if pred is None:
                num_parse_fail += 1

            item = {
                "idx": idx,
                "question": question,
                "gold": gold,
                "output_text": pred_text,
                "prediction": pred,
                "correct": bool(correct),
                "raw_output": pred_text,
            }
            results.append(item)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()

    total = len(results)
    acc = num_correct / total if total > 0 else 0.0
    parse_fail_rate = num_parse_fail / total if total > 0 else 0.0

    print("\n========== Results ==========")
    print(f"Total examples   : {total}")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Parse fail rate  : {parse_fail_rate:.4f}")
    print(f"Saved predictions: {args.output_path}")


if __name__ == "__main__":
    main()