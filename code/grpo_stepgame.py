import os
import re
import json
from dataclasses import dataclass, field

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser, get_peft_config


# =========================
# Script arguments
# =========================
@dataclass
class ScriptArguments:
    data_path: str = field(
        default="./datasets/StepGame/data_nhop6/train.jsonl",
        metadata={"help": "Path to the StepGame train jsonl file."},
    )
    story_key: str = field(
        default="story",
        metadata={"help": "Which field to feed as the fact list: 'story' (noisy) or 'story_clean'."},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "Reward functions to use. Options: 'accuracy', 'format'."},
    )


# =========================
# Prompt
# =========================
LABELS = [
    "left", "right", "above", "below",
    "lower-left", "lower-right", "upper-left", "upper-right",
    "overlap",
]

SYSTEM_MSG = (
    "You are a spatial reasoning assistant.\n\n"
    "Given a list of spatial facts describing relative positions between objects, "
    "and a question about the relation between two specific objects, determine the "
    "correct spatial relation by reasoning step by step.\n\n"
    "The answer must be exactly one of the following 9 labels:\n"
    "  left, right, above, below, lower-left, lower-right, upper-left, upper-right, overlap.\n\n"
    "Output requirements:\n"
    "1. Think step by step inside <think> </think> tags.\n"
    "2. After thinking, output the final answer wrapped in <answer> </answer> tags,\n"
    "   containing exactly one of the 9 labels above.\n"
)

QUESTION_TEMPLATE = (
    "Facts:\n{facts}\n\n"
    "Question: {question}\n\n"
    "Think step by step inside <think> </think>, then give the final label inside "
    "<answer> </answer>."
)


# =========================
# Reward functions
# =========================
_LABEL_SYNONYMS = {
    "top": "above", "bottom": "below",
    "top-left": "upper-left", "top-right": "upper-right",
    "bottom-left": "lower-left", "bottom-right": "lower-right",
}


def _extract_answer(content: str):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    cand = m.group(1).strip().lower().replace("_", "-").replace(" ", "-")
    cand = _LABEL_SYNONYMS.get(cand, cand)
    return cand if cand in LABELS else None


def accuracy_reward(completions, solution, **kwargs):
    """1.0 if the label inside <answer>...</answer> matches the ground truth."""
    contents = [c[0]["content"] for c in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        pred = _extract_answer(content)
        rewards.append(1.0 if pred is not None and pred == str(sol).strip().lower() else 0.0)

    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH", "grpo_debug.log")
        with open(log_path, "a") as f:
            for content, sol, r in zip(contents, solution, rewards):
                f.write(f"[accuracy] reward={r} | gt={sol} | output={content[:300]}\n")

    return rewards


# Expect: <think>...</think> followed by <answer>label</answer>.
_FORMAT_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>\s*[A-Za-z\-]+\s*</answer>\s*$",
    re.DOTALL,
)


def format_reward(completions, **kwargs):
    """1.0 if the completion matches <think>...</think><answer>...</answer>."""
    contents = [c[0]["content"] for c in completions]
    rewards = [1.0 if _FORMAT_RE.match(c) else 0.0 for c in contents]

    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH", "grpo_debug.log")
        with open(log_path, "a") as f:
            for content, r in zip(contents, rewards):
                f.write(f"[format] reward={r} | output={content[:300]}\n")

    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    # "format": format_reward,
}


# =========================
# Dataset
# =========================
def load_data(path: str, story_key: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            facts = "\n".join(f"- {s}" for s in ex[story_key])
            records.append(
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_MSG},
                        {
                            "role": "user",
                            "content": QUESTION_TEMPLATE.format(
                                facts=facts, question=ex["question"]
                            ),
                        },
                    ],
                    "solution": ex["answer"].strip().lower(),
                }
            )
    return Dataset.from_list(records)


# =========================
# Main
# =========================
def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[f] for f in script_args.reward_funcs]

    dataset = load_data(script_args.data_path, script_args.story_key)
    print(f"Loaded {len(dataset)} training examples from {script_args.data_path}")

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if getattr(training_args, "gradient_checkpointing", False):
        gc_kwargs = getattr(training_args, "gradient_checkpointing_kwargs", None)
        if not gc_kwargs:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    main(script_args, training_args, model_args)
