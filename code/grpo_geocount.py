import os
import re
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser, get_peft_config


# =========================
# Script arguments
# =========================
@dataclass
class ScriptArguments:
    data_path: str = field(
        default="./datasets/GeoCount/shapes_2/train_task.json",
        metadata={"help": "Path to the train_task.json file"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "Reward functions to use. Options: 'accuracy', 'format'"},
    )


# =========================
# Prompt
# =========================
SYSTEM_MSG = (
    "You are a mathematical reasoning assistant.\n\n"
    "Solve geometry problems by reasoning carefully and step by step.\n"
    "Intersection points occur when the boundaries of shapes cross each other.\n"
    "Count all unique intersection points.\n\n"
    "Output requirements:\n"
    "1. Think step by step inside <think> </think> tags.\n"
    "2. The final line of your response must be exactly in this format:\n"
    "Final answer: <integer>\n"
)

QUESTION_TEMPLATE = (
    "{question}\n\n"
    "Think step by step and determine the number of intersection points."
)


# =========================
# Reward functions
# =========================
def accuracy_reward(completions, solution, **kwargs):
    """1.0 if the extracted 'Final answer: X' matches the ground truth."""
    contents = [c[0]["content"] for c in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        match = re.search(r"Final answer:\s*(\d+)", content, re.IGNORECASE)
        pred = match.group(1).strip() if match else None
        rewards.append(1.0 if pred == str(sol) else 0.0)

    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH", "grpo_debug.log")
        with open(log_path, "a") as f:
            for content, sol, r in zip(contents, solution, rewards):
                f.write(f"[accuracy] reward={r} | gt={sol} | output={content[:200]}\n")

    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
}


# =========================
# Dataset
# =========================
def get_answer(target_scores: dict) -> str:
    return max(target_scores, key=lambda k: target_scores[k])


def load_data(path: str) -> Dataset:
    with open(path) as f:
        raw = json.load(f)
    records = []
    for ex in raw["examples"]:
        records.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_MSG},
                    {
                        "role": "user",
                        "content": QUESTION_TEMPLATE.format(question=ex["input"]),
                    },
                ],
                "solution": get_answer(ex["target_scores"]),
            }
        )
    return Dataset.from_list(records)


# =========================
# Main
# =========================
def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[f] for f in script_args.reward_funcs]

    dataset = load_data(script_args.data_path)

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
    main(script_args, training_args, model_args)