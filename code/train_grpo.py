import os
import json
import re
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, TaskType


# =========================
# Config
# =========================
MODEL_NAME = '/playpen-shared/haochenz/DeepSeek-R1-Distill-Qwen-7B'
DATA_PATH = "./datasets/shapes_3/train_task.json"
OUTPUT_DIR = "./grpo_ckpt/shapes_3"


WANDB_PROJECT = "grpo-GeoCount"
WANDB_RUN_NAME = "r1-lora-verifiable"


# =========================
# Prompt
# =========================
def build_prompt(question: str):
    system_msg = (
        "You are a mathematical reasoning assistant.\n\n"
        "Carefully analyze geometry problems and reason step by step.\n"
        "Intersection points occur when the boundaries of shapes cross each other.\n"
        "Count all unique intersection points.\n\n"
        "At the end, output the answer in the following format:\n"
        "Final answer: <integer>"
    )

    user_msg = (
        f"{question}\n\n"
        "Think step by step and determine the number of intersection points."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# =========================
# Utils
# =========================
def extract_answer(target_scores):
    return max(target_scores, key=lambda k: target_scores[k])


def extract_number(text: str):
    # ⭐ 只解析 Final answer
    match = re.search(r"Final answer:\s*(\d+)", text)
    if match:
        return match.group(1)
    return None


def completion_to_text(completion):
    if isinstance(completion, list):
        if len(completion) > 0 and isinstance(completion[0], dict):
            return completion[0].get("content", "")
        return ""
    if isinstance(completion, str):
        return completion
    return str(completion)


# =========================
# Dataset
# =========================
def load_data(path):
    with open(path, "r") as f:
        raw_data = json.load(f)

    data = []
    for item in raw_data:
        answer = extract_answer(item["target_scores"])
        data.append({
            "prompt": build_prompt(item["input"]),
            "answer": answer,
        })

    return Dataset.from_list(data)


# =========================
# Reward (0/1)
# =========================
def reward_fn(prompts, completions, answer, **kwargs):
    rewards = []

    for completion, gt in zip(completions, answer):
        text = completion_to_text(completion)
        pred = extract_number(text)

        # ⭐ 类型统一
        gt = str(gt)

        rewards.append(1.0 if (pred is not None and pred == gt) else 0.0)

    return rewards
# =========================
# Main
# =========================
def main():

    # ========= WandB =========
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_NAME"] = WANDB_RUN_NAME

    # ========= Tokenizer =========
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========= Model =========
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
    )

    # ========= LoRA =========
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,                     # ⭐ rank（建议8~32）
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        bias="none",
    )

    # ========= Dataset =========
    dataset = load_data(DATA_PATH)

    # ========= GRPO Config =========
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,

        remove_unused_columns=False,

        # ===== batch =====
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,

        # ===== optimization =====
        learning_rate=1e-5,
        num_train_epochs=5,

        # ===== logging =====
        logging_steps=10,
        save_steps=100,
        report_to="wandb",

        # ===== precision =====
        bf16=True,

        # ===== generation =====
        max_prompt_length=512,
        max_completion_length=1024,
        num_generations=16,

        temperature=0.7,
        top_p=0.9,

        # ===== GRPO =====
        beta=0.02,  # ⭐ LoRA推荐
        loss_type="dapo",  # ⭐ 必改
        scale_rewards="group",

        epsilon=0.2,
        num_iterations=1,

        importance_sampling_level="sequence",  # ⭐ 关键
        mask_truncated_completions=True,  # ⭐ 稳定性

        top_entropy_quantile=0.2,  # ⭐ reasoning提升

        # ===== stability =====
        disable_dropout=True,
        log_completions=True,
    )

    # ========= Trainer =========
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        # ⭐ LoRA 注入
        peft_config=peft_config,
    )

    # ========= Train =========
    trainer.train()

    # ========= Save =========
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()