import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType


# =========================
# Config
# =========================
MODEL_NAME = "Qwen/Qwen3-8B"
DATA_PATH = "./datasets/GeoCount/shapes_2/sft_thinking.jsonl"
OUTPUT_DIR = "./sft_ckpt/shapes_2_qwen3"

WANDB_PROJECT = "sft-GeoCount"
WANDB_RUN_NAME = "qwen3-8b-lora-thinking"

SYSTEM_MSG = (
    "You are a mathematical reasoning assistant.\n\n"
    "Solve geometry problems by reasoning carefully and step by step.\n"
    "Intersection points occur when the boundaries of shapes cross each other.\n"
    "Count all unique intersection points.\n\n"
    "Output requirements:\n"
    "1. You may reason step by step before giving the final answer.\n"
    "2. The final line of your response must be exactly in this format:\n"
    "Final answer: <integer>\n"
)


# =========================
# Dataset
# =========================
def load_data(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def format_example(example, tokenizer):
    """
    Build the full chat text for one example.
    Assistant turn: <think>{thinking}</think>\nFinal answer: {answer}
    Only the assistant turn is trained on (the rest is masked by SFTTrainer).
    """
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {
            "role": "user",
            "content": (
                f"{example['question']}\n\n"
                "Think step by step and determine the number of intersection points."
            ),
        },
        {
            "role": "assistant",
            "content": (
                f"<think>\n{example['thinking']}\n</think>\n"
                f"Final answer: {example['answer']}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


# =========================
# Main
# =========================
def main():
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_NAME"] = WANDB_RUN_NAME

    # ========= Tokenizer =========
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.padding_side = "right"  # right-padding for SFT
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========= Dataset =========
    raw_dataset = load_data(DATA_PATH)
    dataset = raw_dataset.map(
        lambda ex: {"text": format_example(ex, tokenizer)},
        remove_columns=raw_dataset.column_names,
    )

    # ========= Model =========
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    # ========= LoRA =========
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    # ========= SFT Config =========
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        # batch
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,

        # optimization
        learning_rate=2e-4,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        # sequence length
        max_seq_length=4096,

        # logging / saving
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        report_to="wandb",

        # precision
        bf16=True,

        # packing short sequences together for efficiency
        packing=True,

        dataset_text_field="text",
    )

    # ========= Trainer =========
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # ========= Train =========
    trainer.train()

    # ========= Save =========
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()