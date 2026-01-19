"""
DPO Training Script using HuggingFace TRL
Trains DPO on top of the SFT-trained model
"""

import os
import sys
import json
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig

# Configuration
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
SFT_ADAPTER_PATH = "outputs/sft_hf"  # Your SFT-trained adapter
DPO_OUTPUT_DIR = "outputs/dpo_hf"
DATA_PATH = "data/dpo.jsonl"

# DPO Hyperparameters
BETA = 0.1  # DPO temperature parameter
LEARNING_RATE = 5e-7  # Lower LR for fine-tuning on top of SFT
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
MAX_LENGTH = 512
MAX_PROMPT_LENGTH = 256

# LoRA Configuration for DPO
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]


def load_dpo_dataset(data_path: str):
    """Load and prepare DPO dataset"""
    print(f"Loading dataset from {data_path}...")
    
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    # Split into train/validation
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"Train samples: {len(split_dataset['train'])}")
    print(f"Validation samples: {len(split_dataset['test'])}")
    
    return split_dataset["train"], split_dataset["test"]


def format_dpo_example(example):
    """
    Format DPO examples for TRL DPOTrainer
    Expected format:
    - prompt: the instruction/question
    - chosen: the preferred response
    - rejected: the less preferred response
    """
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def main():
    print("=" * 80)
    print("DPO Training on SFT Model (HuggingFace)")
    print("=" * 80)
    print(f"Base model: {BASE_MODEL}")
    print(f"SFT adapter: {SFT_ADAPTER_PATH}")
    print(f"Output directory: {DPO_OUTPUT_DIR}")
    print(f"Data path: {DATA_PATH}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(DPO_OUTPUT_DIR, exist_ok=True)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for DPO
    
    # Load dataset
    train_dataset, eval_dataset = load_dpo_dataset(DATA_PATH)
    
    # Format datasets
    train_dataset = train_dataset.map(format_dpo_example)
    eval_dataset = eval_dataset.map(format_dpo_example)
    
    # Load base model
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": "mps"},  # Apple Silicon
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Load SFT adapter
    print(f"\nLoading SFT adapter from {SFT_ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, SFT_ADAPTER_PATH)
    
    # Merge SFT adapter into base model
    print("Merging SFT adapter with base model...")
    model = model.merge_and_unload()
    
    # Add new LoRA layers for DPO training
    print("\nAdding new LoRA layers for DPO training...")
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Create reference model (frozen SFT model) - DPOTrainer handles this internally when peft_config is provided
    # ref_model is the same as model (Base+SFT) at this point
    # We don't need to load it explicitly
    
    # DPO Training Configuration
    training_args = DPOConfig(
        output_dir=DPO_OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # DPO specific
        beta=BETA,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        
        # Logging and saving
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        save_total_limit=3,
        eval_strategy="steps",
        
        # Optimization
        gradient_checkpointing=True,
        optim="adamw_torch",
        
        # Other
        remove_unused_columns=False,
        report_to="none",  # Set to "wandb" if you want W&B logging
        seed=42,
    )
    
    # Initialize DPO Trainer
    print("\nInitializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # TRL handles reference model creation for PEFT
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # Train
    print("\n" + "=" * 80)
    print("Starting DPO training...")
    print("=" * 80)
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(DPO_OUTPUT_DIR)
    tokenizer.save_pretrained(DPO_OUTPUT_DIR)
    
    # Save training metadata
    metadata = {
        "base_model": BASE_MODEL,
        "sft_adapter": SFT_ADAPTER_PATH,
        "dpo_config": {
            "beta": BETA,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
        },
        "training_args": training_args.to_dict(),
    }
    
    with open(f"{DPO_OUTPUT_DIR}/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✓ DPO Training Complete!")
    print("=" * 80)
    print(f"Model saved to: {DPO_OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Run evaluation script to compare base vs SFT vs DPO")
    print("  2. Test on your specific use cases")
    print("  3. Deploy the best performing model")


if __name__ == "__main__":
    main()
