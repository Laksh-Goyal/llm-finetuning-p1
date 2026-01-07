import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments
from helpers import load_dataset_base, format_prompt_example

# Config 
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DATA_PATH = "data/sft.jsonl"
OUTPUT_DIR = "outputs/sft_hf_v2"

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

print("Config Complete")

# Data/Model Set up
dataset = load_dataset_base(DATA_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map={"": "mps"},
)

print("Set up Complete")

# Model training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=200,
    fp16=False,
    bf16=False,
    report_to="wandb",
    run_name="mistral7b-fintech-sft-mac"
)

start_time = time.time()

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    formatting_func=format_prompt_example,
    args=training_args
)

trainer.train()

trainer.save_model(OUTPUT_DIR)

print(f"Training complete in {time.time() - start_time:.2f} seconds. Model saved to {OUTPUT_DIR}")
