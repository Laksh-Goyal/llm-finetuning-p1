"""
Supervised Fine-Tuning (SFT) using MLX + LoRA
Model: MISTRAL 7B Instruct (4-bit, MLX)
Dataset: Curated FinTech instruction-response pairs
"""

import time
# import random
from pathlib import Path
# from helpers import format_prompt, load_sft_dataset, save_lora_adapters

# import mlx.core as mx
# import mlx.optimizers as optim
# from mlx_lm import load
# from mlx_lm.tuner import train
# from datasets import load_dataset
from mlx_lm_lora.train import train as lora_train

def save_lora_adapters(model, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    lora_params = {
        k: v
        for k, v in model.parameters().items()
        if "lora" in k.lower()
    }

    mx.save_safetensors(
        str(Path(output_dir) / "lora_adapters.safetensors"),
        lora_params
    )

    print(f"Saved {len(lora_params)} LoRA tensors")

# Config
MODEL_PATH = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
DATA_DIR = "data/sft_final" 
DATA_PATH = "data/sft.jsonl"
OUTPUT_DIR = "models/sft_mlx"

TRAIN_CONFIG = {
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "epochs": 1,
    "max_seq_length": 512,
}

LORA_CONFIG = {
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,
}

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("Starting MLX LoRA SFT training")

    start_time = time.time() # I am adding a timing component to see how it compares to Hugging Face later

    lora_train(
        model=MODEL_PATH,
        data=DATA_DIR,
        output_dir=OUTPUT_DIR,
        lora_config=LORA_CONFIG,
        **TRAIN_CONFIG,
    )

    print(f"Training completion: {((time.time() - start_time) / 60):.2f} minutes")
    print(f"Adapters saved to {OUTPUT_DIR}")


# def main():
#     # Load Model
#     model, tokenizer = load(MODEL_PATH)

#     # Load and configure dataset
#     dataset = load_sft_dataset(DATA_PATH)
#     random.shuffle(dataset)

#     tokenized_dataset = [
#         tokenizer.encode(ex, max_length=MAX_TOKENS, truncation=True) for ex in dataset
#     ]

#     optimizer = optim.AdamW(learning_rate=LEARNING_RATE)

#     # Training Start

#     # Initialize
#     start_time = time.time() # I am adding a timing component to see how it compares to Hugging Face later
#     losses = []
#     step = 0

#     for epoch in range(EPOCHS):
#         print(f"Epoch: {epoch}")
#         for tokens in tokenized_dataset:
#             loss = tuner.train(
#                 model,
#                 optimizer,
#                 tokens,
#                 batch_size=BATCH_SIZE,
#                 grad_accum_steps=GRAD_ACCUM_STEPS,
#                 lora_config=LORA_CONFIG
#             )

#             losses.append(float(loss))
#             step += 1

#             if step % LOGGING_STEPS == 0:
#                 avg_loss = np.mean(losses[-LOGGING_STEPS:]) # Get the average loss for the latest [LOGGING_STEPS] amount of steps
#                 print(f"Step: {step}| Avg Loss: {avg_loss:.4f}| Time: {time.time() - start_time:.2f}s")

#             # Flushing Compute Graph
#             mx.eval(model.parameters())

#     print(f"\nTraining completion: {((time.time() - start_time) / 60):.2f} minutes")
    
#     # Saving Model Parameters
#     Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
#     save_lora_adapters(model, OUTPUT_DIR)
#     print(f"Saved LoRA adapters to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
