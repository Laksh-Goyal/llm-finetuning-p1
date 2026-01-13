import sys
import time
from mlx_lm_lora.train import main as mlx_train

def main():
    start_time = time.time()

    sys.argv = [
        "train_sft_mlx.py",
        "--model", "mlx-community/Mistral-7B-Instruct-v0.2-4bit",
        "--train",
        "--data", "data/sft_mlx",
        "--output-dir", "outputs/sft_mlx",

        # LoRA config
        "--lora-r", "16",
        "--lora-alpha", "32",
        "--lora-dropout", "0.05",

        # Training config
        "--batch-size", "1",
        "--lr", "2e-4",
        "--iters", "600",
        "--save-interval", "200",

        # Context
        "--max-seq-length", "1024",
        "--seed", "42",
    ]

    mlx_train()

    print(f"Training complete in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
