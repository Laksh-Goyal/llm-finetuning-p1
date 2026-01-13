import sys
import time
from mlx_lm_lora.train import main as mlx_train

def main():
    start_time = time.time()

    sys.argv = [
        "train_sft_mlx.py",

        # Model
        "--model", "mlx-community/Mistral-7B-Instruct-v0.2-4bit",

        # Training mode
        "--train",
        "--train-type", "lora",
        "--train-mode", "sft",

        # Data
        "--data", "data/sft_mlx",

        # Output (LoRA adapters)
        "--adapter-path", "outputs/sft_mlx",

        # Training config
        "--batch-size", "1",
        "--learning-rate", "3e-5",
        "--iters", "300",
        "--save-every", "100",

        # Context
        "--max-seq-length", "1024",
        "--grad-checkpoint",
        "--seed", "42",
    ]

    mlx_train()
    print(f"Training complete in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
