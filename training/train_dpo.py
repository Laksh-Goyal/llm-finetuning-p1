from datasets import load_dataset
from transformers import AutoTokenizer
from mlx_lm import Trainer, LoraConfig, load, save

model, tokenizer = load("models/sft-mlx")

prefs = load_dataset("json", data_files="data/dpo.jsonl", split="train")

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"]
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=prefs,
    lora=lora_cfg,
    objective="dpo",
    beta=0.1,
    epochs=2,
    batch_size=1,
    lr=5e-6,
)

trainer.train()

save(model, "models/dpo-mlx")
