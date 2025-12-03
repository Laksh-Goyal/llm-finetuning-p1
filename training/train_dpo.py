from trl import DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset

prefs = load_dataset("json", data_files="data/dpo.jsonl", split="train")
model = AutoModelForCausalLM.from_pretrained("models/sft")

training_args = TrainingArguments(
    output_dir="models/dpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    num_train_epochs=2
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    beta=0.1,
    train_dataset=prefs
)

trainer.train()
