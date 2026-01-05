# Helper functions useful to all training
#  
def format_prompt(instruction: str, response: str) -> str:
    return f"<s>[INST] {instruction.strip()} [/INST] {response.strip()}</s>"


def load_sft_dataset(path: str):
    dataset = load_dataset("json", data_files=path, split="train")
    return [
        format_prompt(ex["instruction"], ex["response"])
        for ex in dataset
    ]

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
