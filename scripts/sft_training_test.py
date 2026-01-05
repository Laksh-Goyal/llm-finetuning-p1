from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER = "outputs/sft_hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Base model
base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": "mps"},
    dtype=torch.float16,
)

# SFT model
sft = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": "mps"},
    dtype=torch.float16,
)
sft.load_adapter(ADAPTER)

prompts = [
    "<s>[INST] Explain debt-to-income ratio in lending. [/INST]",
    "<s>[INST] A borrower has a DTI of 48%. How might this affect loan approval? [/INST]",
    "<s>[INST] Why do lenders set maximum DTI thresholds? [/INST]",
    "<s>[INST] What is the typical interest rate for a payday loan? [/INST]",
    "<s>[INST] Can a borrower with high income still be rejected due to DTI? [/INST]",
    "<s>[INST] How does DTI differ from credit utilization? [/INST]",
]

def run(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    out = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(out[0], skip_special_tokens=True)


for prompt in prompts:
    print("=== BASE ===")
    print(run(base, prompt))
    print("\n=== SFT ===")
    print(run(sft, prompt))
