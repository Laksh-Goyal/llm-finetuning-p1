from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# ADAPTER = "outputs/sft_hf"
ADAPTER = "outputs/sft_hf_v2"

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
    "<s>[INST]A borrower applies for a $28,000 auto loan with a 72-month term. They earn $5,200 per month, have $2,300 in existing monthly debt, and a credit score of 675. How would a lender likely assess approval and interest rate?[/INST]",
    "<s>[INST]A borrower requests a $45,000 unsecured personal loan. They earn $11,000 per month, already carry $6,200 in monthly debt, and have a credit score of 690. How would a lender likely decide?[/INST]",
    "<s>[INST]A borrower applies for a $22,000 secured loan backed by a savings account. They earn $4,100 per month, have $1,900 in monthly debt, and a credit score of 610. How would a lender evaluate this application?[/INST]",
    "<s>[INST]A borrower applies for a $15,000 unsecured installment loan. They earn $3,400 per month, have $2,100 in existing monthly debt, and a credit score of 620. How would a lender assess this request?[/INST]",
    "<s>[INST]Why might a lender place more weight on debt-to-income ratio than credit utilization when evaluating a new loan application?[/INST]"
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
