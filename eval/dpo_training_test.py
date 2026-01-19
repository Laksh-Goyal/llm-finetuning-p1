"""
DPO Training Test - HuggingFace Version
Compares Base Model vs DPO-Optimized Model (SFT + DPO)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import time

# Model Paths
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
SFT_ADAPTER = "outputs/sft_hf"
DPO_ADAPTER = "outputs/dpo_hf"
MAX_TOKENS = 150

QUESTIONS = [
    "A borrower applies for a $28,000 auto loan with a 72-month term. "
    "They earn $5,200 per month, have $2,300 in existing monthly debt, and a credit score of 675. "
    "How would a lender likely assess approval and interest rate?",

    "A borrower requests a $45,000 unsecured personal loan. "
    "They earn $11,000 per month, already carry $6,200 in monthly debt, and have a credit score of 690. "
    "How would a lender likely decide?",

    "A borrower applies for a $22,000 secured loan backed by a savings account. "
    "They earn $4,100 per month, have $1,900 in monthly debt, and a credit score of 610. "
    "How would a lender evaluate this application?",

    "A borrower applies for a $15,000 unsecured installment loan. "
    "They earn $3,400 per month, have $2,100 in existing monthly debt, and a credit score of 620. "
    "How would a lender assess this request?",

    "Why might a lender place more weight on debt-to-income ratio than credit utilization "
    "when evaluating a new loan application?",
]

print("=" * 80)
print("Loading models...")
print("=" * 80)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Load BASE model
print("\nLoading BASE model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map={"": "mps"},
    torch_dtype=torch.float16,
)
print("✓ Base model loaded")

# Load DPO model (SFT + DPO)
print("\nLoading DPO-optimized model (SFT + DPO)...")
dpo_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map={"": "mps"},
    torch_dtype=torch.float16,
)
# Load SFT adapter first
dpo_model = PeftModel.from_pretrained(dpo_model, SFT_ADAPTER)
# Merge SFT
dpo_model = dpo_model.merge_and_unload()
# Load DPO adapter on top
dpo_model = PeftModel.from_pretrained(dpo_model, DPO_ADAPTER)
print("✓ DPO model loaded (SFT + DPO)")

print("\n" + "=" * 80)
print("Models loaded. Starting evaluation...")
print("=" * 80)


def generate(model, prompt):
    """Generate response from model"""
    formatted_prompt = f"<s>[INST]{prompt}[/INST]"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("mps")
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(formatted_prompt, "").strip()
    
    return response, elapsed


for i, question in enumerate(QUESTIONS, 1):
    print("\n" + "=" * 80)
    print(f"PROMPT {i}/{len(QUESTIONS)}")
    print("=" * 80)
    print(f"{question[:80]}...")
    
    # BASE model
    print("\n--- BASE MODEL RESPONSE ---")
    base_response, base_time = generate(base_model, question)
    print(base_response)
    print(f"\nGeneration time: {base_time:.2f}s")
    
    # DPO model
    print("\n--- DPO-OPTIMIZED MODEL RESPONSE (SFT + DPO) ---")
    dpo_response, dpo_time = generate(dpo_model, question)
    print(dpo_response)
    print(f"\nGeneration time: {dpo_time:.2f}s")
    
    print("\n" + "-" * 80)

print("\n" + "=" * 80)
print("✓ DPO evaluation complete!")
print("=" * 80)
print("\nNote: The DPO model includes both SFT and DPO optimizations")
print("For a three-way comparison, use: python eval/compare_all_models.py")

