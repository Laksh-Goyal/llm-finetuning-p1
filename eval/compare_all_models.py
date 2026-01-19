"""
Comprehensive Evaluation Script
Compares Base Model vs SFT Model vs DPO Model
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import time
import gc

# Model Paths
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
SFT_ADAPTER = "outputs/sft_hf"
DPO_ADAPTER = "outputs/dpo_hf"

# Test Prompts (same as your SFT tests)
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


def load_model(model_type):
    """
    Load a specific model configuration.
    model_type: "base", "sft", or "dpo"
    """
    print(f"\nloading {model_type.upper()} model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": "mps"},
        torch_dtype=torch.float16,
    )
    
    if model_type == "base":
        print("   ✓ Base model loaded")
        return tokenizer, model

    elif model_type == "sft":
        model = PeftModel.from_pretrained(model, SFT_ADAPTER)
        print(f"   ✓ SFT model loaded from {SFT_ADAPTER}")
        return tokenizer, model

    elif model_type == "dpo":
        # First load SFT adapter
        model = PeftModel.from_pretrained(model, SFT_ADAPTER)
        # Merge SFT
        model = model.merge_and_unload()
        # Then load DPO adapter on top
        model = PeftModel.from_pretrained(model, DPO_ADAPTER)
        print(f"   ✓ DPO model loaded (SFT + DPO from {DPO_ADAPTER})")
        return tokenizer, model
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_response(model, tokenizer, prompt, max_tokens=150):
    """Generate response from a model"""
    # Format prompt using chat template
    formatted_prompt = f"<s>[INST]{prompt}[/INST]"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("mps")
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generation_time = time.time() - start_time
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from response
    response = response.replace(formatted_prompt, "").strip()
    response = "\n" + response
    return response, generation_time


def main():
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("Comparing: Base → SFT → DPO")
    print("=" * 80)
    
    # Storage for results
    # Structure: { question_index: { 'question': str, 'base': (resp, time), 'sft': ..., 'dpo': ... } }
    results = {i: {'question': q} for i, q in enumerate(QUESTIONS)}
    
    model_types = ["base", "sft", "dpo"]
    
    for model_type in model_types:
        print(f"\n{'='*30} PROCESSING {model_type.upper()} {'='*30}")
        
        # Load model
        tokenizer, model = load_model(model_type)
        
        # Evaluate on all questions
        for i, question in enumerate(QUESTIONS):
            print(f"Processing prompt {i+1}/{len(QUESTIONS)}...", end="\r")
            response, gen_time = generate_response(model, tokenizer, question)
            results[i][model_type] = (response, gen_time)
            
        print(f"\n✓ {model_type.upper()} evaluation complete")
        
        # Cleanup
        del model
        del tokenizer
        gc.collect()
        torch.mps.empty_cache()
        print("✓ Memory cleaned up")

    # Display Results
    for i in range(len(QUESTIONS)):
        print("\n\n" + "=" * 80)
        print(f"PROMPT {i+1}/{len(QUESTIONS)}")
        print("=" * 80)
        print(f"Question: {results[i]['question'][:100]}...")
        print("\n" + "-" * 80)
        
        # Base
        resp, t = results[i]['base']
        print("\n📌 BASE MODEL RESPONSE:")
        print("-" * 80)
        print(resp)
        print(f"\n⏱️  Generation time: {t:.2f}s")
        
        # SFT
        resp, t = results[i]['sft']
        print("\n" + "-" * 80)
        print("📌 SFT MODEL RESPONSE:")
        print("-" * 80)
        print(resp)
        print(f"\n⏱️  Generation time: {t:.2f}s")
        
        # DPO
        resp, t = results[i]['dpo']
        print("\n" + "-" * 80)
        print("📌 DPO MODEL RESPONSE (SFT + DPO):")
        print("-" * 80)
        print(resp)
        print(f"\n⏱️  Generation time: {t:.2f}s")
        
        print("\n" + "=" * 80)
    
    print("\n\n" + "=" * 80)
    print("✓ EVALUATION COMPLETE!")
    print("=" * 80)
    print("\nModel Progression:")
    print("  1. BASE: Original Mistral-7B-Instruct-v0.2")
    print("  2. SFT:  Fine-tuned on lending domain data")
    print("  3. DPO:  Preference-optimized on top of SFT")
    print("=" * 80)


if __name__ == "__main__":
    main()
