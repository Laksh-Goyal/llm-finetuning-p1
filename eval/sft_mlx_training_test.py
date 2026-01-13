import time
from mlx_lm import load, generate

BASE_MODEL = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
ADAPTER_PATH = "outputs/sft_mlx"
MAX_TOKENS = 256

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

print("Loading model + adapter...")
model, tokenizer = load(
    BASE_MODEL,
    adapter_path=ADAPTER_PATH
)
print("Loaded.\n")

for i, question in enumerate(QUESTIONS, 1):
    print("=" * 80)
    print(f"PROMPT {i}")

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )

    print(prompt)
    print("\n--- RESPONSE ---")

    start = time.time()
    output = generate(
        model,
        tokenizer,
        prompt,
        max_tokens=MAX_TOKENS,
    )
    elapsed = time.time() - start

    print(output)
    print(f"\nGeneration time: {elapsed:.2f}s")

print("\n MLX evaluation complete")
