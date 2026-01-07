# LLM Fine-Tuning Project (SFT + DPO) — FinTech / Domain-Specific LLM

This repository contains a full end-to-end pipeline for fine-tuning a large language model using:

- **Supervised Fine-Tuning (SFT)**
- **Direct Preference Optimization (DPO)**
- **Custom domain datasets (FinTech, lending, credit models, etc.)**
- **Evaluation toolkit**
- **Deployment-ready inference server (FastAPI + VLLM)**

The project is optimized for **Apple Silicon (M4 Pro / M3 / M2)** and **GPU cloud deployment** for final inference.

---

# Project Overview

This project demonstrates an end-to-end, **production-oriented fine-tuning pipeline** for large language models (LLMs) applied to **FinTech and lending-domain tasks**. The focus is on improving **domain relevance, decision-oriented reasoning, and response completeness** using parameter-efficient fine-tuning techniques.

The pipeline is structured into two primary training stages:

1. **Supervised Fine-Tuning (SFT)** using Hugging Face + TRL + LoRA
2. **Direct Preference Optimization (DPO)** (planned / next stage)

The repository also includes dataset preparation, evaluation methodology, and deployment-ready inference components.

The pipeline reflects what ML Engineers and Applied Scientists do at companies like **G42, e&, AIQ, OpenAI partners, Stripe, Klarna, and Goldman Sachs**.

---

# Architecture

          ┌────────────────────────────┐
          │   Raw Dataset (Domain)     │
          └──────────────┬─────────────┘
                         ▼
          ┌────────────────────────────┐
          │   Data Cleaning + Prep     │
          │  - SFT jsonl               │
          │  - DPO preference pairs    │
          └──────────────┬─────────────┘
                         ▼
         ┌───────────────────────────────┐
         │  Supervised Fine-Tuning (SFT) │
         │  LoRA + 4-bit QLoRA           │
         └──────────────┬────────────────┘
                        ▼
          ┌────────────────────────────┐
          │        DPO Training        │
          │ (Align to preferred output)│
          └──────────────┬─────────────┘
                         ▼
        ┌────────────────────────────────┐
        │         Evaluation Suite       │
        │  - Domain eval set             │
        │  - Hallucination tests         │
        │  - MMLU subset                 │
        │  - Response quality scoring    │
        └──────────────┬─────────────────┘
                       ▼
    ┌────────────────────────────────────────┐
    │ Deployment (FastAPI + VLLM)            │
    │ - GPU inference                        │
    │ - Low-latency server                   │
    └────────────────────────────────────────┘


---

## Dataset

The supervised fine-tuning dataset was curated from a large open instruction corpus (~70,000 samples) using a multi-stage filtering and cleaning process:

- Domain-specific keyword filtering (FinTech, lending, credit risk)
- Response quality and relevance constraints
- Blacklist-based cleanup to remove meta, off-domain, and low-quality responses
- Manual spot checks for correctness and clarity

The final SFT dataset contains **~500 high-quality instruction–response pairs** focused on:
- Lending metrics (DTI, credit utilization)
- Loan approval logic
- Risk and policy reasoning
- Financial product explanations

Each example follows the schema:

```json
{
  "instruction": "...",
  "response": "..."
}
```

---

## Supervised Fine-Tuning (SFT)

### Training Setup

Supervised fine-tuning was performed using the **Hugging Face ecosystem** with parameter-efficient fine-tuning via **LoRA (PEFT)**.

**Model:**
- `mistralai/Mistral-7B-Instruct-v0.2`

**Training stack:**
- `transformers`
- `trl` (SFTTrainer)
- `peft` (LoRA)
- `datasets`
- `accelerate`

**Hardware:**
- Apple Silicon (Mac M4 Pro) using PyTorch MPS backend

Only LoRA adapter weights were trained; the base model weights remained frozen.

---

### LoRA Configuration

```text
Rank (r):        16
Alpha:           32
Dropout:         0.05
Target modules:  q_proj, v_proj
```

---

### Training Metrics

Training was run for **1 epoch**, which was sufficient given the curated dataset size.

Key observations:

- Training loss decreased steadily from ~1.8 to ~1.0
- Mean token accuracy improved from ~0.65 to ~0.74
- No instability, NaNs, or divergence observed
- Training completed in ~8–9 minutes on Apple Silicon

---

## 🔍 Evaluation & Results

### Evaluation Methodology

Evaluation focused on **behavioral improvements**, not just loss reduction. The fine-tuned model was compared against the base model using **side-by-side qualitative evaluation**.

Prompts were selected to test:
- Definitions of financial concepts
- Applied lending decisions
- Policy reasoning
- Edge cases (e.g. high income but high DTI)
- Comparative reasoning (DTI vs credit utilization)
- Out-of-domain sanity checks

### Evaluation Criteria

Each response was evaluated on:
- Domain relevance
- Decision clarity
- Completeness (no truncation)
- Professional tone
- Factual correctness

---

### Results Summary

Across all tested prompts, the fine-tuned model consistently outperformed the base model.

| Category | Base Model | Fine-Tuned Model |
|-------|------------|------------------|
| Definitions | Generic, sometimes truncated | Clear and domain-aligned |
| Applied decisions | Often incomplete | Concise and decisive |
| Policy reasoning | Rambling or cut off | Focused risk framing |
| Edge cases | Inconsistent | Structured explanations |
| Comparisons | Frequently truncated | Complete and accurate |

**Overall result:** The fine-tuned model showed **improved completeness, stronger domain framing, and more decision-oriented responses** without introducing hallucinations or factual degradation.

---

### Example Comparison

**Prompt:**
> Explain debt-to-income ratio in lending.

**Base model (excerpt):**
> "...The resulting ratio expresses the proportion of income that goes towards debt repayment... Lenders use DTI to evaluate the borrower's ability to—"

*(response truncated)*

**Fine-tuned model (excerpt):**
> "Debt-to-income ratio (DTI) is a measure of a borrower's ability to repay their debts. Lenders use DTI to assess creditworthiness and may apply maximum DTI thresholds when approving loans."

---

### Conclusion

After a single epoch of LoRA-based SFT, the model demonstrated **clear behavioral improvements** across all evaluation categories. Given the strength and consistency of results, no additional SFT epochs were run to avoid overfitting.

The SFT phase is considered **complete and successful**, and the project proceeds to **Direct Preference Optimization (DPO)** as the next stage for refining response nuance and uncertainty handling.

---

## Evaluation

### Setup

We evaluated the fine-tuned model against the base model using a fixed set of held-out prompts designed to test **lending decision reasoning**, rather than factual recall. The prompts reflect realistic underwriting scenarios, including:

- borderline debt-to-income (DTI) cases  
- high income combined with excessive leverage  
- secured loans with weak credit profiles  
- clear high-risk denial scenarios  
- conceptual questions framed from a lender’s perspective  

All prompts were evaluated side by side to isolate behavioral differences introduced by supervised fine-tuning.

---

### Evaluation Criteria

Responses were assessed qualitatively across the following dimensions:

- **Decision correctness** – Plausibility of approval, denial, or pricing outcome  
- **Numeric reasoning** – Correct use of DTI, income, debt, and credit score  
- **Risk tradeoff explanation** – Clear articulation of why a lender would make a given decision  
- **Denial clarity** – Ability to clearly reject high-risk cases without excessive hedging  
- **Tone and framing** – Professional, lender-centric, non-advisory style  

---

### Results

The fine-tuned model demonstrates a clear and consistent improvement over the base model.

**Key improvements observed:**

- **Stronger numeric grounding**: The SFT model consistently computes and references DTI and other quantitative risk factors, whereas the base model often remains descriptive or generic.
- **Lender-centric reasoning**: Responses are framed from an underwriting perspective, focusing on repayment capacity, exposure, and risk, rather than educational explanations.
- **Improved pricing logic**: In borderline cases, the model reliably distinguishes between outright denial and approval with elevated interest rates.
- **More consistent structure and tone**: Outputs are concise, structured, and aligned with the reasoning patterns seen in the training data.

---

### Limitations

Some limitations remain and are explicitly documented:

- **Denial finality**: In extreme risk scenarios, the model occasionally uses cautious language (e.g., “approval is uncertain”) rather than issuing a hard denial.
- **Approval bias**: The training data intentionally mirrors real-world lending distributions, with approvals and pricing outcomes more common than denials. As a result, denial behavior improves relative to the base model but is not perfectly balanced.

These behaviors are expected given the dataset composition and are acceptable for the scope of this project.

---

### Summary

Supervised fine-tuning produced a meaningful behavioral shift from generic financial explanations to **numeric, risk-based lending decision reasoning**. The evaluation confirms that the model learned domain-specific judgment rather than memorizing definitions, validating both the dataset construction and the training approach.


