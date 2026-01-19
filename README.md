# Domain-Specialized LLM Alignment with SFT + DPO (Lending Risk)

## Overview
This project demonstrates an **end-to-end large language model alignment pipeline** for a real-world decision-making domain: **consumer lending and credit risk assessment**.

The goal is not to chase benchmark scores, but to show how modern alignment techniques—**Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)**—can be applied *correctly* to produce **clearer, more decisive, and more consistent model behavior** in high-stakes scenarios.

The repository is intentionally designed to mirror how these techniques are used in practice:
- Domain-specific SFT to establish grounding and numerical correctness
- Careful DPO to refine **decision preferences**, not rewrite behavior
- Qualitative evaluation focused on reasoning quality and risk tradeoffs

This project is suitable as a **portfolio example for applied ML / LLM engineering roles**, particularly those involving alignment, fine-tuning, or decision-support systems.

---

## End-to-End Pipeline

```
Raw prompts
   ↓
Supervised Fine-Tuning (LoRA)
   ↓
Preference Dataset Construction
   ↓
Direct Preference Optimization (DPO)
   ↓
Side-by-side Evaluation (Base vs SFT vs DPO)
```

Each stage is isolated, reproducible, and evaluated independently.

---

## Dataset Construction

The dataset consists of lending-related prompts designed to test:
- Debt-to-income (DTI) reasoning
- Credit score interpretation
- Secured vs unsecured loan risk
- Borderline approval vs denial cases
- Conceptual lender decision logic

For SFT, each prompt was created using a gold standard generated from ChatGPT 5.2 and several variants generated from the same. The variants were sampled from the same distribution as the gold standard.

For DPO, each prompt includes:
- A **preferred (chosen)** response (from the SFT model)
- One or more **less-preferred (rejected)** responses (sampled variants)

This reflects a common real-world setup where preference data is synthetic but anchored to a strong SFT baseline.

---

## Training

### Supervised Fine-Tuning (SFT)

- Base model: `Mistral-7B-Instruct-v0.2`
- Method: LoRA-based SFT
- Objective: Domain grounding and numerical consistency

After SFT, the model reliably:
- Computes and references DTI correctly
- Uses lender-appropriate terminology
- Produces concise, structured explanations

---

### Direct Preference Optimization (DPO)

- Starting point: SFT-trained model
- Objective: **Refine decision preferences**, not content
- Key behaviors encouraged:
  - Reduced hedging in high-risk cases
  - Clearer approval vs denial decisions
  - Better prioritization of repayment capacity

Special care was taken to:
- Correctly mask prompt tokens during training
- Apply loss only to response tokens
- Keep DPO pressure intentionally low

This avoids common failure modes such as repetition, collapse, or hallucination.

---

## Evaluation: Base vs SFT vs DPO

### Evaluation Objective

The evaluation focuses on **behavioral differences**, not benchmark scores. Specifically, we assess whether each training stage improves:
- Decision clarity
- Numeric grounding
- Risk tradeoff articulation
- Structural completeness

---

### Models Compared

| Model | Description |
|------|-------------|
| **Base** | Mistral-7B-Instruct (no fine-tuning) |
| **SFT** | Base + LoRA supervised fine-tuning |
| **DPO** | SFT model refined with preference optimization |

All models are evaluated using identical prompts and decoding parameters.

---

### Prompt - Response Pairs

Prompt: 
A borrower requests a $45,000 unsecured personal loan. They earn $11,000 per month, already carry $6,200 in monthly debt, and have a credit score of 690.
How would a lender likely decide?

---

**Response (Base)**:
A lender evaluates a borrower's application for a loan using several factors, including income, debt-to-income ratio, credit score, and other financial information. Based on the provided information, here's how a lender might make a decision:

Debt-to-Income Ratio: The borrower's monthly debt of $6,200 and monthly income of $11,000 results in a debt-to-income ratio of 56.36%, which is relatively high and indicates that the borrower has a significant amount of existing debt relative to their income.

**Observation (Base)**:
Fluent and correct, but largely descriptive. The model explains the risk but avoids committing to a clear approval or rejection.

---

**Response (SFT)**:
A lender would likely consider the borrower's monthly debt-to-income ratio, credit score, and income to determine whether they can afford the monthly payments on a $45,000 unsecured personal loan.

The borrower's debt-to-income ratio is 56.36% ($6,200 ÷ $11,000), which is relatively high. Lenders typically prefer borrowers with a debt-to-income ratio below 36%, making this loan request higher risk.

**Observation (SFT)**:
Clear improvement in domain grounding and numeric reasoning, but still cautious in drawing a firm conclusion.

---

**Response (DPO)**:
Based on the borrower's income, debt, and credit score, a lender would likely decline the loan request. The borrower's debt-to-income ratio is already high, and their credit score is below average. A lender would view this borrower as high risk and may not be willing to extend a loan of this size.

**Observation (DPO)**:
The model makes a clear lending decision, prioritizing repayment capacity over explanatory detail.
This reflects DPO’s role as a decision preference refiner, not a content generator.



---

### Qualitative Results Summary

| Dimension | Base | SFT | DPO |
|---------|------|-----|-----|
| Domain relevance | Generic | Strong | Strong |
| Numeric reasoning | Inconsistent | Consistent | Consistent |
| Decision clarity | Hedged | Moderate | **High** |
| Denial confidence | Weak | Improved | **Clear** |
| Fluency | High | High | High |

---

### Results

The fine-tuned model demonstrates a clear and consistent improvement over the base model.

**Key improvements observed:**

- **Stronger numeric grounding**: The SFT model consistently computes and references DTI and other quantitative risk factors, whereas the base model often remains descriptive or generic.
- **Lender-centric reasoning**: Responses are framed from an underwriting perspective, focusing on repayment capacity, exposure, and risk, rather than educational explanations.
- **Improved pricing logic**: In borderline cases, the model reliably distinguishes between outright denial and approval with elevated interest rates.
- **More consistent structure and tone**: Outputs are concise, structured, and aligned with the reasoning patterns seen in the training data.

---

### Key Observations

- **Base model**: Fluent but generic; often avoids firm decisions
- **SFT model**: Domain-aligned and numerically grounded; still cautious
- **DPO model**: More decisive in high-risk scenarios with minimal stylistic drift

Importantly, DPO does *not* force changes where the preference signal is weak (e.g., conceptual questions), which is the desired behavior.

---

## Limitations & Learnings

- Preference optimization is sensitive to masking and alignment details
- DPO should be applied conservatively; visible stylistic changes are often a red flag
- In narrow domains, SFT provides most of the gain; DPO provides **polish**, not transformation
- In extreme risk scenarios, the model occasionally uses cautious language (e.g., “approval is uncertain”) rather than issuing a hard denial.
- The training data intentionally mirrors real-world lending distributions, with approvals and pricing outcomes more common than denials. As a result, denial behavior improves relative to the base model but is not perfectly balanced.

These behaviors are expected given the dataset composition and are acceptable for the scope of this project.


These lessons are documented intentionally, as they reflect real-world alignment work.

---

### MLX (Apple Silicon) LoRA SFT

The same learning objective was re-implemented using **MLX**, optimized for Apple Silicon as an extra enhancement.

**Key differences vs Hugging Face**
- CLI-first workflow with stricter dataset contracts
- Implicit LoRA defaults (fewer exposed knobs)
- No automatic chat template handling
- Deterministic greedy decoding by default

**Engineering work required**
- Rebuilt the dataset to remove embedded chat tokens (`<s>`, `[INST]`, `[/INST]`)
- Diagnosed tokenizer–training mismatches causing `<unk>` generation
- Resolved numerical instability by lowering learning rate and enabling gradient checkpointing
- Ensured fair evaluation by explicitly applying the tokenizer chat template at inference time

**Observed behavior**
- More concise, decision-oriented responses
- Clear approval vs denial boundaries
- Comparable quality to Hugging Face after stabilization

---
## Project Demonstrationss

This repository demonstrates:
- Practical application of SFT and DPO
- Correct handling of common alignment pitfalls
- Evaluation focused on reasoning quality, not just outputs

It is intended to showcase **applied LLM engineering judgment**

---

## Reproducibility

All training and evaluation scripts are included. The pipeline can be reproduced end-to-end with modest compute using LoRA adapters.

---

## Contact

If you are reviewing this project as part of a hiring or collaboration process and would like additional details on design decisions, tradeoffs, or extensions, feel free to reach out to lakshgoyal0812@gmail.com

