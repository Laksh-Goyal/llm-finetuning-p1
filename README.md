# LLM Fine-Tuning Project (SFT + DPO) — FinTech / Domain-Specific LLM

This repository contains a full end-to-end pipeline for fine-tuning a large language model using:

- **Supervised Fine-Tuning (SFT)**
- **Direct Preference Optimization (DPO)**
- **Custom domain datasets (FinTech, lending, credit models, etc.)**
- **Evaluation toolkit**
- **Deployment-ready inference server (FastAPI + VLLM)**

The project is optimized for **Apple Silicon (M4 Pro / M3 / M2)** and **GPU cloud deployment** for final inference.

---

# 🚀 Project Overview

The goal is to fine-tune an LLM (Llama-3-8B or Mistral-7B) to perform accurate, reliable domain reasoning for:

- **Credit risk explanations**
- **Loan underwriting**
- **PD / LGD modeling Q&A**
- **Financial compliance**
- **User-facing FinTech assistant tasks**

The pipeline reflects what ML Engineers and Applied Scientists do at companies like **G42, e&, AIQ, OpenAI partners, Stripe, Klarna, and Goldman Sachs**.

---

# 🧱 Architecture

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

# 🛠 Mac-Optimized Training (Apple Silicon)

This repo uses the **mlx** framework for fast training on Apple Silicon.

## Install mlx:

```bash
pip install mlx-lm
pip install transformers datasets peft bitsandbytes accelerate trl

# LLM Fine-Tuning Project — FinTech Domain (MLX + LoRA)

This project demonstrates end-to-end supervised fine-tuning (SFT) of an open-weight Large Language Model using a curated FinTech dataset. The goal is to adapt a general-purpose instruction model to produce more precise, domain-specific responses for credit, lending, and risk-related questions.

The project emphasizes **data quality, reproducibility, and framework-level engineering decisions**, rather than benchmark chasing.

---

## Project Overview

- **Base Model:** Llama-3 8B Instruct (4-bit, MLX)
- **Fine-Tuning Method:** LoRA (Low-Rank Adaptation)
- **Framework:** MLX (Apple Silicon–optimized)
- **Domain:** FinTech (credit risk, lending, underwriting)
- **Dataset Size:** ~484 curated instruction–response pairs

---

## Why MLX?

MLX was selected to enable efficient LoRA fine-tuning on Apple Silicon hardware while preserving the same conceptual workflow used in Hugging Face–based training pipelines. This allows rapid iteration and experimentation without access to large GPU instances.

A Hugging Face–based replication and comparison is planned as a follow-up step.

---

## Dataset

The dataset (`data/sft.jsonl`) was curated from a large open instruction corpus (~70k samples) using:

- Domain keyword filtering
- Length constraints
- Manual review
- Blacklist-based cleanup to remove meta, off-domain, and low-quality responses

Each example follows a simple schema:

```json
{
  "instruction": "...",
  "response": "..."
}
