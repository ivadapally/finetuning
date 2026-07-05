# Customer Support AI Assistant — Fine-Tuned with Unsloth

## Project title

**Domain-Specific AI Assistant for Customer Support**, built by taking a small open-source LLM through non-instruction fine-tuning → instruction fine-tuning (SFT) → DPO preference alignment, using [Unsloth](https://github.com/unslothai/unsloth).

## Domain selected

**Customer Support Assistant.**

## Business problem

An internal AI assistant that can answer customer-support questions — refunds, order tracking, cancellations, replacements, payment issues, invoices, delivery, account management, complaints, and escalation to a human agent — in the terminology and tone of this business, rather than generic advice a base LLM would give for "contact your retailer."

## Dataset details

All three datasets are derived from the public **[Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)** (CC BY 4.0, ~26.9k rows across 27 customer-support intents such as `cancel_order`, `get_refund`, `track_order`, `payment_issue`, `check_invoice`, `contact_human_agent`, etc.), processed locally by `scripts/build_datasets.py`:

| File | Rows/paragraphs | How it was built |
|---|---|---|
| `data/non_instruction_data.txt` | 54 paragraphs | For each of the 27 intents, 2 distinct real responses are merged into a knowledge-base-style prose paragraph (no Q&A framing) — raw domain text for Stage 1. |
| `data/instruction_dataset.jsonl` | 135 pairs | Stratified sample (~5 per intent) of `{"instruction", "response"}` pairs for Stage 2 SFT. |
| `data/preference_dataset.jsonl` | 54 triples | `{"prompt", "chosen", "rejected"}` — `chosen` is a real Bitext response, `rejected` is one of a small set of deliberately dismissive/unhelpful templates, for Stage 3 DPO. |
| `data/eval_questions.json` | 10 questions | Fixed set used identically by all three notebooks so before/after tables line up. |

**Cleaning applied:** Bitext wraps personalization slots in `{{Placeholder}}` tokens. Rows containing name/greeting placeholders (`{{Person Name}}`, `{{Salutation}}`, etc.) were dropped rather than patched, since substituting a name cleanly needs grammar-aware rewriting. The remaining, non-personal placeholders (`{{Order Number}}`, `{{Invoice Number}}`, `{{Website URL}}`, `{{Customer Support Phone Number}}`, `{{Customer Support Hours}}`, `{{Online Order Interaction}}`, `{{Online Company Portal Info}}`, `{{Date Range}}`) were substituted with concrete, plausible values (e.g. `ORD-58204`, `1-800-555-0199`) and a small regex pass fixed article clashes the substitution introduced (`"your the X"` → `"the X"`). Rows leaking the dataset's own internal PII-anonymization notes were filtered out. See `scripts/build_datasets.py` for the exact logic — re-running it is fully deterministic (seeded).

**License note:** Bitext-customer-support-llm-chatbot-training-dataset is CC BY 4.0 — free to use with attribution, which is given here and in `scripts/build_datasets.py`.

## Base model used

**`unsloth/Qwen2.5-0.5B`** — the plain (non-instruct) base checkpoint, loaded 4-bit via Unsloth. Chosen from the assignment's recommended list for being the smallest/fastest, while still fitting comfortably on a free Colab T4 for all three training stages.

## Three-stage approach

```
Base Model (Qwen2.5-0.5B)
    -> Stage 1: Non-instruction fine-tuning on data/non_instruction_data.txt
       (plain causal-LM objective, no instruction framing - teaches domain
       vocabulary and phrasing)
    -> Stage 2: Instruction fine-tuning (SFT) on data/instruction_dataset.jsonl
       (Alpaca-style prompt/response template - teaches the model to actually
       answer questions)
    -> Stage 3: DPO alignment on data/preference_dataset.jsonl
       (prompt/chosen/rejected triples - teaches the model to prefer the
       correct/professional answer over a weak one)
    -> Final Domain-Specific Customer Support Assistant
```

Each stage resumes from the previous stage's saved adapter automatically (the notebooks check whether `outputs/stage{N-1}_adapter` exists and load from it if so, falling back to the plain base model otherwise) — see `notebooks/instruction_finetuning.ipynb` and `notebooks/dpo_alignment.ipynb`.

## LoRA / QLoRA configuration

All three stages use 4-bit quantization (QLoRA) with the same LoRA shape, and stage-appropriate learning rates:

| Setting | Value |
|---|---|
| LoRA rank `r` | 16 |
| LoRA `alpha` | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Stage 1 / Stage 2 learning rate | 2e-4 |
| Stage 3 (DPO) learning rate | 5e-6, `beta=0.1` |
| Batch size / grad accumulation | 2 / 4 (effective 8) |
| Max sequence length | 1024 |

Full reasoning for these choices is in [`reports/fine_tuning_explanation.md`](reports/fine_tuning_explanation.md).

## Training screenshots or logs

_TODO (fill in after running the notebooks on Colab): paste `trainer.train()` loss curves / screenshots for each of the three stages here._

## Before vs. after output comparison

Generated automatically by the notebooks — see:
- [`reports/base_model_evaluation.md`](reports/base_model_evaluation.md) — base model on 10 fixed questions
- [`reports/sft_model_comparison.md`](reports/sft_model_comparison.md) — base vs. instruction-tuned
- [`reports/final_evaluation.md`](reports/final_evaluation.md) — base vs. SFT vs. DPO, three-way

## How to run

This repo was assembled on a machine with **no GPU**, so the dataset files were built and verified locally, but the three training notebooks need to be *run* on a GPU runtime.

1. **Datasets are already built** (`data/*.txt`, `data/*.jsonl`, `data/eval_questions.json`). To rebuild them from scratch: `pip install datasets huggingface_hub pandas && python scripts/build_datasets.py`.
2. **Open `notebooks/` on Google Colab** (Runtime → change runtime type → T4 GPU), upload/clone this repo so the relative paths (`../data`, `../outputs`, `../reports`, `../src`) resolve, and run in order:
   1. `non_instruction_finetuning.ipynb`
   2. `instruction_finetuning.ipynb` (auto-fills `base_model_evaluation.md` and `sft_model_comparison.md`)
   3. `dpo_alignment.ipynb` (auto-fills `final_evaluation.md`)
3. **Query the final model** locally or on Colab: `python src/inference.py "How can I get a refund?"` (or run it with no arguments for an interactive loop). It loads `outputs/stage3_dpo_model`, produced by step 2.3.

## Final observations

- Domain adaptation for a customer-support assistant is largely about *vocabulary and structure* (order numbers, policy names, "My Orders" page, escalation phrasing) rather than deep reasoning, which is exactly the kind of gap non-instruction + instruction fine-tuning is well suited to close on a small model.
- Keeping the same fixed prompt template and the same 10 evaluation questions across all three stages was important for the before/after tables to be a fair comparison rather than an artifact of prompt differences.
- DPO's learning rate needs to be much smaller than SFT's — it's refining an already-competent model's preferences, not teaching it a new skill from scratch.

## Challenges faced

- The Bitext dataset's own synthetic responses occasionally have odd colloquial phrasing (an artifact of how the source dataset itself was generated) and personalization placeholders that don't substitute cleanly into plain text — handled by filtering rather than trying to patch every case.
- No local GPU meant the notebooks could be authored and reviewed for correctness but not executed end-to-end in this environment; the auto-report-filling design (notebooks write straight into `reports/*.md`) exists specifically so a single Colab run leaves the repo in the same state a fully-local run would.

## Future improvements

- Expand the preference dataset with harder negatives (plausible-but-subtly-wrong answers) rather than obviously dismissive ones, which would give DPO a more meaningful signal than "obviously good vs. obviously bad."
- Try a slightly larger base model (Llama-3.2-1B) once the pipeline is validated on 0.5B, and compare.
- Replace the hand-rotated "Which is better?" / "Reason" defaults in the auto-filled reports with an actual LLM-as-judge pass for a less biased comparison.

## Repository structure

```
data/                  non_instruction_data.txt, instruction_dataset.jsonl,
                        preference_dataset.jsonl, eval_questions.json
scripts/               build_datasets.py (one-off local dataset builder)
notebooks/             non_instruction_finetuning.ipynb, instruction_finetuning.ipynb,
                        dpo_alignment.ipynb
reports/               base_model_evaluation.md, sft_model_comparison.md,
                        final_evaluation.md, fine_tuning_explanation.md
src/                   report_utils.py (shared by notebooks), inference.py
outputs/               saved adapters/models (git-ignored, created by running the notebooks)
```
