# Customer Support AI Assistant — Build Summary & Next Steps

## What this repo is

A complete submission scaffold for the "Practical Fine-Tuning Assignment" — a **Customer Support Assistant** fine-tuned with Unsloth through three stages (non-instruction FT → instruction FT/SFT → DPO alignment), built on top of the public [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).

## What's already done (built and verified on this machine, no GPU needed)

| Deliverable | Status |
|---|---|
| `data/non_instruction_data.txt` (54 paragraphs) | Done, validated |
| `data/instruction_dataset.jsonl` (135 pairs) | Done, validated |
| `data/preference_dataset.jsonl` (54 triples) | Done, validated |
| `data/eval_questions.json` (10 questions) | Done |
| `scripts/build_datasets.py` | Done — reproducible, seeded |
| `notebooks/non_instruction_finetuning.ipynb` | Written, syntax-checked — **not executed** (no local GPU) |
| `notebooks/instruction_finetuning.ipynb` | Written, syntax-checked — **not executed** |
| `notebooks/dpo_alignment.ipynb` | Written, syntax-checked — **not executed** |
| `reports/fine_tuning_explanation.md` | Done (fully written, no execution needed) |
| `reports/base_model_evaluation.md` | Template only — auto-fills when notebook 2 runs |
| `reports/sft_model_comparison.md` | Template only — auto-fills when notebook 2 runs |
| `reports/final_evaluation.md` | Template only — auto-fills when notebook 3 runs |
| `src/inference.py`, `src/report_utils.py` | Done, dry-run tested |
| `README.md`, `requirements.txt`, `.gitignore` | Done |

## What YOU need to do next

1. **Run the notebooks on a GPU runtime** (Google Colab's free T4 is enough for the 0.5B model):
   - Upload/clone this repo to Colab so relative paths (`../data`, `../outputs`, `../reports`, `../src`) resolve correctly.
   - Run in order — each stage depends on the previous one's saved adapter:
     1. `notebooks/non_instruction_finetuning.ipynb`
     2. `notebooks/instruction_finetuning.ipynb` (fills in `base_model_evaluation.md` and `sft_model_comparison.md`)
     3. `notebooks/dpo_alignment.ipynb` (fills in `final_evaluation.md`)
2. **Review the auto-filled reports by hand.** The "Which is Better?" / "Reason" / "Best Answer" columns are auto-filled with reasonable defaults — read the actual model answers and correct these where the default guess is wrong.
3. **Add training screenshots/logs** to the "Training screenshots or logs" section of `README.md` (currently a TODO placeholder) — copy the loss curves or `trainer.train()` output from each notebook.
4. **Push to GitHub** and submit the repo link via the assignment's Google Form before **11 July 2026**.

## Reference

Full project write-up, dataset details, LoRA/QLoRA config, and "how to run" instructions are in [`README.md`](README.md). Concept explanations (LoRA/QLoRA/SFT/DPO) are in [`reports/fine_tuning_explanation.md`](reports/fine_tuning_explanation.md).
