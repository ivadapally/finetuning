# Base Model Evaluation

**Base model:** `unsloth/Qwen2.5-0.5B` (plain base checkpoint, before any fine-tuning — or after Stage 1 non-instruction fine-tuning if `outputs/stage1_adapter` exists, but still *before* any instruction tuning).

Tested on the 10 fixed customer-support questions in `data/eval_questions.json`. This table is generated automatically by the first cells of `notebooks/instruction_finetuning.ipynb`, before instruction fine-tuning starts — re-running that notebook regenerates it.

<!-- AUTO:base_eval:START -->
_Not yet generated. Run `notebooks/instruction_finetuning.ipynb` on a GPU runtime (e.g. Google Colab) to populate this table._
<!-- AUTO:base_eval:END -->

## Observation

The base model has general language ability but no exposure to this business's specific refund/order/payment policies or tone. Expect answers that are plausible-sounding but generic, occasionally off-topic, and not phrased the way a real support agent for this store would answer (e.g. no mention of the actual "My Orders" page, no consistent escalation path to a human agent). This is the "before" baseline that Stage 2 (instruction fine-tuning) and Stage 3 (DPO alignment) are compared against in `sft_model_comparison.md` and `final_evaluation.md`.
