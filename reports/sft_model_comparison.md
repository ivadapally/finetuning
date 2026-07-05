# Base Model vs. Instruction Fine-Tuned (SFT) Model

Same 10 questions from `data/eval_questions.json`, asked of the base model and of the Stage 2 instruction-fine-tuned (`outputs/stage2_adapter`) model with the identical prompt template, so the only variable is the training. Generated automatically by the last cells of `notebooks/instruction_finetuning.ipynb`.

<!-- AUTO:sft_comparison:START -->
_Not yet generated. Run `notebooks/instruction_finetuning.ipynb` on a GPU runtime (e.g. Google Colab) to populate this table._
<!-- AUTO:sft_comparison:END -->

## Evaluation criteria

Each row's "Which is Better?" / "Reason" columns are auto-filled with a reasonable default (fine-tuned model, more domain-specific) and are meant to be **reviewed and edited by hand** after reading the actual generations, judging on:

- Correctness (does it match our actual refund/order/payment process?)
- Domain accuracy (right terminology: "My Orders" page, order number, invoice, etc.)
- Clarity and helpfulness
- Safety / professionalism
- Whether it avoids a generic, could-apply-to-any-business answer
