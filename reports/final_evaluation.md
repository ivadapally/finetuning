# Final Evaluation: Base vs. SFT vs. DPO-Aligned Model

All three stages answering the same 10 questions from `data/eval_questions.json`, with the identical prompt template. Generated automatically by the last cells of `notebooks/dpo_alignment.ipynb`, which independently loads the plain base model, the Stage 2 SFT model (`outputs/stage2_adapter`), and the Stage 3 DPO model (`outputs/stage3_dpo_model`).

<!-- AUTO:final_eval:START -->
_Not yet generated. Run `notebooks/dpo_alignment.ipynb` on a GPU runtime (e.g. Google Colab) to populate this table._
<!-- AUTO:final_eval:END -->

## Evaluation criteria

"Best Answer" / "Reason" are auto-filled with a reasonable default (DPO) and should be **reviewed and edited by hand**, judging on:

- Correctness and domain accuracy
- Helpfulness and clarity
- Safety and professional tone
- Hallucination reduction (does it invent policies that don't exist?)
- Overall response quality vs. the base model's generic answer

## Summary

Fill in after reviewing the table above:

- Did instruction fine-tuning (SFT) noticeably improve domain-specificity over the base model?
- Did DPO alignment noticeably improve tone/helpfulness/safety over SFT, or were the preference pairs too easy (rejected responses too obviously bad) to teach a meaningful signal?
- Any questions where the base model was still competitive, and why?
