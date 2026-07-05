"""Simple inference script for the final DPO-aligned Customer Support Assistant.

Usage:
    python src/inference.py "How can I request a refund?"
    python src/inference.py            # interactive loop

Requires the Stage 3 model to exist at outputs/stage3_dpo_model, which is
produced by running notebooks/dpo_alignment.ipynb on a GPU runtime.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "outputs" / "stage3_dpo_model"

PROMPT_TEMPLATE = (
    "Below is a customer support question. Write a helpful, professional, "
    "domain-specific response.\n\n### Question:\n{}\n\n### Response:\n"
)

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}.\n"
            "Run the three notebooks in notebooks/ (on a GPU runtime, e.g. Colab) "
            "in order first:\n"
            "  1. non_instruction_finetuning.ipynb\n"
            "  2. instruction_finetuning.ipynb\n"
            "  3. dpo_alignment.ipynb\n"
            "The last one saves the final model to outputs/stage3_dpo_model."
        )

    from unsloth import FastLanguageModel

    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MODEL_PATH),
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(_model)
    return _model, _tokenizer


def generate_answer(question: str, max_new_tokens: int = 150) -> str:
    """Answer a single customer-support question with the final DPO-aligned model."""
    model, tokenizer = _load_model()
    prompt = PROMPT_TEMPLATE.format(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=False,
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("### Response:")[-1].strip()


def main():
    try:
        if len(sys.argv) > 1:
            question = " ".join(sys.argv[1:])
            answer = generate_answer(question)
            print(answer)
            return

        print("Customer Support Assistant — type a question (or 'quit' to exit)")
        while True:
            try:
                question = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question or question.lower() in {"quit", "exit"}:
                break
            answer = generate_answer(question)
            print(f"Answer: {answer}")
    except FileNotFoundError as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
