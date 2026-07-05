"""
One-off local data-prep script for the Customer Support AI Assistant assignment.

Downloads the public Bitext customer-support dataset from Hugging Face
(https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset,
CC BY 4.0) and derives the three files the assignment asks for:

  data/non_instruction_data.txt   - 50+ prose knowledge-base paragraphs (raw domain text)
  data/instruction_dataset.jsonl  - 100+ {"instruction","response"} pairs
  data/preference_dataset.jsonl   - 50+ {"prompt","chosen","rejected"} triples
  data/eval_questions.json        - 10 fixed questions used by all three notebooks

Run once locally:  python scripts/build_datasets.py
"""
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Bitext wraps personalization slots in {{...}}. Tokens that name a concrete,
# non-personal thing are safe to substitute with a plausible fixed value.
# Tokens that stand in for a person's name/salutation are dropped instead,
# because substituting them cleanly requires grammar-aware rewriting that's
# out of scope here - we simply skip any row that contains one of them.
SAFE_SUBSTITUTIONS = {
    "Order Number": "ORD-58204",
    "Invoice Number": "INV-30221",
    "Online Order Interaction": "'My Orders' page",
    "Online Company Portal Info": "'My Account' portal",
    "Website URL": "website",
    "Customer Support Phone Number": "1-800-555-0199",
    "Customer Support Hours": "Monday to Friday, 9 AM to 6 PM",
    "Customer Support Email": "support@example.com",
    "Customer Support Team": "our support team",
    "Date Range": "the last 30 days",
    "Account Number": "your account number",
    "Cancel Purchase": "Cancel Purchase",
    "Refund Policy": "Refund Policy",
    "Return Policy": "Return Policy",
    "Cancellation Policy": "Cancellation Policy",
    "Settings": "Settings",
    "My Purchases": "My Purchases",
}

# Any row containing one of these name/greeting placeholders is skipped
# entirely rather than patched, to avoid mangled personalized greetings.
SKIP_IF_CONTAINS = {
    "Person Name", "Client Last Name", "Client First Name", "Client Name",
    "Salutation", "Company Account", "Company Name", "Company",
    "Confirm Cancellation", "Remove", "Account Name", "Billing Category",
    "Date of the Invoice", "Date of the Bill", "Timeframe", "Year", "Billing",
    "additional details", "month",
}

PLACEHOLDER_RE = re.compile(r"\{\{([^}]+)\}\}")


def has_skip_token(text: str) -> bool:
    return any(f"{{{{{tok}}}}}" in text for tok in SKIP_IF_CONTAINS)


def substitute(text: str) -> str:
    def repl(m):
        key = m.group(1)
        return SAFE_SUBSTITUTIONS.get(key, key)

    text = PLACEHOLDER_RE.sub(repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    # fix article clashes created by substitution, e.g. "your 'My Account' portal"
    # preceded by another article/possessive, or "our our website"
    text = re.sub(r"\b(our|your|the)\s+(our|your|the)\b", r"\2", text, flags=re.IGNORECASE)
    return text


LEAKED_META_MARKERS = ("anonymize", "personally identifiable", " pii ", "placeholders")


def is_clean(row) -> bool:
    combined = row["instruction"] + " " + row["response"]
    if has_skip_token(combined):
        return False
    lowered = combined.lower()
    if any(marker in lowered for marker in LEAKED_META_MARKERS):
        return False
    # any placeholder token not in our safe map would leak literal {{...}}
    for tok in PLACEHOLDER_RE.findall(combined):
        if tok not in SAFE_SUBSTITUTIONS:
            return False
    return True


def main():
    print("Downloading Bitext customer-support dataset ...")
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")["train"]

    by_intent = {}
    for row in ds:
        if not is_clean(row):
            continue
        by_intent.setdefault(row["intent"], []).append(row)

    intents = sorted(by_intent.keys())
    print(f"{len(intents)} intents, {sum(len(v) for v in by_intent.values())} clean rows")

    # ------------------------------------------------------------------
    # 1) non_instruction_data.txt - prose knowledge-base paragraphs
    # ------------------------------------------------------------------
    paragraphs = []
    for intent in intents:
        rows = by_intent[intent]
        # de-dup responses, keep the longer/more informative ones first
        seen = set()
        uniq = []
        for r in sorted(rows, key=lambda r: -len(r["response"])):
            resp = substitute(r["response"])
            if resp not in seen:
                seen.add(resp)
                uniq.append(resp)
        # build 2 paragraphs per intent from distinct response pairs
        topic = intent.replace("_", " ")
        for i in range(2):
            chunk = uniq[i * 2 : i * 2 + 2]
            if len(chunk) < 2:
                chunk = uniq[:2]
            if not chunk:
                continue
            heading = f"[{topic.upper()}]"
            paragraph = heading + " " + " ".join(chunk)
            paragraphs.append(paragraph)

    non_instruction_path = DATA_DIR / "non_instruction_data.txt"
    non_instruction_path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    print(f"Wrote {len(paragraphs)} paragraphs -> {non_instruction_path}")

    # ------------------------------------------------------------------
    # 2) instruction_dataset.jsonl - instruction/response pairs
    # ------------------------------------------------------------------
    instruction_examples = []
    seen_instructions = set()
    for intent in intents:
        rows = list(by_intent[intent])
        random.shuffle(rows)
        picked = 0
        for r in rows:
            instr = substitute(r["instruction"])
            key = instr.lower()
            if key in seen_instructions:
                continue
            resp = substitute(r["response"])
            seen_instructions.add(key)
            instruction_examples.append({"instruction": instr, "response": resp})
            picked += 1
            if picked >= 5:
                break

    instruction_path = DATA_DIR / "instruction_dataset.jsonl"
    with instruction_path.open("w", encoding="utf-8") as f:
        for ex in instruction_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(instruction_examples)} instruction examples -> {instruction_path}")

    # ------------------------------------------------------------------
    # 3) preference_dataset.jsonl - prompt/chosen/rejected triples
    # ------------------------------------------------------------------
    BAD_TEMPLATES = [
        "I don't know, figure it out yourself.",
        "Just ignore it, it's not a big deal.",
        "We don't handle that here, try somewhere else.",
        "That's not possible.",
        "Whatever, do what you want.",
        "Read the website, I'm not going to explain it.",
        "That's your problem, not ours.",
    ]

    used_instructions = seen_instructions.copy()
    preference_examples = []
    for intent in intents:
        rows = list(by_intent[intent])
        random.shuffle(rows)
        picked = 0
        for r in rows:
            instr = substitute(r["instruction"])
            key = instr.lower()
            if key in used_instructions:
                continue
            used_instructions.add(key)
            chosen = substitute(r["response"])
            rejected = random.choice(BAD_TEMPLATES)
            preference_examples.append(
                {"prompt": instr, "chosen": chosen, "rejected": rejected}
            )
            picked += 1
            if picked >= 2:
                break

    preference_path = DATA_DIR / "preference_dataset.jsonl"
    with preference_path.open("w", encoding="utf-8") as f:
        for ex in preference_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(preference_examples)} preference examples -> {preference_path}")

    # ------------------------------------------------------------------
    # 4) eval_questions.json - 10 fixed questions shared across notebooks
    # ------------------------------------------------------------------
    eval_questions = [
        "How can I get a refund for my order?",
        "How do I track my order?",
        "Can I cancel an order after placing it?",
        "How do I request a replacement for a damaged product?",
        "My payment failed, what should I do?",
        "How long does delivery usually take?",
        "How can I get a copy of my invoice?",
        "How do I delete my account?",
        "I want to file a complaint about my order, how do I do that?",
        "How can I speak to a human customer support agent?",
    ]
    eval_path = DATA_DIR / "eval_questions.json"
    eval_path.write_text(json.dumps(eval_questions, indent=2), encoding="utf-8")
    print(f"Wrote {len(eval_questions)} eval questions -> {eval_path}")


if __name__ == "__main__":
    main()
