# Fine-Tuning Concepts — In My Own Words

## Why full fine-tuning is expensive

Full fine-tuning updates every weight in the model. For a 0.5B-1B parameter model that's still hundreds of millions of gradients, optimizer states (Adam keeps two extra float32 buffers per parameter), and activations to hold in GPU memory simultaneously — easily 4-8x the model's raw size in VRAM. Scale that to a 7B+ model and you need multiple high-end GPUs just to fit one training run, plus the storage cost of saving a full new checkpoint per experiment. For a domain-adaptation task like this one, where we only need to nudge behavior rather than relearn language from scratch, that cost is disproportionate to the benefit.

## What LoRA does

LoRA (Low-Rank Adaptation) freezes the original weight matrices and, for selected layers (here: the attention `q/k/v/o_proj` and MLP `gate/up/down_proj`), adds a pair of small trainable matrices `A` (shape `d x r`) and `B` (shape `r x d`) whose product `A@B` is added to the frozen weight at inference time. Only `A` and `B` are trained — with `r=16` that's a tiny fraction of the original parameter count. Because the base weights never change, you can swap adapters in and out, keep several task-specific adapters for one base model, and the optimizer state is proportional to the adapter size, not the model size.

## What QLoRA does

QLoRA is LoRA applied on top of a **4-bit quantized** frozen base model (via bitsandbytes' NF4 format), with the LoRA adapters themselves kept in higher precision (bf16/fp16) for stable gradients. This cuts the memory needed to hold the frozen base weights by ~4x compared to fp16, on top of LoRA's already-small trainable footprint. Unsloth's `load_in_4bit=True` in `FastLanguageModel.from_pretrained` is exactly this.

## Why QLoRA is useful on a limited GPU

It's the combination that makes fine-tuning a language model feasible on a single free-tier GPU (e.g. a Colab T4 with ~15GB VRAM): the frozen base model's memory footprint shrinks 4x from quantization, and the trainable parameter count (and its optimizer state) shrinks by orders of magnitude from LoRA. Without it, even a "small" 0.5B-1B model's full fine-tuning optimizer state plus activations can be tight on a T4; QLoRA leaves comfortable headroom for a reasonable batch size and sequence length.

## What is non-instruction fine-tuning?

It's continued pre-training: plain next-token prediction on raw, unstructured domain text (here, `data/non_instruction_data.txt` — prose knowledge-base paragraphs about refunds, cancellations, payments, etc.), with **no** instruction/response framing. The goal isn't to teach the model to follow instructions — it's to shift its internal language statistics toward domain vocabulary, phrasing, and recurring concepts (order numbers, "My Orders" page, refund policy, etc.) before it ever sees a question-answer pair.

## What is instruction fine-tuning?

Also called SFT (Supervised Fine-Tuning). The model is trained on paired `(instruction, response)` examples formatted with a consistent prompt template, so it learns the *behavior* of answering a question in the expected format and tone, rather than just continuing text. This is where the model actually becomes a usable assistant instead of a text-completion engine.

## What is DPO?

Direct Preference Optimization trains directly on `(prompt, chosen, rejected)` triples: it increases the model's relative preference for the `chosen` response over the `rejected` one for the same prompt, using a frozen reference model (in our case, the same LoRA model with its adapter disabled) as an anchor so the policy doesn't drift arbitrarily far while optimizing preference. There's no separate reward model to train first (unlike RLHF/PPO) — the preference data is used directly in the loss, which is why it's simpler to run than full RLHF while still capturing "which answer is better" signal that plain SFT can't.

## Difference between SFT and DPO

SFT only ever sees "correct" examples and learns to imitate them — it has no notion of *what a bad answer looks like* or *how much better one answer is than another*. DPO is trained on contrastive pairs and explicitly pushes the good answer's probability up and the bad answer's probability down for the same prompt. In practice: SFT teaches the model to be a domain assistant at all; DPO refines *how it behaves at the margins* — tone, safety, avoiding the generic/dismissive answers that SFT alone doesn't explicitly penalize.

## Hyperparameters used in this project

| Stage | Setting | Value |
|---|---|---|
| LoRA (all stages) | rank `r` | 16 |
| LoRA (all stages) | `lora_alpha` | 16 |
| LoRA (all stages) | `lora_dropout` | 0.05 |
| LoRA (all stages) | target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Stage 1 (non-instruction) | learning rate | 2e-4 |
| Stage 1 (non-instruction) | batch size / grad accum | 2 / 4 (effective 8) |
| Stage 1 (non-instruction) | epochs | 3 |
| Stage 2 (instruction / SFT) | learning rate | 2e-4 |
| Stage 2 (instruction / SFT) | batch size / grad accum | 2 / 4 (effective 8) |
| Stage 2 (instruction / SFT) | epochs | 3 |
| Stage 3 (DPO) | learning rate | 5e-6 |
| Stage 3 (DPO) | beta | 0.1 |
| Stage 3 (DPO) | batch size / grad accum | 2 / 4 (effective 8) |
| Stage 3 (DPO) | epochs | 2 |
| All stages | max sequence length | 1024 |
| All stages | quantization | 4-bit (QLoRA) |

The DPO learning rate is deliberately ~40x smaller than the SFT rate — DPO is refining an already-competent model's preferences, and a learning rate as high as SFT's would risk overwriting what Stage 2 just taught it.
