"""
sili_peridot/model/eval_pruning.py
────────────────────────────────────
Does a given pruning decision actually degrade MiniCPM5-1B-Base's
next-token prediction quality by an unacceptable amount? Loads the real
HuggingFace model, evaluates it dense, then loads an already-pruned
dense state dict (built by model/prune.py -- no CSR conversion, no
folding, no column-averaging, no sili runtime involved at all here --
purely "does zeroing these specific weights hurt", isolated from every
later conversion step) and compares next-token loss/perplexity and
top-1 accuracy on a small held-out text sample.

This module has NO pruning-construction logic of its own -- see
model/prune.py for that (prune_state_dict / prune_state_dict_by_role,
plus prune.sparse_state_to_dense_state_dict to turn either's output back
into plain tensors loadable via model.load_state_dict). Keeping the two
concerns separate avoids two parallel implementations of "which weights
get zeroed."

torch/transformers-only; not part of the sili runtime path, only a
validation step for the conversion pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

# Short, diverse, hand-written passages -- no external dataset
# dependency. Plain declarative English is exactly what a BASE model
# (no instruction tuning) should already predict well; this isn't meant
# to be a rigorous benchmark, just a sanity-scale check that pruning
# hasn't broken the model in an obvious way.
EVAL_TEXTS: List[str] = [
    "The capital of France is Paris, a city known for its museums and "
    "architecture. Many tourists visit every year to see the Eiffel Tower.",

    "Water boils at one hundred degrees Celsius at sea level. As "
    "altitude increases, the boiling point of water decreases.",

    "She opened the old wooden door and stepped into the quiet library. "
    "Dust floated in the afternoon light as she searched the shelves.",

    "The mitochondria is often called the powerhouse of the cell because "
    "it produces most of the cell's supply of adenosine triphosphate.",

    "In the morning, the fishermen pushed their small boats into the "
    "gray water and rowed out past the harbor wall toward open sea.",
]

# A second, disjoint set (different topics/style) -- never used during
# threshold search, only to check a chosen set of thresholds isn't
# overfit to the specific EVAL_TEXTS snippets above. Confirmed on the
# final DEFAULT_TARGET_SPARSITY_BY_ROLE thresholds: pruned accuracy held
# steady across both sets (0.482 vs. 0.478) even though the dense
# baseline itself varies more between them (0.503 vs. 0.584) -- see
# JOURNAL.md.
EVAL_TEXTS_HELDOUT: List[str] = [
    "Mount Everest is the tallest mountain above sea level on Earth, "
    "located in the Himalayas on the border between Nepal and Tibet.",

    "He measured the flour carefully, then folded it into the batter "
    "along with two eggs and a pinch of salt before heating the pan.",

    "Photosynthesis is the process by which plants convert sunlight, "
    "water, and carbon dioxide into glucose and release oxygen.",

    "The train pulled slowly out of the station as passengers waved "
    "goodbye through the rain-streaked windows of the carriage.",

    "A honeybee colony typically has one queen, thousands of worker "
    "bees, and a much smaller number of drones during the summer months.",
]


@dataclass
class EvalResult:
    per_text_loss: List[float]
    per_text_accuracy: List[float]

    @property
    def perplexity(self) -> float:
        avg_loss = sum(self.per_text_loss) / len(self.per_text_loss)
        return float(torch.exp(torch.tensor(avg_loss)))

    @property
    def accuracy(self) -> float:
        return sum(self.per_text_accuracy) / len(self.per_text_accuracy)


def evaluate_next_token_prediction(model, tokenizer, texts: List[str] = EVAL_TEXTS) -> EvalResult:
    """Teacher-forced next-token loss (HF's own shifted cross-entropy via
    labels=input_ids) and top-1 accuracy, per text."""
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt")
            out = model(**ids, labels=ids["input_ids"])
            losses.append(float(out.loss))

            logits = out.logits[0, :-1]          # predict token t+1 from position t
            targets = ids["input_ids"][0, 1:]
            preds = logits.argmax(dim=-1)
            accs.append(float((preds == targets).float().mean()))
    return EvalResult(per_text_loss=losses, per_text_accuracy=accs)


def compare_dense_vs_pruned(
    model,
    tokenizer,
    pruned_dense_state_dict: Dict[str, torch.Tensor],
    texts: List[str] = EVAL_TEXTS,
) -> dict:
    """
    Evaluate `model` dense, then with `pruned_dense_state_dict` loaded
    in, then restore its original weights -- so the caller's model isn't
    left mutated. `pruned_dense_state_dict` is typically
    prune.sparse_state_to_dense_state_dict(prune.prune_state_dict_by_role(...)[0]).

    Returns a plain dict so callers/tests don't need EvalResult imported.
    """
    original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    try:
        model.load_state_dict(original_state_dict)   # ensure a known-clean start
        dense_result = evaluate_next_token_prediction(model, tokenizer, texts)

        model.load_state_dict(pruned_dense_state_dict)
        pruned_result = evaluate_next_token_prediction(model, tokenizer, texts)
    finally:
        model.load_state_dict(original_state_dict)

    return {
        "dense_perplexity":  dense_result.perplexity,
        "pruned_perplexity": pruned_result.perplexity,
        "dense_accuracy":    dense_result.accuracy,
        "pruned_accuracy":   pruned_result.accuracy,
        "dense_per_text_loss":  dense_result.per_text_loss,
        "pruned_per_text_loss": pruned_result.per_text_loss,
    }
