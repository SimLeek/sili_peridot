"""See docs/research/eval_pruning.rst:module_overview."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# See docs/research/eval_pruning.rst:eval_texts_design.
EVAL_TEXTS: list[str] = [
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

# See docs/research/eval_pruning.rst:eval_texts_heldout_design.
EVAL_TEXTS_HELDOUT: list[str] = [
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
    per_text_loss: list[float]
    per_text_accuracy: list[float]

    @property
    def perplexity(self) -> float:
        avg_loss = sum(self.per_text_loss) / len(self.per_text_loss)
        return float(torch.exp(torch.tensor(avg_loss)))

    @property
    def accuracy(self) -> float:
        return sum(self.per_text_accuracy) / len(self.per_text_accuracy)


def evaluate_next_token_prediction(model, tokenizer, texts: list[str] = EVAL_TEXTS) -> EvalResult:
    """Teacher-forced next-token loss (HF's own shifted cross-entropy via
    labels=input_ids) and top-1 accuracy, per text."""
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt")
            out = model(**ids, labels=ids["input_ids"])
            losses.append(float(out.loss))

            logits = out.logits[0, :-1]  # predict token t+1 from position t
            targets = ids["input_ids"][0, 1:]
            preds = logits.argmax(dim=-1)
            accs.append(float((preds == targets).float().mean()))
    return EvalResult(per_text_loss=losses, per_text_accuracy=accs)


def compare_dense_vs_pruned(
    model,
    tokenizer,
    pruned_dense_state_dict: dict[str, torch.Tensor],
    texts: list[str] = EVAL_TEXTS,
) -> dict:
    """See docs/research/eval_pruning.rst:compare_dense_vs_pruned_restore."""
    original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    try:
        model.load_state_dict(original_state_dict)  # ensure a known-clean start
        dense_result = evaluate_next_token_prediction(model, tokenizer, texts)

        model.load_state_dict(pruned_dense_state_dict)
        pruned_result = evaluate_next_token_prediction(model, tokenizer, texts)
    finally:
        model.load_state_dict(original_state_dict)

    return {
        "dense_perplexity": dense_result.perplexity,
        "pruned_perplexity": pruned_result.perplexity,
        "dense_accuracy": dense_result.accuracy,
        "pruned_accuracy": pruned_result.accuracy,
        "dense_per_text_loss": dense_result.per_text_loss,
        "pruned_per_text_loss": pruned_result.per_text_loss,
    }
