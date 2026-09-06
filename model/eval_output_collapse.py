"""See docs/research/eval_output_collapse.rst:module_overview."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

PredictFn = Callable[[int], tuple[list[int], list[int], list[np.ndarray]]]


@dataclass
class OutputCollapseReport:
    n_samples: int
    accuracy: float
    unique_prediction_fraction: float
    most_common_prediction_fraction: float
    prediction_entropy_bits: float
    max_entropy_bits: float  # log2(vocab_size), for normalizing
    mean_logit_std: float  # average WITHIN-sample logit spread
    cross_sample_logit_std: float  # average ACROSS-sample logit spread per class

    @property
    def normalized_entropy(self) -> float:
        """See docs/research/eval_output_collapse.rst:normalized_entropy_threshold."""
        return self.prediction_entropy_bits / self.max_entropy_bits if self.max_entropy_bits > 0 else 0.0


def check_output_collapse(
    predict_fn: PredictFn, *, vocab_size: int, n_batches: int = 1, seed: int = 0
) -> OutputCollapseReport:
    """See docs/research/eval_output_collapse.rst:check_output_collapse_design."""
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_logits: list[np.ndarray] = []
    for b in range(n_batches):
        preds, targets, logits = predict_fn(seed + b)
        all_preds.extend(preds)
        all_targets.extend(targets)
        all_logits.extend(logits)

    n = len(all_preds)
    if n == 0:
        raise ValueError("predict_fn returned zero samples")

    correct = sum(p == t for p, t in zip(all_preds, all_targets, strict=False))
    accuracy = correct / n

    counts: dict = {}
    for p in all_preds:
        counts[p] = counts.get(p, 0) + 1
    unique_fraction = len(counts) / n
    most_common_fraction = max(counts.values()) / n

    entropy = 0.0
    for c in counts.values():
        p = c / n
        entropy -= p * math.log2(p)
    max_entropy = math.log2(vocab_size) if vocab_size > 1 else 0.0

    logit_stack = np.stack(all_logits, axis=0)  # (n, vocab_size)
    mean_logit_std = float(np.mean(np.std(logit_stack, axis=1)))
    cross_sample_logit_std = float(np.mean(np.std(logit_stack, axis=0)))

    return OutputCollapseReport(
        n_samples=n,
        accuracy=accuracy,
        unique_prediction_fraction=unique_fraction,
        most_common_prediction_fraction=most_common_fraction,
        prediction_entropy_bits=entropy,
        max_entropy_bits=max_entropy,
        mean_logit_std=mean_logit_std,
        cross_sample_logit_std=cross_sample_logit_std,
    )
