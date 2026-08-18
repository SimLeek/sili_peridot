"""
sili_peridot/model/eval_output_collapse.py
─────────────────────────────────────────────
Does a trained model's output actually depend on its input, or has it
collapsed to predicting a narrow (or single) set of tokens regardless of
what's fed in? A bare accuracy number can't tell these apart --
scripts/l1_sparsity_probe.py's own evaluate() docstring already flags
this exact concern ("the model is degenerate/guessing a narrow set of
tokens" vs "these particular held-out sequences happened to skew toward
one target token") and works around it with a `verbose=` flag that
prints the last 5 (pred, target) pairs -- a partial, qualitative check.
This module makes it a real, quantitative diagnostic instead.

Domain-agnostic like eval_lr.py/eval_eigenvalues.py: callers supply a
predict_fn(seed) -> (predictions, targets, logits) callback that runs
some held-out eval batch and returns the raw per-sample data, instead of
collapsing straight to a scalar the way evaluate() does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

PredictFn = Callable[[int], Tuple[List[int], List[int], List[np.ndarray]]]


@dataclass
class OutputCollapseReport:
    n_samples: int
    accuracy: float
    unique_prediction_fraction: float
    most_common_prediction_fraction: float
    prediction_entropy_bits: float
    max_entropy_bits: float  # log2(vocab_size), for normalizing
    mean_logit_std: float          # average WITHIN-sample logit spread
    cross_sample_logit_std: float  # average ACROSS-sample logit spread per class

    @property
    def normalized_entropy(self) -> float:
        """0.0 = fully collapsed (always predicts the same token), 1.0 =
        maximally diverse (predictions uniform over the vocab). No hard
        threshold -- below ~0.3 is a reasonable starting point to treat
        as a real collapse signal worth investigating, not a pass/fail
        line."""
        return (self.prediction_entropy_bits / self.max_entropy_bits
                if self.max_entropy_bits > 0 else 0.0)


def check_output_collapse(predict_fn: PredictFn, *, vocab_size: int,
                           n_batches: int = 1, seed: int = 0) -> OutputCollapseReport:
    """predict_fn(seed) -> (predictions, targets, logits) over some
    held-out eval set -- same shape/semantics as l1_sparsity_probe.py's
    own evaluate(), just returning the raw per-sample data instead of
    reducing straight to an accuracy scalar. n_batches>1 calls
    predict_fn with different seeds and pools everything together (use
    this if a single call's eval set is small enough that per-call
    sampling noise would dominate the diversity metrics).

    cross_sample_logit_std specifically distinguishes two different
    failure modes accuracy alone conflates: a model that's collapsed
    onto one confident-but-wrong ARGMAX every time (low
    unique_prediction_fraction) could still have logits that genuinely
    shift with input (real signal, wrong decision boundary) -- that's a
    very different bug than a model whose raw OUTPUT REPRESENTATION
    barely moves at all regardless of input (near-zero
    cross_sample_logit_std), which is collapse in a much deeper sense."""
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_logits: List[np.ndarray] = []
    for b in range(n_batches):
        preds, targets, logits = predict_fn(seed + b)
        all_preds.extend(preds)
        all_targets.extend(targets)
        all_logits.extend(logits)

    n = len(all_preds)
    if n == 0:
        raise ValueError("predict_fn returned zero samples")

    correct = sum(p == t for p, t in zip(all_preds, all_targets))
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
        n_samples=n, accuracy=accuracy,
        unique_prediction_fraction=unique_fraction,
        most_common_prediction_fraction=most_common_fraction,
        prediction_entropy_bits=entropy, max_entropy_bits=max_entropy,
        mean_logit_std=mean_logit_std, cross_sample_logit_std=cross_sample_logit_std,
    )
