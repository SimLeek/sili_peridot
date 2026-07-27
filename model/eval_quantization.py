"""
sili_peridot/model/eval_quantization.py
─────────────────────────────────────────
Does B5's real FP4 quantization (model/quantize.py's real
FoldedLayer.from_descriptor, sili__new PR #10) degrade
MiniCPM5-1B-Base's next-token prediction quality by an unacceptable
amount, ON TOP OF B3's already-validated pruning? Same methodology as
model/eval_pruning.py's compare_dense_vs_pruned (B3b) -- load the real
HF model, evaluate with B3's pruned weights (the accepted baseline),
then with those SAME weights additionally FP4-quantized, and compare
next-token loss/perplexity/accuracy -- isolating quantization's own
effect from pruning's (already measured separately).
"""
from __future__ import annotations

from typing import Dict, List

import torch

from .eval_pruning import EVAL_TEXTS, evaluate_next_token_prediction


def compare_pruned_vs_quantized(
    model,
    tokenizer,
    pruned_dense_state_dict: Dict[str, torch.Tensor],
    quantized_dense_state_dict: Dict[str, torch.Tensor],
    texts: List[str] = EVAL_TEXTS,
) -> dict:
    """
    Evaluate `model` with pruned_dense_state_dict loaded (B3's already
    -validated baseline), then with quantized_dense_state_dict applied
    on top (partial -- only the suffixes model.quantize actually
    touches, loaded with strict=False), then restore the model's
    original weights so the caller isn't left with a mutated model.
    """
    original_state_dict = {k: model.state_dict()[k].clone() for k in pruned_dense_state_dict}
    try:
        model.load_state_dict(pruned_dense_state_dict)
        pruned_result = evaluate_next_token_prediction(model, tokenizer, texts)

        model.load_state_dict(quantized_dense_state_dict, strict=False)
        quantized_result = evaluate_next_token_prediction(model, tokenizer, texts)
    finally:
        model.load_state_dict(original_state_dict)

    return {
        "pruned_perplexity":       pruned_result.perplexity,
        "quantized_perplexity":    quantized_result.perplexity,
        "pruned_accuracy":         pruned_result.accuracy,
        "quantized_accuracy":      quantized_result.accuracy,
        "pruned_per_text_loss":    pruned_result.per_text_loss,
        "quantized_per_text_loss": quantized_result.per_text_loss,
    }
