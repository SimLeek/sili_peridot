"""See docs/research/eval_quantization.rst:module_overview."""

from __future__ import annotations

import torch

from .eval_pruning import EVAL_TEXTS, evaluate_next_token_prediction


def compare_pruned_vs_quantized(
    model,
    tokenizer,
    pruned_dense_state_dict: dict[str, torch.Tensor],
    quantized_dense_state_dict: dict[str, torch.Tensor],
    texts: list[str] = EVAL_TEXTS,
) -> dict:
    """See docs/research/eval_quantization.rst:compare_pruned_vs_quantized_partial_load."""
    original_state_dict = {k: model.state_dict()[k].clone() for k in pruned_dense_state_dict}
    try:
        model.load_state_dict(pruned_dense_state_dict)
        pruned_result = evaluate_next_token_prediction(model, tokenizer, texts)

        model.load_state_dict(quantized_dense_state_dict, strict=False)
        quantized_result = evaluate_next_token_prediction(model, tokenizer, texts)
    finally:
        model.load_state_dict(original_state_dict)

    return {
        "pruned_perplexity": pruned_result.perplexity,
        "quantized_perplexity": quantized_result.perplexity,
        "pruned_accuracy": pruned_result.accuracy,
        "quantized_accuracy": quantized_result.accuracy,
        "pruned_per_text_loss": pruned_result.per_text_loss,
        "quantized_per_text_loss": quantized_result.per_text_loss,
    }
