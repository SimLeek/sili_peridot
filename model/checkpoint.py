"""
sili_peridot/model/checkpoint.py
──────────────────────────────────
Load MiniCPM5-1B-Base's checkpoint into a flat {name: torch.Tensor}
state dict, and verify its structure actually matches MiniCPM5Config
before any conversion code trusts it -- a shape mismatch should fail
loudly here, not surface later as a cryptic matmul error during folding.
"""
from __future__ import annotations

from typing import Dict

import torch
from sili.conversion.sparse_prune import load_state_dict as _load_state_dict

from .config import MiniCPM5Config


def load_minicpm5_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    """Load `path` (a directory of safetensors shards, or a single
    shard/file) into a flat {name: tensor} state dict. Thin wrapper over
    sili.conversion.sparse_prune.load_state_dict -- no MiniCPM5-specific
    logic here, see verify_checkpoint_matches_config for that."""
    return _load_state_dict(path)


def verify_checkpoint_matches_config(state_dict: Dict[str, torch.Tensor],
                                     cfg: MiniCPM5Config) -> None:
    """Assert the checkpoint's tensor names/shapes match `cfg`, raising
    on the first mismatch found."""

    def shape(name: str):
        assert name in state_dict, f"missing tensor: {name}"
        return tuple(state_dict[name].shape)

    assert shape("model.embed_tokens.weight") == (cfg.vocab_size, cfg.hidden_size)
    assert shape("lm_head.weight")            == (cfg.vocab_size, cfg.hidden_size)
    assert shape("model.norm.weight")         == (cfg.hidden_size,)

    if not cfg.tie_word_embeddings:
        assert (state_dict["lm_head.weight"].data_ptr()
                != state_dict["model.embed_tokens.weight"].data_ptr()), (
            "config says tie_word_embeddings=False but lm_head/embed_tokens "
            "share storage"
        )

    n_layers_found = len({
        int(name.split(".")[2]) for name in state_dict
        if name.startswith("model.layers.")
    })
    assert n_layers_found == cfg.num_hidden_layers, (
        f"found {n_layers_found} layers in checkpoint, config says "
        f"{cfg.num_hidden_layers}"
    )

    for i in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{i}."
        assert shape(prefix + "input_layernorm.weight")          == (cfg.hidden_size,)
        assert shape(prefix + "post_attention_layernorm.weight") == (cfg.hidden_size,)
        assert shape(prefix + "self_attn.q_proj.weight") == (cfg.q_proj_out,  cfg.attn_in)
        assert shape(prefix + "self_attn.k_proj.weight") == (cfg.kv_proj_out, cfg.attn_in)
        assert shape(prefix + "self_attn.v_proj.weight") == (cfg.kv_proj_out, cfg.attn_in)
        assert shape(prefix + "self_attn.o_proj.weight") == (cfg.attn_out,    cfg.o_proj_in)
        assert shape(prefix + "mlp.gate_proj.weight")    == (cfg.mlp_hidden,  cfg.mlp_in)
        assert shape(prefix + "mlp.up_proj.weight")      == (cfg.mlp_hidden,  cfg.mlp_in)
        assert shape(prefix + "mlp.down_proj.weight")    == (cfg.mlp_out,     cfg.mlp_hidden)

        # No attention bias, no qk-norm -- confirmed absent in the real
        # checkpoint; assert they stay absent so a future MiniCPM5
        # variant that adds one doesn't get silently ignored.
        for absent in ("self_attn.q_proj.bias", "self_attn.k_proj.bias",
                      "self_attn.v_proj.bias", "self_attn.o_proj.bias",
                      "self_attn.q_norm.weight", "self_attn.k_norm.weight"):
            assert prefix + absent not in state_dict, (
                f"unexpected tensor {prefix + absent} -- config assumes no "
                f"attention bias / no qk-norm, but this checkpoint has one"
            )
