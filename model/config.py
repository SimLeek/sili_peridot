"""
sili_peridot/model/config.py
─────────────────────────────
Explicit MiniCPM5-1B-Base hyperparameters, read directly from the
checkpoint's own config.json -- never re-derived from tensor shapes or
assumed from hidden_size/num_heads.

MiniCPM5-1B-Base is NOT hidden_size-square in its attention projections:
num_attention_heads * head_dim = 16 * 128 = 2048 != hidden_size = 1536.
Every projection's in/out dims below are explicit properties, not
computed from hidden_size and num_heads the way a "standard" square
LLaMA would allow -- treat that shortcut as wrong for this model
specifically, not a simplification to reach for later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class MiniCPM5Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    hidden_act: str = "silu"

    @classmethod
    def from_json(cls, path: str) -> MiniCPM5Config:
        with open(path) as f:
            raw = json.load(f)
        return cls(
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw["head_dim"],
            vocab_size=raw["vocab_size"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            tie_word_embeddings=raw["tie_word_embeddings"],
            hidden_act=raw.get("hidden_act", "silu"),
        )

    # ── Explicit per-projection shapes ──────────────────────────────────
    # Every downstream conversion step should read these instead of
    # deriving q/k/v/o shapes from hidden_size/num_heads itself -- see
    # module docstring for why that shortcut is wrong here.

    @property
    def q_proj_out(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_out(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def o_proj_in(self) -> int:
        return self.q_proj_out

    @property
    def attn_in(self) -> int:
        return self.hidden_size

    @property
    def attn_out(self) -> int:
        return self.hidden_size

    @property
    def mlp_in(self) -> int:
        return self.hidden_size

    @property
    def mlp_hidden(self) -> int:
        return self.intermediate_size

    @property
    def mlp_out(self) -> int:
        return self.hidden_size
