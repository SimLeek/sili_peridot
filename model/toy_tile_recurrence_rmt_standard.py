"""
model/toy_tile_recurrence_rmt_standard.py
────────────────────────────────────────────
A genuinely STANDARD/textbook torch implementation of Recurrent Memory
Transformer's mechanism -- NOT another exact port of this project's own
model. Built per direct instruction after BOTH the sili-based control
(task #230: fp4 acc=0.200, fp32 acc=0.100) AND the exact-fidelity torch
port (task #232: acc=0.433) landed well short of the near-1.0 a working
RMT should hit trivially on K=1 MQAR, with torch's own result clearly
"middle of the road" rather than confirming either "the port is fine"
or "the port is broken" -- direct guidance: check the standard-reference
road FIRST (before bit-diffing the two existing ports against each
other), since it answers "is 0.43 even in the right ballpark" before
spending time tracing a divergence between two implementations that
might BOTH be wrong.

Deliberately standard, not matched to this project's own custom
choices:
  - Standard multi-head-capable scaled dot-product attention
    (softmax(QK^T/sqrt(d))V), no Gaussian positional bias term -- that
    bias is this project's own invention, not part of RMT/transformers
    generally.
  - A learned absolute positional embedding (added to every token
    before attention) instead of the Gaussian bias -- the standard way
    transformers give attention positional information at all.
  - Standard LayerNorm (mean+variance normalization, learned affine),
    not this project's own RMSNorm.
  - Standard residual connections, NO hard clip anywhere (LayerNorm's
    own normalization is the standard way scale is controlled).
  - Standard end-to-end backprop: every parameter (including what this
    project calls "weights") trained by one shared torch.optim.Adam
    over the whole model -- no custom per-synapse RMSprop-with-
    importance rule, no L1-sparsity split-backward mechanism.

Kept the SAME as the rest of this investigation, deliberately, so this
stays a fair "is the architecture/task solvable at all" check rather
than changing everything at once:
  - Same task/harness (model/toy_recall_task.py's generate_mqar_sequence,
    same embed_table convention -- fixed random, never trained).
  - Same num_memory_slots concat-into-the-window mechanism (memory
    tokens genuinely concatenated with content tokens, not additively
    merged).
  - Same readout: mean-pool each column_neurons-wide block down to
    embed_width before the final vocab projection (this project's own
    "column" convention) -- not swapped for e.g. a CLS-token readout,
    to isolate the OPTIMIZER/ATTENTION/NORM/CLIP question specifically,
    not the readout question too.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class ToyTileRecurrenceRMTStandard(nn.Module):
    def __init__(self, vocab_size: int, embed_width: int, column_neurons: int,
                 num_tiles: int, num_memory_slots: int):
        super().__init__()
        self.embed_width = embed_width
        self.column_neurons = column_neurons
        self.state_width = embed_width * column_neurons
        self.num_tiles = num_tiles
        self.num_memory_slots = num_memory_slots
        self.total_slots = num_tiles + num_memory_slots
        sw = self.state_width

        self.input_proj = nn.Linear(embed_width, sw)
        self.q_proj = nn.Linear(sw, sw)
        self.k_proj = nn.Linear(sw, sw)
        self.v_proj = nn.Linear(sw, sw)
        self.o_proj = nn.Linear(sw, sw)
        self.lm_head = nn.Linear(embed_width, vocab_size)

        self.input_ln = nn.LayerNorm(sw)
        self.memory_ln = nn.LayerNorm(sw)
        self.state_ln = nn.LayerNorm(sw)

        # Standard learned absolute positional embedding -- replaces
        # the Gaussian positional bias (this project's own invention,
        # not part of standard RMT/transformer designs).
        self.pos_embed = nn.Parameter(torch.zeros(self.total_slots, sw))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x_window: torch.Tensor, memory_prev: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_mem, n_content = self.num_memory_slots, self.num_tiles

        x_wide = self.input_proj(x_window)
        x_normed = self.input_ln(x_wide)
        memory_normed = self.memory_ln(memory_prev)
        combined_normed = torch.cat([memory_normed, x_normed], dim=0)
        combined_normed = combined_normed + self.pos_embed

        q = self.q_proj(combined_normed)
        k = self.k_proj(combined_normed)
        v = self.v_proj(combined_normed)
        d = q.shape[-1]
        scores = (q @ k.transpose(-1, -2)) / (d ** 0.5)
        attn_w = torch.softmax(scores, dim=-1)
        attn = attn_w @ v
        attn = self.o_proj(attn)

        raw_combined = torch.cat([memory_prev, x_wide], dim=0)
        combined_new = raw_combined + attn
        combined_new = self.state_ln(combined_new)

        memory_new = combined_new[:n_mem]
        content_out = combined_new[n_mem:]
        pooled = content_out.reshape(n_content, self.embed_width, self.column_neurons).mean(dim=-1)
        logits = self.lm_head(pooled)
        return memory_new, logits
