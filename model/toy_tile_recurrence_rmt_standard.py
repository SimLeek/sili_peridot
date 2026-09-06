"""
model/toy_tile_recurrence_rmt_standard.py

See docs/research/toy_tile_recurrence_rmt_standard.rst:
toy_tile_recurrence_rmt_standard.module_overview for why this file exists
(standard-torch RMT control, task #230/#232 follow-up),
toy_tile_recurrence_rmt_standard.standard_vs_custom_choices for what's
deliberately standard vs this project's own conventions, and
toy_tile_recurrence_rmt_standard.controlled_variables for what's kept
identical to the rest of the investigation.
"""

from __future__ import annotations

import torch
from torch import nn


class ToyTileRecurrenceRMTStandard(nn.Module):
    def __init__(self, vocab_size: int, embed_width: int, column_neurons: int, num_tiles: int, num_memory_slots: int):
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

        # See docs/research/toy_tile_recurrence_rmt_standard.rst:
        # toy_tile_recurrence_rmt_standard.standard_vs_custom_choices.
        self.pos_embed = nn.Parameter(torch.zeros(self.total_slots, sw))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x_window: torch.Tensor, memory_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        scores = (q @ k.transpose(-1, -2)) / (d**0.5)
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
