from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from sili.tensor import Tensor, gaussian_attention, exp, reduce_sum, tensor_abs, gather, concat
from sili.sparse_rnn import DISLDOLayer

from .toy_recall_models import rmsnorm_tensor


class ToyTileRecurrenceRMT:
    """Faithful reference implementation of Recurrent Memory Transformer's
    actual mechanism (Bulatov et al., "Recurrent Memory Transformer",
    NeurIPS 2022, arXiv:2207.06881) -- built from the SAME sili__new
    primitives (disldo_cls layers, Tensor autograd, gaussian_attention,
    RMSNorm) as ToyTileRecurrenceRealFP4, NOT plain torch. This IS the
    control for the ongoing MQAR investigation (see conversation): same
    engine, same precision handling (FP4 via disldo_cls, or fp32 via a
    plain-float disldo_cls), so a pass/fail result on the exact same
    K=1 MQAR task/harness isolates the ARCHITECTURE question specifically
    -- if this also fails to learn, the problem is the task/embedding/
    harness, not ToyTileRecurrenceRealFP4's own recurrence wiring.
    Plain torch is a deliberately DEFERRED fallback, only worth building
    if THIS itself fails, to separate "the engine is broken" from "even
    a proven architecture fails here somehow" -- building torch first
    would confound engine differences with architecture differences and
    defeat the point of a control (direct instruction).

    Core mechanism, matching RMT exactly: `num_memory_slots` dedicated
    memory-token positions are CONCATENATED into the same attention
    window as the real content tile positions (never additively merged
    -- see ToyTileRecurrenceRealFP4's own "Known differences" docstring,
    items 1 and 4), so gaussian_attention processes memory and content
    together via ordinary self-attention, exactly like RMT's own memory
    tokens being literally re-inserted into the input/output sequence.
    The memory tokens' own output positions after this step become next
    step's memory, carried purely by being re-inserted into the window
    on the next call -- no separate gate/combine mechanism at all (RMT
    itself has none; the memory update is handled entirely by ordinary
    attention + residual, same as any other token position)."""

    def __init__(self, vocab_size: int, embed_width: int, column_neurons: int,
                 num_tiles: int, num_memory_slots: int, max_weights: int,
                 num_cpus: int = 2, rms_eps: float = 1e-6, disldo_cls=DISLDOLayer,
                 dense: bool = False, clip_range: float = 6.0,
                 l1_sparsity_coef: float = 0.0,
                 synapse_kwargs: Optional[dict] = None,
                 rng: Optional[np.random.Generator] = None):
        """num_memory_slots: RMT's own paper uses a small handful of
        memory tokens (their experiments: as few as 1-16 depending on
        task) -- default kept small here to match, not tuned against
        this specific task yet.

        Everything else mirrors ToyTileRecurrenceRealFP4's own
        conventions exactly (embed_width/column_neurons/state_width,
        disldo_cls/dense/l1_sparsity_coef, per-layer independent rng
        seeding) so a comparison between the two isolates the
        architecture question, not incidental convention differences."""
        self.embed_width = embed_width
        self.column_neurons = column_neurons
        self.state_width = embed_width * column_neurons
        self.num_tiles = num_tiles
        self.num_memory_slots = num_memory_slots
        self.total_slots = num_tiles + num_memory_slots
        self.rms_eps = rms_eps
        self.clip_range = clip_range
        self.l1_sparsity_coef = l1_sparsity_coef
        self.num_cpus = num_cpus
        # min_decay_frac/max_abs_delta/max_ci passthrough (see
        # sili.sparse_rnn.DISLDOLayer.forward's own docstring) -- lets a
        # caller toggle e.g. max_abs_delta=1e30 (effectively off) for a
        # real ablation on this actual model without a C++ rebuild or
        # touching every one of this step()'s own 6 disldo_cls.forward()
        # call sites by hand.
        self.synapse_kwargs = synapse_kwargs or {}

        state_width = self.state_width
        if rng is None:
            rng = np.random.default_rng()
        n_layer_seeds = 6  # input_proj, q, k, v, o_proj, lm_head
        layer_seeds = iter(int(s) for s in rng.integers(0, 2**31 - 1, size=n_layer_seeds))
        dense_kwargs = {"dense": True} if dense else {}

        self.input_proj = disldo_cls(embed_width, state_width, max_weights, num_cpus,
                                     rng=np.random.default_rng(next(layer_seeds)), **dense_kwargs)
        self.q_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                 rng=np.random.default_rng(next(layer_seeds)), **dense_kwargs)
        self.k_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                 rng=np.random.default_rng(next(layer_seeds)), **dense_kwargs)
        self.v_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                 rng=np.random.default_rng(next(layer_seeds)), **dense_kwargs)
        self.o_proj = disldo_cls(state_width, state_width, max_weights, num_cpus,
                                 rng=np.random.default_rng(next(layer_seeds)), **dense_kwargs)
        self.lm_head = disldo_cls(embed_width, vocab_size, max_weights, num_cpus,
                                  rng=np.random.default_rng(next(layer_seeds)), **dense_kwargs)

        # Separate RMSNorm gains for memory vs content tokens -- unlike
        # ToyTileRecurrenceRealFP4 (which reuses one input_ln for both
        # sides of an additive combine), memory and content are now
        # genuinely different KINDS of token (never summed together),
        # so giving them their own learned scale is the more faithful
        # choice, matching how RMT's own memory tokens get their own
        # learned embeddings.
        self.input_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.memory_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.state_ln = Tensor(np.ones(state_width, dtype=np.float32))
        self.centers = Tensor(np.array([i + 0.5 for i in range(self.total_slots)], dtype=np.float32))
        self.log_sigmas = Tensor(np.zeros(self.total_slots, dtype=np.float32))

    def parameters_for_optimizer(self) -> List[Tensor]:
        return [self.input_ln, self.memory_ln, self.state_ln, self.centers, self.log_sigmas]

    def _l1_sparsity_split(self, layer, input_t: Tensor, lr: float, coef: float) -> Tensor:
        """Exact port of ToyTileRecurrenceRealFP4's own helper -- see its
        l1_sparsity_coef docstring for the full split-backward rationale."""
        out_aux = layer.forward(input_t, lr, damp_by_importance=False, **self.synapse_kwargs)
        n = float(np.asarray(out_aux.data).size)
        return reduce_sum(tensor_abs(out_aux)) * (coef / n)

    def step(self, x_window: np.ndarray, memory_prev: np.ndarray,
             learning_rate: float) -> Tuple[np.ndarray, Tensor, Optional[Tensor]]:
        """x_window: [num_tiles, embed_width] -- same convention as
        ToyTileRecurrenceRealFP4 (a real per-tile embedding, zeros for
        "nothing here yet"). memory_prev: [num_memory_slots, state_width]
        -- genuinely dedicated memory, DETACHED (no BPTT across steps,
        matching this whole project's convention), unlike
        ToyTileRecurrenceRealFP4's ambiguous per-window-slot rolling
        state (see its own docstring item 4). Returns (memory_new numpy
        [num_memory_slots, state_width], logits Tensor [num_tiles,
        vocab_size], aux_loss)."""
        sw = self.state_width
        n_mem, n_content = self.num_memory_slots, self.num_tiles

        x_window_t = Tensor(x_window.astype(np.float32))
        x_wide = self.input_proj.forward(x_window_t, learning_rate, **self.synapse_kwargs)  # [n_content, sw]
        x_normed = rmsnorm_tensor(x_wide, self.input_ln, self.rms_eps)

        memory_prev_t = Tensor(memory_prev.astype(np.float32))
        memory_normed = rmsnorm_tensor(memory_prev_t, self.memory_ln, self.rms_eps)

        combined_normed = concat([memory_normed, x_normed], axis=0)          # [total_slots, sw]

        q = self.q_proj.forward(combined_normed, learning_rate, **self.synapse_kwargs)
        k = self.k_proj.forward(combined_normed, learning_rate, **self.synapse_kwargs)
        v = self.v_proj.forward(combined_normed, learning_rate, **self.synapse_kwargs)
        sigmas = exp(self.log_sigmas)
        attn = gaussian_attention(q, k, v, self.centers, sigmas,
                                  num_cpus=self.num_cpus, causal=False)
        attn = self.o_proj.forward(attn, learning_rate, **self.synapse_kwargs)
        attn.data = np.clip(attn.data, -self.clip_range, self.clip_range)

        aux_loss = None
        if self.l1_sparsity_coef > 0.0:
            l1_terms = [
                self._l1_sparsity_split(self.input_proj, x_window_t, learning_rate, self.l1_sparsity_coef),
                self._l1_sparsity_split(self.q_proj, combined_normed, learning_rate, self.l1_sparsity_coef),
                self._l1_sparsity_split(self.k_proj, combined_normed, learning_rate, self.l1_sparsity_coef),
                self._l1_sparsity_split(self.v_proj, combined_normed, learning_rate, self.l1_sparsity_coef),
                self._l1_sparsity_split(self.o_proj, gaussian_attention(
                    q, k, v, self.centers, sigmas, num_cpus=self.num_cpus, causal=False),
                    learning_rate, self.l1_sparsity_coef),
            ]
            for term in l1_terms:
                aux_loss = term if aux_loss is None else aux_loss + term

        # Residual against the RAW (pre-RMSNorm) memory/content values,
        # matching ToyTileRecurrenceRealFP4's own convention (residual
        # from raw M_prev, not the normed qkv_source).
        raw_combined = concat([memory_prev_t, x_wide], axis=0)
        combined_new = raw_combined + attn
        combined_new = rmsnorm_tensor(combined_new, self.state_ln, self.rms_eps)
        combined_new.data = np.clip(combined_new.data, -self.clip_range, self.clip_range)

        # Split back into memory (next step's carry, plain numpy -- no
        # BPTT across steps, matching this project's convention) and
        # content (stays differentiable, feeds the readout below) via
        # gather+reshape (no slicing op exists on Tensor -- see
        # cross_entropy_sum's own docstring for why gather is this
        # codebase's established workaround).
        memory_new = combined_new.data[:n_mem].copy()
        content_flat_idx = [(n_mem + t) * sw + c for t in range(n_content) for c in range(sw)]
        content_out = gather(combined_new, content_flat_idx).reshape((n_content, sw))

        pooled = content_out.reshape((n_content, self.embed_width, self.column_neurons))
        pooled = reduce_sum(pooled, axis=-1) * (1.0 / self.column_neurons)
        if self.l1_sparsity_coef > 0.0:
            lm_l1 = self._l1_sparsity_split(self.lm_head, pooled, learning_rate, self.l1_sparsity_coef)
            aux_loss = lm_l1 if aux_loss is None else aux_loss + lm_l1
        logits = self.lm_head.forward(pooled, learning_rate, **self.synapse_kwargs)

        return memory_new, logits, aux_loss
