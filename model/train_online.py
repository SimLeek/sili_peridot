"""
sili_peridot/model/train_online.py
────────────────────────────────────
Minimal online (single-sample, no batching) training probe: does any
training after pruning/quantization recover next-token accuracy? Trains
entirely through sili's own machinery (sili.tensor's Tensor autograd +
_cpu.SparseLinearLayer's inline-learning-rate forward_dense/
backward_dense/synap_step, the same primitives sili__new's
FoldedLayer/FoldedColumnLayer already use for real, working training --
see sili__new's tests/integration/test_folded_column_layer.py) -- no
torch, no dense weight duplication, no batch dimension anywhere in the
trainable path.

Scope, deliberately reduced from "train the whole last fold step": only
the MLP (gate_proj/up_proj/down_proj) of the LAST fold step (step
cfg.num_hidden_layers - 1) is trained. Attention (q/k/v/o) stays frozen/
inference, computed exactly as sili_block.apply_fold_step already does
it -- training it online would need a genuine incremental causal
KV-cache (each new token's attention depends on all previous tokens'
K/V, so "online, one token at a time" for attention means maintaining a
running cache, not just calling forward_dense once per token the way
the position-independent MLP allows). That's real, buildable work, but
not needed to answer the first question ("does training recover any
accuracy at all") -- left as a follow-up if this probe shows signal.
Frozen prefix (fold steps 0..-2), RMSNorm, final norm, and lm_head are
all unchanged inference, exactly as sili_model.py/sili_block.py compute
them today.

Loss (softmax cross-entropy) and the frozen final-RMSNorm/lm_head
backward are hand-derived closed forms, computed OUTSIDE the Tensor
graph -- lm_head is large ([vocab, hidden], sparse) and isn't being
trained here, so there's no reason to materialize it dense or wire it
into the autograd graph; the one gradient needed from it
(d_loss/d_hidden) is a single sparse-vector matmul either way.

No synaptogenesis (structural growth) or pruning in this probe, on
purpose, not just for speed: both need real per-synapse importance
signal to be a meaningful structural decision (which existing synapses
are actually weak enough to remove, which candidate positions are
actually worth adding), and this probe hasn't run long enough for that
signal to accumulate yet -- growing/pruning blind, before there's
anything to base the decision on, isn't a step worth taking regardless
of its cost. (It also happens to be extremely expensive on these
specific layers -- measured directly: synap_step alone costs 25-188ms/
call on the real checkpoint's MLP layers, scaling with each row's
ABSOLUTE existing content size, not its density; gate_proj/up_proj rows
average ~3,500 real entries and cost 170-188ms, dominating the whole
per-token loop. That's a secondary, real finding worth fixing later --
see sili__new/prototypes/synaptogenesis_subrow_interleaving/README.md
for the idea already sketched for it -- but not why it's skipped here.)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse

from sili.energy import EnergyDynamics
from sili.tensor import Tensor, banded_attention, mul, silu as t_silu, _acc

from .config import MiniCPM5Config
from .eval_pruning import EvalResult
from .sili_block import apply_fold_step, apply_rotary, rmsnorm, rope_cos_sin

_MLP_SUFFIXES = (".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight")


# ── Frozen prefix: fold steps 0..-2, unchanged inference ─────────────────────

def _run_frozen_prefix(
    x: np.ndarray, sili_model: dict, cfg: MiniCPM5Config,
    half_bandwidth: int, num_cpus: int,
) -> np.ndarray:
    """state=0; for step in 0..n-2: out=block(x+state); state+=out --
    same recurrence as sili_block.run_folded_recurrence, stopped one
    step short of the last (that step is handled by the trainable path
    below). Returns `state` after n-1 steps; the caller adds `x` to get
    the last step's real block input, matching run_folded_recurrence's
    own convention."""
    cos, sin = rope_cos_sin(x.shape[0], cfg.head_dim, cfg.rope_theta)
    step_layers = sili_model["step_layers"]
    input_ln    = sili_model["input_ln"]
    post_ln     = sili_model["post_ln"]
    state = np.zeros_like(x)
    for i in range(cfg.num_hidden_layers - 1):
        out = apply_fold_step(
            x + state, step_layers[i], input_ln[i], post_ln[i],
            cfg, cos, sin, half_bandwidth, num_cpus, activation_density=None)
        state = state + out
    return state


def _last_step_attention(
    sili_model: dict, cfg: MiniCPM5Config, x_full: np.ndarray, state: np.ndarray,
    half_bandwidth: int, num_cpus: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Frozen attention half of the LAST fold step (q/k/v/o, RoPE,
    banded causal GQA attention) -- duplicates apply_fold_step's
    attention logic directly (activation_density is always None here,
    so this is exactly apply_fold_step's own code path, just stopped
    before the MLP so normed2 can be reused per-token by the trainable
    path instead of once for the whole sequence).

    Returns (normed2, x_after_attn), both [T, hidden]. x_after_attn =
    block_input + attn_out; the caller derives whatever it needs
    relative to x_full/state from that."""
    last = cfg.num_hidden_layers - 1
    layers   = sili_model["step_layers"][last]
    input_ln = sili_model["input_ln"][last]
    post_ln  = sili_model["post_ln"][last]

    T = x_full.shape[0]
    cos, sin = rope_cos_sin(T, cfg.head_dim, cfg.rope_theta)
    n_heads, n_kv_heads, head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    groups = n_heads // n_kv_heads

    block_input = x_full + state
    normed = rmsnorm(block_input, input_ln, cfg.rms_norm_eps)

    q = layers[".self_attn.q_proj.weight"].forward_dense(normed, learning_rate=0.0)
    k = layers[".self_attn.k_proj.weight"].forward_dense(normed, learning_rate=0.0)
    v = layers[".self_attn.v_proj.weight"].forward_dense(normed, learning_rate=0.0)

    q = q.reshape(T, n_heads, head_dim)
    k = k.reshape(T, n_kv_heads, head_dim)
    v = v.reshape(T, n_kv_heads, head_dim)

    attn_out = np.empty((T, n_heads, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_h = h // groups
        qh = Tensor(apply_rotary(q[:, h, :], cos, sin))
        kh = Tensor(apply_rotary(k[:, kv_h, :], cos, sin))
        vh = Tensor(np.ascontiguousarray(v[:, kv_h, :]))
        out_h = banded_attention(qh, kh, vh, half_bandwidth=half_bandwidth,
                                 num_cpus=num_cpus, causal=True)
        attn_out[:, h, :] = out_h.data

    attn_out = attn_out.reshape(T, n_heads * head_dim)
    attn_out = layers[".self_attn.o_proj.weight"].forward_dense(attn_out, learning_rate=0.0)

    x_after_attn = block_input + attn_out
    normed2 = rmsnorm(x_after_attn, post_ln, cfg.rms_norm_eps)
    return normed2, x_after_attn


# ── Frozen-prefix memoization cache ───────────────────────────────────────────
#
# Steps 0..-2 plus the last step's attention are ALL frozen (only the
# last step's MLP is trained) -- their output for a given text is
# invariant for the whole duration of one training run. Profiling
# showed this recomputation (not attention itself, which is cheap) was
# the dominant cost: eval_checkpoint calls re-evaluate the SAME 5 texts
# repeatedly, and the training loop revisits the same ~50-sentence
# corpus constantly once it's been through once. This is NOT a true
# incremental (token-by-token, growing-sequence) KV-cache -- it's a
# simple memoization keyed by whole-text token ids, valid only because
# nothing it depends on ever changes during a run. Doesn't help once
# training moves to a large, non-repeating corpus -- see
# sili_peridot/todolist.md Phase B10 for the real incremental-cache
# scope that would need at that point.

_FrozenCache = Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def _get_frozen(
    cache: _FrozenCache, ids: np.ndarray, sili_model: dict, cfg: MiniCPM5Config,
    half_bandwidth: int, num_cpus: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (x_full, state, normed2, x_after_attn) for this token-id
    sequence, computed once and reused on every later call with the
    same ids (see cache docstring above for why that's valid here)."""
    key = tuple(int(i) for i in ids)
    cached = cache.get(key)
    if cached is not None:
        return cached
    embed_tokens = sili_model["embed_tokens"]
    x_full = embed_tokens[ids]
    if scipy.sparse.issparse(x_full):
        x_full = x_full.toarray()
    state = _run_frozen_prefix(x_full, sili_model, cfg, half_bandwidth, num_cpus)
    normed2, x_after_attn = _last_step_attention(
        sili_model, cfg, x_full, state, half_bandwidth, num_cpus)
    result = (x_full, state, normed2, x_after_attn)
    cache[key] = result
    return result


# ── Trainable last-step MLP: single-token Tensor-graph forward ──────────────

def _trainable_forward(layer, x: Tensor, learning_rate: float) -> Tensor:
    """Single-sample (1-D, no batch) SparseLinearLayer forward wrapped
    into sili.tensor's autograd graph -- mirrors sili__new's
    FoldedLayer.forward's proven per-layer wrapping pattern.
    forward_dense/backward_dense always return [1, cols] even for a
    bare 1-D input (see sili__new's DISLDOLayer.forward, same fix);
    squeeze back to 1-D here for the same reason."""
    x_np   = np.asarray(x.data, dtype=np.float32)
    out_np = layer.forward_dense(x_np, learning_rate).squeeze(0)
    out    = Tensor(out_np, (x,), "trainable_forward", x.backend)

    def _bwd():
        if out.grad is not None:
            dy = np.asarray(out.grad, dtype=np.float32)
            dx = layer.backward_dense(dy, learning_rate, lr_per_row_nnz=True).squeeze(0)
            _acc(x, dx)

    out._backward = _bwd
    return out


def _mlp_forward_online(
    mlp_layers: Dict[str, object], normed2_t: np.ndarray, learning_rate: float,
    energy: Optional[EnergyDynamics],
) -> Tuple[Tensor, Optional[Tensor]]:
    """One token's MLP: down_proj(silu(gate_proj(x)) * up_proj(x)).
    normed2_t: [hidden] -- already-normed input for THIS token (frozen
    RMSNorm/attention, computed by the caller via _last_step_attention).
    Returns (mlp_out: Tensor[hidden], aux_loss: Tensor scalar or None
    if energy is disabled)."""
    x = Tensor(np.asarray(normed2_t, dtype=np.float32))
    gate = _trainable_forward(mlp_layers[".mlp.gate_proj.weight"], x, learning_rate)
    up   = _trainable_forward(mlp_layers[".mlp.up_proj.weight"],   x, learning_rate)
    h = mul(t_silu(gate), up)
    aux_loss = None
    if energy is not None:
        h, aux_loss, _ = energy(h)
    mlp_out = _trainable_forward(mlp_layers[".mlp.down_proj.weight"], h, learning_rate)
    return mlp_out, aux_loss


def _mlp_forward_inference(mlp_layers: Dict[str, object], normed2_t: np.ndarray) -> np.ndarray:
    """Pure numpy, learning_rate=0, no Tensor graph -- for evaluation,
    where nothing will ever call .backward() on the result, so building
    one is pure overhead. Energy gating is deliberately NOT applied
    here even when the training run uses it: EnergyDynamics is a
    training-time regularization/exploration mechanism (forced firing,
    exploration noise) with its own persistent, mutating state -- an
    eval call to it would perturb that state (contaminating the NEXT
    real training step) purely to measure the model, and there's no
    clean "peek without mutating" option on it. Same convention as
    dropout/noise being disabled at eval time elsewhere in ML."""
    x = np.asarray(normed2_t, dtype=np.float32)
    gate = mlp_layers[".mlp.gate_proj.weight"].forward_dense(x, 0.0).squeeze(0)
    up   = mlp_layers[".mlp.up_proj.weight"].forward_dense(x, 0.0).squeeze(0)
    h = (gate / (1.0 + np.exp(-gate))) * up  # silu(gate) * up
    return mlp_layers[".mlp.down_proj.weight"].forward_dense(h, 0.0).squeeze(0)


# ── Frozen post-processing: RMSNorm + lm_head, closed-form loss/grad ────────

def _rmsnorm_backward(x: np.ndarray, weight: np.ndarray, eps: float, d_out: np.ndarray) -> np.ndarray:
    """d_loss/d_x for rmsnorm's y = x * rsqrt(mean(x^2)+eps) * weight,
    standard closed form. x/weight/d_out: [hidden]."""
    n  = x.shape[-1]
    ms = np.mean(x.astype(np.float32) ** 2)
    r  = 1.0 / np.sqrt(ms + eps)
    wx_dy = np.sum(weight * x * d_out)
    return (r * weight * d_out - (r ** 3 * x / n) * wx_dy).astype(np.float32)


def _softmax_cross_entropy(logits: np.ndarray, target: int) -> Tuple[float, np.ndarray]:
    """Standard numerically-stable softmax CE, closed-form gradient
    (probs - onehot(target)) -- same definition as
    sili_model._cross_entropy_and_accuracy, single-token here."""
    m = float(np.max(logits))
    shifted = logits - m
    exp_shifted = np.exp(shifted)
    sum_exp = float(np.sum(exp_shifted))
    probs = exp_shifted / sum_exp
    loss = float(np.log(sum_exp) - shifted[target])
    d_logits = probs.copy()
    d_logits[target] -= 1.0
    return loss, d_logits.astype(np.float32)


def _lm_head_matvec(lm_head, hidden: np.ndarray) -> np.ndarray:
    if scipy.sparse.issparse(lm_head):
        return np.asarray(lm_head @ hidden, dtype=np.float32).ravel()
    return (lm_head @ hidden).astype(np.float32)


def _lm_head_backward(lm_head, d_logits: np.ndarray) -> np.ndarray:
    """d_loss/d_hidden = d_logits @ lm_head -- sparse-vector-friendly,
    lm_head (large, [vocab, hidden]) is never densified since it isn't
    being trained here."""
    if scipy.sparse.issparse(lm_head):
        return np.asarray(d_logits @ lm_head, dtype=np.float32).ravel()
    return (d_logits @ lm_head).astype(np.float32)


def _evaluate_cached(
    sili_model: dict, cfg: MiniCPM5Config, tokenizer, texts: List[str], cache: _FrozenCache,
    half_bandwidth: int, num_cpus: int,
) -> EvalResult:
    """sili_model.evaluate_next_token_prediction_sili's counterpart for
    this training probe -- same teacher-forced loss/top-1-accuracy
    definition and EvalResult shape (directly comparable), but routes
    the frozen prefix through the memoization cache and the trainable
    MLP through the pure-inference path instead of recomputing
    everything from sili_model.py's general (uncached) path."""
    last = cfg.num_hidden_layers - 1
    mlp_layers = {s: sili_model["step_layers"][last][s] for s in _MLP_SUFFIXES}
    per_text_loss, per_text_acc = [], []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0].numpy()
        if len(ids) < 2:
            continue
        x_full, state, normed2, x_after_attn = _get_frozen(
            cache, ids, sili_model, cfg, half_bandwidth, num_cpus)
        state_plus_attn = x_after_attn - x_full
        losses = []
        n_correct = 0
        for t in range(len(ids) - 1):
            target = int(ids[t + 1])
            mlp_out = _mlp_forward_inference(mlp_layers, normed2[t])
            final_state_t = state_plus_attn[t] + mlp_out
            normed_final = rmsnorm(
                final_state_t[None, :], sili_model["final_norm"], cfg.rms_norm_eps)[0]
            logits = _lm_head_matvec(sili_model["lm_head"], normed_final)
            loss, _ = _softmax_cross_entropy(logits, target)
            losses.append(loss)
            n_correct += int(np.argmax(logits) == target)
        per_text_loss.append(float(np.mean(losses)))
        per_text_acc.append(n_correct / len(losses))
    return EvalResult(per_text_loss=per_text_loss, per_text_accuracy=per_text_acc)


# ── Training report ───────────────────────────────────────────────────────────

@dataclass
class OnlineTrainingReport:
    step_losses: List[float] = field(default_factory=list)
    eval_checkpoints: List[dict] = field(default_factory=list)  # {elapsed_s, accuracy, perplexity}
    wall_clock_s: float = 0.0


# ── Main online training loop ─────────────────────────────────────────────────

def train_last_step_mlp_online(
    sili_model: dict,
    cfg: MiniCPM5Config,
    tokenizer,
    train_texts: List[str],
    eval_texts: List[str],
    half_bandwidth: int,
    num_cpus: int = 1,
    learning_rate: float = 0.01,
    use_energy: bool = False,
    energy_kwargs: Optional[dict] = None,
    wall_clock_budget_s: float = 900.0,
    eval_every_s: float = 120.0,
    on_eval_checkpoint: Optional[Callable[[OnlineTrainingReport], None]] = None,
) -> OnlineTrainingReport:
    """
    Online (single-token, no batching) training of the last fold step's
    MLP (gate/up/down) -- see module docstring for full scope. Runs
    until `wall_clock_budget_s` elapses (looping back over train_texts
    if exhausted first), evaluating on `eval_texts` via _evaluate_cached
    roughly every `eval_every_s` so a real accuracy-vs-time curve is
    visible, not just one before/after number. Mutates sili_model's
    last-step MLP layers in place; the SAME sili_model dict can be
    re-evaluated afterward with no rebuild.

    Frozen-prefix results (steps 0..-2 + last-step attention) are
    memoized per text for the duration of this call -- see the
    _FrozenCache docstring for why that's valid (only the last step's
    MLP changes here) and its limits (a real, non-repeating training
    corpus wouldn't benefit the same way). The cache lives for exactly
    ONE call to this function -- a caller wanting incremental progress
    over a long run should pass on_eval_checkpoint (called with the
    in-progress `report` each time an eval checkpoint is recorded, so
    results can be persisted without needing to chunk this call into
    several smaller ones, which would reset the cache each time and
    defeat its purpose.

    use_energy=True gates the MLP's silu(gate)*up activation through
    EnergyDynamics (default low-drive kwargs, see energy_kwargs to
    override) -- its aux_loss is backprop'd as a SEPARATE call after
    the task-loss backward, not fused into one gradient (documented
    simplification, see the design plan this implements). Eval always
    runs without energy gating regardless of this setting -- see
    _mlp_forward_inference.
    """
    last = cfg.num_hidden_layers - 1
    layers = sili_model["step_layers"][last]
    mlp_layers = {s: layers[s] for s in _MLP_SUFFIXES}

    energy = None
    if use_energy:
        kwargs = dict(drive=0.02, activation_cost=0.05, precision=0.02,
                      density=0.05, p=0.1, exploration=0.001)
        kwargs.update(energy_kwargs or {})
        energy = EnergyDynamics(**kwargs)

    frozen_cache: _FrozenCache = {}
    report = OnlineTrainingReport()
    t_start = time.perf_counter()
    last_eval = t_start

    def _record_eval():
        result = _evaluate_cached(
            sili_model, cfg, tokenizer, eval_texts, frozen_cache, half_bandwidth, num_cpus)
        report.eval_checkpoints.append({
            "elapsed_s": time.perf_counter() - t_start,
            "accuracy": result.accuracy,
            "perplexity": result.perplexity,
        })
        if on_eval_checkpoint is not None:
            on_eval_checkpoint(report)

    _record_eval()  # baseline, before any training

    text_i = 0
    while time.perf_counter() - t_start < wall_clock_budget_s:
        text = train_texts[text_i % len(train_texts)]
        text_i += 1
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0].numpy()
        if len(ids) < 2:
            continue

        x_full, state, normed2, x_after_attn = _get_frozen(
            frozen_cache, ids, sili_model, cfg, half_bandwidth, num_cpus)
        state_plus_attn = x_after_attn - x_full  # = state + attn_out, see _mlp_forward_online usage below

        for t in range(len(ids) - 1):
            target = int(ids[t + 1])
            mlp_out, aux_loss = _mlp_forward_online(
                mlp_layers, normed2[t], learning_rate, energy)

            final_state_t = state_plus_attn[t] + mlp_out.data
            normed_final = rmsnorm(
                final_state_t[None, :], sili_model["final_norm"], cfg.rms_norm_eps)[0]
            logits = _lm_head_matvec(sili_model["lm_head"], normed_final)
            loss, d_logits = _softmax_cross_entropy(logits, target)
            report.step_losses.append(loss)

            d_normed_final = _lm_head_backward(sili_model["lm_head"], d_logits)
            d_final_state = _rmsnorm_backward(
                final_state_t, sili_model["final_norm"], cfg.rms_norm_eps, d_normed_final)

            # d(final_state)/d(mlp_out) = 1 -- state_plus_attn is a
            # frozen constant w.r.t. this step's trainable MLP.
            mlp_out.grad = d_final_state
            mlp_out.backward()
            if aux_loss is not None:
                aux_loss.backward()

            if time.perf_counter() - t_start >= wall_clock_budget_s:
                break

        now = time.perf_counter()
        if now - last_eval >= eval_every_s:
            _record_eval()
            last_eval = now

    _record_eval()  # final
    report.wall_clock_s = time.perf_counter() - t_start
    return report
