"""
scripts/train_mqar_rmt_optimizer_ablation.py
──────────────────────────────────────────────
Follow-up to train_mqar_rmt_ablation.py's LR sweep (task #235): that
sweep found DISLDO's own custom optimizer has a narrow stable-LR
window (0.001..0.03 improving, 0.1 collapsing to noise) and never
reaches Adam's clean acc=1.0000 even at its best tested lr (0.03,
acc=0.700). Direct question raised: is the narrow window caused by
plain-RMSprop-vs-Adam's-momentum, or by one of the "stability"
additions layered on top of plain RMSprop?

DISLDOTorchLinear's real update (delta_csr_types.hpp's
BoundedRMSpropSynapsePolicy, ported exactly) has FOUR properties that
differ from textbook Adam, each now an independent toggle on
DISLDOTorchLinear (model/toy_tile_recurrence_rmt_torch.py) so they can
be ablated in torch -- fast -- before touching the real C++ policy:
  - bias_correct_ci: production has NO bias correction on the ci EMA,
    unlike RMSpropScalePolicy's OWN _scale_update (value_scale/
    output_scale), which DOES bias-correct. With beta2=0.999, ci
    starts at 0 and stays under-estimated for ~O(1/(1-beta2))=1000
    steps, inflating -g/(sqrt(ci)+eps) early -- a real candidate for
    both the narrow window and the lr=0.1 collapse.
  - use_momentum: production has no first-moment/momentum term at all
    (plain RMSprop on ci, update = -g/denom) -- Adam's own -m_hat/denom
    smooths the raw per-step gradient before dividing.
  - include_contrib_in_ci: production folds contrib^2 (per-synapse
    forward-activation contribution) into the SAME denominator as g^2
    -- not present in Adam's v at all.
  - clip_raw_delta: production hard-clips the raw update to
    +-max_abs_delta=2.0 BEFORE scaling by eff_lr -- Adam has no
    analogous clip.

Each OPT_CONFIGS entry changes ONE of these from the production
default, all still under use_custom_optimizer=True (never Adam) so
the row_degree-normalized eff_lr convention stays intact -- this is a
sub-ablation of the DISLDO optimizer itself, not a repeat of the
optimizer-vs-Adam swap already done. Everything else matches
baseline_a exactly (use_hard_clip/use_gaussian_bias/use_rmsnorm=True,
l1_sparsity_coef=0.05).

Run: python3 scripts/train_mqar_rmt_optimizer_ablation.py <opt_config_name> [lr] [train_steps] [seed]
  opt_config_name: production | bias_correct_ci | add_momentum
                  | drop_contrib | drop_raw_clip | bias_correct_and_momentum
                  | all_fixes
"""
from __future__ import annotations

import sys
import time
import json
from typing import Optional

import numpy as np
import torch

torch.set_num_threads(1)  # see train_mqar_rmt_ablation.py -- avoids N x M thread oversubscription

sys.path.insert(0, ".")

from model.toy_recall_task import generate_mqar_sequence
from model.toy_tile_recurrence_rmt_ablation import ToyTileRecurrenceRMTAblation, clip_grad_norm_
from scripts.train_tile_curriculum import _build_tile_window
from scripts.train_mqar_rmt_ablation import (
    lr_schedule, predicted_token, seq_len_for_k, _build_targets,
    EMBED_WIDTH, COLUMN_NEURONS, NUM_MEMORY_SLOTS, VOCAB, WARMUP_STEPS, MAX_GRAD_NORM, EVAL_SEQUENCES,
)

DEFAULT_LR = 0.03  # best single value found by the prior LR sweep (task #235)

OPT_CONFIGS = {
    "production": dict(),
    "bias_correct_ci": dict(bias_correct_ci=True),
    "add_momentum": dict(use_momentum=True),
    "drop_contrib": dict(include_contrib_in_ci=False),
    "drop_raw_clip": dict(clip_raw_delta=False),
    "bias_correct_and_momentum": dict(bias_correct_ci=True, use_momentum=True),
    "all_fixes": dict(bias_correct_ci=True, use_momentum=True,
                      include_contrib_in_ci=False, clip_raw_delta=False),
    # Raw-space max_abs_delta sweep (production=2.0) -- sili__new's own
    # sweep_synapse_policy_min_decay_frac.cpp found safety vs max_abs_delta
    # is NOT monotonic on ITS task (8x8 permutation regression, lr=0.05):
    # flat-safe to ~4.0 raw, degrading-but-safe to ~10, an unsafe resonance
    # pocket at 12, safe again 16-18, unsafe again ~24. Those exact pockets
    # are task/lr-specific and don't transfer here -- this sweep maps our
    # OWN task's safe range rather than assuming theirs applies.
    "clip_4": dict(max_abs_delta=4.0),
    "clip_8": dict(max_abs_delta=8.0),
    "clip_16": dict(max_abs_delta=16.0),
}

MODEL_CFG = dict(use_custom_optimizer=True, use_hard_clip=True,
                 use_gaussian_bias=True, use_rmsnorm=True, l1_sparsity_coef=0.05)


def train_and_eval(opt_config_name: str, num_kv_pairs: int, seed: int, train_steps: int,
                   peak_lr: float, log_every: int = 500, log_fn=None) -> dict:
    opt_kwargs = OPT_CONFIGS[opt_config_name]
    seq_len = seq_len_for_k(num_kv_pairs)
    num_tiles = seq_len
    state_width = EMBED_WIDTH * COLUMN_NEURONS

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    model_rng = np.random.default_rng(seed)

    model = ToyTileRecurrenceRMTAblation(
        VOCAB, EMBED_WIDTH, COLUMN_NEURONS, num_tiles, NUM_MEMORY_SLOTS,
        rng=model_rng, optimizer_kwargs=opt_kwargs, **MODEL_CFG)
    # RMSNorm weights + Gaussian-bias centers/log_sigmas are ALWAYS Adam-
    # trained (model.parameters_for_optimizer()), even when
    # use_custom_optimizer=True -- only the main DISLDOTorchLinear layers
    # skip Adam. Missing this Adam/opt.step() call (an earlier version of
    # this script did) silently freezes those params at init and produces
    # a completely different, wrong training dynamic -- matches
    # train_mqar_rmt_ablation.py's own opt handling exactly.
    adam_params = model.parameters_for_optimizer()
    opt = torch.optim.Adam(adam_params) if adam_params else None
    embed_table = rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3

    def _quick_eval(n_sequences: int) -> float:
        correct, total = 0, 0
        with torch.no_grad():
            for _ in range(n_sequences):
                eval_tokens, eval_pairs = generate_mqar_sequence(rng, VOCAB, seq_len, num_kv_pairs)
                eval_by_pos = dict(eval_pairs)
                memory_eval = np.zeros((NUM_MEMORY_SLOTS, state_width), dtype=np.float32)
                for i in range(seq_len):
                    window = _build_tile_window(embed_table, eval_tokens, i, num_tiles)
                    _mp, eval_logits, _ = model.step(window, memory_eval, 0.0)
                    memory_eval = model.extract_memory()
                    if i in eval_by_pos:
                        pred = predicted_token(eval_logits, num_tiles - 1)
                        correct += int(pred == eval_by_pos[i])
                        total += 1
        return correct / total if total else 0.0

    t0 = time.time()
    recent_query_loss = []
    trajectory = []
    for step in range(1, train_steps + 1):
        lr = lr_schedule(step, train_steps, peak_lr, WARMUP_STEPS)
        if opt is not None:
            for g in opt.param_groups:
                g["lr"] = lr
        tokens, mqar_pairs = generate_mqar_sequence(rng, VOCAB, seq_len, num_kv_pairs)
        targets = _build_targets(tokens, mqar_pairs, num_kv_pairs)
        query_positions = set(pos for pos, _ in mqar_pairs)
        memory = np.zeros((NUM_MEMORY_SLOTS, state_width), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, num_tiles)
            _mp, logits, aux = model.step(window, memory, lr)
            if i in targets:
                target = torch.tensor([targets[i]], dtype=torch.long)
                loss = torch.nn.functional.cross_entropy(logits[num_tiles - 1:num_tiles], target)
                if i in query_positions:
                    recent_query_loss.append(float(loss))
                total_loss = loss if aux is None else loss + aux
                model.zero_grad()
                total_loss.backward()
                memory = model.extract_memory()
                model.apply_updates()
                if opt is not None:
                    clip_grad_norm_(adam_params, MAX_GRAD_NORM)
                    opt.step()
            else:
                memory = model.extract_memory()

        if step % log_every == 0 or step == train_steps:
            mean_q_loss = float(np.mean(recent_query_loss)) if recent_query_loss else float("nan")
            recent_query_loss = []
            quick_acc = _quick_eval(40)
            trajectory.append((step, mean_q_loss, quick_acc))
            if log_fn is not None:
                log_fn(step, train_steps, time.time() - t0, mean_q_loss, quick_acc)

    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(EVAL_SEQUENCES):
            tokens, mqar_pairs = generate_mqar_sequence(rng, VOCAB, seq_len, num_kv_pairs)
            mqar_by_pos = dict(mqar_pairs)
            memory = np.zeros((NUM_MEMORY_SLOTS, state_width), dtype=np.float32)
            for i in range(seq_len):
                window = _build_tile_window(embed_table, tokens, i, num_tiles)
                _mp, logits, _aux = model.step(window, memory, 0.0)
                memory = model.extract_memory()
                if i in mqar_by_pos:
                    pred = predicted_token(logits, num_tiles - 1)
                    correct += int(pred == mqar_by_pos[i])
                    total += 1

    return {"opt_config": opt_config_name, "acc": correct / total if total else 0.0,
            "elapsed_s": time.time() - t0, "trajectory": trajectory}


def main():
    opt_config_name = sys.argv[1] if len(sys.argv) > 1 else "production"
    peak_lr = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_LR
    train_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1000

    print(f"# ToyTileRecurrenceRMTAblation opt_config={opt_config_name} peak_lr={peak_lr} "
          f"train_steps={train_steps} seed={seed} opt_kwargs={OPT_CONFIGS[opt_config_name]}", flush=True)

    def log_fn(step, total_steps, elapsed, mean_q_loss, quick_acc):
        print(f"  step={step:>6}/{total_steps}  mean_query_loss={mean_q_loss:.4f}  "
              f"quick_acc={quick_acc:.4f}  ({elapsed:.0f}s elapsed)", flush=True)

    r = train_and_eval(opt_config_name, 1, seed, train_steps, peak_lr, log_fn=log_fn)
    print(f"\nFINAL opt_config={opt_config_name} peak_lr={peak_lr} acc={r['acc']:.4f} ({r['elapsed_s']:.0f}s)",
          flush=True)
    print("TRAJECTORY_JSON " + json.dumps(r["trajectory"]), flush=True)


if __name__ == "__main__":
    main()
