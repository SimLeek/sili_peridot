"""
scripts/train_toy_precision_comparison.py
───────────────────────────────────────────
Adam (fp32-matched-to-FP4) vs importance (real FP4), matched precision
-- see model/toy_precision_models.py's own module docstring and the
approved plan (fuzzy-plotting-starlight.md) for the full design
rationale.

A proper 2x2, per [[feedback_do_science_correctly]] (an earlier version
of this comparison gave EnergyDynamics to the real-FP4 arm only,
confounding "optimizer" with "energy's own added noise"):

                    no energy          + energy
  Adam+artificial-FP4   (A)                (B)
  importance+real-FP4   (C)                (D)

Same architecture shape (q/k/v/o/gate/up/down, causal attention,
RMSNorm), same MQAR task, same UNMODIFIED target convention
(`mqar_pairs` only -- this is a precision/optimizer/energy question,
not the column-mechanism question a separate run already handled).

PLUS a third row: `ToySmallTransformerRealFP4RowScaleAdam` --
importance still trains individual FP4 weight VALUES, but the
coarser per-row value_scale gets its own Adam-style adaptive
normalization on top (see AdamRowScaleDISLDOLayer's own docstring for
the mechanism and its one real approximation). Compared against arm
(C) specifically -- isolates "does Adam-normalizing just the row-scale
fix the divergence seen in arm (C)/(D)" from any energy question.

Tracks BOTH the training loss curve (sampled, not every step) and
final MQAR recall accuracy for every arm.

Run: python -m scripts.train_toy_precision_comparison
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from model.toy_precision_models import (
    ToySmallTransformerArtificialFP4,
    ToySmallTransformerRealFP4,
    ToySmallTransformerRealFP4RowScaleAdam,
)
from model.toy_recall_models import (
    AdamOptimizer,
    clip_grad_norm_,
    cross_entropy_sum,
    lr_schedule,
    predicted_token,
)
from model.toy_recall_task import generate_mqar_sequence

TRAIN_STEPS = 3000
WARMUP_STEPS = 100
PEAK_LR = 0.02
MAX_GRAD_NORM = 1.0
EVAL_SEQUENCES = 60
LOSS_SAMPLE_EVERY = 200

CONFIGS = [
    (16, 2, 20),
    (32, 4, 40),
]


def train_and_eval_artificial_fp4(seq_len, num_kv_pairs, vocab, hidden, mlp_hidden, use_energy, seed):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    model = ToySmallTransformerArtificialFP4(vocab, hidden, mlp_hidden, n_layers=2, use_energy=use_energy, num_cpus=2)
    opt = AdamOptimizer()
    embed_table = rng.randn(vocab, hidden).astype(np.float32) * 0.3

    loss_curve = []
    for step in range(TRAIN_STEPS):
        lr = lr_schedule(step, TRAIN_STEPS, PEAK_LR, WARMUP_STEPS)
        tokens, mqar_pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
        embedded = embed_table[tokens]
        logits, aux_loss = model.forward(embedded)
        loss = cross_entropy_sum(logits, mqar_pairs)
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()
        clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        opt.step(model.parameters(), lr=lr)
        if step % LOSS_SAMPLE_EVERY == 0:
            loss_curve.append(float(loss.data))

    correct, total = 0, 0
    for _ in range(EVAL_SEQUENCES):
        tokens, mqar_pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
        embedded = embed_table[tokens]
        logits, _aux = model.forward(embedded)
        for pos, target in mqar_pairs:
            correct += int(predicted_token(logits, pos) == target)
            total += 1
    return correct / total, loss_curve


def train_and_eval_real_fp4(
    seq_len, num_kv_pairs, vocab, hidden, mlp_hidden, use_energy, seed, model_cls=ToySmallTransformerRealFP4
):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    max_weights = hidden * mlp_hidden
    model = model_cls(vocab, hidden, mlp_hidden, n_layers=2, max_weights=max_weights, use_energy=use_energy, num_cpus=2)
    opt = AdamOptimizer()
    embed_table = rng.randn(vocab, hidden).astype(np.float32) * 0.3

    loss_curve = []
    for step in range(TRAIN_STEPS):
        lr = lr_schedule(step, TRAIN_STEPS, PEAK_LR, WARMUP_STEPS)
        tokens, mqar_pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
        embedded = embed_table[tokens]
        logits, aux_loss = model.forward(embedded, learning_rate=lr)
        loss = cross_entropy_sum(logits, mqar_pairs)
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.grad = np.array(1.0, dtype=np.float32)
        loss.backward()
        opt.step(model.parameters_for_optimizer(), lr=lr)
        if step % LOSS_SAMPLE_EVERY == 0:
            loss_curve.append(float(loss.data))

    correct, total = 0, 0
    for _ in range(EVAL_SEQUENCES):
        tokens, mqar_pairs = generate_mqar_sequence(rng, vocab, seq_len, num_kv_pairs)
        embedded = embed_table[tokens]
        logits, _aux = model.forward(embedded, learning_rate=0.0)
        for pos, target in mqar_pairs:
            correct += int(predicted_token(logits, pos) == target)
            total += 1
    return correct / total, loss_curve


def _fmt_curve(curve):
    return " ".join(f"{v:.1f}" for v in curve)


def main():
    hidden, mlp_hidden = 32, 48
    print(
        f"hidden={hidden} mlp_hidden={mlp_hidden} train_steps={TRAIN_STEPS} "
        f"warmup={WARMUP_STEPS} peak_lr={PEAK_LR} eval_sequences={EVAL_SEQUENCES}\n"
    )
    results = []
    for seq_len, num_kv_pairs, vocab in CONFIGS:
        print(f"=== seq_len={seq_len} kv_pairs={num_kv_pairs} vocab={vocab} ===")
        t0 = time.time()
        a_acc, a_curve = train_and_eval_artificial_fp4(
            seq_len, num_kv_pairs, vocab, hidden, mlp_hidden, False, seed=3000 + seq_len
        )
        t1 = time.time()
        b_acc, b_curve = train_and_eval_artificial_fp4(
            seq_len, num_kv_pairs, vocab, hidden, mlp_hidden, True, seed=3500 + seq_len
        )
        t2 = time.time()
        c_acc, c_curve = train_and_eval_real_fp4(
            seq_len, num_kv_pairs, vocab, hidden, mlp_hidden, False, seed=4000 + seq_len
        )
        t3 = time.time()
        d_acc, d_curve = train_and_eval_real_fp4(
            seq_len, num_kv_pairs, vocab, hidden, mlp_hidden, True, seed=4500 + seq_len
        )
        t4 = time.time()
        e_acc, e_curve = train_and_eval_real_fp4(
            seq_len,
            num_kv_pairs,
            vocab,
            hidden,
            mlp_hidden,
            False,
            seed=5000 + seq_len,
            model_cls=ToySmallTransformerRealFP4RowScaleAdam,
        )
        t5 = time.time()

        print(f"  (A) Adam+artificial-FP4, no energy:      acc={a_acc:.2f}  ({t1 - t0:.1f}s)")
        print(f"      loss curve: {_fmt_curve(a_curve)}")
        print(f"  (B) Adam+artificial-FP4, + energy:        acc={b_acc:.2f}  ({t2 - t1:.1f}s)")
        print(f"      loss curve: {_fmt_curve(b_curve)}")
        print(f"  (C) importance+real-FP4, no energy:       acc={c_acc:.2f}  ({t3 - t2:.1f}s)")
        print(f"      loss curve: {_fmt_curve(c_curve)}")
        print(f"  (D) importance+real-FP4, + energy:        acc={d_acc:.2f}  ({t4 - t3:.1f}s)")
        print(f"      loss curve: {_fmt_curve(d_curve)}")
        print(f"  (E) importance+real-FP4, row-scale-Adam:  acc={e_acc:.2f}  ({t5 - t4:.1f}s)")
        print(f"      loss curve: {_fmt_curve(e_curve)}\n")
        results.append((seq_len, num_kv_pairs, vocab, a_acc, b_acc, c_acc, d_acc, e_acc))

    print(f"{'seq_len':>8}  {'kv':>4}  {'vocab':>6}  {'A':>6}  {'B':>6}  {'C':>6}  {'D':>6}  {'E':>6}")
    for seq_len, num_kv_pairs, vocab, a, b, c, d, e in results:
        print(f"{seq_len:>8}  {num_kv_pairs:>4}  {vocab:>6}  {a:>6.2f}  {b:>6.2f}  {c:>6.2f}  {d:>6.2f}  {e:>6.2f}")
    return results


if __name__ == "__main__":
    main()
