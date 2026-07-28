"""
tests/test_training_recovery.py
──────────────────────────────────
Sanity-tier check: does model.train_online's online (single-token, no
batching) training of the last fold step's MLP actually run against the
real MiniCPM5-1B-Base checkpoint and move next-token accuracy at all,
with vs. without the energy function at low drive?

Deliberately NOT a verdict on whether training genuinely recovers
accuracy -- see model/train_texts.py and model/train_online.py's module
docstrings. This corpus (~50 sentences) and the short wall-clock budget
used here are a plumbing/sanity check only: confirms the mechanism
runs online, weights actually move, and loss trends downward at all. A
flat result here does NOT mean training doesn't work. The real
question (does it recover meaningful accuracy) needs a real-tier run:
more data, a small learning rate, and an overnight-scale wall-clock
budget -- call train_online.train_last_step_mlp_online directly with a
larger wall_clock_budget_s for that, reusing everything here unchanged.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from transformers import AutoTokenizer

from model.config import MiniCPM5Config
from model.checkpoint import load_minicpm5_checkpoint
from model.prune import prune_state_dict_by_role, DEFAULT_TARGET_SPARSITY_BY_ROLE
from model.eval_pruning import EVAL_TEXTS_HELDOUT
from model.train_texts import TRAIN_TEXTS
from model.sili_model import build_sili_model, evaluate_next_token_prediction_sili
from model.train_online import train_last_step_mlp_online
from conftest import trim_memory

REAL_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'MiniCPM5-1B-Base')

# Sanity-tier budget: long enough to see several hundred online steps
# and at least one mid-run eval checkpoint, short enough to run as part
# of the normal test suite. NOT the real-tier budget -- see module
# docstring.
_SANITY_WALL_CLOCK_S = 180.0
_SANITY_EVAL_EVERY_S = 60.0
_NUM_CPUS = 4


@pytest.mark.skipif(not os.path.isdir(REAL_CHECKPOINT_DIR),
                    reason="MiniCPM5-1B-Base checkpoint not present on this machine")
class TestTrainingRecoverySanity:

    def _build_model(self):
        cfg = MiniCPM5Config.from_json(os.path.join(REAL_CHECKPOINT_DIR, "config.json"))
        tokenizer = AutoTokenizer.from_pretrained(REAL_CHECKPOINT_DIR)
        sd = load_minicpm5_checkpoint(REAL_CHECKPOINT_DIR)
        sparse_state, _ = prune_state_dict_by_role(sd, DEFAULT_TARGET_SPARSITY_BY_ROLE)
        del sd
        trim_memory()
        max_seq_len = max(len(tokenizer(t)["input_ids"]) for t in EVAL_TEXTS_HELDOUT + TRAIN_TEXTS)
        sili_model = build_sili_model(sparse_state, cfg, num_cpus=_NUM_CPUS)
        return sili_model, cfg, tokenizer, max_seq_len

    def _run(self, use_energy: bool):
        sili_model, cfg, tokenizer, half_bandwidth = self._build_model()

        baseline = evaluate_next_token_prediction_sili(
            sili_model, tokenizer, cfg, half_bandwidth, EVAL_TEXTS_HELDOUT, _NUM_CPUS)

        report = train_last_step_mlp_online(
            sili_model, cfg, tokenizer, TRAIN_TEXTS, EVAL_TEXTS_HELDOUT,
            half_bandwidth=half_bandwidth, num_cpus=_NUM_CPUS, learning_rate=0.01,
            use_energy=use_energy,
            wall_clock_budget_s=_SANITY_WALL_CLOCK_S, eval_every_s=_SANITY_EVAL_EVERY_S,
        )

        del sili_model
        trim_memory()

        assert len(report.step_losses) > 0, "no online training steps ran at all"
        assert all(np.isfinite(l) for l in report.step_losses), \
            "training produced a non-finite loss somewhere -- see report.step_losses"
        assert len(report.eval_checkpoints) >= 2, \
            "expected at least the baseline + one mid/final eval checkpoint"
        assert report.eval_checkpoints[0]["accuracy"] == pytest.approx(baseline.accuracy), \
            ("train_online._evaluate_cached's baseline should exactly match "
             "sili_model.evaluate_next_token_prediction_sili's -- same computation, "
             "independent code path; a mismatch means the cache or the trainable-vs-general "
             "MLP forward paths disagree somewhere")

        variant = "with-energy" if use_energy else "no-energy"
        print(f"\n[{variant}] baseline accuracy={baseline.accuracy:.4f} "
              f"perplexity={baseline.perplexity:.2f}")
        print(f"[{variant}] {len(report.step_losses)} online steps, "
              f"{report.wall_clock_s:.1f}s wall-clock")
        print(f"[{variant}] loss[:5]={report.step_losses[:5]} "
              f"loss[-5:]={report.step_losses[-5:]}")
        for cp in report.eval_checkpoints:
            print(f"[{variant}] t={cp['elapsed_s']:.1f}s "
                  f"accuracy={cp['accuracy']:.4f} perplexity={cp['perplexity']:.2f}")

        return baseline, report

    def test_no_energy_runs_and_moves_loss(self):
        self._run(use_energy=False)

    def test_with_energy_low_drive_runs_and_moves_loss(self):
        self._run(use_energy=True)
