"""
sili_peridot/scripts/run_overnight_training.py
──────────────────────────────────────────────────
Real-tier online training run: does enough training (small learning
rate, real text data, hours of wall-clock) recover next-token accuracy
after pruning/quantization, with vs. without the energy function at
low drive? See model/train_online.py's module docstring for the
mechanism this exercises and tests/test_training_recovery.py for the
sanity-tier precursor (which already confirmed the mechanism runs
correctly, just not for long enough to say anything about recovery).

Meant to run unattended for hours. Writes progress to
scripts/overnight_run_results.json after every eval checkpoint (not
just at the end), so results are inspectable at any point without
needing the process to still be running or to finish cleanly. Run it
detached from the shell, e.g.:

    cd sili_peridot
    nohup python3 scripts/run_overnight_training.py \
        > scripts/overnight_run.log 2>&1 &
    disown

Runs the two variants (no-energy, then with-energy) SEQUENTIALLY, not
in parallel -- this machine has 16GB RAM; two full model copies running
at once would risk real memory pressure for no throughput benefit
(CPU-bound work, would just contend for the same cores).
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer

from model.config import MiniCPM5Config
from model.checkpoint import load_minicpm5_checkpoint
from model.prune import prune_state_dict_by_role, DEFAULT_TARGET_SPARSITY_BY_ROLE
from model.eval_pruning import EVAL_TEXTS_HELDOUT
from model.train_texts import load_real_tier_train_texts
from model.sili_model import build_sili_model, evaluate_next_token_prediction_sili
from model.train_online import train_last_step_mlp_online
from tests.conftest import trim_memory

REAL_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'MiniCPM5-1B-Base')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'overnight_run_results.json')

NUM_CPUS = 4
LEARNING_RATE = 0.001          # small, per the real-tier plan (10x below the sanity run's 0.01)
TOTAL_BUDGET_S = 8.0 * 3600.0  # 8 hours total, split evenly between the two variants
PER_VARIANT_BUDGET_S = TOTAL_BUDGET_S / 2.0
EVAL_EVERY_S = 900.0           # 15 min -- frequent enough for a real curve, cheap thanks to the frozen-prefix cache
MAX_TRAIN_TEXTS = 4000         # WikiText-2 yields ~3,500 usable lines after filtering; this is effectively "all of them"

_results: dict = {"variants": {}, "learning_rate": LEARNING_RATE, "per_variant_budget_s": PER_VARIANT_BUDGET_S}


def _save_results() -> None:
    with open(RESULTS_PATH, "w") as f:
        json.dump(_results, f, indent=2)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_variant(name: str, use_energy: bool, train_texts, tokenizer, cfg, half_bandwidth: int) -> None:
    _log(f"=== starting variant: {name} (use_energy={use_energy}) ===")
    sd = load_minicpm5_checkpoint(REAL_CHECKPOINT_DIR)
    sparse_state, _ = prune_state_dict_by_role(sd, DEFAULT_TARGET_SPARSITY_BY_ROLE)
    del sd
    trim_memory()
    sili_model = build_sili_model(sparse_state, cfg, num_cpus=NUM_CPUS)
    _log(f"[{name}] model built")

    baseline = evaluate_next_token_prediction_sili(
        sili_model, tokenizer, cfg, half_bandwidth, EVAL_TEXTS_HELDOUT, NUM_CPUS)
    _log(f"[{name}] baseline accuracy={baseline.accuracy:.4f} perplexity={baseline.perplexity:.2f}")

    _results["variants"][name] = {
        "baseline_accuracy": baseline.accuracy,
        "baseline_perplexity": baseline.perplexity,
        "eval_checkpoints": [],
        "n_steps": 0,
        "wall_clock_s": 0.0,
        "status": "running",
    }
    _save_results()

    def _on_eval_checkpoint(report) -> None:
        # Called from inside train_last_step_mlp_online every time it
        # records an eval checkpoint -- lets us persist progress without
        # chunking the call (which would reset the frozen-prefix cache
        # each time, see train_online.py's docstring).
        _results["variants"][name]["eval_checkpoints"] = report.eval_checkpoints
        _results["variants"][name]["n_steps"] = len(report.step_losses)
        _results["variants"][name]["wall_clock_s"] = report.wall_clock_s
        _save_results()
        cp = report.eval_checkpoints[-1]
        _log(f"[{name}] {len(report.step_losses)} steps, {cp['elapsed_s']:.0f}s elapsed, "
             f"accuracy={cp['accuracy']:.4f}, perplexity={cp['perplexity']:.2f}")

    report = train_last_step_mlp_online(
        sili_model, cfg, tokenizer, train_texts, EVAL_TEXTS_HELDOUT,
        half_bandwidth=half_bandwidth, num_cpus=NUM_CPUS, learning_rate=LEARNING_RATE,
        use_energy=use_energy, wall_clock_budget_s=PER_VARIANT_BUDGET_S, eval_every_s=EVAL_EVERY_S,
        on_eval_checkpoint=_on_eval_checkpoint,
    )

    _results["variants"][name]["eval_checkpoints"] = report.eval_checkpoints
    _results["variants"][name]["n_steps"] = len(report.step_losses)
    _results["variants"][name]["wall_clock_s"] = report.wall_clock_s
    _results["variants"][name]["status"] = "done"
    _save_results()
    _log(f"=== finished variant: {name}: {len(report.step_losses)} steps, "
         f"{report.wall_clock_s:.0f}s ===")

    del sili_model, report
    trim_memory()
    gc.collect()


def main() -> None:
    cfg = MiniCPM5Config.from_json(os.path.join(REAL_CHECKPOINT_DIR, "config.json"))
    tokenizer = AutoTokenizer.from_pretrained(REAL_CHECKPOINT_DIR)

    _log("Loading real-tier training corpus (WikiText-2)...")
    train_texts = load_real_tier_train_texts(max_texts=MAX_TRAIN_TEXTS)
    _log(f"Loaded {len(train_texts)} training texts")
    _results["n_train_texts"] = len(train_texts)

    half_bandwidth = max(
        max(len(tokenizer(t)["input_ids"]) for t in EVAL_TEXTS_HELDOUT),
        max(len(tokenizer(t)["input_ids"]) for t in train_texts),
    )
    _log(f"half_bandwidth={half_bandwidth}")
    _results["half_bandwidth"] = half_bandwidth
    _save_results()

    for name, use_energy in (("no_energy", False), ("with_energy", True)):
        try:
            run_variant(name, use_energy, train_texts, tokenizer, cfg, half_bandwidth)
        except Exception as e:
            _log(f"variant {name} FAILED: {e!r}")
            _results["variants"].setdefault(name, {})["status"] = f"failed: {e!r}"
            _save_results()

    _log("=== overnight run complete ===")


if __name__ == "__main__":
    main()
