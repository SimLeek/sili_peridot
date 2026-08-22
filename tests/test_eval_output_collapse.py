"""Tests for model/eval_output_collapse.py.

Fast tests use hand-built prediction/logit sets with KNOWN collapse
properties (fully collapsed, fully diverse, confident-but-wrong-argmax
with real logit movement) to check the metrics do what they claim. One
opt-in test wires check_output_collapse to the actual `baseline`
OriginalArchModel's held-out eval, mirroring evaluate()'s own eval-loop
shape but returning raw predictions/targets/logits instead of a bare
accuracy scalar.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from model.eval_output_collapse import check_output_collapse

RUN_ENV_VAR = "SILI_RUN_LR_SEARCH"
VOCAB = 10


def _make_predict_fn(preds, targets, logits):
    def predict_fn(seed):
        return preds, targets, logits
    return predict_fn


class TestOutputCollapseSynthetic:
    def test_fully_collapsed_has_zero_entropy(self):
        n = 50
        preds = [3] * n
        targets = [i % VOCAB for i in range(n)]
        logits = [np.eye(VOCAB)[3] * 10.0 for _ in range(n)]  # identical every time
        report = check_output_collapse(
            _make_predict_fn(preds, targets, logits), vocab_size=VOCAB)
        assert report.prediction_entropy_bits == pytest.approx(0.0, abs=1e-9)
        assert report.normalized_entropy == pytest.approx(0.0, abs=1e-9)
        assert report.unique_prediction_fraction == pytest.approx(1.0 / n)
        assert report.most_common_prediction_fraction == pytest.approx(1.0)
        assert report.cross_sample_logit_std == pytest.approx(0.0, abs=1e-9)

    def test_fully_diverse_uniform_predictions_has_max_entropy(self):
        n = VOCAB * 20
        rng = np.random.default_rng(0)
        preds = [i % VOCAB for i in range(n)]  # exactly uniform
        targets = [int(rng.integers(0, VOCAB)) for _ in range(n)]
        logits = [rng.standard_normal(VOCAB) for _ in range(n)]
        report = check_output_collapse(
            _make_predict_fn(preds, targets, logits), vocab_size=VOCAB)
        assert report.normalized_entropy == pytest.approx(1.0, rel=1e-6)
        assert report.unique_prediction_fraction == pytest.approx(VOCAB / n)

    def test_confident_wrong_argmax_but_real_logit_movement(self):
        # The specific failure-mode distinction this module exists for:
        # ALWAYS predicts the same class (argmax collapsed), but the
        # underlying logits genuinely shift with input -- a real,
        # different bug from a model whose output barely moves at all.
        n = 30
        rng = np.random.default_rng(1)
        preds = [0] * n
        targets = [i % VOCAB for i in range(n)]
        logits = []
        for _ in range(n):
            v = rng.standard_normal(VOCAB) * 3.0  # real per-sample variation
            v[0] += 10.0  # but class 0 always dominates regardless
            logits.append(v)
        report = check_output_collapse(
            _make_predict_fn(preds, targets, logits), vocab_size=VOCAB)
        assert report.normalized_entropy == pytest.approx(0.0, abs=1e-9)  # argmax fully collapsed
        assert report.cross_sample_logit_std > 1.0  # but logits genuinely vary

    def test_accuracy_computed_correctly(self):
        preds = [1, 2, 3, 4]
        targets = [1, 2, 0, 0]
        logits = [np.zeros(VOCAB) for _ in range(4)]
        report = check_output_collapse(
            _make_predict_fn(preds, targets, logits), vocab_size=VOCAB)
        assert report.accuracy == pytest.approx(0.5)

    def test_n_batches_pools_across_calls(self):
        calls = []
        def predict_fn(seed):
            calls.append(seed)
            return [seed % VOCAB], [seed % VOCAB], [np.zeros(VOCAB)]
        report = check_output_collapse(predict_fn, vocab_size=VOCAB, n_batches=5, seed=100)
        assert calls == [100, 101, 102, 103, 104]
        assert report.n_samples == 5

    def test_raises_on_zero_samples(self):
        with pytest.raises(ValueError):
            check_output_collapse(_make_predict_fn([], [], []), vocab_size=VOCAB)


@pytest.mark.skipif(not os.environ.get(RUN_ENV_VAR),
                    reason=f"real short training run, opt in via {RUN_ENV_VAR}=1")
class TestOutputCollapseRealModel:
    def test_baseline_config_after_training(self):
        from scripts.l1_sparsity_probe import (
            OriginalArchModel, generate_copy_sequence, run,
            _build_tile_window, NUM_TILES,
        )

        model = OriginalArchModel(
            1000, dense=True, o_proj_coef=0.0, all_layer_coef=0.0,
            l1_sparsity_coef=0.05, use_energy=False, all_zero_init=False,
        )
        # Same calibrated lr find_optimal_lr found for this config
        # (conversation) -- a real, if short, training run before
        # checking for collapse, not an untrained model.
        run(model, 1500, 1000, verbose=False, peak_lr=0.0483)

        embed_table = np.random.RandomState(1000).randn(VOCAB, 8).astype(np.float32) * 0.3
        state_width = model.state_width

        def predict_fn(seed):
            eval_rng = np.random.RandomState(seed)
            preds, targets, logits = [], [], []
            for _ in range(30):
                tokens, pairs = generate_copy_sequence(eval_rng, VOCAB, NUM_TILES)
                tgt_map = dict(pairs)
                M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
                for i in range(NUM_TILES):
                    window = _build_tile_window(embed_table, tokens, i, NUM_TILES, 4)
                    M, out_logits, aux = model.step(window, M, 0.0)
                    if i in tgt_map:
                        preds.append(int(np.argmax(out_logits.data[NUM_TILES - 1])))
                        targets.append(tgt_map[i])
                        logits.append(np.asarray(out_logits.data[NUM_TILES - 1]))
            return preds, targets, logits

        report = check_output_collapse(predict_fn, vocab_size=VOCAB, n_batches=3, seed=999_983)
        print(f"\noutput collapse report: accuracy={report.accuracy:.3f} "
              f"normalized_entropy={report.normalized_entropy:.3f} "
              f"unique_prediction_fraction={report.unique_prediction_fraction:.3f} "
              f"most_common_prediction_fraction={report.most_common_prediction_fraction:.3f} "
              f"cross_sample_logit_std={report.cross_sample_logit_std:.4f}")
        # Sanity, not a tight pass/fail bound -- a genuinely collapsed
        # model (normalized_entropy near 0 AND cross_sample_logit_std
        # near 0 together) would be the real signal to act on; this
        # just confirms the pipeline runs end-to-end and produces finite,
        # sane numbers on a real (if lightly) trained model.
        assert np.isfinite(report.normalized_entropy)
        assert np.isfinite(report.cross_sample_logit_std)
        assert report.n_samples > 0
