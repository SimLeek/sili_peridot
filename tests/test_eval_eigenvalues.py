"""Tests for model/eval_eigenvalues.py's eigenvalue/spectral-norm probes.

Two families of ground truth, matching the module's two families of
measurement:
  - SpectralProbe (cheap, iterative, square-layers-only) approximates a
    spectral-RADIUS-like quantity -- tested against a hand-built matrix
    with a real, well-separated dominant eigenvalue (an upper-triangular
    matrix's eigenvalues ARE its diagonal entries, guaranteed real,
    regardless of the off-diagonal terms).
  - exact_spectral_norm/exact_spectral_radius (exact, SVD/eigval-based)
    are tested against a generic random (non-symmetric, complex-
    dominant-eigenvalue) matrix via np.linalg.svd/eigvals directly --
    no convergence tolerance needed, these should match to float
    precision.

One opt-in test wires track_spectral_health to the actual `baseline`
OriginalArchModel to confirm it runs end-to-end against a real training
loop and produces a sane, non-exploding trajectory.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from sili.tensor import Tensor
from model.eval_eigenvalues import (
    SpectralProbe, probe_layers, measure_snapshot, track_spectral_health,
    dense_weight_matrix, exact_spectral_norm, exact_spectral_radius,
    exact_spectral_snapshot,
)

RUN_ENV_VAR = "SILI_RUN_LR_SEARCH"


class _LinearStub:
    """Minimal .forward(x, learning_rate)-compatible stub -- exactly the
    interface this module needs, nothing else, so these tests don't
    depend on any real DISLDOLayer machinery."""
    def __init__(self, W: np.ndarray):
        self.W = W.astype(np.float32)
        self.in_features = W.shape[0]
        self.out_features = W.shape[1]

    def forward(self, x, learning_rate: float = 0.0) -> Tensor:
        data = np.asarray(x.data).reshape(1, -1) @ self.W
        return Tensor(data.astype(np.float32))


def _triangular_matrix_with_real_dominant_eigenvalue(n, dominant, seed=0):
    """Upper-triangular -> eigenvalues ARE the diagonal entries, exactly,
    regardless of off-diagonal content -- a clean way to build a matrix
    with a KNOWN, REAL, well-separated dominant eigenvalue (unlike a
    generic random matrix, whose dominant eigenvalue is usually complex
    -- see this module's own docstring)."""
    rng = np.random.default_rng(seed)
    diag = np.array([dominant] + [dominant * f for f in (0.5, 0.3, 0.2, 0.1)][:n - 1])
    W = np.triu(rng.standard_normal((n, n)).astype(np.float64) * 0.3, k=1)
    np.fill_diagonal(W, diag[:n])
    return W.astype(np.float32), float(dominant)


def _random_matrix(in_f, out_f, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((in_f, out_f)).astype(np.float32)


class TestSpectralProbeSynthetic:
    def test_converges_to_spectral_radius_for_real_dominant_eigenvalue(self):
        W, true_radius = _triangular_matrix_with_real_dominant_eigenvalue(16, 5.0, seed=1)
        layer = _LinearStub(W)
        probe = SpectralProbe(dim=16, seed=0, ema_decay=0.0)  # ema_decay=0 -> raw each step
        sigma = None
        for _ in range(300):
            sigma = probe.measure(layer)
        assert sigma == pytest.approx(true_radius, rel=0.02)
        # And confirm it's not measuring the (generally different) top
        # singular value -- the sharper version of this comparison,
        # with a much bigger norm/radius gap on a generic random matrix,
        # is test_exact_spectral_norm_exceeds_radius_for_nonsymmetric_
        # matrix below; this triangular construction's small off-
        # diagonal perturbation only pushes them slightly apart.
        true_norm = float(np.linalg.svd(W, compute_uv=False)[0])
        assert true_norm >= true_radius

    def test_ema_smooths_relative_to_raw(self):
        W, _ = _triangular_matrix_with_real_dominant_eigenvalue(16, 5.0, seed=3)
        layer = _LinearStub(W)
        probe = SpectralProbe(dim=16, seed=0, ema_decay=0.9)
        vals = [probe.measure(layer) for _ in range(50)]
        assert all(v > 0 and np.isfinite(v) for v in vals)
        tail = vals[-10:]
        assert (max(tail) - min(tail)) < 0.5 * np.mean(tail)

    def test_probe_layers_reads_in_features(self):
        W, _ = _triangular_matrix_with_real_dominant_eigenvalue(4, 5.0, seed=4)
        layers = {"a": _LinearStub(W), "b": _LinearStub(W)}
        probes = probe_layers(layers, seed=0)
        assert set(probes.keys()) == {"a", "b"}
        assert probes["a"].u.shape == (4,)

    def test_probe_layers_raises_without_in_features(self):
        class _NoShape:
            def forward(self, x, learning_rate=0.0):
                return x
        with pytest.raises(ValueError):
            probe_layers({"bad": _NoShape()})

    def test_probe_layers_rejects_rectangular_layer(self):
        # Confirmed directly: feeding a rectangular layer's OUTPUT back
        # in as the next step's INPUT can't work (dimension mismatch) --
        # probe_layers must reject this up front, not let it crash
        # partway through a training loop.
        W = _random_matrix(8, 32, seed=2)
        with pytest.raises(ValueError, match="rectangular"):
            probe_layers({"bad": _LinearStub(W)})

    def test_measure_snapshot_and_trajectory_shape(self):
        W, _ = _triangular_matrix_with_real_dominant_eigenvalue(4, 5.0, seed=5)
        layers = {"a": _LinearStub(W)}
        probes = probe_layers(layers, seed=0)
        snap = measure_snapshot(probes, layers, step=100)
        assert snap.step == 100
        assert "a" in snap.sigma_ema and "a" in snap.sigma_raw

    def test_track_spectral_health_generic_loop(self):
        W, true_radius = _triangular_matrix_with_real_dominant_eigenvalue(8, 5.0, seed=6)
        layer = _LinearStub(W)
        traj = track_spectral_health(
            model_step_fn=lambda: None,  # stub "training" does nothing
            layers_fn=lambda: {"only": layer},
            n_steps=1000, probe_every=100, seed=0, ema_decay=0.0,
        )
        assert len(traj.snapshots) == 10
        assert traj.final("only") == pytest.approx(true_radius, rel=0.05)


class TestExactSpectralMeasurements:
    def test_dense_weight_matrix_reconstructs_exactly(self):
        W = _random_matrix(6, 10, seed=7)
        layer = _LinearStub(W)
        reconstructed = dense_weight_matrix(layer)
        assert reconstructed.shape == (10, 6)
        np.testing.assert_allclose(reconstructed, W.T, atol=1e-5)

    def test_exact_spectral_norm_matches_svd(self):
        W = _random_matrix(12, 12, seed=8)
        layer = _LinearStub(W)
        expected = float(np.linalg.svd(W, compute_uv=False)[0])
        assert exact_spectral_norm(layer) == pytest.approx(expected, rel=1e-4)

    def test_exact_spectral_norm_works_on_rectangular_layer(self):
        # The whole point of having this alongside SpectralProbe -- no
        # square-only constraint, since it's a real SVD, not iteration.
        W = _random_matrix(6, 20, seed=9)
        layer = _LinearStub(W)
        expected = float(np.linalg.svd(W, compute_uv=False)[0])
        assert exact_spectral_norm(layer) == pytest.approx(expected, rel=1e-4)

    def test_exact_spectral_radius_matches_eigvals(self):
        W = _random_matrix(12, 12, seed=10)
        layer = _LinearStub(W)
        expected = float(np.max(np.abs(np.linalg.eigvals(W.T))))
        assert exact_spectral_radius(layer) == pytest.approx(expected, rel=1e-4)

    def test_exact_spectral_radius_raises_on_rectangular_layer(self):
        W = _random_matrix(6, 20, seed=11)
        layer = _LinearStub(W)
        with pytest.raises(ValueError, match="square"):
            exact_spectral_radius(layer)

    def test_exact_spectral_norm_exceeds_radius_for_nonsymmetric_matrix(self):
        # Direct confirmation of the finding that motivated this whole
        # module: for a generic (non-symmetric) matrix, norm > radius,
        # often substantially.
        W = _random_matrix(16, 16, seed=12)
        layer = _LinearStub(W)
        norm = exact_spectral_norm(layer)
        radius = exact_spectral_radius(layer)
        assert norm > radius

    def test_exact_spectral_snapshot_shape(self):
        W_sq = _random_matrix(8, 8, seed=13)
        W_rect = _random_matrix(8, 20, seed=14)
        layers = {"square": _LinearStub(W_sq), "rect": _LinearStub(W_rect)}
        snap = exact_spectral_snapshot(layers)
        assert set(snap.keys()) == {"square", "rect"}
        assert snap["square"]["radius"] is not None
        assert snap["rect"]["radius"] is None
        assert snap["rect"]["norm"] is not None


@pytest.mark.skipif(not os.environ.get(RUN_ENV_VAR),
                    reason=f"real short training run, opt in via {RUN_ENV_VAR}=1")
class TestSpectralHealthRealModel:
    def test_baseline_config_produces_sane_trajectory(self):
        from scripts.l1_sparsity_probe import OriginalArchModel, generate_copy_sequence, VOCAB
        import numpy as np

        model = OriginalArchModel(
            1000, dense=True, o_proj_coef=0.0, all_layer_coef=0.0,
            l1_sparsity_coef=0.05, use_energy=False, all_zero_init=False,
        )
        task_rng = np.random.RandomState(1000)
        embed_table = task_rng.randn(VOCAB, 8).astype(np.float32) * 0.3
        state_width = model.state_width

        # model.step() is FORWARD-ONLY -- confirmed directly (its own
        # docstring: "Returns (M_new, logits, aux_loss)", no gradient
        # applied). A first version of this test called only that in a
        # loop with no .backward()/opt.step() afterward, so its
        # "trajectory over training" was actually just the SpectralProbe
        # power-iteration converging against a STATIC, never-updated
        # model -- not measuring anything about training dynamics at
        # all. Fixed to match run()'s own real pattern: accumulate loss
        # across the tile sequence, backward() + opt.step() once per
        # outer step, same as the actual training loop this repo uses
        # everywhere else.
        from model.toy_recall_models import cross_entropy_sum, AdamOptimizer, clip_grad_norm_
        from scripts.l1_sparsity_probe import _build_tile_window, NUM_TILES
        opt = AdamOptimizer()

        def step():
            tokens, pairs = generate_copy_sequence(task_rng, VOCAB, NUM_TILES)
            targets = dict(pairs)
            M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
            total_loss = None
            for i in range(NUM_TILES):
                window = _build_tile_window(embed_table, tokens, i, NUM_TILES, 4)
                M, logits, aux = model.step(window, M, 0.0483)
                if i in targets:
                    tgt_loss = cross_entropy_sum(logits, [(NUM_TILES - 1, targets[i])])
                    total_loss = tgt_loss if total_loss is None else total_loss + tgt_loss
            if total_loss is not None:
                total_loss.backward()
                clip_grad_norm_(model.parameters_for_optimizer(), 1.0)
                opt.step(model.parameters_for_optimizer(), lr=0.0483)

        traj = track_spectral_health(
            model_step_fn=step,
            layers_fn=lambda: {"q_proj": model.q_proj, "k_proj": model.k_proj,
                                "v_proj": model.v_proj, "o_proj": model.o_proj},
            n_steps=500, probe_every=100,
        )
        print(f"\nspectral (radius-like) trajectory over 500 steps: "
              f"{[(n, traj.series(n)) for n in traj.layer_names()]}")
        for name in traj.layer_names():
            series = traj.series(name)
            assert all(np.isfinite(s) for s in series)
            # Sanity bound, not a tight assertion -- catches genuine
            # blow-up (the kind of thing spectral-norm regulation exists
            # to prevent), not meant to pin an exact healthy range.
            assert max(series) < 1000.0

        exact = exact_spectral_snapshot(
            {"q_proj": model.q_proj, "k_proj": model.k_proj,
             "v_proj": model.v_proj, "o_proj": model.o_proj})
        print(f"exact final snapshot: {exact}")
        for name, vals in exact.items():
            assert np.isfinite(vals["norm"])
            assert vals["radius"] is not None and np.isfinite(vals["radius"])
            # norm >= radius always (spectral norm upper-bounds radius).
            assert vals["norm"] >= vals["radius"] - 1e-4
