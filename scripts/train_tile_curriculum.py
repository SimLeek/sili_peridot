"""Fast curriculum test harness for ToyTileRecurrenceRealFP4.
See docs/research/train_tile_curriculum.rst:train_tile_curriculum.module_overview.

Usage: python3 train_tile_curriculum.py <arm> <use_energy 0|1> <use_attention 0|1> <total_steps> [checkpoint_every] [seed]
  arm: rank1 | rank2 | fp8
"""

from __future__ import annotations

import functools
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from sili import _cpu
from sili.sparse_rnn import (
    DISLDOLayer,
    DISLDOLayer8,
    DISLDOLayer32,
    DISLDOLayerDeterministic,
    DISLDOLayerNoScale,
    DISLDOLayerNoScaleDeterministic,
    DISLDOLayerResync,
    DISLDOLayerResyncDeterministic,
)

from model.toy_precision_models import (
    PeriodicSeedRank1DISLDOLayer8,
    QuantizedDISLDOLayer32,
    SeededDISLDOLayer8AdaMax,
    SeededDISLDOLayer8Resync,
    SeededRank1DISLDOLayer8,
    TrueMultiDigitDenseLayer,
    TrueMultiDigitLayer,
)
from model.toy_recall_models import AdamOptimizer, clip_grad_norm_, cross_entropy_sum, lr_schedule, predicted_token
from model.toy_tile_precision_models import ToyTileRecurrenceRealFP4


def generate_copy_sequence(rng: np.random.RandomState, vocab: int, seq_len: int):
    """See docs/research/train_tile_curriculum.rst:train_tile_curriculum.generate_copy_sequence_task_design."""
    tokens = rng.randint(0, vocab, size=seq_len)
    pairs = [(seq_len - 1, int(tokens[0]))]
    return tokens, pairs


EMBED_WIDTH = 8
COLUMN_NEURONS = 4
MLP_HIDDEN_MULT = 2  # unused (MLP removed), kept for API compat
# NUM_TILES/VOCAB/MAX_WEIGHTS_PER_LAYER/MAX_GRAD_NORM sizing: see
# docs/research/train_tile_curriculum.rst:train_tile_curriculum.tuning_constants_and_grad_clip.
NUM_TILES = 4
VOCAB = 10
MAX_WEIGHTS_PER_LAYER = 128
NUM_CPUS = 1
PEAK_LR = 0.002
WARMUP_STEPS = 50
MAX_GRAD_NORM = 1.0
EVAL_SEQUENCES = 60

SEQ_LEN_START = 2
SEQ_LEN_MAX = NUM_TILES
STEPS_PER_STAGE_DEFAULT = 500

ENERGY_KWARGS = {
    "drive": 0.00535,
    "activation_cost": 0.005,
    "precision": 0.001,
    "density": 0.005,
    "p": 0.995,
    "reactivity": 0.0001,
}

# ARMS registry overview: see
# docs/research/train_tile_curriculum.rst:train_tile_curriculum.arms_registry_overview.
ARMS = {
    "rank1": DISLDOLayer,
    "rank2": functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="rankn", rank=2, quantize_importance=True),
    "fp8": DISLDOLayer8,
    "fp32": DISLDOLayer32,
    "rank1_8bit": functools.partial(QuantizedDISLDOLayer32, bits=8, scheme="rank1", quantize_importance=True),
    # 4-bit scale-representation arms: see
    # docs/research/train_tile_curriculum.rst:train_tile_curriculum.arms_4bit_scale_schemes.
    "row_4bit": functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="row", quantize_importance=True),
    "rank1_4bit": functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="rank1", quantize_importance=True),
    "rank4_4bit": functools.partial(QuantizedDISLDOLayer32, bits=4, scheme="rankn", rank=4, quantize_importance=True),
    "multi_fp4": functools.partial(
        QuantizedDISLDOLayer32, bits=4, scheme="residual", n_stages=2, quantize_importance=True
    ),
    # FP8 cold-start/scale-staleness arms: see
    # docs/research/train_tile_curriculum.rst:train_tile_curriculum.arms_fp8_scale_coldstart.
    "fp8_seeded": SeededRank1DISLDOLayer8,
    "fp8_reseeded": functools.partial(PeriodicSeedRank1DISLDOLayer8, reseed_every=250),
    "fp8_resync": SeededDISLDOLayer8Resync,
    "fp8_adamax": SeededDISLDOLayer8AdaMax,
    # fixed_digit_2/3/4 (zero trained/fitted scale): see
    # docs/research/train_tile_curriculum.rst:train_tile_curriculum.arms_4bit_scale_schemes.
    "fixed_digit_2": functools.partial(
        QuantizedDISLDOLayer32, bits=4, scheme="fixed_digit_residual", n_stages=2, base=4.0, quantize_importance=True
    ),
    "fixed_digit_3": functools.partial(
        QuantizedDISLDOLayer32, bits=4, scheme="fixed_digit_residual", n_stages=3, base=4.0, quantize_importance=True
    ),
    "fixed_digit_4": functools.partial(
        QuantizedDISLDOLayer32, bits=4, scheme="fixed_digit_residual", n_stages=4, base=4.0, quantize_importance=True
    ),
    # true_multi_digit_lr0/lr1/lr2, fp32_ref, dense: see
    # docs/research/train_tile_curriculum.rst:train_tile_curriculum.arms_true_multi_digit_lr_power.
    "true_multi_digit_lr0": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayer, n_stages=3, base=4.0, lr_power=0.0
    ),
    "true_multi_digit_lr1": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayer, n_stages=3, base=4.0, lr_power=1.0
    ),
    "true_multi_digit_lr2": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayer, n_stages=3, base=4.0, lr_power=2.0
    ),
    "true_multi_digit_fp32_ref": functools.partial(
        TrueMultiDigitLayer,
        digit_cls=DISLDOLayer32,
        n_stages=3,
        base=4.0,
        lr_power=0.0,
        simulate_quantize=True,
        bits_per_stage=4,
    ),
    "true_multi_digit_dense": functools.partial(TrueMultiDigitDenseLayer, n_stages=3, base=4.0, lr_power=0.0),
    # Stale value_scale bug isolation: see
    # docs/research/train_tile_curriculum.rst:train_tile_curriculum.arms_stale_scale_bug_isolation.
    "row_4bit_resync": DISLDOLayerResync,
    "row_4bit_noscale": DISLDOLayerNoScale,
    "true_multi_digit_resync": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerResync, n_stages=3, base=4.0, lr_power=0.0
    ),
    "true_multi_digit_noscale": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerNoScale, n_stages=3, base=4.0, lr_power=0.0
    ),
    # Stochastic vs. deterministic rounding: see
    # docs/research/train_tile_curriculum.rst:train_tile_curriculum.arms_stochastic_vs_deterministic_rounding.
    "row_4bit_deterministic": DISLDOLayerDeterministic,
    "row_4bit_resync_deterministic": DISLDOLayerResyncDeterministic,
    "row_4bit_noscale_deterministic": DISLDOLayerNoScaleDeterministic,
    # base=12.0 default and the unseeded-preseed bug found mid-sweep: see
    # docs/research/train_tile_curriculum.rst:train_tile_curriculum.arms_base_sweep_and_unseeded_bug.
    "true_multi_digit_deterministic": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic, n_stages=3, base=12.0, lr_power=0.0
    ),
    # STOCHASTIC rounding is the preferred choice for real runs: see
    # docs/research/train_tile_curriculum.rst:train_tile_curriculum.arms_stochastic_vs_deterministic_rounding.
    "true_multi_digit_stochastic": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayer, n_stages=3, base=12.0, lr_power=0.0
    ),
    # "fp4+fp4 dual": see
    # docs/research/train_tile_curriculum.rst:train_tile_curriculum.arms_stochastic_vs_deterministic_rounding.
    "true_multi_digit_dual": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayer, n_stages=2, base=12.0, lr_power=0.0
    ),
    "true_multi_digit_deterministic_base4": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic, n_stages=3, base=4.0, lr_power=0.0
    ),
    "true_multi_digit_deterministic_base6": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic, n_stages=3, base=6.0, lr_power=0.0
    ),
    "true_multi_digit_deterministic_base24": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic, n_stages=3, base=24.0, lr_power=0.0
    ),
    # lr_power retest under deterministic rounding: see
    # docs/research/train_tile_curriculum.rst:train_tile_curriculum.arms_base_sweep_and_unseeded_bug.
    "true_multi_digit_deterministic_lr1": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic, n_stages=3, base=12.0, lr_power=1.0
    ),
    "true_multi_digit_deterministic_lr2": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic, n_stages=3, base=12.0, lr_power=2.0
    ),
    # Dense-connectivity variants (testing the sparse echo-network preseed itself): see
    # docs/research/train_tile_curriculum.rst:train_tile_curriculum.arms_dense_connectivity_variants.
    "true_multi_digit_deterministic_dense": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic, n_stages=3, base=12.0, lr_power=0.0, dense=True
    ),
    "true_multi_digit_stochastic_dense": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayer, n_stages=3, base=12.0, lr_power=0.0, dense=True
    ),
    "true_multi_digit_dual_dense": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayer, n_stages=2, base=12.0, lr_power=0.0, dense=True
    ),
    "true_multi_digit_deterministic_base4_dense": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic, n_stages=3, base=4.0, lr_power=0.0, dense=True
    ),
    "true_multi_digit_deterministic_base6_dense": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic, n_stages=3, base=6.0, lr_power=0.0, dense=True
    ),
    "true_multi_digit_deterministic_base24_dense": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerDeterministic, n_stages=3, base=24.0, lr_power=0.0, dense=True
    ),
    "true_multi_digit_noscale_deterministic": functools.partial(
        TrueMultiDigitLayer, digit_cls=DISLDOLayerNoScaleDeterministic, n_stages=3, base=12.0, lr_power=0.0
    ),
    # Shared-connectivity hypothesis: see
    # docs/research/train_tile_curriculum.rst:train_tile_curriculum.arms_shared_connectivity_hypothesis.
    "true_multi_digit_shared_conn": functools.partial(
        TrueMultiDigitLayer,
        digit_cls=DISLDOLayerDeterministic,
        n_stages=3,
        base=4.0,
        lr_power=0.0,
        share_connectivity=True,
    ),
}


def _maybe_synaptogenesis(model, k: int = 4, importance_cutoff: float = 0.01):
    """See docs/research/train_tile_curriculum.rst:train_tile_curriculum.maybe_synaptogenesis_k4_design."""
    for attr in ("q_proj", "k_proj", "v_proj", "lm_head"):
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        if hasattr(sub, "_max_row_weights"):
            sub.synaptogenesis(k, importance_cutoff, sub._max_row_weights)
        elif hasattr(sub, "synaptogenesis"):
            sub.synaptogenesis(k, importance_cutoff)
    o_proj = getattr(model, "o_proj", None)
    o_layers = o_proj if isinstance(o_proj, list) else ([o_proj] if o_proj is not None else [])
    for sub in o_layers:
        if hasattr(sub, "_max_row_weights"):
            sub.synaptogenesis(k, importance_cutoff, sub._max_row_weights)
        elif hasattr(sub, "synaptogenesis"):
            sub.synaptogenesis(k, importance_cutoff)


def _build_tile_window(
    embed_table: np.ndarray, tokens: np.ndarray, i: int, num_tiles: int, column_neurons: int | None = None
) -> np.ndarray:
    """See docs/research/train_tile_curriculum.rst:train_tile_curriculum.build_tile_window_not_tiled_correction."""
    embed_width = embed_table.shape[1]
    window = np.zeros((num_tiles, embed_width), dtype=np.float32)
    for j in range(num_tiles):
        src = i - (num_tiles - 1) + j
        if src >= 0:
            window[j] = embed_table[tokens[src]]
    return window


def current_seq_len(step: int, steps_per_stage: int) -> int:
    stage = step // steps_per_stage
    return min(SEQ_LEN_START + stage, SEQ_LEN_MAX)


def evaluate(model, rng, embed_table: np.ndarray, seq_len: int) -> float:
    state_width = embed_table.shape[1] * COLUMN_NEURONS
    correct, total = 0, 0
    for _ in range(EVAL_SEQUENCES):
        tokens, pairs = generate_copy_sequence(rng, VOCAB, seq_len)
        targets = dict(pairs)
        M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, COLUMN_NEURONS)
            M, logits, _aux = model.step(window, M, 0.0)
            if i in targets:
                pred = predicted_token(logits, NUM_TILES - 1)
                correct += int(pred == targets[i])
                total += 1
    return correct / total if total else 0.0


# See docs/research/train_tile_curriculum.rst:train_tile_curriculum.arm_value_bits_approximation.
ARM_VALUE_BITS = {
    "rank1": 4,
    "rank2": 4,
    "fp8": 8,
    "fp32": 32,
    "rank1_8bit": 8,
    "row_4bit": 4,
    "rank1_4bit": 4,
    "rank4_4bit": 4,
    "multi_fp4": 8,
    "fp8_seeded": 8,
    "fp8_reseeded": 8,
    "fp8_resync": 8,
    "fp8_adamax": 8,
    "fixed_digit_2": 8,
    "fixed_digit_3": 12,
    "fixed_digit_4": 16,
    "true_multi_digit_lr0": 12,
    "true_multi_digit_lr1": 12,
    "true_multi_digit_lr2": 12,
    "true_multi_digit_fp32_ref": 32,
    "true_multi_digit_dense": 32,
    "row_4bit_resync": 4,
    "row_4bit_noscale": 4,
    "true_multi_digit_resync": 12,
    "true_multi_digit_noscale": 12,
    "row_4bit_deterministic": 4,
    "row_4bit_resync_deterministic": 4,
    "row_4bit_noscale_deterministic": 4,
    "true_multi_digit_deterministic": 12,
    "true_multi_digit_stochastic": 12,
    "true_multi_digit_stochastic_dense": 12,
    "true_multi_digit_dual": 8,
    "true_multi_digit_dual_dense": 8,
    "true_multi_digit_noscale_deterministic": 12,
    "true_multi_digit_deterministic_base4": 12,
    "true_multi_digit_deterministic_base6": 12,
    "true_multi_digit_deterministic_base24": 12,
    "true_multi_digit_deterministic_lr1": 12,
    "true_multi_digit_deterministic_lr2": 12,
    "true_multi_digit_deterministic_dense": 12,
    "true_multi_digit_deterministic_base4_dense": 12,
    "true_multi_digit_deterministic_base6_dense": 12,
    "true_multi_digit_deterministic_base24_dense": 12,
    "true_multi_digit_shared_conn": 12,
}


def estimate_value_bits(
    arm: str, state_width: int, embed_width: int, vocab: int, max_weights: int, use_attention: bool
) -> int:
    bits = ARM_VALUE_BITS[arm]
    o_proj = min(max_weights, state_width * state_width)
    lm_head = min(max_weights, embed_width * vocab)
    total_entries = o_proj + lm_head
    if use_attention:
        total_entries += 3 * min(max_weights, state_width * state_width)  # q,k,v
    return total_entries * bits


def main():
    global EMBED_WIDTH, COLUMN_NEURONS, MAX_WEIGHTS_PER_LAYER, SEQ_LEN_MAX, PEAK_LR

    arm = sys.argv[1]
    use_energy = bool(int(sys.argv[2]))
    use_attention = bool(int(sys.argv[3]))
    train_steps = int(sys.argv[4])
    checkpoint_every = int(sys.argv[5]) if len(sys.argv) > 5 else max(train_steps // 20, 50)
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 1000
    steps_per_stage = int(sys.argv[7]) if len(sys.argv) > 7 else STEPS_PER_STAGE_DEFAULT
    # optional overrides for memory-footprint-matched comparisons -- see
    # docs/research/train_tile_curriculum.rst:train_tile_curriculum.arm_value_bits_approximation.
    if len(sys.argv) > 8:
        EMBED_WIDTH = int(sys.argv[8])
    if len(sys.argv) > 9:
        COLUMN_NEURONS = int(sys.argv[9])
    if len(sys.argv) > 10:
        MAX_WEIGHTS_PER_LAYER = int(sys.argv[10])
    # See docs/research/train_tile_curriculum.rst:train_tile_curriculum.cli_seq_len_max_out_of_context.
    SEQ_LEN_MAX = int(sys.argv[11]) if len(sys.argv) > 11 else NUM_TILES
    # See docs/research/train_tile_curriculum.rst:train_tile_curriculum.cli_peak_lr_per_row_nnz_fp4.
    if len(sys.argv) > 12:
        PEAK_LR = float(sys.argv[12])
    # See docs/research/train_tile_curriculum.rst:train_tile_curriculum.cli_o_proj_depth_cascaded_idea.
    o_proj_depth = int(sys.argv[13]) if len(sys.argv) > 13 else 1
    # See docs/research/train_tile_curriculum.rst:train_tile_curriculum.cli_use_synaptogenesis_intent.
    use_synaptogenesis = bool(int(sys.argv[14])) if len(sys.argv) > 14 else False
    # See docs/research/train_tile_curriculum.rst:train_tile_curriculum.cli_clip_range_finding.
    clip_range = float(sys.argv[15]) if len(sys.argv) > 15 else 6.0
    # See docs/research/train_tile_curriculum.rst:train_tile_curriculum.cli_magnitude_penalty_coef.
    magnitude_penalty_coef = float(sys.argv[16]) if len(sys.argv) > 16 else 0.0
    # spectral_norm_target/l1_sparsity_coef: see
    # docs/research/train_tile_curriculum.rst:train_tile_curriculum.cli_spectral_norm_vs_l1_sparsity.
    spectral_norm_target = float(sys.argv[17]) if len(sys.argv) > 17 else None
    l1_sparsity_coef = float(sys.argv[18]) if len(sys.argv) > 18 else 0.0

    state_width = EMBED_WIDTH * COLUMN_NEURONS
    mlp_hidden = state_width * MLP_HIDDEN_MULT

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    # RNG seeding (both the stochastic-rounding RNG and model construction):
    # see docs/research/train_tile_curriculum.rst:train_tile_curriculum.rng_seeding_stochastic_and_model_construction.
    if hasattr(_cpu, "seed_fp4_stochastic_rng"):
        _cpu.seed_fp4_stochastic_rng(seed)
    model_rng = np.random.default_rng(seed)
    model = ToyTileRecurrenceRealFP4(
        VOCAB,
        EMBED_WIDTH,
        COLUMN_NEURONS,
        mlp_hidden,
        NUM_TILES,
        MAX_WEIGHTS_PER_LAYER,
        num_cpus=NUM_CPUS,
        disldo_cls=ARMS[arm],
        use_energy=use_energy,
        energy_kwargs=ENERGY_KWARGS if use_energy else None,
        use_attention=use_attention,
        o_proj_depth=o_proj_depth,
        rng=model_rng,
        clip_range=clip_range,
        magnitude_penalty_coef=magnitude_penalty_coef,
        spectral_norm_target=spectral_norm_target,
        l1_sparsity_coef=l1_sparsity_coef,
    )
    opt = AdamOptimizer()
    embed_table = rng.randn(VOCAB, EMBED_WIDTH).astype(np.float32) * 0.3

    value_bits = estimate_value_bits(arm, state_width, EMBED_WIDTH, VOCAB, MAX_WEIGHTS_PER_LAYER, use_attention)
    print(
        f"# arm={arm} use_energy={use_energy} use_attention={use_attention} "
        f"train_steps={train_steps} checkpoint_every={checkpoint_every} seed={seed} "
        f"vocab={VOCAB} num_tiles={NUM_TILES} embed_width={EMBED_WIDTH} "
        f"column_neurons={COLUMN_NEURONS} state_width={state_width} "
        f"max_weights={MAX_WEIGHTS_PER_LAYER} o_proj_depth={o_proj_depth} "
        f"peak_lr={PEAK_LR} est_value_bits={value_bits} "
        f"(~{value_bits / 8:.0f} bytes) use_synaptogenesis={use_synaptogenesis} "
        f"seq_len={SEQ_LEN_START}->{SEQ_LEN_MAX} (+1/{steps_per_stage} steps)",
        flush=True,
    )

    t0 = time.time()
    for step in range(1, train_steps + 1):
        seq_len = current_seq_len(step, steps_per_stage)
        lr = lr_schedule(step, train_steps, PEAK_LR, WARMUP_STEPS)
        tokens, pairs = generate_copy_sequence(rng, VOCAB, seq_len)
        targets = dict(pairs)
        M = np.zeros((NUM_TILES, state_width), dtype=np.float32)
        for i in range(seq_len):
            window = _build_tile_window(embed_table, tokens, i, NUM_TILES, COLUMN_NEURONS)
            M, logits, aux = model.step(window, M, lr)
            if i in targets:
                loss = cross_entropy_sum(logits, [(NUM_TILES - 1, targets[i])])
                if aux is not None:
                    loss = loss + aux
                loss.backward()
                clip_grad_norm_(model.parameters_for_optimizer(), MAX_GRAD_NORM)
                opt.step(model.parameters_for_optimizer(), lr=lr)

        if use_synaptogenesis:
            _maybe_synaptogenesis(model)

        if step % checkpoint_every == 0:
            acc = evaluate(model, rng, embed_table, seq_len)
            elapsed = time.time() - t0
            print(
                f"step={step:>7}  seq_len={seq_len}  acc={acc:.4f}  "
                f"({elapsed:.0f}s elapsed, {elapsed / step:.4f}s/step)",
                flush=True,
            )

    print(
        f"# DONE arm={arm} use_energy={use_energy} use_attention={use_attention} ({time.time() - t0:.0f}s total)",
        flush=True,
    )


if __name__ == "__main__":
    main()
