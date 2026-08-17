"""Graduated wall-clock test ladder for the genuine zero-weight-init
design (empty CSR + synaptogenesis growth, sili/sparse_rnn.py's
_preseed_empty + DISLDOLayer(empty_init=True) + OriginalArchModel.
synaptogenesis_all()) -- per direct instruction: build a doubling
ladder (~1, 2, 4, 8, 16, 32, 64 minutes), each tier validating
something the previous, faster tier couldn't.

IMPORTANT, per direct correction: step counts below are NOT derived by
extrapolating an early-run per-step rate to a target duration -- with
synaptogenesis growing the network's live connectivity over the course
of a run (and topology changes shifting cache/branch-prediction
behavior even at matched sparsity), per-step cost is not constant, so
that kind of estimate is unreliable. Each tier is run to COMPLETION
(never interrupted mid-run for a time budget -- a truncated run doesn't
give a valid result), and its real wall-clock time is measured and
reported honestly. If a tier's actual time lands far from its target
bucket, that's a signal to shrink ITS OWN config (fewer steps/seeds)
for the next run, not to cut a run short. The tier numbers are labels
picked to roughly reach 1/2/4/8/... minutes on THIS machine, calibrated
by iterating actual measured runs.

Tiers 1/2/4 are cheap sanity/reproducibility checks; tier 8 is the
first real apples-to-apples comparison against the established
landmark baselines (baseline/baseline_energy, see l1_sparsity_probe.py
module docstring). Tiers 16/32/64 are reserved for the recurrent-scale
model (ToyTileRecurrenceRealFP4/train_tile_curriculum.py) -- empty_init
is NOT YET wired there, that's the next follow-up once this ladder
confirms the mechanism works at toy scale.

Usage: PYTHONPATH=<sili_peridot root> python scripts/empty_init_synaptogenesis_ladder.py <tier>
  tier in {1, 2, 4, 8}
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
from scripts.l1_sparsity_probe import OriginalArchModel, run, evaluate


def read_nnz(model):
    return {name: sum(d._c.nnz for d in layer.digits)
            for name, layer in (("q", model.q_proj), ("k", model.k_proj),
                                 ("v", model.v_proj), ("o", model.o_proj),
                                 ("lm", model.lm_head))}


def tier1_smoke(seed=1000, n_steps=8000):
    """~1 min, 1 seed: does the full model grow real synapses and have
    them escape weight=0, with no crashes, under empty_init=True?"""
    t0 = time.time()
    model = OriginalArchModel(seed, dense=False, o_proj_coef=0.0, all_layer_coef=0.0,
                              l1_sparsity_coef=0.05, use_energy=False, empty_init=True,
                              synap_k=3)
    last_accs, skips, total, avg_step_time = run(model, n_steps, seed, verbose=False)
    nnz = read_nnz(model)
    acc = evaluate(model, 50, seed, verbose=True)
    dt = time.time() - t0
    print(f"[tier1] seed={seed} n_steps={n_steps} wall={dt:.1f}s "
          f"nnz={nnz} eval_acc={acc:.4f} skips={skips}/{total}", flush=True)
    return dt


def tier2_multiseed(seeds=(1000, 1001, 1002), n_steps=5500):
    """~2 min, 3 seeds: is escape-from-zero reproducible, not a
    single-seed fluke?"""
    t0 = time.time()
    for seed in seeds:
        model = OriginalArchModel(seed, dense=False, o_proj_coef=0.0, all_layer_coef=0.0,
                                  l1_sparsity_coef=0.05, use_energy=False, empty_init=True,
                                  synap_k=3)
        run(model, n_steps, seed, verbose=False)
        nnz = read_nnz(model)
        acc = evaluate(model, 50, seed)
        print(f"[tier2] seed={seed} nnz={nnz} eval_acc={acc:.4f}", flush=True)
    dt = time.time() - t0
    print(f"[tier2] total wall={dt:.1f}s ({len(seeds)} seeds)", flush=True)
    return dt


def tier4_learning_signal(seeds=(1000, 1001, 1002), n_steps=12000):
    """~4 min, 3 seeds, full curriculum length (STEPS_PER_STAGE=500 x
    NUM_TILES=4 -> context complete by step ~1500): does accuracy
    exceed a genuinely untrained (0-step) reference -- real learning
    signal, not just weight movement."""
    t0 = time.time()
    results = []
    for seed in seeds:
        untrained = OriginalArchModel(seed, dense=False, o_proj_coef=0.0, all_layer_coef=0.0,
                                      l1_sparsity_coef=0.05, use_energy=False, empty_init=True,
                                      synap_k=3)
        untrained_acc = evaluate(untrained, 100, seed)
        model = OriginalArchModel(seed, dense=False, o_proj_coef=0.0, all_layer_coef=0.0,
                                  l1_sparsity_coef=0.05, use_energy=False, empty_init=True,
                                  synap_k=3)
        run(model, n_steps, seed, verbose=False)
        trained_acc = evaluate(model, 100, seed)
        results.append((untrained_acc, trained_acc))
        print(f"[tier4] seed={seed} untrained_acc={untrained_acc:.4f} "
              f"trained_acc={trained_acc:.4f} delta={trained_acc-untrained_acc:+.4f} "
              f"nnz={read_nnz(model)}", flush=True)
    dt = time.time() - t0
    mean_delta = np.mean([t - u for u, t in results])
    print(f"[tier4] total wall={dt:.1f}s mean_delta={mean_delta:+.4f}", flush=True)
    return dt


def tier8_landmark_scale(seeds=(1000, 1001, 1002, 1003, 1004), n_steps=15000):
    """~8 min, 5 seeds x 15000 steps -- matches the established landmark
    scale exactly (see l1_sparsity_probe.py module docstring's own
    baseline/baseline_energy numbers), first real apples-to-apples
    comparison of empty_init+synaptogenesis against those."""
    t0 = time.time()
    accs = []
    for seed in seeds:
        model = OriginalArchModel(seed, dense=False, o_proj_coef=0.0, all_layer_coef=0.0,
                                  l1_sparsity_coef=0.05, use_energy=False, empty_init=True,
                                  synap_k=3)
        run(model, n_steps, seed, verbose=False)
        acc = evaluate(model, 100, seed)
        accs.append(acc)
        print(f"[tier8] seed={seed} eval_acc={acc:.4f} nnz={read_nnz(model)}", flush=True)
    dt = time.time() - t0
    print(f"[tier8] total wall={dt:.1f}s mean_acc={np.mean(accs):.4f} "
          f"std={np.std(accs):.4f} accs={[f'{a:.4f}' for a in accs]}", flush=True)
    return dt


TIERS = {"1": tier1_smoke, "2": tier2_multiseed, "4": tier4_learning_signal, "8": tier8_landmark_scale}

if __name__ == "__main__":
    tier = sys.argv[1] if len(sys.argv) > 1 else "1"
    fn = TIERS[tier]
    t0 = time.time()
    fn()
    print(f"=== tier {tier} grand total wall={time.time()-t0:.1f}s ===", flush=True)
