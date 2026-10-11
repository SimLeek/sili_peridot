import sys

sys.path.insert(0, ".")
import scripts.train_mqar_curriculum as m

r = m.train_curriculum(
    "fp32",
    3000,
    1000,
    0.015,
    16,
    10,
    additive_rank=1,
    dynamic_rank_control=True,
    rank_grace_period_steps=50,
    embed_width=36,
    wrong_streak_threshold=100000000,
    k_first_target=3,
    log_every=25,
)
print(
    f"FINAL {r['final_vocab']} {r['final_k']} {r['final_phase']} steps/sec {r['steps_per_sec']:.2f} elapsed_s {r['elapsed_s']:.0f}"
)
print("PEAK", r["peak_stage"])
print("STAGE_HISTORY", r["stage_history"][:10])
