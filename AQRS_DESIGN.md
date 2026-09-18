# AQRS: Adaptive Quantized Residual Subspace

Design note capturing a formal specification (12 theorems, worked out in an
earlier session with Kimi.ai) for why sili's quantized layers need BOTH a
multiplicative and an additive low-rank correction on top of the quantized
base weight, not just one. Recorded here rather than left in chat history
because it directly explains an empirical finding from this session (fp8's
MQAR collapse) and independently re-derives a bug this session already
fixed (the never-zero live-quantize work). Implementation deferred until
the system package manager finishes recovering -- this is planning/
documentation only, no C++ or build changes made yet.

## The decomposition

    W_eff = W_q (had) M(theta_s)  +  A(theta_o)

- `W_q`: the b-bit quantized base weight (FP4/FP8 storage as it exists today).
- `(had)`: elementwise (Hadamard) product.
- `M(theta_s) = ones + sum_k gamma_s_k * u_k v_k^T`: the MULTIPLICATIVE
  branch, rank `r_s`.
- `A(theta_o) = sum_k gamma_o_k * u_k v_k^T`: the ADDITIVE branch, rank `r_o`.

## Where sili already stands

**The multiplicative branch already exists.** `value_scale` (per-row) and
`output_scale` (per-column), already generalized to rank-N (tasks #245/246,
`RMSpropScalePolicy`, the rank-2 scale envelope work), is exactly
`M(theta_s)` above. Theorem 2 (Hadamard Rank-1 Factorization) is the formal
justification for why this can be applied without ever materializing a
dense m x n matrix:

    (u v^T (had) W) X  =  u (had) (W (v (had) X))

i.e. a rank-1 multiplicative correction costs one extra elementwise scale
before and after the existing matmul, not an extra dense matrix multiply.
This is precisely what value_scale/output_scale already do.

**The additive branch does not exist as a kernel feature.** There is no
`A(theta_o)` anywhere in disldo_forward/disldo_backward today. This is the
real gap, and per Theorem 4 below, it isn't a "nice to have" -- it's a
structural impossibility, not a training-difficulty problem.

## Why this matters: Theorem 3 + Theorem 4 (structural deadness / additive necessity)

If a live weight's quantized code lands on the sentinel "zero" value at
position (i,j) -- `W_q[i,j] = 0` -- then for ANY multiplicative parameters,
at ANY rank:

    W_eff[i,j] = W_q[i,j] * M(theta_s)[i,j] = 0 * M(theta_s)[i,j] = 0

Every partial derivative of `W_eff[i,j]` with respect to every
multiplicative parameter is *exactly* zero (Theorem 3) -- not small, not
slow to train, structurally zero. No amount of multiplicative rank fixes
this. Only an additive term can write a nonzero value there (Theorem 4).

**This is a strong, independently-derived explanation for the fp8
"input-independent collapse" found via direct logit dump earlier this
session** (two different MQAR eval sequences, different correct answers,
byte-identical top-5 logits). Without an additive branch, the model has no
way to express the *specific* per-key value at all -- it can only rescale
what's already representable via W_q, so it converges to modeling the
marginal distribution over plausible outputs (what W_q + multiplicative
rescaling CAN express) instead of the exact per-key association it needs
(what only an additive branch could write).

## Cross-validation: Theorem 6/7 = the never-zero live-quantize fix

Independently, this document proves (Theorem 6) that if the quantization
grid *allows* an exact zero code, the system has a stable fixed point where
the quantized weight sits at 0 forever while an additive-style correction
(if present) chases the target alone -- a limit cycle: correction grows to
compensate, stochastic rounding kicks the code off zero, overshoot, correction
shrinks, code falls back to zero, repeat. Theorem 7 proves a **no-zero grid**
(every code has |q| >= eps > 0) eliminates this trap entirely -- gradient
flow to the quantized weight is never structurally zero, so weight and
correction co-adapt smoothly instead of fighting each other.

This is *exactly* the never-zero live-quantize fix already built and shipped
this session (fp4_quantize_live/fp8_quantize_live, tasks #248-258): live
synapses can never quantize to the zero code, matching Theorem 7's
prescription precisely. Good independent confirmation that fix was the
right one, not just an empirically-motivated patch.

## Implementation target: fused forward pass (Theorem 11)

    Y = W_q X
        + sum_k gamma_s_k * [ u_s_k (had) (W_q (v_s_k (had) X)) ]   # multiplicative (already have this via value_scale/output_scale)
        + U_o diag(gamma_o) V_o^T X                                  # additive (NEW)

The additive term is a genuine low-rank matmul (`U_o` is m x r_o, `V_o` is
n x r_o) -- cheap relative to the base matmul as long as `r_o << min(m,n)`,
and composes with the existing scattered/block4 dispatch without touching
either. This should live in disldo_forward/disldo_backward as real C++,
not be composed after the fact in Python, per direct instruction -- Python
composition works for demonstrating usage but doesn't give the fused,
single-pass kernel this formula describes.

**On `gamma`/`diag(gamma)` and non-square layers** (see conversation --
user directly asked how this works when n_inputs != n_outputs): `diag(gamma)`
is ALWAYS `r x r` (the rank, whatever `r_o`/`r_s` happens to be that step),
never `m x m` or `n x n` -- it lives entirely in the small "rank space"
between the down- and up-projection, completely decoupled from the layer's
actual input/output widths. Tracing the fused computation makes this
concrete: `p = U_o^T @ X` projects the m-dim input down to an r_o-dim
vector; `diag(gamma_o) @ p` is JUST `p[k] *= gamma_o[k]` elementwise on
that small r_o-dim vector; `V_o @ (that)` projects back up to n-dim. `U_o`'s
columns have length m (n_inputs), `V_o`'s columns have length n
(n_outputs), `gamma_o` is a plain length-r_o vector -- none of the three
pieces care whether m==n. Exactly analogous to the middle diagonal matrix
Sigma in an SVD (`W = U Sigma V^T`), which is always `r x r` regardless of
whether `W` itself is square. NOTE: this project's own #275-278
implementation currently does NOT include a separate gamma (matches the
existing multiplicative branch's own two-plain-vector convention, no
redundant capacity needed at fixed rank) -- gamma is reintroduced
specifically for #273 (dynamic rank control), where it's not about
capacity but about having one clean scalar per channel to prune/grow
against (see #273's own updated description).

## Next: dynamic rank control (Section 8 of the source doc, not started)

The intended end state doesn't fix `r_s`/`r_o` -- both ranks should grow and
shrink based on measurable signal, with proven stability properties. This
section was under-specified in the first pass of this doc (see conversation
-- user directly asked "did you get the exact signals and how to avoid
noise" and the honest answer was no, not fully). Corrected here.

**Exact trigger conditions (Theorem 10)** -- per branch, per rank channel
`i`, with `gamma in R^r` the channel's own scale parameter(s):

    C_i = |gamma_i| / ||gamma||_1        # this channel's share of the group's total L1 mass

    Apoptosis (shrink):   A(gamma_i) = (|gamma_i| < tau_death)  AND  (C_i < tau_death)
    Neurogenesis (grow):  N(gamma, grad) = (min_j |gamma_j| > tau_active)  AND  (||grad||_F > theta)

    with tau_death < tau_active (the hysteresis gap itself)

Apoptosis needs BOTH the channel's absolute magnitude and its relative
share of the group below threshold -- stops a channel being killed just
for being smaller than siblings when the whole group is legitimately
small. Neurogenesis needs every EXISTING channel already pulling its
weight (`min_j |gamma_j| > tau_active`) AND real leftover gradient
pressure (`||grad||_F > theta`) -- i.e. don't grow if there's already an
idle/redundant channel, and don't grow if there's nothing left to explain.

**Direction for a new channel, once neurogenesis triggers**: Theorem 9
says align the new rank-1 channel with the top singular vector of the
residual `R = W_target - W_approx`, provably reducing error by exactly
`sigma_1(R)^2`. STILL UNRESOLVED: `W_target` is only observable in the
source document's synthetic layer-fitting tests (a known ground-truth
matrix); real end-to-end task training has no such target, only a task
loss gradient. Practical proxy needed -- candidate is sili's existing
`neuron_grad_accum`/importance signal (accumulated gradient direction is
a reasonable descent-direction stand-in for "residual direction," but
this is an assumption, not something the source proof covers -- verify
before relying on it).

**TODO/open question (per direct user instruction, see conversation):**
seeding a new channel once at grow-time (LoRA-style: one side small-
nonzero, the other left at its zero default, confirmed necessary in
sili__new/tests/unit/test_aqrs_rank_growth_shrink.cpp -- both the
multiplicative and additive branches hit a genuine symmetric zero-init
deadlock otherwise) breaks symmetry ONCE, at birth. The multiplicative and
additive branches are effectively stacked correction "layers" on top of
W_q -- for LIFELONG/continual learning specifically (not a single bounded
training run), a one-time random/aligned seed may not be enough: two
channels could still drift back into a degenerate/symmetric relationship
over a long horizon, or a channel revived after apoptosis+later regrowth
may need FRESH symmetry-breaking, not just its birth-time seed. sili's
existing EnergyDynamics noise mechanism (per-neuron stochastic
firing/wake, see [[project_energy_wake_gated_annealing_idea]] and
[[project_energy_drive_activation_cost_balance]]) is a candidate ongoing
source of exactly this kind of continual symmetry-breaking pressure,
applied BETWEEN channels/layers rather than only at construction --
unexplored, not designed yet. Revisit once #273's fixed-point-based
apoptosis/neurogenesis is working and lifelong (not just bounded-horizon)
training is actually being tested.

**Noise mitigation -- CORRECTED per direct instruction (see conversation):
the source spec's own "every N steps" protocol (section 10.2) is REJECTED
here, not adopted.** Reasoning: periodic checking on a raw/instantaneous
value isn't actually a noise filter, it's a "luck filter" -- it doesn't
integrate/average anything, it just samples less often, so whichever
value happens to land on the checkpoint step (itself possibly a noisy
outlier) determines the decision. EMA is the part that does genuine
averaging; periodicity alone does none of that work. Separately, batching
the decision into N-step bursts is a real-time/latency problem in a
system with no fixed training horizon (matches this project's existing
move away from step-scheduled control, e.g. LR already went from
step-based to accuracy-driven this session) -- prefer reacting every step
wherever possible.

Corrected design -- two mechanisms, checked/updated EVERY STEP, not N:

1. **L1 penalty creates a genuine attracting fixed point at exactly
   `gamma=0`** (Theorem 8), not just "gets small." Matters specifically
   for noise: without a real fixed point, ordinary gradient noise keeps a
   dying channel jittering around some small-but-nonzero value
   indefinitely, and a bare magnitude threshold would flicker every time
   noise crossed the line. With the fixed point, "small" is a settled
   state, not a moving target.
2. **EMA-smooth `|gamma_i|`, `C_i`, and `||grad||_F` (or its proxy) every
   single step** -- same `decay=0.98`-style pattern already used for
   `loss_ema`/`acc_ema` in the MQAR curriculum (train_mqar_curriculum.py)
   -- and evaluate `A(gamma_i)`/`N(gamma,grad)` against the EMA values
   every step too, not gated behind a step counter. This is the actual
   noise filter (the source spec doesn't specify it at all -- the
   theorems are stated for deterministic/noise-free gradient flow, not
   real SGD).

Cost of checking every step should be kept low by piggybacking on
whatever's ALREADY touched/sparse that step -- sili's per-synapse
importance/ci updates already only run on rows touched by the current
batch (synap_row_step), not a full dense sweep, so the EMA update +
threshold check for a rank channel should ride along with that same
already-sparse per-step work rather than requiring a separate periodic
maintenance pass over the whole matrix.

**The hysteresis gap** (`tau_death < tau_active`, distinct thresholds,
Theorem 10) is UNCHANGED and still needed -- it solves a different
problem (stopping a channel from immediately regrowing the instant it's
pruned) than the noise-filtering above, and stays valid regardless of
per-step vs per-N-step checking. Same anti-oscillation idea already used
for the MQAR curriculum's level-up/level-down mechanism (task #272), just
applied to rank instead of curriculum stage.

This is real follow-up scope, not something to fold into the additive-branch
kernel work above -- get the fixed-rank additive branch working and
validated first (matches the source document's own recommendation: "get
the fixed-rank additive branch working and validated first").

## Documented limitation, not a test (per prior decision)

If the quantization residual `W - W_q` has a genuinely flat singular-value
spectrum (no low-rank structure at all, in either the additive or the
Hadamard-scaled multiplicative sense), both branches need rank
`~min(m,n)` to help at all, at which point there's no compression
advantage over fp32 -- see the source document's Theorem/Corollary on
"uniform residuals." Real transformer weight matrices empirically have
heavy-tailed SVD spectra (a few large singular values, long decay), so
this is a documented assumption, not expected to bite in practice. Already
decided (in the source conversation) to note this rather than build a
dedicated test for it -- the test would mostly demonstrate a synthetic
worst case, not verify anything about the actual implementation's
correctness.

## Status

Planning/documentation only. No C++ changes made. Deferred until the
system package manager recovers (currently mid-upgrade, ~3.7GB/17GB
install). See tasks #268 (superseded/sharpened by this doc) and the new
task for the fused additive-branch kernel.
