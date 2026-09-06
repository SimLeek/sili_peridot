``stochastic_stability_vs_scale_sparsity.py`` research notes
================================================================

Companion doc to ``scripts/stochastic_stability_vs_scale_sparsity.py``. Source
comments point back here by anchor ID (``*ID:* `` marker under each heading
below). See ``sili__new/docs/research/sparse_rnn.rst`` for the pattern this
follows (semantic dotted anchor IDs, visible ID markers via the ``*ID:*``
line since plain ``.. _id:`` targets alone render invisible on GitHub,
frozen code snippets on real-bug/non-obvious-derivation sections only).

.. _stochastic_stability_vs_scale_sparsity.research_question_and_ooc_task_design:

Research question and why this uses an out-of-context task
--------------------------------------------------------------------------

*ID:* ``stochastic_stability_vs_scale_sparsity.research_question_and_ooc_task_design``

At what model scale (``state_width``) or input/activation sparsity does
stochastic rounding's inherent per-step noise become more "stable to work
with" -- and stop harming PEAK TRAINED accuracy as much as it might at small
scale/low input sparsity?

Uses an OUT-OF-CONTEXT task (``seq_len = NUM_TILES + OOC_MARGIN``), not the
in-context ``seq_len==NUM_TILES`` setup other probes in this investigation
used. ``generate_copy_sequence``'s own docstring and ``_build_tile_window``'s
sliding-window math both confirm: at ``seq_len<=NUM_TILES``, the tile window
at the final (query) position reaches all the way back to position 0 (the
key) directly, so the task is solvable from local attention alone with NO
real dependency on the carried recurrent state ``M`` -- a genuinely
"in-context" task. The stochastic-rounding issues this whole investigation
is chasing showed up on tasks that specifically REQUIRE recurrence (state
carried across tile boundaries beyond the local window's reach), not this
in-context setup -- so this grid needs ``seq_len > NUM_TILES``, forcing the
key to be carried forward through ``M`` past where the local window can see
it directly. Per project convention (grow past the in-context ceiling ONE
increment at a time, not a big out-of-context jump), ``OOC_MARGIN`` is kept
small (+2) rather than testing some large out-of-context gap.

.. _stochastic_stability_vs_scale_sparsity.stuck_weights_followup_motivation:

Motivation: direct follow-up to the stuck-weights investigation
--------------------------------------------------------------------------

*ID:* ``stochastic_stability_vs_scale_sparsity.stuck_weights_followup_motivation``

Direct follow-up to the stuck-weights investigation: the (row,col)-keyed
``check_stuck_weights`` redesign (``model/eval_stuck_weights.py``) just
confirmed stochastic rounding produces real nonzero movement on already-live
synapses where deterministic rounding is bit-exact-frozen
(``mean_delta_w=0.002259`` vs ``0.000000``, toy scale, 800 steps) -- so
stochastic rounding is doing real, useful work. But it's also, by
construction, noisy (every near-threshold update has some chance of
rounding either way each step). This script measures whether that noise is
a real cost in FINAL ACCURACY (not just a per-synapse curiosity) and
whether the cost shrinks as ``state_width`` or INPUT sparsity changes.

.. _stochastic_stability_vs_scale_sparsity.input_density_not_connectivity_density:

Corrected axis: input/activation sparsity, not connectivity/weight sparsity
--------------------------------------------------------------------------

*ID:* ``stochastic_stability_vs_scale_sparsity.input_density_not_connectivity_density``

IMPORTANT, corrected per direct feedback: an earlier version of this script
swept CONNECTIVITY/WEIGHT sparsity (``dense=False``, ``max_weights`` as a
fraction of ``n_in*n_out`` on the layer's own synapses) under the label
"density" -- that's a different axis from what was actually asked ("input
sparsity"). Connectivity now stays FULLY DENSE (``dense=True``) at every
grid point; the swept axis here is genuine INPUT/ACTIVATION sparsity
instead: each vocab token's embedding is given a fixed random subset of
active (nonzero) dimensions at the requested ``input_density`` fraction, so
the SIGNAL a synapse ever sees is sparse, independent of how densely
connected the layer itself is. This is the intended "sparse input" reading
-- per direct correction, NOT weight sparsity. Only vary the one intended
axis per comparison: connectivity density is now held fixed rather than
co-varying with input sparsity.

.. _stochastic_stability_vs_scale_sparsity.two_axis_grid_and_gap_metric:

Grid design: two independent axes, both compared against their own baseline
--------------------------------------------------------------------------

*ID:* ``stochastic_stability_vs_scale_sparsity.two_axis_grid_and_gap_metric``

Two axes, both swept independently against the SAME deterministic baseline
at each grid point (comparing stochastic to the deterministic arm run at
that exact width/input-sparsity, not to a single fixed baseline):

- ``state_width``, via ``(embed_width, column_neurons)`` pairs, same
  convention as ``weight_update_magnitude_vs_width.py``.
- ``input_density``: fraction of each vocab token's embedding dimensions
  that are nonzero (1.0 = fully dense input, same as before this axis
  existed; lower = sparser input signal, same active dims reused every
  time that token appears, so it's a fixed per-token sparse representation,
  not per-step dropout noise).

For each grid point, ``N_SEEDS`` independent seeds per arm (deterministic,
stochastic), each trained for ``N_STEPS`` then evaluated on held-out
copy-task accuracy (real metric, not a proxy). Reports per grid point:

- ``stoch_mean``, ``stoch_std`` (variance across seeds -- the "how noisy"
  signal)
- ``det_mean`` (deterministic baseline at the same grid point)
- ``gap = det_mean - stoch_mean`` (positive = stochastic underperforms;
  shrinking toward 0 as width/input_density increases is the "becomes
  safe" signal the user asked about)

Kept fast on purpose (per-project convention: these diagnostics should run
in 5-10 minutes, not hours) -- small grid, few seeds, short training.

.. _stochastic_stability_vs_scale_sparsity.sparse_embed_table_fixed_mask_design:

``_sparse_embed_table``: a fixed per-token sparse mask, seeded independently
--------------------------------------------------------------------------

*ID:* ``stochastic_stability_vs_scale_sparsity.sparse_embed_table_fixed_mask_design``

Dense random embedding table, then (if ``input_density<1.0``) each vocab
row gets a FIXED random subset of active dims zeroed out to the requested
density -- same active dims every time that token is looked up, so this is
a stable sparse input REPRESENTATION, not per-step dropout noise. Mask RNG
is seeded independently of ``task_rng`` (token sequence generation) so the
two axes don't entangle.
