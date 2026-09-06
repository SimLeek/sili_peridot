``eval_superposition.py`` research notes
===========================================

Companion doc to ``model/eval_superposition.py``. Source comments point back
here by anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers, frozen code snippets on
real-bug/non-obvious-derivation sections only).

.. _eval_superposition.module_overview:

Module purpose: does FP4 quantization structurally cap superposition below float32
--------------------------------------------------------------------------------------

*ID:* ``eval_superposition.module_overview``

Can FP4-quantized weights pack more independent SPARSE features than they
have raw dimensions -- the same "superposition" phenomenon Anthropic's Toy
Models of Superposition (Elhage et al. 2022) found in dense float weights --
or does quantization structurally cap this below what float32 achieves at
the same width? Real dense transformer weights are known to rely heavily on
superposition to represent far more features/circuits than their raw
dimension count, so this is directly relevant to the actual MiniCPM5-FP4
conversion goal, more so than a synthetic rank/recall test.

Distinct axis from ``eval_rank_floor.py`` (does gradient descent reach a
rank the architecture is theoretically capable of) and from a
recurrent/MQAR-style capacity test (temporal capacity over TIME, a separate
planned test) -- this is about SPATIAL packing precision: can a bottleneck
of width ``hidden_width`` represent ``n_features`` independent SPARSE
directions at an acceptable reconstruction cost, and does that cost degrade
under FP4 relative to float32 at the SAME width/sparsity.

Setup follows the original Toy Models of Superposition paper directly:
linear encoder (``n_features -> hidden_width``, no nonlinearity), ReLU
decoder (``hidden_width -> n_features``), trained to reconstruct sparse
input vectors under an importance-weighted MSE loss (earlier features
matter more -- ``feature_importance``'s own geometric decay). Superposition
is the STRATEGY of packing multiple features into non-orthogonal directions
in the bottleneck, tolerable because features are sparse (they rarely
co-activate, so the resulting interference is rare too) -- dense input
(``density=1.0``) gives the model no reason to ever attempt it, sparser
input makes it increasingly worthwhile, which is why the real comparison
sweeps density rather than testing at one fixed sparsity level.

.. _eval_superposition.sample_sparse_features_convention:

``sample_sparse_features``: matching the paper's own sampling convention
------------------------------------------------------------------------------

*ID:* ``eval_superposition.sample_sparse_features_convention``

One sparse feature vector: each of ``n_features`` independently active with
probability ``density``, magnitude ~ Uniform(0,1) when active, 0 otherwise
-- matches Toy Models of Superposition's own convention exactly, so results
here are comparable to the paper's own findings rather than an arbitrary
reinterpretation.

.. _eval_superposition.feature_importance_geometric_decay:

``feature_importance``: why a geometric decay, not uniform weighting
------------------------------------------------------------------------

*ID:* ``eval_superposition.feature_importance_geometric_decay``

Geometric per-feature importance weighting (``I_i = decay**i``) -- earlier
features matter more, matching Toy Models of Superposition's own
convention. Without some importance gradient, every feature is
interchangeable and there's nothing to prioritize when interference under a
tight bottleneck is unavoidable -- the model would have no principled basis
for choosing which features to represent well and which to sacrifice.

.. _eval_superposition.no_superposition_baseline_derivation:

``no_superposition_baseline``: a closed-form EY-floor-style reference number
----------------------------------------------------------------------------------

*ID:* ``eval_superposition.no_superposition_baseline_derivation``

Closed-form expected weighted loss for the best NO-superposition strategy:
with only ``hidden_width`` orthogonal directions available, perfectly
represent the ``hidden_width`` most-important features (their own
importance-sorted order -- ``importance`` is assumed already sorted
descending, matching ``feature_importance``'s own geometric-decay
convention) and give up entirely on the rest (predict 0 for them, ReLU's
own natural "no signal" output).

A dropped feature ``i`` is 0 with probability ``(1-density)`` and
Uniform(0,1) with probability ``density``, so
``E[x_i**2] = density * E[U**2] = density/3`` (``U ~ Uniform(0,1)``).

.. code-block:: python

   # E[x_i^2] for a dropped feature i:
   #   0 with probability (1 - density)
   #   Uniform(0,1)^2 with probability density, E[U^2] = 1/3
   # => E[x_i^2] = density / 3
   dropped = importance[hidden_width:]
   baseline_loss = float(np.sum(dropped) * density / 3.0)

This is the EY-floor-style reference number for this module: a real,
principled lower bound on achievable loss WITHOUT superposition, so "did
training beat this" is a rigorous claim, not just "did the loss go down".

.. _eval_superposition.measure_superposition_single_forward_no_amplification:

``measure_superposition``: one forward pass per layer per step, no amplification risk
-------------------------------------------------------------------------------------------

*ID:* ``eval_superposition.measure_superposition_single_forward_no_amplification``

Trains ``encoder`` (``n_features -> hidden_width``) and ``decoder``
(``hidden_width -> n_features``) -- each any object with ``.forward(x:
Tensor, learning_rate) -> Tensor`` and (for a plain, non-quantized layer)
``.trainable_params() -> List[Tensor]`` -- to reconstruct randomly sampled
sparse feature vectors through a ReLU decoder, matching Toy Models of
Superposition's exact architecture. Pass ``opt=None`` (both encoder and
decoder ignore ``trainable_params``, i.e. real DISLDOLayer-family layers)
for arms whose weights update inline during ``backward()``; pass a real
optimizer + ``opt_step`` closure for the plain-float sanity arm
(``FullRankDenseLayer`` encoder/decoder).

Each step samples ONE sparse feature vector, calls ``encoder.forward`` then
``decoder.forward`` exactly once each -- a normal single-pass 2-layer
network, unlike ``eval_rank_floor.py``'s per-basis-vector regression, which
called each layer N times within one accumulated ``backward()`` and had to
work around a resulting update-amplification bug (see
``eval_rank_floor.measure_rank_floor_per_example_updates``). No such risk
here, since each layer is genuinely only touched once per step.

.. _eval_superposition.measure_superposition_lr_decay_reuse:

``lr_decay``: reusing eval_rank_floor's RMSprop-divergence fix
--------------------------------------------------------------------

*ID:* ``eval_superposition.measure_superposition_lr_decay_reuse``

See ``eval_rank_floor.py``'s own ``measure_rank_floor`` docstring (and
``eval_rank_floor.measure_rank_floor_lr_decay_rmsprop_divergence`` in
``docs/research/eval_rank_floor.rst``) for why this matters for
DISLDOLayer-family arms specifically: late-training RMSprop divergence once
per-synapse ``ci`` decays to match a small residual gradient. Same
mechanism, same fix, reused here rather than re-derived.
