``eval_eigenvalues.py`` research notes
==========================================

Companion doc to ``model/eval_eigenvalues.py``. Source comments point back
here by anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers, frozen code snippets on
real-bug/non-obvious-derivation sections only).

.. _eval_eigenvalues.module_overview:

Module purpose: read-only RNN-health diagnostics decoupled from the training-time policy
---------------------------------------------------------------------------------------------

*ID:* ``eval_eigenvalues.module_overview``

Standalone eigenvalue/spectral-norm ("RNN health") diagnostics for
sili_peridot's tile-recurrence layers -- read-only, works on any layer
regardless of whether the model was built with spectral-norm regulation
active. The power-iteration probe here started as an extraction of
``ToyTileRecurrenceRealFP4._spectral_rescale_factor``
(``model/toy_tile_precision_models.py``), which only ever measures a layer
when ``spectral_norm_target`` is set -- so a config like ``baseline`` (no
spectral-norm mechanism at all) previously got zero eigenvalue visibility.
Keeping each piece of code doing its own job (per conversation): the
training-time file keeps its rescaling POLICY, this module holds the pure
measurement, usable on any layer independent of whether anything downstream
regulates it.

Two DIFFERENT quantities are provided here, and they must not be conflated:

- ``SpectralProbe``/``track_spectral_health``: cheap, iterative, ONE extra
  forward pass per measurement -- suitable for tracking every N steps during
  real training. Despite being modeled on "power iteration for the dominant
  singular value," this is FORWARD-ONLY iteration
  (``u_{k+1} = layer(u_k)/||layer(u_k)||``, no transpose step), which only
  converges to the true top singular value when the underlying map is
  symmetric. For a generic (non-symmetric) weight matrix -- the normal case
  here -- the dominant eigenvalue is generically a COMPLEX pair, and this
  instead approximates something close to the SPECTRAL RADIUS (max
  |eigenvalue|), not the spectral norm. Confirmed directly: for a random
  16x16 Gaussian matrix this iteration converges to ~3.43 while the true top
  singular value is ~7.13 and the true spectral radius is ~3.49 -- tracking
  the latter, not the former (the same correction is now applied to
  ``_spectral_rescale_factor``'s own docstring, which had this mislabeled
  the same way). Kept because spectral radius actually IS the theoretically
  correct quantity for recurrent-dynamics stability (spectral norm is a
  conservative, often much larger, upper bound on it), and because it's
  cheap enough to run every training step.
- ``exact_spectral_norm``/``exact_spectral_radius``: EXACT (not iterative),
  via a real ``np.linalg.svd``/``eigvals`` call on the layer's reconstructed
  dense weight matrix. Costs ``in_features`` forward passes to rebuild the
  matrix (each layer here is a purely linear map at ``learning_rate=0.0`` --
  no activation function inside a single ``DISLDOLayer``/
  ``TrueMultiDigitLayer`` forward call -- so probing with each standard
  basis vector and collecting the outputs as columns reconstructs W exactly,
  then numpy does exact linear algebra on it), so this is fine for periodic
  diagnostic snapshots (every few hundred steps) but too expensive to call
  every single training step the way ``SpectralProbe`` is designed for.

.. _eval_eigenvalues.spectral_probe_forward_only_iteration:

``SpectralProbe``: persistent power-iteration vector, and why it needs a square layer
-------------------------------------------------------------------------------------------

*ID:* ``eval_eigenvalues.spectral_probe_forward_only_iteration``

One persistent probe vector + EMA state for a single SQUARE layer
(``in_features == out_features`` -- a state-to-state recurrent map, e.g.
``o_proj``). ``.measure(layer)`` reuses the SAME vector across calls (not a
fresh random one each time), which is what makes this power iteration
rather than a single noisy one-shot estimate. It approximates a
spectral-RADIUS-like quantity (max |eigenvalue|), NOT the spectral
norm/top singular value -- see
:ref:`eval_eigenvalues.module_overview` for the full explanation and the
``exact_spectral_norm``/``exact_spectral_radius`` alternative when a precise
number matters more than per-step cost.

ONLY works for square layers: each step feeds the layer's OUTPUT back in as
the NEXT step's input, which requires ``in_features == out_features`` --
confirmed directly (a rectangular layer crashes on the second
``.measure()`` call with a shape mismatch, not just measures something
different). This is exactly why the training-time mechanism this was
extracted from only ever applied it to ``o_proj``. For a genuinely
rectangular layer (e.g. ``lm_head``), use ``exact_spectral_norm`` instead --
real SVD has no such constraint.

``layer`` must expose ``.forward(x: Tensor, learning_rate: float) -> Tensor``
(the ``DISLDOLayer``-family convention used throughout this repo) --
``forward(..., 0.0)`` is the zero-side-effect convention already used
elsewhere (``evaluate()``, the original training-time probe): no
backward/optimizer call, no weight mutation, so this is safe to call at any
point during or after training without disturbing it.

``sigma_raw`` (last unsmoothed estimate) is kept alongside ``sigma_ema`` for
convergence checks.

.. _eval_eigenvalues.probe_layers_square_requirement:

``probe_layers``: rejecting rectangular layers up front with a clear error
-------------------------------------------------------------------------------

*ID:* ``eval_eigenvalues.probe_layers_square_requirement``

Builds one ``SpectralProbe`` per named layer, sized to each layer's own
input width (reads ``.in_features``/``.out_features``, matching every
``DISLDOLayer``-family layer's own attributes). Requires SQUARE layers
(``in_features == out_features``) -- see
:ref:`eval_eigenvalues.spectral_probe_forward_only_iteration` for why; a
rectangular layer here would crash on its second ``measure()`` call, so
this rejects it up front with a clear error instead.

.. _eval_eigenvalues.measure_snapshot_periodic_cadence:

``measure_snapshot``: one measurement pass, matching the periodic-eval cadence
-------------------------------------------------------------------------------

*ID:* ``eval_eigenvalues.measure_snapshot_periodic_cadence``

One measurement pass across every probed layer -- call this periodically
from inside a training loop (same cadence pattern as ``run()``'s own
``periodic_eval``) to build up a ``SpectralTrajectory``.

.. _eval_eigenvalues.track_spectral_health_layers_fn_recomputed:

``track_spectral_health``: generic training-loop wrapper, and why ``layers_fn`` is called fresh each snapshot
------------------------------------------------------------------------------------------------------------------

*ID:* ``eval_eigenvalues.track_spectral_health_layers_fn_recomputed``

Generic training-loop wrapper: calls ``model_step_fn()`` once per step (the
caller's own single training step -- forward+backward+optimizer.step,
whatever that model needs), and every ``probe_every`` steps takes a
spectral-norm snapshot of ``layers_fn()``'s current layers. ``layers_fn`` is
called fresh each snapshot (not once up front) so this works even for
models that replace/grow layers over time (e.g. synaptogenesis) -- probes
themselves are keyed by name and persist across snapshots regardless.

Domain-agnostic like ``find_optimal_lr``'s ``trial_fn`` -- this module knows
nothing about ``OriginalArchModel`` or any specific architecture. See
``tests/test_eval_eigenvalues.py`` for a worked adapter.

.. _eval_eigenvalues.dense_weight_matrix_reconstruction:

``dense_weight_matrix``: exact reconstruction via standard-basis probing
-------------------------------------------------------------------------------

*ID:* ``eval_eigenvalues.dense_weight_matrix_reconstruction``

Exact dense reconstruction of ``layer``'s linear map at
``learning_rate=0.0``, via forwarding each standard basis vector and
collecting the outputs as columns: ``W[:, i] = layer.forward(e_i, 0.0)``.
Only valid for layers that are genuinely LINEAR at lr=0 -- true for every
``DISLDOLayer``-family layer in this repo (no activation function inside a
single forward call; ``TrueMultiDigitLayer``'s residual sum of linear digit
layers is still linear overall). Costs ``in_features`` forward passes; fine
for periodic diagnostic snapshots, NOT something to call every training
step (that's what ``SpectralProbe`` is for).

.. _eval_eigenvalues.exact_spectral_norm_radius_square_requirement:

``exact_spectral_norm`` / ``exact_spectral_radius``: exact SVD/eigenvalue alternative
-------------------------------------------------------------------------------------------

*ID:* ``eval_eigenvalues.exact_spectral_norm_radius_square_requirement``

``exact_spectral_norm`` gives the exact top singular value (``np.linalg.svd``
on the reconstructed dense matrix) -- the quantity ``SpectralProbe`` does
NOT actually measure (see
:ref:`eval_eigenvalues.spectral_probe_forward_only_iteration`).

``exact_spectral_radius`` gives the exact max |eigenvalue|
(``np.linalg.eigvals`` on the reconstructed dense matrix) -- needs a SQUARE
weight matrix (``in_features == out_features``), which every state-to-state
recurrent layer in this codebase's tile-recurrence architecture is
(q/k/v/o_proj all map ``state_width -> state_width``). Raises for a
genuinely rectangular layer (e.g. ``lm_head``, ``embed_width -> vocab``)
since eigenvalues aren't defined for a non-square matrix -- use
``exact_spectral_norm`` there instead.

.. _eval_eigenvalues.exact_spectral_snapshot_companion:

``exact_spectral_snapshot``: one-shot exact companion to ``measure_snapshot``
-------------------------------------------------------------------------------

*ID:* ``eval_eigenvalues.exact_spectral_snapshot_companion``

One-shot EXACT measurement across every named layer -- returns
``{name: {"norm": exact top singular value, "radius": exact spectral
radius, or None if that layer's matrix isn't square}}``. Companion to
``measure_snapshot``'s cheap/approximate per-step version -- use this one
when a precise answer matters more than call cost (e.g. a final
post-training health check, not every-N-steps tracking).
