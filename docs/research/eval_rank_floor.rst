``eval_rank_floor.py`` research notes
============================================

Companion doc to ``model/eval_rank_floor.py``. Source comments point back
here by anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers, frozen code snippets on
real-bug/non-obvious-derivation sections only).

.. _eval_rank_floor.module_overview:

Module purpose: does the scale-rank constraint actually cap what gradient descent reaches
--------------------------------------------------------------------------------------------

*ID:* ``eval_rank_floor.module_overview``

Does a DISLDOLayer-family layer's SCALE-rank constraint (``scale_rank=1`` vs
``2`` vs higher -- the rank of the ``value_scale`` :math:`\otimes`
``output_scale`` envelope, see ``TrueMultiDigitLayer``/``DISLDOLayer``)
actually cap what gradient descent can reach in practice, given this
project's own finding that the discrete per-synapse FP4 codes are hard to
move (often bit-exact frozen, see ``eval_stuck_weights.py``)?

**Important caveat**: ``scale_rank=k`` does NOT bound the layer's
REPRESENTABLE matrix rank the way a literal low-rank factorization would.
The codes themselves are per-synapse independent and nominally
full-rank-capable; a rank-k scale envelope applied elementwise is
equivalent to ``diag(row_scale) @ codes @ diag(col_scale)`` -- a
similarity-like rescaling that leaves ``rank(codes)`` untouched. So the
Eckart-Young floor computed here is NOT a hard representational ceiling for
the real FP4 layer -- it's the floor for gradient descent's easy-to-move
part (the scale) alone. If the real FP4 arm's loss converges near (or
above) the rank-k floor and doesn't beat it with real training, that's
rigorous, quantitative evidence the codes aren't contributing real rank in
practice (a sharper version of this project's earlier ``mean_delta_w=0.0``
finding). If it clearly beats the floor, the codes ARE eventually
contributing real rank, just slower than the scale envelope.

**Three-arm comparison**, same target/training budget, only the layer
differs (see [[feedback_do_science_correctly]]):

- ``LowRankDenseLayer(rank=k)``: plain ``x @ U @ V`` factorization, no
  quantization at all. Should land ~exactly on the EY floor with enough
  training -- this is the harness's own sanity check.
- real FP4 ``DISLDOLayer``/``DISLDOLayerDeterministic`` at ``scale_rank=k``:
  the actual question.
- ``DISLDOLayer32`` (float32 backend, same kernels/training dynamics, exact
  storage): upper bound, should approach ~0 loss regardless of
  ``scale_rank`` (nothing there to get stuck on). ``FullRankDenseLayer``
  (plain ``x @ W``, no quantization) is this same upper-bound role for the
  dense side.

.. _eval_rank_floor.permutation_matrix_shift_choice:

``permutation_matrix``: why a cyclic shift, not the identity
-------------------------------------------------------------------

*ID:* ``eval_rank_floor.permutation_matrix_shift_choice``

Cyclic shift by ``shift`` on ``n`` slots. Orthogonal (all n singular values
exactly 1), so its best rank-k approximation (Eckart-Young) has a clean,
closed-form squared-Frobenius floor of exactly ``n-k`` -- no need to invoke
``shift=0`` (identity, already rank-deficient in a trivial way) or any
non-orthogonal target.

.. _eval_rank_floor.low_rank_layer_sanity_check:

``LowRankDenseLayer``: the harness's own sanity-check arm
-----------------------------------------------------------------

*ID:* ``eval_rank_floor.low_rank_layer_sanity_check``

Plain ``x @ U @ V`` factorization (``U``: n_in x rank, ``V``: rank x
n_out), no quantization. Should reach ~the exact Eckart-Young floor for
``rank`` with enough training, confirming the training loop itself is
capable of hitting a known answer before trusting any FP4-arm comparison
against it.

.. _eval_rank_floor.measure_rank_floor_injected_optimizer:

``measure_rank_floor``: optimizer as an injected dependency, not an import
-------------------------------------------------------------------------------

*ID:* ``eval_rank_floor.measure_rank_floor_injected_optimizer``

Trains ``layer`` (any object with ``.forward(x: Tensor, learning_rate) ->
Tensor`` and ``.trainable_params() -> List[Tensor]``) to reproduce
``target`` (n x n) via full-batch regression against all n standard basis
vectors every step -- exact, no sampling noise; ``target`` is small enough
(permutation matrices tested at n<=32) that this is cheap.

``opt``/``opt_step``/``clip_grad_norm`` let the caller supply its own
``AdamOptimizer``/``clip_grad_norm_`` (kept as injected dependencies, not
imported here, so this module doesn't need to know which project
convention is in use); pass ``opt=None`` for a layer whose weights update
inline (real DISLDOLayer-family layers, ``lr`` is threaded through
``forward``'s ``learning_rate`` instead).

``beat_floor_tol``: ``best_sse`` must be below ``beat_floor_tol * ey_floor``
to count as ``beat_floor=True`` in the returned ``RankFloorReport`` -- a
real, meaningful escape from the rank-k floor, not just landing on it
within numerical noise.

.. _eval_rank_floor.measure_rank_floor_per_example_updates:

Per-example online updates: why not one batched loss across all n columns
-------------------------------------------------------------------------------

*ID:* ``eval_rank_floor.measure_rank_floor_per_example_updates``

Per-example online updates, NOT one accumulated-then-batched update across
all n basis vectors -- real DISLDOLayer-family layers update inline as a
side effect of each individual ``forward(..., learning_rate)`` call's
backward, using that call's own LOCAL gradient. Summing all n columns'
losses into one ``total_loss`` and calling ``.backward()`` once would still
fire n separate full-lr inline updates (one per forward call in the graph,
each seeing only its own local gradient, not an n-way-averaged one) --
silently amplifying the DISLDO arms' effective step size ~n-fold relative
to the dense arms' single batched Adam step, which reliably diverged in
practice. Per-example updates for every arm keep the comparison
apples-to-apples (same update count, same per-update gradient scale)
regardless of which arm owns the update mechanism.

.. _eval_rank_floor.measure_rank_floor_lr_decay_rmsprop_divergence:

``lr_decay``: a real RMSprop-without-decay divergence, root-caused
-------------------------------------------------------------------------

*ID:* ``eval_rank_floor.measure_rank_floor_lr_decay_rmsprop_divergence``

Per-outer-step exponential decay applied to ``lr``
(``effective_lr = lr * lr_decay**step``), default 0.99.

Root-caused, not just papered over with ``best_sse`` tracking: traced a
real ``DISLDOLayer32`` run step-by-step and found it converges cleanly to
SSE below the rank-2 floor by step ~20, then destabilizes and diverges
around step ~350-400 at a CONSTANT lr. Cause is the standard, well-known
RMSprop-without-decay pathology -- ``linear_disldo.hpp``'s own update is
plain ``ci = beta2*ci + (1-beta2)*(g**2+contrib**2)`` (``beta2=0.999``, no
bias correction on this per-synapse ``ci``) then
``delta_w = -lr*g/(sqrt(ci)+eps)``.

.. code-block:: text

   ci = beta2 * ci + (1 - beta2) * (g**2 + contrib**2)   # beta2=0.999, no bias correction
   delta_w = -lr * g / (sqrt(ci) + eps)

Once the model is near-converged, ``ci`` (a SLOW ~1000-step EMA of the
ONCE-larger gradients) stays elevated relative to the now-tiny residual
gradient for a long time, which is what keeps the step naturally small
during the long stable plateau. But ``ci`` eventually decays down to match
the small residual gradient too -- once it does, ``g/sqrt(ci)`` stops
shrinking with the error and returns to ~full lr-sized steps REGARDLESS of
how small the actual error is, so it overshoots, the resulting larger
gradient pushes ``ci`` back up, and the cycle repeats/compounds. This is
exactly why RMSprop/Adam are essentially never run without an external lr
schedule in practice -- nothing sili-specific, and ``ci``'s own
bias-correction (already fixed elsewhere for
``value_scale_importance``/``output_scale_importance``) wouldn't fix THIS
failure mode -- that fixes ``ci`` being too SMALL early on; this is ``ci``
becoming well-matched to a small gradient LATE, which is the opposite
problem.

``lr_decay`` keeps the effective step shrinking faster than ``ci`` can
"catch up", confirmed directly to eliminate the divergence entirely (flat
SSE=3.0000 from step 100 through 410 at ``lr_decay`` in {0.985, 0.99,
0.995}, vs. spiking past 150+ with no decay). ``best_sse``/``final_sse``
are both still tracked and reported separately -- ``lr_decay`` makes them
converge to the same value in practice, but the distinction stays
meaningful for any caller who passes ``lr_decay=1.0`` (undecayed) or a
decay too slow for their own step budget.
