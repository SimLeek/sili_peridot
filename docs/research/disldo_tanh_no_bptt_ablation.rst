``disldo_tanh_no_bptt_ablation.py`` research notes
=====================================================

Companion doc to ``scripts/disldo_tanh_no_bptt_ablation.py``. Source
comments point back here by anchor ID (``*ID:* `` marker under each heading
below). See ``sili__new/docs/research/sparse_rnn.rst`` for the pattern this
follows.

.. _disldo_tanh_no_bptt_ablation.module_overview:

Fixing the unbounded residual accumulate found in the sibling ablation
------------------------------------------------------------------------------

*ID:* ``disldo_tanh_no_bptt_ablation.module_overview``

Direct follow-up to ``scripts/disldo_no_bptt_ablation.py``, which found a
real, decisive root cause via weight/state tracing (not guessed): that
ablation's cell used ``h_new = h_prev + cell([x, h_prev])`` -- an UNBOUNDED
residual accumulate with no squashing nonlinearity anywhere. Traced directly
with FROZEN (``lr=0.0``, untrained) weights: ``h_norm`` grows ~1.7-1.9x
EVERY tick regardless of training (0.85 -> 1.4 -> 2.6 -> ... -> 1.97e12 by
tick 50) -- a pure linear-feedback instability (spectral radius > 1),
nothing to do with FP4/DISLDO's own training dynamics. This is exactly why
torch's ``nn.RNN`` control never showed this: ``nn.RNN``'s actual formula is
``h_new = tanh(Whx@x + Whh@h)`` -- full OVERWRITE, not accumulate, squashed
through tanh every tick, which provably bounds ``||h||`` regardless of the
weights.

This script makes the DISLDO cell structurally IDENTICAL to that formula --
``h_new = tanh(cell([x, h_prev]))``, no residual add -- the single
closest-to-``nn.RNN`` change possible while still using ``DISLDOLayer`` as
the linear part. Confirmed first (see JOURNAL.md) that this alone fixes the
frozen-weight forward-pass blowup (``h_norm`` stays ~1.2-1.4 indefinitely,
same regime as the working torch control). This script answers the real
remaining question: does it also fix LEARNING, or was the forward-pass
instability a red herring alongside a separate, still-unfixed
training-dynamics problem?

Run: ``python -m scripts.disldo_tanh_no_bptt_ablation``

.. _disldo_tanh_no_bptt_ablation.lr_per_row_nnz_bug:

Root-causing and fixing the silent lr_per_row_nnz division bug
--------------------------------------------------------------------

*ID:* ``disldo_tanh_no_bptt_ablation.lr_per_row_nnz_bug``

This was drifting from "minimal change to the base RNN, just DISLDO" -- the
earlier version of this file used a ``peak_lr=0.05`` warmup+cosine schedule
(borrowed from a DIFFERENT, unrelated script's convention) instead of
torch's own flat Adam ``lr=1e-3``, and separately (in the sibling ``_sparse``
variant) reduced density below ``nn.RNN``'s real parameter count to route
around a training-rate problem instead of fixing it.

Root cause of THAT problem, found by tracing weight/value_scale updates
directly (see JOURNAL.md): ``DISLDOLayer.forward()`` hardcoded
``lr_per_row_nnz=True`` in its backward closure, with NO way to override it
-- silently dividing whatever ``learning_rate`` is passed by the row's own
connection count (``nnz_this_row``), a normalization whose real purpose
(keeping updates comparable across rows when synaptogenesis makes degree
vary WITHIN a layer) does nothing useful at uniform density and just
crushes the effective rate by ~128x here. Fixed upstream in sili__new
(``DISLDOLayer.forward`` gained an ``lr_per_row_nnz`` param, default
``True`` for backward compat elsewhere).

This script now: full density (real ``nn.RNN`` parameter parity, no
``PER_ROW_K``), flat ``lr=1e-3`` (literally torch's own Adam lr, no
schedule, no ``PEAK_LR``), ``lr_per_row_nnz=False`` (a real, literal
learning rate, not one silently rescaled by density) -- the actual minimal
ablation: same RNN, same lr, same training regime, ONLY the cell type
differs.
