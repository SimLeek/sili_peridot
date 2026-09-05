``curriculum.py`` research notes
===================================

Companion doc to ``model/curriculum.py``. Source comments point back here by
anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers, frozen code snippets on
real-bug/non-obvious-derivation sections only).

.. _curriculum.module_overview:

Module overview: B8a's backward-growing column-averaging curriculum
-------------------------------------------------------------------------

*ID:* ``curriculum.module_overview``

B8a's backward-growing column-averaging curriculum: an explicit stage list
(stage i = predict from the average of the last i+1 fold positions) plus
``WindowState``, which drives ``sili_block.grow_window_layer`` to build up
each suffix's combined window matrix ONE position at a time as the
curriculum advances -- never for positions outside the current window (see
``sili_block.module_overview`` for why).

Stage 0 (``window_size=1``) is the sanity-check stage: a single position has
nothing to recur with, so there is nothing for a combined matrix to usefully
hold yet. ``advance_window`` still builds one (``grow_window_layer``'s very
first call, from ``existing_window_layer=None``, is unavoidable plumbing --
``window_size=2`` needs SOME 1-position base to grow from, and that base is
bit-identical to the position's own plain layer, cheap to build), but
forward-pass callers should special-case ``window_size==1`` and use
``step_layers[window_positions[0]][suffix]`` directly instead of routing
through ``window_state.suffix_windows`` -- simpler, and avoids paying for
the window machinery's extra indirection when recurrence is structurally
impossible anyway.

.. _curriculum.window_state_ordering:

``WindowState``: field conventions -- position ordering, and Phase 2.7's centers/log_sigmas
---------------------------------------------------------------------------------------------

*ID:* ``curriculum.window_state_ordering``

``suffix_windows[suffix]`` holds a combined ``SparseLinearLayer`` for every
``window_size>=1`` (see ``curriculum.module_overview`` for why
``window_size==1``'s is usually better bypassed by forward-pass callers, not
why it's absent). ``window_positions`` holds the ABSOLUTE fold-step indices
currently in the window, in window-growth order (index 0 = first position
added = the LAST fold-step, matching ``grow_window_layer``'s own convention)
-- e.g. for a 24-layer model, ``window_size=3`` means
``window_positions == [23, 22, 21]``.

``centers``/``log_sigmas`` (Phase 2.7): trainable ``Tensor`` leaves, one pair
per window position, in the SAME order as ``window_positions`` -- the
learnable Gaussian center/spread ``apply_window_step``'s
``gaussian_attention`` call uses (see
``sili_block.apply_window_step.major_pivot_design``). Both ``None`` only
when ``window_size==0`` (nothing built yet). Grown one position at a time by
``advance_window``, exactly like ``suffix_windows`` -- an already-trained
position's own center/log_sigma carries forward unchanged (as data, not a
live autograd graph -- there's no backward call spanning a curriculum stage
transition) when the window widens.

.. _curriculum.advance_window_growth:

``advance_window``: incremental growth, and the Gaussian init formula for new positions
-------------------------------------------------------------------------------------------

*ID:* ``curriculum.advance_window_growth``

Grows the window by exactly one position -- the position immediately before
``window_state``'s current earliest one (or the LAST fold-step,
``n_folds-1``, if the window is empty/stage 0). Calls ``grow_window_layer``
once per suffix, each reusing that suffix's own previous combined matrix (or
``None``, for the very first position -- see ``grow_window_layer``'s own
``None``-``existing_window_layer`` case) so already-trained in-window
recurrent connections carry forward. ``grow_window_layer`` works the same
regardless of what ``value_scale_mode`` built ``step_layers`` (``per_row``
or ``rank1``) -- no mode argument needed here.

Does NOT mutate ``window_state`` -- returns a new one, matching the
functional style ``grow_window_layer`` itself already uses (old layers
untouched, caller replaces its reference).

Also grows ``centers``/``log_sigmas`` by exactly one pair, matching
``grow_window_layer``'s own "old rows reused verbatim" discipline: the new
position's index in the window is ``p = window_state.window_size``
(0-indexed, before the increment), so its Gaussian center is initialized to
``2p + 0.5`` (own fresh-token/carried-state pair's midpoint in
``apply_window_step``'s interleaved key space) and ``log_sigma`` to ``0.0``
(``sigma=1.0``) -- see
``sili_block.default_window_helpers.placeholder_and_init_conventions`` for
the same formula applied to a whole window at once. Every earlier position's
own center/log_sigma value is carried forward UNCHANGED (as data, not a live
autograd graph -- there is no backward call spanning a curriculum stage
transition).
