``eval_output_collapse.py`` research notes
=============================================

Companion doc to ``model/eval_output_collapse.py``. Source comments point
back here by anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows.

.. _eval_output_collapse.module_overview:

Turning a qualitative collapse check into a quantitative diagnostic
--------------------------------------------------------------------------

*ID:* ``eval_output_collapse.module_overview``

Does a trained model's output actually depend on its input, or has it
collapsed to predicting a narrow (or single) set of tokens regardless of
what's fed in? A bare accuracy number can't tell these apart --
``scripts/l1_sparsity_probe.py``'s own ``evaluate()`` docstring already
flags this exact concern ("the model is degenerate/guessing a narrow set of
tokens" vs "these particular held-out sequences happened to skew toward one
target token") and works around it with a ``verbose=`` flag that prints the
last 5 (pred, target) pairs -- a partial, qualitative check. This module
makes it a real, quantitative diagnostic instead.

Domain-agnostic like ``eval_lr.py``/``eval_eigenvalues.py``: callers supply
a ``predict_fn(seed) -> (predictions, targets, logits)`` callback that runs
some held-out eval batch and returns the raw per-sample data, instead of
collapsing straight to a scalar the way ``evaluate()`` does.

.. _eval_output_collapse.normalized_entropy_threshold:

``normalized_entropy``: a starting point, not a pass/fail line
--------------------------------------------------------------------

*ID:* ``eval_output_collapse.normalized_entropy_threshold``

0.0 = fully collapsed (always predicts the same token), 1.0 = maximally
diverse (predictions uniform over the vocab). No hard threshold -- below
~0.3 is a reasonable starting point to treat as a real collapse signal worth
investigating, not a pass/fail line.

.. _eval_output_collapse.check_output_collapse_design:

``check_output_collapse``: pooling and the two collapse failure modes
---------------------------------------------------------------------------

*ID:* ``eval_output_collapse.check_output_collapse_design``

``predict_fn(seed) -> (predictions, targets, logits)`` over some held-out
eval set -- same shape/semantics as ``l1_sparsity_probe.py``'s own
``evaluate()``, just returning the raw per-sample data instead of reducing
straight to an accuracy scalar. ``n_batches>1`` calls ``predict_fn`` with
different seeds and pools everything together (use this if a single call's
eval set is small enough that per-call sampling noise would dominate the
diversity metrics).

``cross_sample_logit_std`` specifically distinguishes two different failure
modes accuracy alone conflates: a model that's collapsed onto one
confident-but-wrong ARGMAX every time (low ``unique_prediction_fraction``)
could still have logits that genuinely shift with input (real signal, wrong
decision boundary) -- that's a very different bug than a model whose raw
OUTPUT REPRESENTATION barely moves at all regardless of input (near-zero
``cross_sample_logit_std``), which is collapse in a much deeper sense.
