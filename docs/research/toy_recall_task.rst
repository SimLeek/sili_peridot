``toy_recall_task.py`` research notes
========================================

Companion doc to ``model/toy_recall_task.py``. Source comments point back
here by anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers via the ``*ID:*`` line since
plain ``.. _id:`` targets alone render invisible on GitHub, frozen code
snippets on real-bug/non-obvious-derivation sections only).

.. _toy_recall_task.module_overview:

Module overview: why this specific synthetic task
-------------------------------------------------------

*ID:* ``toy_recall_task.module_overview``

Synthetic associative-recall ("induction head") task -- see the approved
plan (``fuzzy-plotting-starlight.md``) for why this specific task: it
directly tests genuine long-range, content-based retrieval, the property
tile-recurrence exists to have more of than the old single-carried-vector
window mechanism. Feeds ``model/toy_recall_models.py`` (see
``toy_recall_models.module_overview``).

``generate_sequence`` (the original, single-bigram design): one cue-response
bigram (A, B) is planted early in an otherwise random token sequence; A is
planted again ``lag`` positions later, and the well-defined recall target at
that second occurrence is B (also planted as the following real token, so
this is an ordinary "predict the next real token" target, not a synthetic
label bolted on separately). ``lag`` must be ``>= 2`` (room for
``A, B, ..., A, B`` without the two A/B pairs colliding) and ``seq_len``
must be ``>= lag + 3`` (room for at least one valid cue position).
``induction_correct`` just checks the predicted token against
``tokens[induction_pos + 1]``.

.. _toy_recall_task.mqar_adoption_and_port:

``generate_mqar_sequence``: adopting the standard MQAR benchmark instead of extending the hand-rolled task
-------------------------------------------------------------------------------------------------------------

*ID:* ``toy_recall_task.mqar_adoption_and_port``

Standard Multi-Query Associative Recall task (Arora, Eyuboglu et al.,
"Zoology: Measuring and Improving Recall in Efficient Language Models",
2023, arXiv:2312.04927) -- per direct decision, adopted instead of
continuing to extend the hand-rolled single-bigram ``generate_sequence``
above: this is the established benchmark for exactly the property under
test here (does an efficient/recurrent architecture recall associations as
well as full attention), used throughout the linear-attention/SSM/RWKV
architecture literature.

Ported from HazyResearch/zoology's own reference implementation
(``zoology/data/multiquery_ar.py``, fetched directly from the repo) to
plain single-example numpy -- no torch, no batching (this project trains
online, one example at a time, and batching a single CPU doesn't buy
anything here regardless).

A context of ``num_kv_pairs`` unique (key, value) bigrams is laid down
first (keys drawn from the lower half of the vocab, values from the upper
half -- so a key can never be mistaken for a value or vice versa). Then, at
power-law-distributed gap positions after the context (small gaps much
likelier than large ones, matching real language's own repeated-bigram
statistics -- see the original zoology docstring's own explanation), each
key is repeated as a QUERY; the correct prediction immediately after a
query is that key's own value, requiring genuine retrieval from the
earlier context (never duplicated directly in the input). Non-query filler
positions get random noise tokens (``random_non_queries``, matching the
reference default) rather than a fixed placeholder, so the model can't use
"is this a special filler symbol" as a shortcut.

Returns ``(tokens [seq_len] int, pairs)`` where ``pairs`` is a list of
``(position, correct_next_token)`` -- directly usable with
``cross_entropy_sum``'s own ``(row, target)`` convention (see
``toy_recall_models.cross_entropy_sum_and_predicted_token``), one entry per
query. ``seq_len`` must be even and ``>= 4*num_kv_pairs`` (context and
queries both need ``2*num_kv_pairs`` slots).

.. _toy_recall_task.forced_keys_curriculum_guarantee:

``forced_keys``: guaranteeing curriculum vocab-growth coverage
-----------------------------------------------------------------

*ID:* ``toy_recall_task.forced_keys_curriculum_guarantee``

``forced_keys`` (curriculum "at least 1 new vocab per N queries" guarantee):
without this, keys are drawn uniformly from the WHOLE current key range, so
a just-grown vocab's newest token can go untested for many steps by pure
chance, letting a level-up streak complete on old vocab alone. When
``forced_keys`` is given, up to ``num_kv_pairs`` of them are folded into the
key set first (subsampled without replacement if there are more forced keys
than slots), and the remaining slots are filled from the rest of the key
range as usual, then shuffled together so the forced keys don't all land in
predictable positions.
