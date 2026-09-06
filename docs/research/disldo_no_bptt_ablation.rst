``disldo_no_bptt_ablation.py`` research notes
================================================

Companion doc to ``scripts/disldo_no_bptt_ablation.py``. Source comments
point back here by anchor ID (``*ID:* `` marker under each heading below).
See ``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows.

.. _disldo_no_bptt_ablation.module_overview:

Isolating whether DISLDO itself (not BPTT) explains the out-of-context struggle
------------------------------------------------------------------------------------

*ID:* ``disldo_no_bptt_ablation.module_overview``

Ablation control, direct follow-up to ``scripts/torch_rnn_control.py``'s
no-BPTT result (both ``nn.RNN`` and ``nn.LSTM`` stayed near-100% even with
the hidden state detached every tick -- BPTT itself isn't what explains the
from-scratch system's out-of-context struggle). Copy that no-BPTT structure
to a new file and swap ONLY the recurrent cell -- PyTorch's ``nn.RNN`` out,
sili's real ``DISLDOLayer`` in -- same task (``generate_deviation_sequence``),
same no-curriculum uniform ``n_bits`` sampling, same
train_steps/eval_sequences/n_bits range, same tick-by-tick/detached-hidden-
state training regime. If THIS still holds up near-100%, DISLDO itself
isn't the culprit either, narrowing down what actually breaks the full
from-scratch system (energy gating, peak-synapse correction, curriculum
dependence, or something else entirely). If it collapses toward chance
here, DISLDO's own quantization/training dynamics are a real candidate,
worth isolating further.

Run: ``python -m scripts.disldo_no_bptt_ablation``

.. _disldo_no_bptt_ablation.parameter_matching:

Parameter-matching DISLDO to nn.RNN's recurrent weight count
------------------------------------------------------------------

*ID:* ``disldo_no_bptt_ablation.parameter_matching``

Parameter-matched to ``nn.RNN``'s own recurrent weight count as closely as
reasonably possible without inventing new machinery: torch's control uses
``nn.Embedding(vocab=3, hidden=128)`` -> ``nn.RNN(input_size=128,
hidden_size=128)`` (Whx 128x128 + Whh 128x128 = 32768 recurrent params) ->
``nn.Linear(128, 3)``. Here: a FIXED (not trained) random one-hot lift
``onehot(3) @ E -> hidden(128)`` stands in for the embedding table (E is
deterministic/seeded, not gradient-updated -- a change of basis, not a
learned representation; DISLDO has no established trainable-embedding
convention in this codebase to reuse, and wiring one up would need new
plumbing this ablation doesn't need to isolate the actual question: does
swapping in the RECURRENT CELL itself break things). The cell is
``DISLDOLayer(HIDDEN*2, HIDDEN, max_weights=HIDDEN*2*HIDDEN)`` -- 32768
params at full density, exactly matching Whx+Whh's combined count (ignoring
``nn.RNN``'s small bias terms). Head is ``DISLDOLayer(HIDDEN, VOCAB_SIZE,
max_weights=HIDDEN*VOCAB_SIZE)`` -- 384 params, close to
``nn.Linear(128,3)``'s 387 (ignoring its 3-param bias).

.. _disldo_no_bptt_ablation.lr_choice:

Learning rate: DISLDO's own tuned schedule, not torch's borrowed Adam lr
------------------------------------------------------------------------------

*ID:* ``disldo_no_bptt_ablation.lr_choice``

NOT torch's Adam ``lr=1e-3`` (DISLDO's own inline C++ per-row training isn't
Adam, so blindly reusing that number would confound "does DISLDO collapse"
with "is DISLDO untuned at this lr"). Uses this project's own already-
validated ``lr_schedule`` (``peak_lr=0.05``, warmup) from
``prototype_peak_synapse_learning_comparison.py`` instead -- DISLDO
evaluated under its own best-known hyperparameters, not an arbitrary
borrowed number.

No ``EnergyDynamics``, no peak-synapse correction, no curriculum -- the
plainest possible DISLDO recurrent cell, matching "just disldo". Note:
``loss.backward()`` is NOT preceded by a manual
``loss.grad = np.array(1.0, ...)`` assignment (see JOURNAL.md for why that
line was removed from the sibling prototype script -- verified functionally
identical to ``Tensor.backward()``'s own default root-grad init for a
scalar loss, but removed anyway to match this codebase's actual convention
going forward).

.. _disldo_no_bptt_ablation.no_bptt_tick_semantics:

``step``: no-BPTT tick semantics
-------------------------------------

*ID:* ``disldo_no_bptt_ablation.no_bptt_tick_semantics``

``h_prev`` is a plain numpy array (fresh ``Tensor`` leaf every call, no
graph connection to prior ticks) -- exactly matching ``torch_rnn_control.py``'s
detach-every-tick convention.
