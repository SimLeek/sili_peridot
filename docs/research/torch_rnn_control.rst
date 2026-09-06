``torch_rnn_control.py`` research notes
==========================================

Companion doc to ``scripts/torch_rnn_control.py``. Source comments point
back here by anchor ID (``*ID:* `` marker under each heading below). See
``sili__new/docs/research/sparse_rnn.rst`` for the pattern this follows
(semantic dotted anchor IDs, visible ID markers via the ``*ID:*`` line since
plain ``.. _id:`` targets alone render invisible on GitHub, frozen code
snippets on real-bug/non-obvious-derivation sections only).

.. _torch_rnn_control.module_overview_task_characterization:

Module overview: a diagnostic ceiling using PyTorch's own real-BPTT RNNs
----------------------------------------------------------------------------

*ID:* ``torch_rnn_control.module_overview_task_characterization``

Diagnostic CONTROL, not part of the real model -- a genuine "this should
just work" ceiling for the out-of-context deviation-detection task
(``model/toy_beyond_context_task.py``'s ``generate_deviation_sequence``),
using PyTorch's OWN built-in, real-BPTT recurrent modules (``nn.RNN``,
``nn.LSTM``) and a standard optimizer (Adam), on the EXACT SAME task
generator the from-scratch peak-eligibility experiment uses.

The task is functionally a detection/latch problem (did any deviation occur
anywhere in the sequence -- 1 bit of carried state, not a k-symbol copy),
closer to Hochreiter & Schmidhuber's original long-lag/latch problems (1997,
"Long Short-Term Memory", Neural Computation 9(8)) than to the harder
standard copy task (Le, Jaitly & Hinton 2015, arXiv:1504.00941; reference
PyTorch implementation with LSTM/GRU baselines: Bai, Kolter & Koltun 2018,
arXiv:1803.01271, github.com/locuslab/TCN). At ``n_bits<=6`` this is nowhere
near the several-dozen-step regime where vanilla RNNs' vanishing gradients
become a real barrier -- both ``nn.RNN`` AND ``nn.LSTM`` are run here
deliberately: if even the plain, no-gating vanilla RNN solves this cleanly,
that's the strongest possible sanity check that the TASK itself is easy for
anything with genuine BPTT, independent of gating architecture.
``num_layers=1``, ``hidden=128``, Adam ``lr=1e-3`` -- standard literature
defaults for a synthetic RNN memory task at this scale, not tuned to make
either arm look good.

.. _torch_rnn_control.full_sequence_bptt_design:

Full-BPTT training: whole sequence through one fused RNN call
----------------------------------------------------------------

*ID:* ``torch_rnn_control.full_sequence_bptt_design``

The whole sequence (body + query token) is fed to the RNN module in one
call -- torch's fused RNN implementations backprop through the entire
unrolled sequence for free, so there is no truncation and no curriculum
needed (unlike the from-scratch experiment, real BPTT doesn't have a
credit-assignment gap to work around).

.. _torch_rnn_control.no_bptt_variant_isolation:

No-BPTT variant: per-tick detachment, matching the from-scratch system's convention
----------------------------------------------------------------------------------------

*ID:* ``torch_rnn_control.no_bptt_variant_isolation``

Also runs a NO-BPTT variant, per direct instruction, to isolate whether BPTT
specifically is what makes the difference (not architecture, not optimizer,
not anything else): the sequence is fed one tick at a time, and the hidden
state is DETACHED after every step before being passed to the next --
exactly matching how ``M_prev`` is a fresh detached leaf every tick in the
from-scratch DISLDO system, same architecture/optimizer/task otherwise.
Only the query tick's own forward computation is differentiable;
``loss.backward()`` is called once, same as the from-scratch system's
``query_step`` convention.

.. _torch_rnn_control.bptt_hypothesis_rejected_result:

RESULT (seed=1000, train_steps=2000): the BPTT-per-se hypothesis is WRONG
------------------------------------------------------------------------------

*ID:* ``torch_rnn_control.bptt_hypothesis_rejected_result``

No-BPTT ``nn.RNN`` still hit 100% at every ``n_bits`` (2/3/4/6); no-BPTT
``nn.LSTM`` hit 100%/100%/91%/90% -- NOT a drop to chance. Per direct
correction, this matches the user's own prior hands-on experience ("BPTT
does practically nothing... it's not a chance vs 100% thing ever") -- the
hypothesis that BPTT-per-se explains the from-scratch system's
out-of-context failure is WRONG.

Why no-BPTT still works here: the recurrent weight matrix is SHARED across
every tick and across every training sequence (varying ``query_pos``,
varying ``n_bits``), so even though any single training example only
differentiates through its own last tick, the same weights get a gradient
nudge toward the correct one-step transition rule from many different
"positions in the recursion" across the training set -- sufficient to learn
a stateless composable update (accumulate-deviation is exactly such a rule)
without ever needing multi-tick BPTT.

This is architecturally the SAME regime the from-scratch ``PlainCell``/
``PeakSynapseCell`` training loop already uses (``train()`` calls
``cell.step()`` with a real lr at EVERY tick, not just the query tick -- see
``scripts/prototype_peak_synapse_learning_comparison.py``'s ``train()``) --
so BPTT was never the missing ingredient there either.

Next step (per direct instruction, not yet built here): ablation-style --
start from this working no-BPTT PyTorch control and incrementally swap in
components of the from-scratch system (DISLDO's sparse/quantized weights,
``EnergyDynamics`` gating, the residual state update, FP4 rounding noise)
one at a time until something makes it drop toward chance, isolating the
actual cause.

.. _torch_rnn_control.evaluate_forward_only_shared:

``evaluate()``: forward-only, shared across both training regimes
----------------------------------------------------------------------

*ID:* ``torch_rnn_control.evaluate_forward_only_shared``

Eval is just forward passes, no backward -- BPTT vs no-BPTT only affects
TRAINING, so this same ``evaluate()`` works for models trained either way.
It runs a full-sequence-at-once forward, matching how the from-scratch
system's own ``evaluate()`` also runs plain forward ticks at ``lr=0``.
