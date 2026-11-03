# The basis grid

Each row is one Reproduction Gallery entry; each column is one of the six verbs on a trace.
A filled cell means the entry's script uses that verb. The grid shows that the entries are
compositions of the same few verbs; it does not show that every future method is one, and it
says nothing about the scientific validity of a method.

<!-- JMT review: plain factual prose, written for review. -->

The six entries are being built. Until an entry lands, its verb cells come from its
specification and its last two cells read "pending"; when it lands they are filled from its
script: the number of load-bearing lines against the original repository, and the
architectures the entry was rerun on.

| Entry | capture | select | read | do | compare | validate | Load-bearing lines against the original repo | Rerun on |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Causal tracing, with a wrong-site control (Meng et al. 2022) | planned | planned | planned | planned | planned | planned | pending | pending |
| 2. CNN receptive and projective fields, used as an intervention site | planned | planned | planned | planned | planned | planned | pending | pending |
| 3. Neuro feature extraction to per-layer RDMs, with an order-error control | planned | planned | planned | | planned | planned | pending | pending |
| 4. Jacobian lens, also on an architecture its reference repo cannot run | planned | planned | planned | | planned | planned | pending | pending |
| 5. IOI circuit by edge attribution (Wang et al. 2022) | planned | planned | planned | planned | planned | planned | pending | pending |
| 6. Bundles plus a two-model harness | planned | planned | planned | planned | planned | planned | pending | pending |

## The verbs

| Verb | On a trace |
| --- | --- |
| capture | `trace = tl.trace(model, x)` |
| select | `tl.module(...)`, `tl.func(...)`, `tl.in_module(...)`, receptive-field regions, combined with `&` and `\|` |
| read | `trace["layer3"].out`, `tl.extract`, `tl.extract_dataset` |
| do | `trace.fork().do(site, edit, model=model, inputs=x)`, or `tl.when(site, edit).bind(model)` live |
| compare | two traces or a bundle of runs, site by site |
| validate | `tl.validate(model, x)`: the replay check of the recording against the real forward |

Every entry starts from the same capture as the [running examples](../field-guide/running_examples.md).
