# The basis grid

Each row is one Reproduction Gallery entry; each column is one of the six verbs on a trace.
A filled cell means the entry's script uses that verb. The grid shows that the entries are
compositions of the same few verbs; it does not show that every future method is one, and it
says nothing about the scientific validity of a method.

<!-- JMT review: plain factual prose, written for review. -->

Entries 1, 2, 3 and 5 have landed (see the [Reproduction Gallery](index.md)); their verb
cells come from each script's declared verbs, and their last two cells from its marked
regions and runs. The line count is the script's load-bearing TorchLens lines beside its
plain-hooks reimplementation; the original repositories' lengths were not counted. Entries 4
and 6 moved to the next batch, so their verb cells still come from their specification.

| Entry | capture | select | read | do | compare | validate | Load-bearing lines against the original repo | Rerun on |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Causal tracing, with a wrong-site control (Meng et al. 2022) | used | used | used | used | used |  | 37 TorchLens, 30 plain hooks | GPT-2 small |
| 2. CNN receptive and projective fields, used as an intervention site | used | used | used | used | used |  | 233 TorchLens, 37 plain hooks | ResNet-18 |
| 3. Neuro feature extraction to per-layer RDMs, with an order-error control | used | used | used |  | used |  | 252 TorchLens, 56 plain hooks | ResNet-18, ViT-B/16 |
| 4. Jacobian lens, also on an architecture its reference repo cannot run | planned | planned | planned | | planned | planned | next batch | next batch |
| 5. IOI circuit by edge attribution (Wang et al. 2022) | used |  | used |  |  |  | 55 TorchLens, 19 plain (scoring, not a reimplementation) | GPT-2 small |
| 6. Bundles plus a two-model harness | planned | planned | planned | planned | planned | planned | next batch | next batch |

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
