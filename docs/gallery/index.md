# Reproduction Gallery

Each entry reproduces one published claim with TorchLens and checks the result against an
independent implementation (an oracle) whose rule was fixed before the first run. Every entry
also runs controls that must fail, saves a portable artifact that reloads and replays, and
records a stamp (TorchLens version, commit, libraries, machine, date). The entry's "Vary it"
section has small examples that run in seconds and are executed by the test suite on every
change.

An entry whose oracle does not pass is not shown here; it moves to the next batch with the
reason recorded below.

## Entries

| Entry | Claim | Result |
|---|---|---|
| [Causal tracing of MLP outputs](causal_tracing.md) | Meng et al. 2022: restoring MLP outputs at the last subject token recovers the most answer probability | passed: TorchLens and plain hooks agree exactly over 72 cells; the peak is at the last subject token, at layer 0 on GPT-2 small |
| [CNN fields as an intervention site](cnn_fields.md) | Araujo et al. 2019: a CNN unit's receptive field bounds what can influence it | passed: the ResNet-18 field matches the analytic box exactly, gradient and projective support IoU 1.0, zeroing outside the field changes the unit by exactly 0 |
| [Neuro extraction with an order-error control](neuro_rdms.md) | Per-layer RDMs from TorchLens agree with independent extraction and rsatoolbox | passed on ResNet-18 and ViT-B/16: Pearson at least 0.99999999999 at every site; a one-sided shuffle drops agreement below 0.5 and is flagged |
| [A Jacobian lens across two architectures](jacobian_lens.md) | A corpus-average Jacobian transports a layer's residual vectors into the final layer's basis, on GPT-2 small and on Mamba-130m, which the reference implementation cannot run | passed: relative error 0.0 at all 11 GPT-2 layers against the reference and at most 4.8e-7 at all 23 Mamba layers against exact autograd; held-out top-ten overlap 1.0 everywhere; both portable artifacts replay; the transposed and wrong-layer controls fail their bars |
| [IOI circuit recovery with EAP-IG](ioi_circuit.md) | Wang et al. 2023: a 26-head circuit in GPT-2 small performs indirect-object identification | passed: on 1,000 examples the EAP-IG head ranking recovers the circuit with ROC AUC 0.926; scrambled names give 0.404 and reversed inputs a correlation of -0.996 |
| [Bundles plus a two-model harness](bundles_harness.md) | A bundle compares two runs only when they line up and differ by the stated edit; a steer reaches a second model only through the tokens it reads | passed: bundle differences equal plain-hook differences at every module call on GPT-2 small and ResNet-18 (max abs difference 0.0), the hand-run pipeline reproduces the recorded tokens and `fed_by` spans, and the three refusing controls refuse |

## Basis grid

The line counts are load-bearing lines (no blanks, comments or docstrings) of each script's
marked regions. "Shared" is setup used by both implementations.

| Entry | Verbs | TorchLens lines | Plain lines | Shared | Architectures | Cost |
|---|---|---|---|---|---|---|
| Causal tracing | capture, select, read, do, compare, save | 37 | 30 | 76 | GPT-2 small | 16 min 52 s, 2 CPU cores, 5.4 GiB |
| CNN fields | capture, select, read, do, compare, save | 233 | 37 | in TorchLens count | ResNet-18 | 21 s, 2 CPU threads, 743 MiB |
| Neuro RDMs | capture, select, read, compare, save | 252 | 56 | in TorchLens count | ResNet-18, ViT-B/16 | 31 s and 61 s, 2 CPU threads, 1.1 and 2.6 GiB |
| Jacobian lens | capture, read, save | 41 | 33 | not separated | GPT-2 small, Mamba-130m | 29 min 38 s, 2 CPU cores, 4.9 GiB |
| IOI circuit | capture, read, save | 55 | 19 | not separated | GPT-2 small | 1 h 41 min, 2 CPU cores, 6.5 GiB |
| Bundles harness | capture, select, read, do, compare, save | 44 | 41 | 19 | GPT-2 small, ResNet-18, DistilGPT2 | 2 min 29 s, 1 CPU thread, 2.7 GiB |
