# Menagerie Ceiling Log

The menagerie aims to render EVERY notable architecture. **No-public-code or gated weights is NOT a reason to
stop:** if an architecture is described in enough detail (paper / spec / repo) to reimplement with reasonable
fidelity, we reimplement it as a `classics/` module (random init). The **only genuine ceiling** is an architecture
**not described in enough detail to reproduce faithfully** — reimplementing it would be unfruitful guessing, and we
will not embarrass ourselves doing so.

Entries below are logged so future agents (and more capable future models) do NOT retrace dead ends. **Revisit any
entry if a better source/spec emerges or a stronger model can infer the architecture.**

## Confirmed ceiling (genuinely underspecified — do not retrace unless new info)
| Model | Reason (why not faithfully reproducible) | Source |
|---|---|---|

## Candidates flagged before this principle (RE-EVALUATE: many may be reimplementable now)
| Model | Old reason | Source |
| --- | --- | --- |
| Neocognitron | insufficient detail in row for a faithful random-init architecture; many incompatible historical variants |  |
| consistency_model | row only names OpenAI consistency_models SongUNet sketch; local source package absent and architecture details insufficient for faithful inline repair |  |
| HDemucs | Installed demucs.HDemucs instantiated, but plain forward failed for tested random-init small configs before TorchLens tracing; no verified metadata-only repair in this slice. |  |
| rt2_reference | closed-source/blog-level RT-2 reference; insufficient public architecture detail for a faithful random-init PyTorch recipe |  |
