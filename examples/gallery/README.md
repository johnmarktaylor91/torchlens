# Reproduction Gallery scripts

Each script here reproduces one cited result with public TorchLens calls and checks it the
same way:

- **Oracles.** Rules fixed before the first run: agreement with the paper's claim, and with
  an independent implementation (plain PyTorch hooks, autograd, or the reference code) on
  the same inputs, to a stated tolerance.
- **Controls.** Runs that must fail: a wrong site, a shuffled order, a seeded wrong
  solution, or a mismatched pair that must be refused. A control that passes when it
  should fail fails the entry.
- **Artifact.** The run is saved as a portable `.tlspec`; the script reloads it with
  `tl.load` and replays one measurement from the reloaded copy.
- **Stamp.** The verdict is printed with the TorchLens version and commit, torch, the
  device, the pinned inputs and the wall time, and written to `<entry>.json`.

An entry passes only when every oracle passes, every control fails as declared and the
replay matches. A script prints `VERDICT: did not pass` otherwise; its rules are never moved
to make it pass, and an entry that does not pass moves to the next batch.

Run one with `python examples/gallery/<entry>.py --out DIR`. The pages under
`docs/gallery/` give each entry's claim, oracles, controls, the recorded stamp and a "vary
it" section whose examples run in the test suite on small stand-in models.

`_gallery.py` holds the shared stamp, report, artifact replay and line-count helpers. The
basis-grid line counts come from regions marked `# region: <name>` ... `# endregion` in each
script: `torchlens` is the analysis written with TorchLens, `plain` the same analysis
written with plain PyTorch hooks or autograd (the independent oracle). Blank, comment and
docstring lines are not counted.

Each script also saves what its figure needs beside `<entry>.json` (a `<entry>_figure.npz`
or `.json` file) and draws the page figure from those saved results through `_figures.py`,
which imports no torch. `python examples/gallery/<entry>.py --figure-only --out DIR` redraws
the PNG and SVG from a finished run without rerunning the model.
