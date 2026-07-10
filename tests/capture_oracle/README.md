# Capture-unification characterization oracle

This package is the non-destructive Stage-0 safety net for capture-pipeline unification. It
runs a deterministic model/configuration matrix in isolated subprocesses and records which
producer ran, how many times the user forward executed, operation-event field population,
final topology, selected activation and gradient payload fingerprints, terminal outcomes,
partial-product metadata, observable side effects, disk artifacts, median wall time, and peak
memory.

The oracle deliberately separates two categories:

- `ground_truth` contains faithful model facts. Any diff fails. This includes executed
  operations, graph relationships, shapes/dtypes, saved payload identities and value digests,
  backward observations, and complete/halted/failed outcomes except for the one stateful
  two-pass carve-out below.
- `expected_to_change` contains only named migration warts with reasons and their Stage-0
  readings. These are the internal producer identity, two-pass double execution and its total
  side effects, the predicate producer's lossy/defaulted event metadata, and only the
  `train_batchnorm__two_pass_negative` failed outcome caused by its stateful second pass. A wart
  diff is accepted only after the legacy producer disappears and a wart-specific validator
  confirms the intended fix (for example, exactly one forward, no `edge_use="unknown"`, or a
  clean exactly-once success for the train-mode BatchNorm case).

Run the oracle from the worktree root:

```bash
PYTHONPATH="$PWD/tests:$PWD" python -m pytest tests/capture_oracle/ -x -q
```

A migration-stage author should root-cause every `ground_truth` failure as a faithfulness
regression. When a known wart flips because it was fixed, keep the faithful projection stable,
document the justification, and re-snapshot. Goldens are generated only by the harness; never
hand-edit them:

```bash
TORCHLENS_UPDATE_CAPTURE_ORACLE=1 PYTHONPATH="$PWD/tests:$PWD" \
  python -m pytest tests/capture_oracle/test_capture_oracle.py \
  -k capture_characterization_matches_golden -q
```

Timing and memory are retained for regression tracking as broad ratios to the committed
baseline, not as machine-specific absolute gates. RNG, model, and input construction are
seeded for every sample. The `followed_by` lookback case is trace-only because current
`record(save=followed_by(...))` rejects that selector explicitly; the matrix does not weaken
that validation.
