# T0 (harness stub): one named module's output

This stub exercises the benchmark harness; it is not a benchmark task.

## Prompt to the agent

`model.py` (in `--inputs`) defines `build_model()` and `make_inputs(seed)`; `config.json` names
the `seed`. Record the output of the submodule named `fc2` for every input row, in input order,
and save it to `outputs.npz` (in `--outputs`) as one float32 array named `out`.

## Oracle and pass bars

The grader recomputes the module output directly and requires exact equality, on the visible
inputs and on a held-out seed.
