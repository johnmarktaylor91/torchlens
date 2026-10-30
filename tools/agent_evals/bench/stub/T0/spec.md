# T0 (harness stub): one named module's output

This stub exercises the benchmark harness; it is not a benchmark task.

`model.py` defines `build_model()` and `make_inputs(seed)`. Write `solution.py` that, run as
`python solution.py --seed SEED --out PATH`, builds the model, makes the inputs for `SEED`, and
saves with `torch.save` a single tensor: the output of the submodule named `fc2` for every input
row, stacked in input order. Then run it once with `--seed 0 --out outputs.pt`.

Pass bar: exact equality with the module's output on the visible seed and on a held-out seed.
