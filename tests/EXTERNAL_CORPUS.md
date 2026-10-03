# External corpus

Correctness checks that cannot execute on any CI leg live here instead of in the suite, so they
stay visible rather than skipping on every runner (tests/test_proofnet_gate_witness.py: the
executes-nowhere tier only burns down). Run them by hand in an environment that has the
dependency; move one back into tests/ once a CI leg can provision it.

## DimeNet (torch_geometric) capture and validation

- Former test: `tests/test_real_world_models.py::test_dimenet` (removed 2026-10-03).
- Why it is here: DimeNet's forward calls `radius_graph`, which current torch_geometric routes
  through the compiled pyg-lib extension and requires `pyg-lib>=0.6.0`. pyg-lib has no PyPI
  wheel; the PyG wheel index publishes 0.6+ CPU builds only for torch 2.8 to 2.12 (and torch 2.13
  for CPython 3.10 only), while the one CI leg that runs slow tests (weekly.yml) uses torch 2.7,
  whose newest build is pyg-lib 0.5.0.
- How to run (CPython 3.11, torch 2.8 CPU):

```bash
pip install "torch==2.8.0+cpu" --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]"
pip install "pyg-lib==0.6.0+pt28cpu" -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
python - <<'PY'
import torch
import torchlens as tl
from torch_geometric.nn import DimeNet
from torchlens.user_funcs import validate_forward_pass

model = DimeNet(6, 3, 4, 2, 6, 3)
z = torch.tensor([6, 1, 1, 1, 1])
pose = torch.tensor(
    [
        [-1.2700e-02, 1.0858e00, 8.0000e-03],
        [2.2000e-03, -6.0000e-03, 2.0000e-03],
        [1.0117e00, 1.4638e00, 3.0000e-04],
        [-5.4080e-01, 1.4475e00, -8.7660e-01],
        [-5.2380e-01, 1.4379e00, 9.0640e-01],
    ]
)
inputs = (z, pose, None)
tl.trace(model, inputs)
assert validate_forward_pass(model, inputs)
print("DimeNet capture and validation OK")
PY
```
