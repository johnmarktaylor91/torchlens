# TorchLens Conventions

## Naming

### Files & Modules
- Snake_case for all Python files
- Subpackage CLAUDE.md files document each package

### Variables & Attributes
- `tl_` prefix on tensor/module attributes during logging
  - Permanent attrs (survive sessions): `tl_module_address`, `tl_module_type`
  - Session attrs (cleaned per-call): `tl_source_model_log`, `tl_module_pass_num`, etc.
- `_raw_` prefix for pre-postprocessing state (e.g., `tl_tensor_label_raw`)
- `_final_` prefix for post-processed state
- `_orig_` prefix for original (pre-decoration) references
- `clean_` prefix for pre-decoration torch function imports (e.g., `clean_clone = torch.clone`)

### Labels
- Source tensors: `{type}_{num}_raw` during capture (e.g., `input_0_raw`, `buffer_1_raw`)
- Function outputs: `{type}_{num}_{counter}_raw` during capture
- Final labels: human-readable after postprocess/labeling.py (e.g., `conv2d_1_5`)
- Pass-qualified: `{label}:{pass_num}` (e.g., `conv2d_1_5:2`)

### Classes
- PascalCase: `ModelLog`, `LayerPassLog`, `LayerLog`, `BufferLog`, `ModuleLog`, `ParamLog`
- Accessors: `LayerAccessor`, `ModuleAccessor`, `ParamAccessor`, `BufferAccessor`
- Internal: `FuncExecutionContext`, `VisualizationOverrides`, `FuncCallLocation`

### Constants
- UPPER_SNAKE_CASE: `FIELD_ORDER`, `ORIG_TORCH_FUNCS`, `IGNORED_FUNCS`
- `_DEVICE_CONSTRUCTOR_NAMES`, `_ATTR_SKIP_SET` for internal sets

## Error Handling
- Validation errors: `MetadataInvariantError(check_name, message)` — named checks A through R
- LayerLog multi-pass access: raises **ValueError** (not AttributeError) to avoid Python's
  property/__getattr__ trap
- `salient_args.py` extractors: try-except returns `{}` on any error (failure-safe)
- Validation replay: exceptions caught and returned as None (Bug #151 — known silent pass)
- `FuncCallLocation`: lazy properties loaded via `linecache` on first access, not at construction

## Testing Patterns

### Fixtures (tests/conftest.py)
- `default_input1` through `default_input4`: `(6,3,224,224)` standard image tensors
- `zeros_input`, `ones_input`: edge-case inputs
- `vector_input` `(5,)`, `input_2d` `(5,5)`, `input_complex` `(3,3)` complex
- `small_input` `(2,3,32,32)`: fast metadata tests
- Deterministic seeding: `torch.manual_seed(0)`, `torch.use_deterministic_algorithms(True)`

### Markers
- `@pytest.mark.slow` — real-world model tests taking >5 min
- `@pytest.mark.smoke` — 18 critical-path tests for fast validation (~6s total)
- `@pytest.mark.rare` — always excluded unless `-m rare` specified

### Test Categories
- **Toy models** (`test_toy_models.py`): `validate_saved_activations()` + `show_model_graph()` for every test
- **Real-world** (`test_real_world_models.py`): `pytest.importorskip()` for optional deps
- **Metadata** (`test_metadata.py`): `log_forward_pass()` directly, assert field properties
- **Aesthetic** (`test_output_aesthetics.py`): generates PDFs for human visual inspection

### Model Definitions
All test models live in `tests/example_models.py` (~5,400 lines). New models go here.

### Output
All test outputs → `tests/test_outputs/` (gitignored):
- `reports/` — coverage, aesthetic report, profiling
- `visualizations/` — PDFs organized by model family subdirectories

## Import Order
stdlib → third-party → local (enforced by ruff)

```python
import os
from typing import Dict, List, Optional

import torch
from torch import nn

from .utils.tensor_utils import safe_copy
from ._state import _logging_enabled
```

## Documentation
- Docstring format: NumPy style
- Type hints on all functions (including internal)
- Top-level file comments on `.py` files where purpose isn't obvious
- Each subpackage has a `CLAUDE.md` with file table, key functions, gotchas, known bugs

## Git

### Commit Messages
Conventional commits for semantic-release:
```
<type>(<scope>): <description> (#<issue>)
```

Types: `fix`, `feat`, `chore`, `docs`, `ci`, `refactor`, `test`, `style`

Scopes (common): `logging`, `vis`, `postprocess`, `capture`, `validation`, `decoration`,
`data`, `state`, `utils`, `ci`, `release`, `types`

### Branch Naming
- Feature branches: `codex/<task-id>` (kebab-case task IDs)
- One branch at a time besides main

### CI/CD
- `lint.yml`: ruff format + check on push/PR, auto-commits fixes
- `quality.yml`: mypy + pip-audit on push/PR
- `release.yml`: semantic-release v9 on push to main, PyPI via OIDC

## Field Management
FIELD_ORDER tuples in `constants.py` define complete field sets. When adding a new field:
1. Add to class definition (LayerPassLog, ModelLog, etc.)
2. Add to corresponding FIELD_ORDER in `constants.py`
3. Add test in `test_metadata.py`
4. Update `to_pandas()` if user-facing
