---
name: torchlens
description: Use when writing, reviewing or debugging Python code that uses TorchLens (`import torchlens as tl`) to capture, save, steer, ablate, patch, compare or draw a PyTorch model's activations, or when inspecting a saved .tlspec artifact.
---

# TorchLens

Before writing TorchLens code, read the router: [ROUTER.md](ROUTER.md). It gives the call to
reach for first for each task, the wrong turn to avoid, and the page with the worked recipe.

When torchlens is installed, prefer the index that matches the installed version. This prints
its path without importing torch:

```bash
python -c "import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec('torchlens').origin).parent / '_docs' / 'AGENT_INDEX.md')"
```

To make every future session in a project find it, run `torchlens agent init` there: it adds a
short marked block to `AGENTS.md` (or each `--file`), and `torchlens agent init --remove` takes
it out again.

To inspect a saved artifact you did not make, use the read-only tools, manifest first:
`torchlens guide` prints their contract.
