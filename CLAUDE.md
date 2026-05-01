# TorchLens Agent Guide

TorchLens logs PyTorch eager execution: run a normal forward pass, record operation metadata and
activations, then inspect the resulting `ModelLog`.

## Install

```bash
pip install torchlens
pip install -e ".[test]"  # local development with test extras
```

Graph rendering also needs Graphviz (`apt install graphviz` on Debian/Ubuntu).

## Common Patterns

```python
import torchlens as tl

log = tl.log_forward_pass(model, x, vis_opt="none")
activation = log["linear_1_1"].activation
print(log.summary())
print(tl.report.explain(log))
```

Use selectors for intervention discovery:

```python
log = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)
site = log.find_sites(tl.func("relu")).first()
edited = log.fork("zero_relu")
edited.attach_hooks(tl.label(site.layer_label), tl.zero_ablate())
edited.rerun(model, x)
```

Before debugging wrapper-specific failures, run:

```python
print(tl.compat.report(model, x).to_markdown())
```

## Anti-Patterns

- Do not log `torch.compile`, TorchScript, or `torch.export` artifacts; log the eager source module.
- Do not run captures concurrently across Python threads or worker processes.
- Do not expect fused kernels to expose hidden internal tensors.
- Do not put opaque callables in portable artifacts unless audit-only behavior is acceptable.
- Do not add new top-level API names casually; use submodules.

## Testing Tiers

```bash
ruff check . --fix
mypy torchlens/
pytest tests/ -m smoke -x --tb=short
pytest tests/ -m "not slow" -x --tb=short  # for public API or boundary changes
```

Use `pytest.importorskip()` for optional migration dependencies. Keep tests deterministic and run
new documentation examples when they are meant to be executable.
