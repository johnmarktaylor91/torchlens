# torchlens-example-provider

A minimal installable TorchLens provider. It registers one export target,
`example_provider_summary`, through the `torchlens.export_targets` entry-point
group. Use it as a template for a provider distribution of your own.

## Install

```bash
python -m pip install -e examples/provider_package          # editable, from a checkout
python -m pip wheel --no-deps -w dist examples/provider_package
python -m pip install dist/torchlens_example_provider-0.1.0-py3-none-any.whl
```

## Use

Installing runs no provider code. Discovery reads the installed metadata only;
activation is one explicit call naming the distribution.

```python
import torch
import torchlens as tl
from torchlens._ecosystem import plugins
from torchlens.export import resolve_export_target

rows = plugins.discover("torchlens.export_targets")  # metadata only, nothing imported
plugins.activate("torchlens-example-provider")       # imports the module, registers the target

trace = tl.trace(torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.ReLU()), torch.randn(1, 3))
summary = resolve_export_target("example_provider_summary")(trace)
print(summary["num_layers"], summary["layer_labels"])
```

`TORCHLENS_PLUGINS=none` refuses every activation. Enabling a provider grants
it ordinary Python process authority; there is no sandbox.

## Files

- `pyproject.toml`: the distribution and its entry point.
- `src/torchlens_example_provider/__init__.py`: the zero-arg factory `export_target()`
  and the exporter `summarize(trace, path=None)`.
