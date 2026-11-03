"""Torch submodule alias wrap targets, picked up once their declaring submodule is imported.

Split out of ``_constants`` (whose ``get_orig_torch_funcs`` appends these
targets); stdlib-only basis module.
"""

from __future__ import annotations

import sys

__tl_layer__ = "L0"

# Submodules that re-export (``from torch import X as Y``) a plain torch
# function under a second name, lazily imported by torch itself rather than
# eagerly by ``import torch`` -- so the one-time ``ORIG_TORCH_FUNCS`` scan at
# ``import torchlens`` time can miss them. Each row is the (submodule, attr)
# pair to track once that submodule has been imported by anyone. Dynamo's own
# ``torch._dynamo.trace_rules`` resolves ``torch.onnx.operators.shape_as_tensor``
# as a tracked overridable; if that submodule's FIRST import lands while
# TorchLens wrappers are installed (e.g. the import happens from inside a
# dynamo rule-table build mid-epoch), its module-level alias binds to the
# wrapper, and -- absent this entry -- neither ``unwrap_torch`` nor the next
# ``wrap_torch`` ever knew that namespace existed to restore or re-wrap it,
# permanently poisoning dynamo's identity-keyed rule map for the rest of the
# process (round-2 CI triage, 2026-10-01). Since torch 2.8 the same module
# also aliases ``reshape_from_tensor_shape = torch._reshape_from_tensor``; once
# the module is in the roster, the public-site audit sees that alias holding
# the unwrapped original unless it is tracked here too.
#
# ``torch.nn.utils.weight_norm`` binds ``from torch import _weight_norm,
# norm_except_dim`` at ``import torch`` time, before any wrapping. The legacy
# ``WeightNorm`` forward pre-hook calls that module global on every forward, so
# without these rows the capture never sees ``_weight_norm``: the plain capture
# needs the mode rescue re-run to record it, and the validation capture (no
# rescue) misses it outright. ``torch.nn.utils.parametrizations.weight_norm``
# looks ``torch._weight_norm`` up at call time and needs no row.
_TORCH_SUBMODULE_ALIAS_TARGETS: tuple[tuple[str, str], ...] = (
    ("torch.onnx.operators", "shape_as_tensor"),
    ("torch.onnx.operators", "reshape_from_tensor_shape"),
    ("torch.nn.utils.weight_norm", "_weight_norm"),
    ("torch.nn.utils.weight_norm", "norm_except_dim"),
)


def _get_torch_submodule_alias_funcs() -> list[tuple[str, str]]:
    """Return alias targets whose declaring submodule is already imported.

    Deliberately uncached (like :func:`_get_torchvision_funcs`): a submodule
    imported after the last wrap is picked up on the very next
    ``get_orig_torch_funcs()`` call, which both ``wrap_torch`` (decoration /
    re-install) and ``unwrap_torch`` (restore) make on every invocation.

    Returns
    -------
    list[tuple[str, str]]
        ``(submodule_name, attr)`` pairs ready for ``_decorate_torch_func_pairs``
        / restoration, limited to submodules already present in
        ``sys.modules``.
    """

    found: list[tuple[str, str]] = []
    for module_name, attr in _TORCH_SUBMODULE_ALIAS_TARGETS:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        spec = getattr(module, "__spec__", None)
        if spec is not None and getattr(spec, "_initializing", False):
            continue  # mid-import (circular import): attribute may not exist yet
        if hasattr(module, attr):
            found.append((module_name, attr))
    return found
