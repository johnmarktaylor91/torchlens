"""r-b4 R26-2: structural gate -- private torch API touches stay inside `_torch_compat`.

The CLAUDE.md rule ("every fragile torch-private-API probe routes through
``torchlens/utils/_torch_compat.py`` and flips a named ``HAS_*`` flag") was
unenforced: probes kept escaping the boundary (the fail-open census probe, the
unguarded dispatcher-schema census, the silent DTensor geometry import, the
``python -O``-stripped TensorBase assert, ...). This AST gate makes the rule
REAL: any private ``torch.*`` attribute access, private ``torch.*`` import, or
``getattr(torch..., "_name")`` literal outside ``_torch_compat`` must appear in
the reason-bearing ledger below, so a new escape is a reviewed contract diff,
never a silent drift.

The ledger is exact-equality in BOTH directions: fixing a touch requires
deleting its row (shrink-only), adding one requires writing a reason here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from _source_corpus import package_ast, package_files

# Module-wide smoke dropped (r3settle2 budget lint): the full-package AST
# ledger scan below measures over the 5s smoke partition; per-test marks.

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "torchlens"

#: The sanctioned boundary modules: the torch compat chokepoint and its TF
#: sibling (the ONLY module allowed to import ``tensorflow.python``).
_BOUNDARY_FILES = {
    "torchlens/utils/_torch_compat.py",
    "torchlens/backends/tf/_tf_compat.py",
}

#: Reason-bearing ledger of every sanctioned private-torch touch OUTSIDE the
#: boundary. Every entry is fail-closed in direction and was individually
#: reviewed in the b4 R26 census; new rows require the same review.
_ALLOWED_PRIVATE_TOUCHES: dict[str, frozenset[str]] = {
    # Fail-closed CVE-surface / unpickler hardening checks: absence REFUSES the
    # load or shrinks the admitted class set, never degrades silently
    # (b4 R26 census, fable inventory).
    "torchlens/_io/_safe_unpickle.py": frozenset(
        {
            "torch._C",
            "getattr(torch, '_storage_classes')",
            "getattr(torch, '_tensor_classes')",
            "getattr(torch.storage, '_StorageBase')",
        }
    ),
    # Tracing-tensor classification fallback: absence falls through to the
    # compat-routed exact-type path plus structural matching; the residual is
    # disclosed by the compat row (HAS_TRACING_TENSOR_TYPES).
    "torchlens/_robustness.py": frozenset({"getattr(torch, '_is_functional_tensor')"}),
    # ``torch.ops.*`` __call__ class enumeration for the cross-thread escape
    # observers; an EMPTY structural scan fails CLOSED into
    # _HOST_ESCAPE_OBSERVER_FAILED.
    "torchlens/backends/torch/_completeness_cross_thread.py": frozenset({"import torch._ops"}),
    # Zero-copy escape-target enumeration: if the private ``_to_dlpack``
    # binding disappears, the escape route it guards disappears with it
    # (fail-neutral; nothing to patch means nothing can escape through it).
    "torchlens/backends/torch/_completeness_finalize.py": frozenset({"getattr(torch, '_C')"}),
    # TorchDispatchMode base class (public-by-usage, import fails the module
    # loudly at import time) for the shared owned-mode marker and pause
    # bracket; the l3 aten wave 0 mode-pop unification MOVED the import here
    # from completeness_witness.py, and the stack getter already routes
    # through _torch_compat (get_current_dispatch_mode_stack).
    "torchlens/backends/torch/_modes.py": frozenset({"from torch.utils._python_dispatch"}),
    # The fail-closed torch.ops enumeration (the TorchDispatchMode base-class
    # import moved to _modes.py with the l3 mode-pop unification).
    "torchlens/backends/torch/completeness_witness.py": frozenset({"import torch._ops"}),
    # Expanded-weights identity shim (SF-53 census): the conv/RNN per-sample-grad
    # picker compares the dispatched func against torch's OWN
    # ``_cudnn_rnn_flatten_weight`` symbol, so the shim must read that exact private
    # binding to normalize the identity basis torch itself uses -- routing through a
    # compat wrapper would change the object identity the comparison depends on.
    # Guarded getattr, fail-neutral: if the symbol disappears, torch's special
    # case disappears with it and the shim falls through to the alias-table path.
    # The three importlib string-literal reads (made visible by the r6/r7 R26
    # scanner extensions) are the SAME shim's module handles: gated on the
    # HAS_EXPANDED_WEIGHTS_CONV_PICKER capability flag and wrapped in a guarded
    # try whose failure abandons the shim (fail-neutral -- no expanded-weights
    # machinery means nothing to shim). The patch must land on torch's OWN
    # ``_expanded_weights`` module objects BY IDENTITY (conv_picker is imported
    # by value between them), so it cannot route through a compat copy.
    "torchlens/backends/torch/identity_shims.py": frozenset(
        {
            "getattr(torch, '_cudnn_rnn_flatten_weight')",
            "importlib.import_module('torch.nn.utils._expanded_weights.conv_utils')",
            "importlib.import_module('torch.nn.utils._expanded_weights.conv_expanded_weights')",
            "importlib.import_module('torch.nn.utils._expanded_weights.expanded_weights_impl')",
        }
    ),
    # Forward-pre-hook provenance interposition (grind-r6/r7 R26, the sol b4
    # escape the aliased-import scanner extension exists to catch): the
    # interposer must read and patch torch's REAL global pre-hook registry
    # (``torch.nn.modules.module._global_forward_pre_hooks``) because the
    # registry OBJECT IDENTITY is what torch's own Module.__call__ consults --
    # a compat-layer copy would observe nothing. Reversible interposition; a
    # registration that bypasses it is disclosed per-snapshot as
    # ``registration_interposition_bypassed``, never silently missed.
    # Fail-LOUD if the registry moves (AttributeError at wrap time).
    # Candidate for a HAS_* compat row (relay filed to the capture lane).
    "torchlens/backends/torch/prehook_provenance.py": frozenset(
        {"torch.nn.modules.module._global_forward_pre_hooks"}
    ),
    # isinstance classification against torch's own module-taxonomy base
    # classes (``_BatchNorm``, ``_DropoutNd``): the private base IS the
    # identity basis torch uses for the whole variant family
    # (BatchNorm1d/2d/3d/Sync, Dropout1d/2d/3d/Alpha), stable across the
    # entire declared 2.1->2.12+ floor. Unguarded on purpose: if torch ever
    # moves them the read fails LOUDLY at call time -- there is no
    # silent-degradation path for a capability flag to disclose.
    "torchlens/data_classes/_trace_validation.py": frozenset(
        {"torch.nn.modules.batchnorm._BatchNorm"}
    ),
    "torchlens/intervention/rerun.py": frozenset(
        {
            "torch.nn.modules.batchnorm._BatchNorm",
            "torch.nn.modules.dropout._DropoutNd",
        }
    ),
    # Same taxonomy-base class, F17 quickstart lineage (PROBEFIX microlane):
    # the pinned-execution norm-buffer snapshot/restore and the train-mode
    # running-stats disclosure classify modules against torch's own norm
    # family bases (``_BatchNorm``, ``_InstanceNorm``) -- the identity basis
    # torch uses for BatchNorm1d/2d/3d/Sync and InstanceNorm1d/2d/3d across
    # the declared 2.1->2.12+ floor. From-import spellings, unguarded on
    # purpose: if torch ever moves them the import fails LOUDLY -- there is
    # no silent-degradation path for a capability flag to disclose.
    "torchlens/quickstart/_primitive.py": frozenset(
        {
            "from torch.nn.modules.batchnorm import _BatchNorm",
            "from torch.nn.modules.instancenorm import _InstanceNorm",
        }
    ),
    "torchlens/user_funcs.py": frozenset(
        {
            "from torch.nn.modules.batchnorm import _BatchNorm",
            "from torch.nn.modules.instancenorm import _InstanceNorm",
        }
    ),
    # TF graph-only static path (r-b4 R26 opus LM, now VISIBLE to the gate):
    # Const NodeDef decoding and ConcreteFunction freezing have no public TF
    # spelling. The decode failure is fail-neutral (constant omitted, region
    # propagates "not interpretable"); the freeze import failure raises out of
    # the static path loudly. Both rows are the shrink-only tripwire for the
    # prescribed HAS_TF_TENSOR_UTIL / HAS_TF_CONVERT_TO_CONSTANTS compat
    # routing (relay filed to the backends/tf owner lane): routing them
    # through `_tf_compat` deletes these rows.
    "torchlens/backends/tf/funcgraph.py": frozenset(
        {
            "from tensorflow.python.framework",
            "from tensorflow.python.framework.convert_to_constants",
        }
    ),
}


def _is_private_segment(segment: str) -> bool:
    """Whether one dotted-path segment is private (single-underscore, not dunder)."""

    return segment.startswith("_") and not (segment.startswith("__") and segment.endswith("__"))


#: TF has no underscore convention; its ENTIRE private surface is the
#: ``tensorflow.python`` namespace, so any import of it is a private touch.
_TF_PRIVATE_ROOT = "tensorflow.python"


def _is_tf_private_module(module: str) -> bool:
    """Whether a dotted module path enters the ``tensorflow.python`` namespace."""

    return module == _TF_PRIVATE_ROOT or module.startswith(_TF_PRIVATE_ROOT + ".")


def _collect_torch_aliases(tree: ast.AST) -> dict[str, str]:
    """Map local names bound by torch imports to their absolute dotted paths.

    Union of the r6 and r7 R26 scanner extensions: ``import torch``,
    ``import torch as t``, ``import torch.x.y [as z]``, and ``from torch.a
    import b [as c]`` at any depth all become recognized chain roots.
    ``import torch.nn.modules.module as torch_module`` followed by
    ``torch_module._global_forward_pre_hooks`` was invisible to the gate (the
    import path has no private segment and the attribute chain roots at the
    alias, not at ``torch``), and ``import torch as t`` re-rooted the whole
    namespace. Deep from-import rooting is scope-blind, so a local variable
    shadowing a bound name could in principle false-positive; the
    whole-package scan is empirically clean and this gate catches drift, not
    a determined adversary.
    """

    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "torch":
                    if alias.asname:
                        aliases[alias.asname] = alias.name
                    else:
                        aliases["torch"] = "torch"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and module.split(".")[0] == "torch":
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    aliases[alias.asname or alias.name] = f"{module}.{alias.name}"
    return aliases


def _private_touches(tree: ast.AST) -> set[str]:
    """Collect private torch / tf-private touches from one module's AST."""

    touches: set[str] = set()
    aliases = _collect_torch_aliases(tree)
    nested_attribute_values = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
    }

    def _torch_chain(node: ast.AST) -> tuple[str, list[str]] | None:
        """Resolve an attribute chain to ``(absolute_root, attr_segments)``.

        The root is ``"torch"`` for literal roots and the alias's absolute
        dotted path for alias roots; private-ness of the ROOT path is the
        import statement's concern (already ledgered there), so callers test
        only the attribute segments plus any private segments the literal
        ``torch.``-rooted spelling names inline.
        """

        parts: list[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name) and current.id in aliases:
            return aliases[current.id], list(reversed(parts))
        if isinstance(current, ast.Name) and current.id == "torch":
            return "torch", list(reversed(parts))
        return None

    def _literal_import_touch(func_repr: str, module: str) -> None:
        segments = module.split(".")
        if (
            segments[0] == "torch"
            and any(_is_private_segment(s) for s in segments[1:])
            or _is_tf_private_module(module)
        ):
            touches.add(f"{func_repr}({module!r})")

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and id(node) not in nested_attribute_values:
            resolved = _torch_chain(node)
            if resolved is not None:
                root, chain = resolved
                if any(_is_private_segment(part) for part in chain):
                    touches.add(root + "." + ".".join(chain))
        elif isinstance(node, ast.Call):
            # getattr(torch..., "_private", ...) literal probes.
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and _is_private_segment(node.args[1].value)
            ):
                base = node.args[0]
                resolved = (
                    _torch_chain(base) if isinstance(base, (ast.Attribute, ast.Name)) else None
                )
                if resolved is not None:
                    root, chain = resolved
                    prefix = root if not chain else root + "." + ".".join(chain)
                    touches.add(f"getattr({prefix}, {node.args[1].value!r})")
            # importlib.import_module("torch._x") / __import__("torch._x") literals.
            elif (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                func = node.func
                if isinstance(func, ast.Name) and func.id in {"import_module", "__import__"}:
                    name = "importlib.import_module" if func.id == "import_module" else "__import__"
                    _literal_import_touch(name, node.args[0].value)
                elif (
                    isinstance(func, ast.Attribute)
                    and func.attr == "import_module"
                    and isinstance(func.value, ast.Name)
                ):
                    _literal_import_touch("importlib.import_module", node.args[0].value)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                segments = alias.name.split(".")
                if (
                    segments[0] == "torch"
                    and any(_is_private_segment(s) for s in segments[1:])
                    or _is_tf_private_module(alias.name)
                ):
                    touches.add(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            segments = module.split(".")
            if node.level != 0:
                continue
            if segments[0] == "torch":
                if any(_is_private_segment(s) for s in segments[1:]):
                    touches.add(f"from {module}")
                # Private NAMES pulled from a public torch module:
                # `from torch import _C` / `from torch.nn import _reduction`.
                for alias in node.names:
                    if _is_private_segment(alias.name):
                        touches.add(f"from {module} import {alias.name}")
            elif _is_tf_private_module(module):
                touches.add(f"from {module}")
    return touches


def _scan_package() -> dict[str, set[str]]:
    """Scan every torchlens module for private torch touches."""

    found: dict[str, set[str]] = {}
    repo_root = _PACKAGE_ROOT.parent
    for path in package_files():
        rel = path.relative_to(repo_root).as_posix()
        if rel in _BOUNDARY_FILES:
            continue
        touches = _private_touches(package_ast(path))
        if touches:
            found[rel] = touches
    return found


@pytest.mark.heavy
def test_private_torch_touches_match_the_ledger_exactly() -> None:
    """Every private torch touch outside `_torch_compat` is ledgered, both ways."""

    found = _scan_package()
    unsanctioned = {
        rel: sorted(touches - _ALLOWED_PRIVATE_TOUCHES.get(rel, frozenset()))
        for rel, touches in found.items()
        if touches - _ALLOWED_PRIVATE_TOUCHES.get(rel, frozenset())
    }
    assert not unsanctioned, (
        "Private torch API touch(es) outside torchlens/utils/_torch_compat.py: "
        f"{unsanctioned}. Route the probe through a _torch_compat accessor with a "
        "named HAS_* flag (CLAUDE.md rule), or -- for a genuinely fail-closed "
        "touch -- add a reason-bearing ledger row in this test."
    )
    stale = {
        rel: sorted(allowed - found.get(rel, set()))
        for rel, allowed in _ALLOWED_PRIVATE_TOUCHES.items()
        if allowed - found.get(rel, set())
    }
    assert not stale, (
        f"Ledgered private-touch row(s) no longer exist in the code: {stale}. "
        "Delete the stale ledger rows (the ledger is shrink-only)."
    )


def test_gate_scanner_detects_planted_offenders() -> None:
    """Planted positives: the scanner sees attribute, import, and getattr forms."""

    planted = ast.parse(
        "import torch._dynamo\n"
        "from torch.utils._python_dispatch import _get_current_dispatch_mode_stack\n"
        "x = torch._C._jit_get_all_schemas()\n"
        "y = getattr(torch._C, '_TensorBase', None)\n"
        "z = getattr(torch, '_VF', None)\n"
    )
    touches = _private_touches(planted)
    assert "import torch._dynamo" in touches
    assert "from torch.utils._python_dispatch" in touches
    assert "torch._C._jit_get_all_schemas" in touches
    assert "getattr(torch._C, '_TensorBase')" in touches
    assert "getattr(torch, '_VF')" in touches


def test_gate_scanner_detects_aliased_and_string_literal_offenders() -> None:
    """grind-r6 b4 R26 (sol MED): the scanner blind spots, planted.

    Before the extension every one of these forms passed the gate silently:
    the aliased module import roots the attribute chain at the alias name,
    the from-import binds a private name off a public module path, and the
    string-literal import never produces an Import node at all.
    """

    planted = ast.parse(
        "import importlib\n"
        "import torch.nn.modules.module as torch_module\n"
        "from torch import nn\n"
        "from torch.utils import _pytree\n"
        "from torch.nn.modules import module as mod_alias\n"
        "a = torch_module._global_forward_pre_hooks\n"
        "b = nn.modules.batchnorm._BatchNorm\n"
        "c = mod_alias._global_backward_hooks\n"
        "d = importlib.import_module('torch.nn.utils._expanded_weights.conv_utils')\n"
        "e = __import__('torch._dynamo')\n"
    )
    touches = _private_touches(planted)
    assert "torch.nn.modules.module._global_forward_pre_hooks" in touches
    assert "torch.nn.modules.batchnorm._BatchNorm" in touches
    assert "torch.nn.modules.module._global_backward_hooks" in touches
    assert "from torch.utils import _pytree" in touches
    assert "importlib.import_module('torch.nn.utils._expanded_weights.conv_utils')" in touches
    assert "__import__('torch._dynamo')" in touches


def test_gate_scanner_detects_planted_bypass_spellings() -> None:
    """r7 R26: the spellings that historically passed the gate clean.

    Each planted case below is a spelling with a LIVE (or plausible) instance
    that the pre-r7 scanner could not see: aliased module imports rooting an
    attribute chain (`prehook_provenance.py` read
    ``torch_module._global_forward_pre_hooks`` invisibly), `import torch as t`
    re-rooting, private NAMES in `from torch import ...`, string-literal
    dynamic imports, and tf-private (``tensorflow.python``) imports.
    """

    planted = ast.parse(
        "import importlib\n"
        "import torch.nn.modules.module as torch_module\n"
        "import torch as t\n"
        "from torch import _VF as V\n"
        "from torch import _C\n"
        "from torch.nn import _reduction\n"
        "h = torch_module._global_forward_pre_hooks\n"
        "g = getattr(torch_module, '_global_backward_hooks', None)\n"
        "w = t._C._nn_module_stack()\n"
        "m = importlib.import_module('torch._dynamo')\n"
        "n = __import__('torch._subclasses')\n"
        "from tensorflow.python.framework import tensor_util\n"
        "import tensorflow.python.eager.context\n"
    )
    touches = _private_touches(planted)
    assert "torch.nn.modules.module._global_forward_pre_hooks" in touches
    assert "getattr(torch.nn.modules.module, '_global_backward_hooks')" in touches
    assert "torch._C._nn_module_stack" in touches
    assert "from torch import _VF" in touches
    assert "from torch import _C" in touches
    assert "from torch.nn import _reduction" in touches
    assert "importlib.import_module('torch._dynamo')" in touches
    assert "__import__('torch._subclasses')" in touches
    assert "from tensorflow.python.framework" in touches
    assert "import tensorflow.python.eager.context" in touches


def test_gate_scanner_ignores_public_alias_use() -> None:
    """Aliased PUBLIC use stays clean: no false positives from the alias rooter."""

    planted = ast.parse(
        "import torch.nn.functional as F\n"
        "import torch.nn as nn\n"
        "from torch import nn as nn2\n"
        "a = F.relu(x)\n"
        "b = nn.Module\n"
        "c = nn2.Linear(2, 2)\n"
        "d = getattr(F, 'conv2d', None)\n"
    )
    assert _private_touches(planted) == set()
