"""AST regression tests banning retired TorchLens host-object attributes."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "torchlens"
LEGACY_ATTR_RE = re.compile(r"^tl__?[a-z]")
RETIRED_NAMES = {
    "tl__label_raw",
    "tl_buffer_address",
    "tl_buffer_source",
    "tl_param_barcode",
    "tl_param_address",
    "tl_call_index",
    "tl_requires_grad",
    "tl_address",
    "tl_module_type",
    "tl_is_decorated_function",
    "tl_forward_call_is_decorated",
    "tl_tensor_replacement_wrapped",
    "tl_tensor_label_raw",
    "requires_grad_original",
}
VISUALIZATION_ALLOWLIST = (
    "tl_legend_",
    "__tl_graph_panel_anchor",
    "__tl_code_panel_node",
)
#: Exact attribute names that deliberately keep the "tl_" prefix as ruled
#: public API -- not a retired name reappearing, and (for the bound-method
#: root) not host-object metadata pollution at all, since the attribute
#: lives on a TorchLens-AUTHORED class rather than a user's model/tensor.
#:
#: F41 bound-method root (foldA D11, ``torchlens/backends/torch/bound_root.py``):
#: ``TLBoundMethodRoot`` is a TorchLens wrapper class, not a host object, and
#: its properties are the ruled public surface consumed from
#: ``model_prep.py``, ``_episode_ledger.py``, and ``user_funcs.py``.
#:
#: ``neuro/_rdms.py``'s ``tl_ledger`` IS attached to a foreign
#: ``rsatoolbox.rdm.RDMs`` result, but it is a deliberate, documented,
#: session-only exception (see that module's docstring) with its own
#: regression coverage in ``tests/test_neuro_pkg_rdms.py``, which asserts
#: the attribute's presence directly.
ALLOWED_TL_PREFIXED_NAMES = frozenset(
    {
        "tl_owner",
        "tl_method_name",
        "tl_owner_class_name",
        "tl_owner_class_qualname",
        "tl_root_entry_point",
        "tl_authored_root",
        "tl_ledger",
    }
)


def _is_docstring_constant(node: ast.Constant, parent: ast.AST | None) -> bool:
    """Return whether an AST string constant is a docstring expression.

    Parameters
    ----------
    node:
        String constant node to classify.
    parent:
        Parent AST node, if known.

    Returns
    -------
    bool
        True when ``node`` is the direct expression value for a docstring.
    """
    if not isinstance(parent, ast.Expr):
        return False
    grandparent = getattr(parent, "_tl_parent", None)
    body = getattr(grandparent, "body", None)
    return bool(body and body[0] is parent and parent.value is node)


def _is_allowed_visualization_name(path: Path, name: str) -> bool:
    """Return whether a legacy-shaped name is an allowed visualization identifier."""
    if "visualization" not in path.relative_to(SOURCE_ROOT).parts:
        return False
    return any(allowed in name for allowed in VISUALIZATION_ALLOWLIST)


@pytest.mark.slow
def test_no_retired_tl_host_object_attrs_in_source() -> None:
    """Source files should not access or store retired TorchLens metadata names."""
    failures: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                setattr(child, "_tl_parent", parent)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and LEGACY_ATTR_RE.match(node.attr):
                if node.attr in ALLOWED_TL_PREFIXED_NAMES:
                    continue
                if _is_allowed_visualization_name(path, node.attr):
                    continue
                failures.append(f"{path.relative_to(ROOT)}:{node.lineno}: .{node.attr}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                parent = getattr(node, "_tl_parent", None)
                if _is_docstring_constant(node, parent):
                    continue
                if node.value in RETIRED_NAMES:
                    failures.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: string {node.value!r}"
                    )

    assert not failures, "\n".join(failures)
