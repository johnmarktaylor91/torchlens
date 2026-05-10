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
    "tl_buffer_parent",
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
    "tl_elk_",
    "__tl_graph_panel_anchor",
    "__tl_code_panel_node",
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


@pytest.mark.smoke
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
