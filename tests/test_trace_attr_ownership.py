"""b5 R50-1a: no package attaches an UNDECLARED private field to a Trace.

Backends, capture, validation, viz, and I/O all reach in and write private
attributes onto the Trace (``trace._mlx_op_captures = ...``,
``setattr(trace, "_validation_replay_status", ...)``). Most of those names
appear in NO authority: not ``Trace.FIELD_POLICY`` (so no policy, no schema
row, no save/load decision), and not even ``_io/scrub.py``'s runtime-only
allowance. SF-39 was one instance of exactly this shape, found and fixed by
hand -- and nothing structural stopped the next one, which is why this is a
GOVERNANCE gate rather than a one-off fix.

The rule: every private attribute written onto a trace-shaped name from
OUTSIDE ``torchlens/data_classes/`` must either

* be declared in ``TRACE_FIELD_OWNERSHIP`` (i.e. it is a real Trace field with
  a ``FieldPolicy`` and an owning component), or
* carry a reason-bearing row in ``TRACE_EXTERNAL_WRITE_EXEMPTIONS``.

Both directions are exact, so the seeded 36 exemptions are shrink-only: the
gate is what makes the 37th attachment a reviewed decision. Enrollment edits
(declaring a field on ``Trace``) belong to the ``data_classes/trace.py`` owner
lane; this gate ships with the census as its seed and shrinks as they land.

Scope: writes only. Private string READS on a trace (``getattr(trace, "_x",
default)``) are the sibling fail-open class, gated by
``tests/test_no_string_private_getattr.py``.
"""

from __future__ import annotations

import ast
import collections
from functools import lru_cache
from pathlib import Path

import pytest
from _source_corpus import package_ast, package_files

from torchlens.data_classes._trace_components import (
    TRACE_EXTERNAL_WRITE_EXEMPTIONS,
    TRACE_FIELD_OWNERSHIP,
)

# Per-test smoke marks (the former module-level pytestmark was additive
# with the census's heavy re-tier and tripped the marker-combination
# lint; every non-census test stays in the smoke tier).

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "torchlens"

#: The one package allowed to define Trace state: the Trace's own home.
_OWNER_PACKAGE = "data_classes"

#: Identifiers that denote a ``Trace``. Kept in sync with the sibling reach-in
#: gate; a new spelling for the same object belongs here, not outside the gate.
_TRACE_IDENTIFIERS = frozenset(
    {
        "trace",
        "new_trace",
        "log",
        "model_log",
        "ml",
        "target_trace",
        "source_trace",
        "refreshed",
        # R50-1: `_io/bundle.py`'s load path binds the trace as `loaded_trace`;
        # the write of `_source_bundle_model_fingerprint` survived three rounds
        # of review purely because this spelling was missing here.
        "loaded_trace",
    }
)


def _is_private_name(name: str) -> bool:
    """Whether an attribute name is single-underscore private (not dunder)."""

    return name.startswith("_") and not name.startswith("__")


def _is_trace_base(base: str) -> bool:
    """Whether an unparsed assignment base denotes a Trace."""

    return base.split(".")[-1].split("[")[0] in _TRACE_IDENTIFIERS


def _external_writes(tree: ast.AST) -> list[tuple[str, int, str]]:
    """Collect ``(attr, lineno, kind)`` private writes onto trace-shaped bases."""

    found: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            targets = [node.target]
        else:
            targets = []
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and _is_private_name(target.attr)
                and _is_trace_base(ast.unparse(target.value))
            ):
                found.append((target.attr, node.lineno, "assign"))
        # `setattr`/`delattr` with a literal name is the same write, spelled to
        # dodge both SLF001 and a naive assignment scan.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"setattr", "delattr"}
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and _is_private_name(node.args[1].value)
            and _is_trace_base(ast.unparse(node.args[0]))
        ):
            found.append((node.args[1].value, node.lineno, node.func.id))
        # ``trace.__dict__["_x"] = v`` and ``trace.__dict__.setdefault("_x",
        # ...)`` attach the same undeclared field while dodging both the
        # attribute-assign scan and setattr (grind-r6 b5 R45:
        # _backward_grad_fn_type_counter broke tl.save through exactly this
        # spelling).
        for target in targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "__dict__"
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
                and _is_private_name(target.slice.value)
                and _is_trace_base(ast.unparse(target.value.value))
            ):
                found.append((target.slice.value, node.lineno, "dict-assign"))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "__dict__"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and _is_private_name(node.args[0].value)
            and _is_trace_base(ast.unparse(node.func.value.value))
        ):
            found.append((node.args[0].value, node.lineno, "dict-setdefault"))
    return found


@lru_cache(maxsize=1)
def _scan_external_writes() -> dict[str, list[str]]:
    """Attribute -> sorted ``file:line kind`` write sites outside data_classes/."""

    found: dict[str, list[str]] = collections.defaultdict(list)
    for path in package_files():
        parts = path.relative_to(_PACKAGE_ROOT).parts
        if parts[0] == _OWNER_PACKAGE:
            continue
        rel = path.relative_to(_PACKAGE_ROOT.parent).as_posix()
        tree = package_ast(path)
        for attr, lineno, kind in _external_writes(tree):
            found[attr].append(f"{rel}:{lineno} ({kind})")
    return {attr: sorted(sites) for attr, sites in found.items()}


# Whole-package AST census cost (shared lru_cache with the census tests
# below, but whichever consumer runs FIRST pays the ~5.7s scan): heavy per
# the 5-20s marker partition, same call as the L1-merge re-tier below.
@pytest.mark.heavy
def test_every_external_private_write_is_declared_or_exempted() -> None:
    """No package attaches an undeclared private field to the Trace."""

    written = _scan_external_writes()
    undeclared = {
        attr: sites
        for attr, sites in written.items()
        if attr not in TRACE_FIELD_OWNERSHIP and attr not in TRACE_EXTERNAL_WRITE_EXEMPTIONS
    }
    assert not undeclared, (
        "Private Trace attribute(s) written from outside data_classes/ with no "
        f"declared owner and no exemption: {undeclared}. Declare the field on "
        "Trace (FIELD_POLICY + a TRACE_FIELD_OWNERSHIP row) so it has a policy "
        "and a schema decision, or -- if it is genuinely transient backend "
        "scratch -- add a reason-bearing row to "
        "TRACE_EXTERNAL_WRITE_EXEMPTIONS in "
        "torchlens/data_classes/_trace_components.py."
    )


# heavy by measured cost, not preference: the package-wide external-write
# scan crossed the 5s smoke boundary as the feature-sprint lanes grew the
# tree (5.7s standalone, 2026-08-17); the 5-20s partition places it heavy.
@pytest.mark.heavy
def test_exemption_ledger_is_shrink_only() -> None:
    """Every exemption row corresponds to a real, still-present external write."""

    written = _scan_external_writes()
    stale = sorted(set(TRACE_EXTERNAL_WRITE_EXEMPTIONS) - set(written))
    assert not stale, (
        f"Exemption row(s) with no external write left: {stale}. Delete them -- "
        "the ledger is shrink-only, and a stale row silently re-opens the class."
    )


def test_exemptions_and_declared_ownership_are_disjoint() -> None:
    """A name is either a declared Trace field or an exemption, never both."""

    both = sorted(set(TRACE_EXTERNAL_WRITE_EXEMPTIONS) & set(TRACE_FIELD_OWNERSHIP))
    assert not both, (
        f"Name(s) both declared and exempted: {both}. Once a field is enrolled "
        "in TRACE_FIELD_OWNERSHIP its exemption row is dead -- delete it, or the "
        "exemption outlives the reason it was written for."
    )


def test_every_exemption_carries_a_reason() -> None:
    """Exemption values are real sentences, not placeholders."""

    thin = {
        attr: reason
        for attr, reason in TRACE_EXTERNAL_WRITE_EXEMPTIONS.items()
        if len(reason.strip()) < 25 or ":" not in reason
    }
    assert not thin, (
        f"Exemption row(s) without a `<owner>: <why>` reason: {thin}. The reason "
        "is the whole point -- it is what a future reviewer reads instead of "
        "re-deriving the lifetime of an undeclared attribute."
    )


# Whole-package AST census: crossed the 5s smoke partition as the tree grew
# through the wave-0 merges (6.8s isolated at the L1 merge); heavy per the
# marker partition (5-20s), not a budget dodge.
@pytest.mark.heavy
def test_declared_writes_still_dominate_the_census() -> None:
    """Sanity floor: most external private writes ARE declared fields.

    Guards against a scanner regression that silently stops seeing writes (a
    broken vocabulary or base-unparse change would make the gate vacuous).
    """

    written = _scan_external_writes()
    declared = [attr for attr in written if attr in TRACE_FIELD_OWNERSHIP]
    assert len(written) >= 100, f"scanner found only {len(written)} written attrs; expected ~108"
    assert len(declared) >= 60, f"only {len(declared)} declared writes seen; scanner may be blind"


def test_gate_scanner_detects_planted_offenders() -> None:
    """Planted positives/negatives across assign, augassign, setattr, delattr."""

    planted = ast.parse(
        "trace._brand_new_attr = 1\n"
        "log._counter += 1\n"
        "ml._annotated: int = 2\n"
        "setattr(model_log, '_via_setattr', 3)\n"
        "delattr(state.trace, '_via_delattr')\n"
        # Negatives: own state, public name, dunder, non-trace base, non-literal.
        "self._own_state = 4\n"
        "trace.public_field = 5\n"
        "trace.__dict__['x'] = 6\n"
        "some_module._private = 7\n"
        "setattr(trace, name, 8)\n"
    )
    attrs = sorted(attr for attr, _, _ in _external_writes(planted))
    assert attrs == [
        "_annotated",
        "_brand_new_attr",
        "_counter",
        "_via_delattr",
        "_via_setattr",
    ]


def test_gate_scanner_detects_dict_spellings_and_loaded_trace() -> None:
    """grind-r6 b5 R45/R50: the ``__dict__`` spellings and the local-name gap.

    ``trace.__dict__["_x"] = v`` and ``trace.__dict__.setdefault("_x", ...)``
    attach the same undeclared field as an attribute assignment while dodging
    the assign/setattr scans (this is exactly how
    ``_backward_grad_fn_type_counter`` broke ``tl.save``), and
    ``loaded_trace``-based writes rooted outside the original 8-identifier
    allowlist were invisible entirely.
    """

    planted = ast.parse(
        "trace.__dict__['_via_dict_assign'] = 1\n"
        "log.__dict__.setdefault('_via_dict_setdefault', {})\n"
        "loaded_trace._via_loaded_trace = 2\n"
        # Negatives: public key, non-literal key, non-trace base.
        "trace.__dict__['public'] = 3\n"
        "trace.__dict__[key] = 4\n"
        "config.__dict__['_not_a_trace'] = 5\n"
    )
    attrs = sorted(attr for attr, _, _ in _external_writes(planted))
    assert attrs == [
        "_via_dict_assign",
        "_via_dict_setdefault",
        "_via_loaded_trace",
    ]
