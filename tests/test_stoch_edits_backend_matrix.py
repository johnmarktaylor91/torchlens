"""F02 backend refusal matrix (edits memo D36; row A6, structural half).

The stochastic/population family executes on torch only. Construction, repr,
equality, and pickle identity work backend-free; EXECUTION on a preview
backend refuses at that backend's helper-resolution preflight, because the
adapters resolve builtin helpers BY NAME against closed tables and raise a
typed backend refusal for anything outside them.

The preview backend libraries are not installed on the gate box, so the
fail-closed property is pinned STRUCTURALLY: the closed adapter tables (MLX,
Paddle) and the curated TF if-chain must not admit any new-family name. A
future adapter row for one of these verbs is a deliberate reviewed change
(with real per-backend semantics + parity oracles), never a silent widening.
The live execution legs run under importorskip when a preview backend exists.
"""

from __future__ import annotations

import ast
import pickle
import re
from pathlib import Path

import pytest
import torch

from torchlens.intervention import (
    compose,
    mean_fill,
    permute_batch,
    reference,
    resample_rows_from,
    sample_from,
    set_direction_mean,
)

_REPO = Path(__file__).resolve().parent.parent

#: Every new-family helper name (the D36 matrix's rows).
_NEW_FAMILY_NAMES = frozenset(
    {
        "patch_from",
        "permute_batch",
        "resample_rows_from",
        "mean_from",
        "mean_fill",
        "set_direction_mean",
        "compose",
    }
)


def _tuple_literal(path: Path, name: str) -> frozenset[str]:
    """Read a module-level string-tuple literal from source (no backend import)."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if name in targets and isinstance(node.value, (ast.Tuple, ast.List)):
                return frozenset(
                    element.value
                    for element in node.value.elts
                    if isinstance(element, ast.Constant) and isinstance(element.value, str)
                )
    raise AssertionError(f"{name} tuple not found in {path}")


def test_mlx_adapter_table_admits_no_new_family_name() -> None:
    """The MLX closed helper table stays disjoint from the stochastic family."""

    table = _tuple_literal(
        _REPO / "torchlens" / "backends" / "mlx" / "interventions.py",
        "_MLX_SUPPORTED_HELPER_NAMES",
    )
    assert table, "the MLX table parsed empty"
    assert not (table & _NEW_FAMILY_NAMES)


def test_paddle_adapter_table_admits_no_new_family_name() -> None:
    """The Paddle closed helper table stays disjoint from the stochastic family."""

    table = _tuple_literal(
        _REPO / "torchlens" / "backends" / "paddle" / "interventions.py",
        "_PADDLE_SUPPORTED_HELPER_NAMES",
    )
    assert table, "the Paddle table parsed empty"
    assert not (table & _NEW_FAMILY_NAMES)


def test_tf_curated_chain_admits_no_new_family_name() -> None:
    """The TF curated helper if-chain stays disjoint from the stochastic family."""

    source = (_REPO / "torchlens" / "backends" / "tf" / "interventions.py").read_text(
        encoding="utf-8"
    )
    compared = set(re.findall(r"helper_name\s*==\s*['\"]([a-z_]+)['\"]", source))
    assert compared, "the TF curated chain parsed empty"
    assert not (compared & _NEW_FAMILY_NAMES)


@pytest.mark.smoke
def test_construction_and_identity_are_backend_free() -> None:
    """D36: construction, repr, equality, and pickle identity need no backend."""

    donors = reference(torch.randn(3, 4), origin="matrix")
    site_shaped = reference(torch.randn(3, 5, 4), origin="site-shaped matrix")
    plan = sample_from(donors, seed=1)
    specs = [
        permute_batch(seed=1, axis=0),
        resample_rows_from(donors, seed=1, axis=0),
        mean_fill(over="all"),
        set_direction_mean(torch.ones(4), site_shaped, feature_axis=1),
        compose(mean_fill(over="all")),
    ]
    del plan
    for spec in specs:
        assert repr(spec)
        assert spec == spec
    # Builtin-portability members round-trip identity through pickle.
    for spec in (permute_batch(seed=1, axis=0), mean_fill(over="all")):
        assert pickle.loads(pickle.dumps(spec)) == spec


def test_mlx_live_leg_refuses_new_family() -> None:
    """Live A6 leg (runs only where MLX is installed): typed refusal, no forward."""

    mlx = pytest.importorskip("mlx.core")
    del mlx
    from torchlens.backends.mlx.interventions import resolve_helper_applier
    from torchlens.errors import BackendUnsupportedError

    with pytest.raises(BackendUnsupportedError):
        resolve_helper_applier(permute_batch(seed=1, axis=0))
