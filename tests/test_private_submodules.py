"""Surface ruling surface.6: only taught submodule paths stay public.

A submodule whose name has no underscore reads as a public import path. The public ones are the
names a namespace lists in ``__all__`` plus the taught paths; every other submodule carries a
leading underscore, and its old attribute spelling refuses with the typed ``facade_redirect``.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

import torchlens
from torchlens._errors import FacadeTeachingError
from torchlens._utils._private_submodules import PRIVATE_SUBMODULES

#: Module paths the walk taught as public import paths (surface.6, option a).
TAUGHT = frozenset(
    {
        "torchlens.backends.jax",
        "torchlens.backends.mlx",
        "torchlens.backends.paddle",
        "torchlens.backends.registry",
        "torchlens.backends.tf",
        "torchlens.backends.tinygrad",
        "torchlens.backends.torch",
        "torchlens.intervention.errors",
        "torchlens.intervention.hooks",
        "torchlens.intervention.stochastic",
    }
)
ROOT = Path(torchlens.__file__).parent


def _literal_all(init: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(init.read_text(encoding="utf-8"))):
        targets = node.targets if isinstance(node, ast.Assign) else []
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            for item in ast.walk(node.value):
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    names.add(item.value)
    return names


def _public_children(package: Path):
    for path in sorted(package.iterdir()):
        if path.name.startswith(("_", ".")):
            continue
        if path.is_dir() and (path / "__init__.py").is_file():
            yield path.name, path
        elif path.suffix == ".py":
            yield path.stem, None


def _unlisted_public_submodules() -> list[str]:
    found: list[str] = []
    for top, top_path in _public_children(ROOT):
        if top_path is None:
            continue
        listed = _literal_all(top_path / "__init__.py")
        for name, _ in _public_children(top_path):
            dotted = f"torchlens.{top}.{name}"
            if dotted not in TAUGHT and name not in listed:
                found.append(dotted)
    return found


def test_every_untaught_unlisted_submodule_is_private() -> None:
    """No non-underscore submodule is public by accident."""

    assert _unlisted_public_submodules() == []


@pytest.mark.parametrize(
    ("owner", "name"),
    sorted((owner, name) for owner, names in PRIVATE_SUBMODULES.items() for name in names),
)
def test_old_submodule_spelling_refuses_typed(owner: str, name: str) -> None:
    """``tl.<parent>.<old leaf>`` raises the typed redirect naming the parent."""

    parent = importlib.import_module(owner)
    with pytest.raises(FacadeTeachingError) as excinfo:
        getattr(parent, name)
    assert excinfo.value.fields["code"] == "facade_redirect"
    assert f"{owner}.{name}" in str(excinfo.value)
    assert not hasattr(parent, name)


@pytest.mark.parametrize("dotted", sorted(TAUGHT))
def test_taught_paths_stay_importable(dotted: str) -> None:
    """The taught module paths still resolve under their public names (no framework import)."""

    spec = importlib.util.find_spec(dotted)
    assert spec is not None
    assert spec.name == dotted


@pytest.mark.parametrize(
    ("owner", "name"),
    [
        ("torchlens.intervention", "compose"),
        ("torchlens.intervention", "population"),
        ("torchlens.receptive_field", "rules"),
        ("torchlens.recording", "dry_run"),
        ("torchlens.recording", "recover"),
        ("torchlens.viz", "feature_maps"),
    ],
)
def test_formerly_shadowed_names_resolve_to_their_public_object(owner: str, name: str) -> None:
    """A public export that once shared its name with a submodule is never the module."""

    value = getattr(importlib.import_module(owner), name)
    assert not isinstance(value, type(torchlens))
