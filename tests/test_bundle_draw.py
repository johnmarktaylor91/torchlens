"""``Bundle.draw`` and the Bundle split: real methods, one draw door, private delta reads."""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import pytest
import torch

import torchlens as tl
from torchlens.errors import FacadeTeachingError

_REAL_METHODS = (
    "store_comparison",
    "stored_comparison",
    "stored_comparison_names",
    "upstream",
    "downstream",
    "token_pairs",
    "token_divergence",
    "add_alternative",
    "draw",
)


class _TinyRelu(torch.nn.Module):
    def __init__(self, offset: float = 0.0) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        self.relu = torch.nn.ReLU()
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x)) + self.offset


def _pair() -> tl.Bundle:
    """Two unrelated members: comparisons refuse members joined by an ordering relation."""
    torch.manual_seed(0)
    x = torch.randn(2, 3)
    clean = tl.trace(_TinyRelu(), x)
    changed = tl.trace(_TinyRelu(offset=0.5), x)
    return tl.Bundle({"clean": clean, "changed": changed}, baseline="clean")


@pytest.mark.smoke
def test_bundle_draw_returns_dot_for_the_union_graph(tmp_path: Path) -> None:
    if shutil.which("dot") is None:
        pytest.skip("graphviz dot not installed")
    bundle = _pair()
    dot = bundle.draw(
        str(tmp_path / "union.svg"),
        node_overrides={"linear_1_1": {"fillcolor": "#ffeeaa"}},
        open_viewer=False,
    )
    assert isinstance(dot, str) and "digraph" in dot
    assert "#ffeeaa" in dot
    assert (tmp_path / "union.svg").exists()


@pytest.mark.smoke
def test_bundle_draw_diff_folds_show_diff(tmp_path: Path) -> None:
    if shutil.which("dot") is None:
        pytest.skip("graphviz dot not installed")
    bundle = _pair()
    dot = bundle.draw(
        graph="diff", members=("clean", "changed"), path=str(tmp_path / "d"), open_viewer=False
    )
    assert dot.startswith("// TorchLens bundle diff")
    assert "clean vs changed" in dot
    assert (tmp_path / "d.svg").exists()
    with pytest.raises(FacadeTeachingError) as excinfo:
        bundle.show_diff
    assert "draw(graph='diff'" in str(excinfo.value)


@pytest.mark.smoke
def test_former_helper_table_methods_are_real_methods() -> None:
    for name in _REAL_METHODS:
        assert inspect.isfunction(inspect.getattr_static(tl.Bundle, name)), name
    assert "__dir__" not in vars(tl.Bundle)
    bundle = _pair()
    name = bundle.store_comparison("relative_l2")
    assert bundle.stored_comparison_names() == (name,)
    assert bundle.stored_comparison(name) == bundle._delta_map("relative_l2")
    bundle.relate("forked_from", "changed", "clean", at_step=0)
    assert bundle.upstream("changed") == ("clean",)
    assert bundle.downstream("clean") == ("changed",)


@pytest.mark.smoke
@pytest.mark.parametrize("name", ["delta_map", "norm_delta", "output_delta", "aligned_pairs"])
def test_delta_family_is_private_behind_compare(name: str) -> None:
    bundle = _pair()
    with pytest.raises(FacadeTeachingError) as excinfo:
        getattr(bundle, name)
    assert excinfo.value.fields["attribute"] == name
    assert callable(getattr(bundle, f"_{name}"))
    rows = bundle.compare("clean", "changed", metric="relative_l2")
    assert rows and set(dict(rows)) <= set(bundle._delta_map("relative_l2"))
