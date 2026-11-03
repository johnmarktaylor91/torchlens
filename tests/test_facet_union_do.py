"""``do(head_a | head_b, edit)`` edits both heads in one call, as two chained ``do`` calls do.

A ``|`` of scoped head selectors used to build a composite that general site
resolution refused (``facet_selector_unsupported``), so only chained ``do`` calls
could edit two heads. The write doors now split the union into its facet
branches: every branch resolves and writes exactly as the single selector, the
recipe holds one facet hook per branch, and ``(module_a | module_b) & tl.head``
distributes over the modules. The model is the GPT-2-shaped two-block attention
model of the ``patch_from`` facet-slice tests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

import torchlens as tl
from tests.test_patch_from_facet_slice import D_HEAD, N_EMBD, SEQ, _final, _TinyGpt
from torchlens._selection.errors import SelectorCapabilityError
from torchlens.intervention.errors import EngineDispatchError
from torchlens.options import CaptureOptions, InterventionOptions

HEAD_A = tl.module("h.0.attn") & tl.head(1, "v")
HEAD_B = tl.module("h.1.attn") & tl.head(0, "v")

Case = tuple[_TinyGpt, torch.Tensor, tl.Trace, tl.Trace]


@pytest.fixture(scope="module")
def case() -> Iterator[Case]:
    model = _TinyGpt().eval()
    generator = torch.Generator().manual_seed(1)
    clean_x = torch.randn(1, SEQ, N_EMBD, generator=generator)
    corrupt_x = torch.randn(1, SEQ, N_EMBD, generator=generator)
    options = CaptureOptions(replayable=True)
    clean = tl.trace(model, clean_x, capture=options)
    corrupt = tl.trace(model, corrupt_x, capture=options)
    try:
        yield model, corrupt_x, clean, corrupt
    finally:
        clean.cleanup()
        corrupt.cleanup()


def _edits(clean: tl.Trace) -> dict[str, Callable[[], Any]]:
    return {"patch_from": lambda: tl.patch_from(clean), "zero_ablate": tl.zero_ablate}


def _do(trace: tl.Trace, site: Any, edit: Any, engine: str, model: Any, x: Any) -> tl.Trace:
    kwargs: dict[str, Any] = {"model": model, "inputs": x} if engine == "rerun" else {}
    return trace.do(site, edit, intervention=InterventionOptions(engine=engine), **kwargs)


def _facet_writes(trace: tl.Trace) -> list[tuple[str, str]]:
    """``(home op label, facet name)`` of every facet hook in the trace's recipe."""

    rows = [dict(spec.metadata) for spec in trace.recipe.hook_specs]
    return [(row["facet_home_label"], row["facet_name"]) for row in rows if row.get("facet_write")]


@pytest.mark.parametrize("engine", ["replay", "rerun"])
@pytest.mark.parametrize("edit_name", ["patch_from", "zero_ablate"])
def test_union_of_two_heads_equals_two_chained_do_calls(
    case: Case, edit_name: str, engine: str
) -> None:
    model, corrupt_x, clean, corrupt = case
    make_edit = _edits(clean)[edit_name]
    chained = _do(corrupt.fork(), HEAD_A, make_edit(), engine, model, corrupt_x)
    chained = _do(chained, HEAD_B, make_edit(), engine, model, corrupt_x)
    union = _do(corrupt.fork(), HEAD_A | HEAD_B, make_edit(), engine, model, corrupt_x)

    torch.testing.assert_close(_final(union), _final(chained), atol=0, rtol=0)
    # The control: both heads moved the output, and each head alone differs from both.
    assert not torch.allclose(_final(union), _final(corrupt), atol=1e-5)
    only_a = _do(corrupt.fork(), HEAD_A, make_edit(), engine, model, corrupt_x)
    assert not torch.allclose(_final(union), _final(only_a), atol=1e-5)


def test_union_records_one_facet_write_per_head_in_one_do(case: Case) -> None:
    _model, _x, clean, corrupt = case
    chained = corrupt.fork().do(HEAD_A, tl.patch_from(clean)).do(HEAD_B, tl.patch_from(clean))
    union = corrupt.fork()
    history_before = len(union.state_history)
    union.do(HEAD_A | HEAD_B, tl.patch_from(clean))

    writes = _facet_writes(union)
    assert len(writes) == 2
    assert writes == _facet_writes(chained)
    assert len({home for home, _ in writes}) == 2
    new_ops = [row["op"] for row in union.state_history[history_before:]]
    assert new_ops.count("do") == 1
    attach = next(r for r in union.state_history[history_before:] if r["op"] == "attach_hooks")
    assert set(attach["sites"]) == {home for home, _ in writes}


def test_module_union_and_head_distributes_over_the_modules(case: Case) -> None:
    _model, _x, _clean, corrupt = case
    head = tl.head(0, "v")
    mixed = corrupt.fork().do(
        (tl.module("h.0.attn") | tl.module("h.1.attn")) & head, tl.zero_ablate()
    )
    chained = (
        corrupt.fork()
        .do(tl.module("h.0.attn") & head, tl.zero_ablate())
        .do(tl.module("h.1.attn") & head, tl.zero_ablate())
    )
    torch.testing.assert_close(_final(mixed), _final(chained), atol=0, rtol=0)
    assert _facet_writes(mixed) == _facet_writes(chained)


def test_union_sets_a_static_value_on_every_head(case: Case) -> None:
    _model, _x, _clean, corrupt = case
    value = torch.full((1, SEQ, D_HEAD), 0.5)
    union = corrupt.fork().do(HEAD_A | HEAD_B, value)
    chained = corrupt.fork().do(HEAD_A, value).do(HEAD_B, value)
    torch.testing.assert_close(_final(union), _final(chained), atol=0, rtol=0)
    assert len(_facet_writes(union)) == 2


def test_union_with_a_plain_module_branch_edits_both(case: Case) -> None:
    _model, _x, _clean, corrupt = case
    proj = tl.module("h.0.attn.c_proj")
    union = corrupt.fork().do(HEAD_B | proj, tl.zero_ablate())
    chained = corrupt.fork().do(HEAD_B, tl.zero_ablate()).do(proj, tl.zero_ablate())
    torch.testing.assert_close(_final(union), _final(chained), atol=0, rtol=0)
    assert len(_facet_writes(union)) == 1


def test_repeated_branch_writes_once(case: Case) -> None:
    _model, _x, _clean, corrupt = case
    twice = corrupt.fork().do(HEAD_A | HEAD_A, tl.zero_ablate())
    once = corrupt.fork().do(HEAD_A, tl.zero_ablate())
    torch.testing.assert_close(_final(twice), _final(once), atol=0, rtol=0)
    assert _facet_writes(twice) == _facet_writes(once)


def test_saved_union_fork_reloads_like_the_chained_fork(case: Case, tmp_path: Path) -> None:
    """A saved and reloaded union fork holds the chained fork's values and reruns alike.

    ``trace.save`` keeps the edited values, not the facet hooks, so a rerun of
    either loaded fork refuses the same way (the single-facet contract).
    """

    model, corrupt_x, clean, corrupt = case
    union = corrupt.fork().do(HEAD_A | HEAD_B, tl.patch_from(clean))
    chained = corrupt.fork().do(HEAD_A, tl.patch_from(clean)).do(HEAD_B, tl.patch_from(clean))
    loaded = {}
    for name, trace in (("union", union), ("chained", chained)):
        trace.save(tmp_path / name, include_custom_attributes=False)
        loaded[name] = tl.load(tmp_path / name)

    torch.testing.assert_close(_final(loaded["union"]), _final(union), atol=0, rtol=0)
    torch.testing.assert_close(_final(loaded["union"]), _final(loaded["chained"]), atol=0, rtol=0)
    refusals = []
    for name in ("union", "chained"):
        with pytest.raises(EngineDispatchError) as caught:
            loaded[name].fork().run(model, corrupt_x)
        refusals.append(caught.value.fields.get("code"))
    assert refusals[0] == refusals[1]


def test_find_sites_over_a_facet_union_still_refuses_and_names_the_do_spelling(
    case: Case,
) -> None:
    _model, _x, _clean, corrupt = case
    with pytest.raises(SelectorCapabilityError) as caught:
        corrupt.find_sites(HEAD_A | HEAD_B)
    assert caught.value.fields["code"] == "facet_selector_unsupported"
    assert "do(module_a & tl.head(i) | module_b & tl.head(j), edit)" in str(caught.value)


def test_union_with_a_facet_inside_another_composite_refuses_before_any_write(
    case: Case,
) -> None:
    """``facet & tl.func(...)`` is not a facet site, so the union refuses as a whole."""

    _model, _x, _clean, corrupt = case
    fork = corrupt.fork()
    with pytest.raises(SelectorCapabilityError) as caught:
        fork.do((HEAD_A & tl.func("linear")) | HEAD_B, tl.zero_ablate())
    assert caught.value.fields["code"] == "facet_selector_unsupported"
    assert _facet_writes(fork) == []
