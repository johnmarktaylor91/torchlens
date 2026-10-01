"""The edit x product matrix (compo memo stratum 1; F36 wave A).

``do()`` against every product/lifecycle state: the push engine serves live
intervention-ready forks and refuses everything else -- each refusal cell
pins its stable code (or its LEDGERED codeless state: the intervention-error
fields-clobber class locks ReplayPreconditionError out of the contract
today, and this matrix is where that burn-down is enumerated).

Ground truth probed live on the merged tree, 2026-08-30.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.heavy, pytest.mark.compo]


#: Memo D12 fixture-economics license (test_galleries.py lint):
COMPO_CONSTRUCTION_LICENSE = "the edit x product matrix constructs the refused product states (loaded/halted/structure-only) that cannot ride the shared live gallery"

EDIT_SITE_BY_PRODUCT = {
    "trace_live_fork": "relu_1_2",
    "trace_live_no_fork": "relu_1_2",
    "trace_loaded": "relu_1_2",
    "trace_structure_only": "relu_1_2",
    "trace_halted": "linear_1_1",
    "trace_not_intervention_ready": "relu_1_2",
}


@dataclass(frozen=True)
class EditCell:
    """One edit x product cell: WORKS[+disclosure] or a refusal contract."""

    state: str  # WORKS | WORKS_WITH_DISCLOSURE | REFUSES_TYPED | REFUSES_CODELESS
    code: str = ""
    exc: str = ""
    gap: str = ""


EDIT_PRODUCT_MATRIX: dict[str, EditCell] = {
    "trace_live_fork": EditCell("WORKS"),
    "trace_live_no_fork": EditCell("WORKS_WITH_DISCLOSURE"),  # MutateInPlaceWarning
    "trace_loaded": EditCell(
        "REFUSES_CODELESS",
        exc="ReplayPreconditionError",
        gap="DIGEST-AUDIT fields-clobber: intervention ctor destroys .fields",
    ),
    "trace_structure_only": EditCell(
        "REFUSES_TYPED",
        code="structure_only_replay_unsupported",
        exc="StructureOnlyCapabilityError",
    ),
    "trace_halted": EditCell("REFUSES_TYPED", code="N5", exc="CaptureOutcomeError"),
    "trace_not_intervention_ready": EditCell(
        "REFUSES_CODELESS",
        exc="ReplayPreconditionError",
        gap="DIGEST-AUDIT fields-clobber: intervention ctor destroys .fields",
    ),
}


@pytest.fixture(scope="module")
def edit_products(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Build the product roster the edit verb is driven against."""

    import torchlens as tl

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
    ).eval()
    example = torch.randn(2, 4)
    live = tl.trace(model, example, capture=tl.options.CaptureOptions(intervention_ready=True))
    save_path = tmp_path_factory.mktemp("editmx") / "live.tlspec"
    tl.save(live, str(save_path))
    products = {
        "trace_live_fork": live,
        "trace_live_no_fork": live,
        "trace_loaded": tl.load(str(save_path)),
        "trace_structure_only": tl.trace(
            model, example, capture=tl.options.CaptureOptions(structure_only=True)
        ),
        "trace_halted": tl.trace(model, example, halt=tl.func("relu")),
        "trace_not_intervention_ready": tl.trace(model, example),
    }
    try:
        yield products
    finally:
        unique_products = {id(value): value for value in products.values()}
        for product in unique_products.values():
            cleanup = getattr(product, "cleanup", None)
            if callable(cleanup):
                cleanup()


@pytest.mark.parametrize("product_name", sorted(EDIT_PRODUCT_MATRIX))
def test_edit_cell_matches_declared_state(product_name: str, edit_products: dict[str, Any]) -> None:
    """Drive do() into one product state and hold it to its cell contract."""

    import torchlens as tl

    cell = EDIT_PRODUCT_MATRIX[product_name]
    product = edit_products[product_name]
    site = EDIT_SITE_BY_PRODUCT[product_name]

    def _edit(target: Any) -> Any:
        return target.do(site, tl.zero_ablate())

    if cell.state == "WORKS":
        fork = product.fork()
        try:
            _edit(fork)
            assert torch.count_nonzero(fork[site].out) == 0, "edit recorded but not applied"
            assert fork.intervention_audit, "no audit row for a successful edit"
        finally:
            fork.cleanup()
        return

    if cell.state == "WORKS_WITH_DISCLOSURE":
        # The cell IS "do() on the source trace without fork()": build a
        # PRIVATE capture (mutating the shared fixture would poison later
        # cells) and assert the in-place disclosure fires on it.
        import torchlens as tl_mod

        torch.manual_seed(0)
        private_model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
        ).eval()
        private = tl_mod.trace(
            private_model,
            torch.randn(2, 4),
            capture=tl_mod.options.CaptureOptions(intervention_ready=True),
        )
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _edit(private)
            names = [type(entry.message).__name__ for entry in caught]
            assert "MutateInPlaceWarning" in names, (
                f"in-place do() on a live trace disclosed {names}: the mutate-"
                "in-place disclosure is the cell's contract"
            )
        finally:
            private.cleanup()
        return

    with pytest.raises(Exception) as excinfo:
        _edit(product.fork() if cell.state == "REFUSES_CODELESS" else product.fork())
    exc = excinfo.value
    assert type(exc).__name__ == cell.exc, (
        f"{product_name}: raised {type(exc).__name__}, cell says {cell.exc}"
    )
    code = getattr(exc, "fields", {}).get("code") if hasattr(exc, "fields") else None
    if cell.state == "REFUSES_TYPED":
        assert code == cell.code, f"{product_name}: code={code!r}, cell says {cell.code!r}"
    else:
        assert cell.gap, f"{product_name}: codeless cell must name its ledger row"
        assert not code, (
            f"{product_name}: the codeless refusal now carries code={code!r} --"
            f" the fields-clobber fix landed; promote the cell and burn down {cell.gap}"
        )


def test_edit_matrix_is_closed() -> None:
    """Every product in the fixture roster has a declared edit cell."""

    assert set(EDIT_PRODUCT_MATRIX) == set(EDIT_SITE_BY_PRODUCT)


def test_selection_lane_edit_on_presenters_refuses_foreign_trace() -> None:
    """A resolved Selection from one trace cannot do() into another product
    (the one explicit cross-run door is align_to; foldB D18 law 1: TRY it)."""

    import torchlens as tl

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
    ).eval()
    example = torch.randn(2, 4)
    options = tl.options.CaptureOptions(intervention_ready=True)
    first = tl.trace(model, example, capture=options)
    second = tl.trace(model, example, capture=options)
    try:
        resolved_on_first = first["relu_1_2"].__selection__().resolve(first)
        fork = second.fork()
        try:
            with pytest.raises(Exception) as excinfo:
                fork.do(resolved_on_first, tl.zero_ablate())
            code = (
                getattr(excinfo.value, "fields", {}).get("code")
                if hasattr(excinfo.value, "fields")
                else None
            )
            assert code == "selection_trace_mismatch", (
                f"foreign resolved-selection do() refused with code={code!r};"
                " the contract pins selection_trace_mismatch (align_to is the"
                " one explicit door)"
            )
        finally:
            fork.cleanup()
    finally:
        first.cleanup()
        second.cleanup()
