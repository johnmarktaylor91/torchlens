"""M1 product/lifecycle x verb seam matrix (compo memo stratum 1; F36 wave A).

Every (product state, report/read/persist verb) cell is declared ORTHOGONAL
to nothing: the matrix is CLOSED. A cell is WORKS (executes with asserted
content), REFUSES_TYPED (stable ``fields['code']`` + the right predicate in
the message -- foldB D18 law 3), or REFUSES_UNTYPED (a ledgered gap: the
DIGEST-AUDIT non-teaching-refusal class, pinned so the typing fix must flip
the cell deliberately). Unlisted cells are RED: the coverage gate fails on
any (product, verb) pair the table does not classify, in both directions.

Ground truth was probed live on the merged tree (2026-08-30); every WORKS
cell asserts content, never just absence-of-raise ("it ran" is not an
oracle).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.heavy, pytest.mark.compo]


#: Memo D12 fixture-economics license (test_galleries.py lint):
COMPO_CONSTRUCTION_LICENSE = "the M1 matrix IS the product-lifecycle roster: it constructs the ten product states (loaded/halted/structure-only/episode/coupled/partial/...) the wave-0 two-product gallery does not serve; construction is module-scoped and private"

PRODUCT_NAMES = (
    "trace_live",
    "trace_loaded",
    "trace_structure_only",
    "trace_halted",
    "trace_episode",
    "trace_episode_coupled",
    "recording",
    "partial_trace",
    "trace_slice",
    "bundle",
)

VERB_NAMES = (
    "repr",
    "summary",
    "explain",
    "agent_json",
    "save_analysis",
    "fork",
    "outcome",
    "len",
    "contains_label",
)

WORKS = "WORKS"
REFUSES_TYPED = "REFUSES_TYPED"
REFUSES_UNTYPED = "REFUSES_UNTYPED"


@dataclass(frozen=True)
class Cell:
    """One classified matrix cell.

    Parameters
    ----------
    state:
        WORKS / REFUSES_TYPED / REFUSES_UNTYPED.
    code:
        Required stable ``fields['code']`` for REFUSES_TYPED cells.
    exc:
        Exception type name required for both refusal states.
    gap:
        DIGEST-AUDIT pointer for REFUSES_UNTYPED cells (the ledgered
        untyped-refusal class; typing it must flip this cell).
    """

    state: str
    code: str = ""
    exc: str = ""
    gap: str = ""


def _w() -> Cell:
    return Cell(WORKS)


#: The closed matrix. Codes are census-visible literals (lane law): each
#: REFUSES_TYPED cell pins code= exactly; each REFUSES_UNTYPED cell names its
#: DIGEST-AUDIT row so the burn-down is enumerated, never folklore.
M1_EXPECTATIONS: dict[tuple[str, str], Cell] = {
    # Live, loaded, structure-only, halted, episode, and coupled-episode
    # Traces serve the full verb set (structure-only/halted/episode content
    # honesty is asserted by their dedicated suites; here the cell contract
    # is "serves with content").
    **{(p, v): _w() for p, v in itertools.product(PRODUCT_NAMES[:6], VERB_NAMES)},
    ("recording", "repr"): _w(),
    ("recording", "summary"): _w(),
    ("recording", "explain"): Cell(
        REFUSES_TYPED, code="report_subject_unsupported", exc="InvalidArgumentError"
    ),
    ("recording", "agent_json"): Cell(
        REFUSES_UNTYPED, exc="AttributeError", gap="DIGEST-AUDIT SG#5 / B2-10 family"
    ),
    ("recording", "save_analysis"): Cell(
        REFUSES_UNTYPED, exc="TorchLensIOError", gap="DIGEST-AUDIT SG#24 (codeless scrub refusal)"
    ),
    ("recording", "fork"): Cell(
        REFUSES_UNTYPED, exc="AttributeError", gap="DIGEST-AUDIT product-secondary-verb sweep"
    ),
    ("recording", "outcome"): _w(),
    ("recording", "len"): _w(),
    ("recording", "contains_label"): _w(),
    ("partial_trace", "repr"): _w(),
    ("partial_trace", "summary"): Cell(
        REFUSES_TYPED, code="partial_trace_member_unavailable", exc="InvalidArgumentError"
    ),
    ("partial_trace", "explain"): _w(),
    ("partial_trace", "agent_json"): Cell(
        REFUSES_TYPED, code="partial_trace_member_unavailable", exc="InvalidArgumentError"
    ),
    ("partial_trace", "save_analysis"): Cell(REFUSES_TYPED, code="N1", exc="CaptureOutcomeError"),
    ("partial_trace", "fork"): Cell(
        REFUSES_UNTYPED, exc="AttributeError", gap="DIGEST-AUDIT product-secondary-verb sweep"
    ),
    ("partial_trace", "outcome"): _w(),
    ("partial_trace", "len"): Cell(
        REFUSES_UNTYPED, exc="TypeError", gap="DIGEST-AUDIT SG#12 dunder-protocol family"
    ),
    ("partial_trace", "contains_label"): Cell(
        REFUSES_UNTYPED, exc="TypeError", gap="DIGEST-AUDIT SG#12 dunder-protocol family"
    ),
    ("trace_slice", "repr"): _w(),
    ("trace_slice", "summary"): _w(),
    ("trace_slice", "explain"): Cell(
        REFUSES_TYPED, code="report_subject_unsupported", exc="InvalidArgumentError"
    ),
    ("trace_slice", "agent_json"): Cell(
        REFUSES_UNTYPED, exc="AttributeError", gap="DIGEST-AUDIT SG#5 presenter family"
    ),
    ("trace_slice", "save_analysis"): Cell(
        REFUSES_TYPED, code="slice_save_unsupported", exc="SelectionError"
    ),
    ("trace_slice", "fork"): Cell(
        REFUSES_UNTYPED, exc="AttributeError", gap="DIGEST-AUDIT product-secondary-verb sweep"
    ),
    ("trace_slice", "outcome"): Cell(
        REFUSES_UNTYPED, exc="AttributeError", gap="DIGEST-AUDIT SG#29 presenter honesty family"
    ),
    ("trace_slice", "len"): _w(),
    ("trace_slice", "contains_label"): _w(),
    ("bundle", "repr"): _w(),
    ("bundle", "summary"): Cell(
        REFUSES_UNTYPED, exc="AttributeError", gap="DIGEST-AUDIT SG#5 presenter family"
    ),
    ("bundle", "explain"): Cell(
        REFUSES_TYPED, code="report_subject_unsupported", exc="InvalidArgumentError"
    ),
    ("bundle", "agent_json"): Cell(
        REFUSES_UNTYPED, exc="AttributeError", gap="DIGEST-AUDIT SG#5 presenter family"
    ),
    ("bundle", "save_analysis"): _w(),
    ("bundle", "fork"): _w(),
    ("bundle", "outcome"): _w(),
    ("bundle", "len"): _w(),
    ("bundle", "contains_label"): _w(),
}


class _StepModule(torch.nn.Module):
    """One toy generation step (the episode-capture substrate)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(x))


class _EpisodeRoot(torch.nn.Module):
    """A wrapper root calling its stepped module N times."""

    def __init__(self) -> None:
        super().__init__()
        self.step = _StepModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.step(x)
        return x


def build_episode_products() -> dict[str, Any]:
    """Build the toy episode + coupled-episode Traces (shared with the
    episode-option table suite; one construction recipe, two consumers)."""

    import torchlens as tl

    torch.manual_seed(0)
    root = _EpisodeRoot().eval()
    example = torch.randn(1, 4)

    def _spec() -> Any:
        return tl.options.EpisodeSpec(
            stepped_module=root.step, n_steps=3, step_output_kind="digest"
        )

    episode = tl.trace(root, example, episode=_spec())
    coupled = tl.trace(
        root,
        example,
        episode=_spec(),
        intervene=tl.when(tl.func("tanh"), tl.zero_ablate()),
    )
    return {"root": root, "example": example, "episode": episode, "coupled": coupled}


@pytest.fixture(scope="module")
def m1_products(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Build the ten-product lifecycle roster ONCE for the whole matrix."""

    import torchlens as tl

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
    ).eval()
    example = torch.randn(2, 4)

    products: dict[str, Any] = {}
    products["trace_live"] = tl.trace(
        model, example, capture=tl.options.CaptureOptions(intervention_ready=True)
    )
    save_path = tmp_path_factory.mktemp("m1") / "live.tlspec"
    tl.save(products["trace_live"], str(save_path))
    products["trace_loaded"] = tl.load(str(save_path))
    products["trace_structure_only"] = tl.trace(
        model, example, capture=tl.options.CaptureOptions(structure_only=True)
    )
    products["trace_halted"] = tl.trace(model, example, halt=tl.func("relu"))
    episode_products = build_episode_products()
    products["trace_episode"] = episode_products["episode"]
    products["trace_episode_coupled"] = episode_products["coupled"]
    products["recording"] = tl.record(model, example, save=tl.func("relu"))

    class _Boom(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("planted forward failure (M1 partial-product substrate)")

    try:
        tl.trace(_Boom(), example)
    except Exception as exc:  # noqa: BLE001 - the partial product IS the payload
        products["partial_trace"] = exc.partial_log
    trace = products["trace_live"]
    products["trace_slice"] = trace.between([trace.layer_labels[0]], [trace.layer_labels[-1]])
    products["bundle"] = tl.sweep(
        model, example, at=tl.func("relu"), values=[0.0, 0.5], include_baseline=True
    )
    assert set(products) >= set(PRODUCT_NAMES)
    try:
        yield products
    finally:
        for product in products.values():
            cleanup = getattr(product, "cleanup", None)
            if callable(cleanup):
                cleanup()


def _run_verb(verb: str, product: Any, tmp_path: Any) -> Any:
    """Execute one verb against one product; the cell contract judges it."""

    import torchlens as tl

    if verb == "repr":
        return repr(product)
    if verb == "summary":
        return product.summary()
    if verb == "explain":
        return tl.report.explain(product)
    if verb == "agent_json":
        return product.to_agent_json()
    if verb == "save_analysis":
        target = tmp_path / "cell.tlspec"
        tl.save(product, str(target))
        assert target.exists(), "save reported success but wrote no artifact"
        return "saved"
    if verb == "fork":
        return product.fork()
    if verb == "outcome":
        return product.outcome
    if verb == "len":
        return len(product)
    if verb == "contains_label":
        return "relu_1_2" in product
    raise AssertionError(f"unknown verb {verb!r}")


def _assert_works(verb: str, result: Any) -> None:
    """Content assertions per verb -- absence-of-raise is never the oracle."""

    if verb in {"repr", "summary", "explain"}:
        assert isinstance(result, str) and len(str(result)) > 20
    elif verb == "agent_json":
        assert isinstance(result, dict) and result
    elif verb == "save_analysis":
        assert result == "saved"
    elif verb == "fork":
        assert result is not None
    elif verb == "outcome":
        assert result.status.name in {
            "COMPLETE",
            "HALTED",
            "ABORTED_NONFINITE",
            "FAILED",
            "UNATTESTED",
            "UNKNOWN",
        }
    elif verb == "len":
        assert isinstance(result, int) and result >= 0
    elif verb == "contains_label":
        assert isinstance(result, bool)


@pytest.mark.parametrize("product_name", PRODUCT_NAMES)
@pytest.mark.parametrize("verb", VERB_NAMES)
def test_m1_cell_matches_declared_state(
    product_name: str, verb: str, m1_products: dict[str, Any], tmp_path: Any
) -> None:
    """Drive one (product, verb) cell and hold it to its declared contract."""

    cell = M1_EXPECTATIONS[(product_name, verb)]
    product = m1_products[product_name]
    if cell.state == WORKS:
        _assert_works(verb, _run_verb(verb, product, tmp_path))
        return
    with pytest.raises(Exception) as excinfo:
        _run_verb(verb, product, tmp_path)
    exc = excinfo.value
    assert type(exc).__name__ == cell.exc, (
        f"cell ({product_name}, {verb}) raised {type(exc).__name__}, expected"
        f" {cell.exc} -- if a lane changed this refusal, update the cell"
        " DELIBERATELY (the matrix is the record of the seam)"
    )
    code = getattr(exc, "fields", {}).get("code") if hasattr(exc, "fields") else None
    if cell.state == REFUSES_TYPED:
        assert code == cell.code, (
            f"cell ({product_name}, {verb}): typed refusal carries code={code!r},"
            f" expected {cell.code!r} (foldB D18 law 3: right code, right reason)"
        )
    else:
        assert cell.gap, "REFUSES_UNTYPED cells must name their DIGEST-AUDIT row"
        assert not code, (
            f"cell ({product_name}, {verb}) now carries code={code!r}: the untyped"
            " gap was FIXED -- promote the cell to REFUSES_TYPED and record the"
            f" burn-down of {cell.gap}"
        )


def test_m1_matrix_is_closed_both_directions() -> None:
    """Unlisted cells refuse: the table covers EXACTLY the cross product."""

    declared = set(M1_EXPECTATIONS)
    universe = set(itertools.product(PRODUCT_NAMES, VERB_NAMES))
    missing = universe - declared
    stray = declared - universe
    assert not missing, f"M1 cells with NO declared contract: {sorted(missing)}"
    assert not stray, f"M1 cells outside the declared universe: {sorted(stray)}"


def test_m1_refusal_states_carry_their_required_fields() -> None:
    """Schema teeth: typed cells pin a code, untyped cells pin a gap row."""

    for key, cell in M1_EXPECTATIONS.items():
        if cell.state == REFUSES_TYPED:
            assert cell.code and cell.exc, f"{key}: REFUSES_TYPED without code/exc"
        elif cell.state == REFUSES_UNTYPED:
            assert cell.gap and cell.exc, f"{key}: REFUSES_UNTYPED without gap/exc"
        else:
            assert cell.state == WORKS, f"{key}: unknown state {cell.state!r}"
