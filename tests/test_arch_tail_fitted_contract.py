"""Fitted contract: the no-optimizer lint + provenance field-set pin (F35; item 20).

The architecture memo's fitted settlement (3.2): fittedness is a CONTRACT,
not a layer -- one provenance-shaped record, one gate at every accepting
seam, refuse-silent-refit, and the AST lint that makes "torchlens never
trains" true by construction. The per-seam refusal behavior is tested by
the owning lanes (tests/test_transforms_lib_projection.py pins
``transform_params_invalid``, width-mismatch refusal, and the
``pca_fitted_digest_mismatch`` tamper refusal). This suite owns the two
package-wide pieces:

- **No-optimizer AST lint**: no ``torch.optim`` optimizer construction and
  no optimizer ``.step()`` call anywhere in ``torchlens/``, behind a
  reason-bearing closed exemption ledger (one row today: the pinned
  memory-profiler parity scenario, which trains its OWN 16-unit MLP to
  exercise torch's categorizer and never touches user state).
- **Provenance field-set pin**: the shipped fitted seam
  (``tl.stats.PCA.fitted() -> tl.transforms.pca_apply``) carries the
  FittedProvenance-shaped identity on the spec it emits -- algorithm
  identity, fitting-data digest, fit scope, and the output geometry a
  planner can price without executing. Dropping a field is a contract
  regression, not a refactor.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO / "torchlens"

#: Reason-bearing exemption ledger for the no-optimizer lint. CLOSED: a new
#: row requires the same justification class (a self-contained diagnostic
#: that constructs its own model and never mutates user state).
OPTIMIZER_LINT_EXEMPT: dict[str, str] = {
    "observability/_memory_parity.py": (
        "pinned torch-memory-profiler parity scenario: builds its own MLP + "
        "SGD to exercise every allocator category; never touches user state"
    ),
}

#: Receivers whose ``.step()`` is optimizer-shaped. TorchLens's own step
#: APIs (collector.step, watch.step) are counters, not parameter updates,
#: and stay out of this pattern deliberately.
_OPTIMIZER_RECEIVER = re.compile(r"(^|\.)_?(optimizer|optimizers?\[[^]]*\]|optim|opt|scaler)$")


def _package_files() -> list[Path]:
    return [path for path in sorted(PACKAGE_ROOT.rglob("*.py")) if "__pycache__" not in path.parts]


def _is_optimizer_construction(node: ast.Call) -> bool:
    """True for ``torch.optim.X(...)`` / ``optim.X(...)`` constructions."""

    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    parent = func.value
    dotted = []
    while isinstance(parent, ast.Attribute):
        dotted.append(parent.attr)
        parent = parent.value
    if isinstance(parent, ast.Name):
        dotted.append(parent.id)
    return "optim" in dotted


def test_no_optimizer_construction_or_step_in_package() -> None:
    """Item 20's lint: torchlens never trains, true by construction."""

    findings: list[str] = []
    for path in _package_files():
        rel = str(path.relative_to(PACKAGE_ROOT))
        if rel in OPTIMIZER_LINT_EXEMPT:
            continue
        source = path.read_text()
        if "optim" not in source and ".step(" not in source:
            continue  # cheap pre-filter; the AST pass is the authority
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            if _is_optimizer_construction(node):
                findings.append(f"{rel}:{node.lineno} optimizer construction")
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "step"
                and _OPTIMIZER_RECEIVER.search(ast.unparse(func.value))
            ):
                findings.append(f"{rel}:{node.lineno} optimizer .step() call")
    assert findings == [], (
        f"no-optimizer lint violations: {findings} -- torchlens observes and "
        "replays, it never fits; a genuinely self-contained diagnostic goes on "
        "OPTIMIZER_LINT_EXEMPT with its reason"
    )


def test_exemption_ledger_rows_still_exist() -> None:
    """The ledger is shrink-only: a stale row must be deleted, not kept."""

    stale = sorted(rel for rel in OPTIMIZER_LINT_EXEMPT if not (PACKAGE_ROOT / rel).exists())
    assert stale == [], f"exemption rows for deleted files: {stale}"


def test_fitted_seam_spec_carries_the_provenance_field_set() -> None:
    """The accepting seam's emitted spec is FittedProvenance-shaped."""

    import torch

    import torchlens as tl

    pca = tl.stats.PCA(n_components=2)
    pca.update(torch.randn(20, 5))
    fitted = pca.fitted(fit_scope="f35-contract-pin")

    # The record itself: algorithm identity + data identity + geometry.
    assert fitted.digest.startswith("sha256:"), "fitting-data identity must be a content digest"
    assert fitted.n_samples == 20 and fitted.n_features == 5

    spec = tl.transforms.pca_apply(fitted)
    params = dict(spec.params)
    missing = {
        "source",  # the fitted record's content digest (refuse-silent-refit key)
        "fit_scope",  # fit population identity
        "in_extent",  # geometry a planner prices without executing
        "out_extent",
        "basis_digest",
    } - set(params)
    assert missing == set(), (
        f"fitted-provenance fields dropped from the accepting seam's spec: "
        f"{sorted(missing)} -- the field set is the contract (memo 3.2), not an "
        "implementation detail"
    )
    assert params["source"] == fitted.digest
    assert params["fit_scope"] == "f35-contract-pin"
    assert (params["in_extent"], params["out_extent"]) == (5, 2)
