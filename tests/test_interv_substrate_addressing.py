"""C03: the ONE addressing repair (site-key-primary targets, guarded join).

Leverage B4/B5 + surgery Build 0a/0b, one substrate:

- Saved spec target manifests carry ``resolved_site_keys`` (site-key-primary;
  labels are display-only disclosure -- the ME ordinal rule).
- ``check_spec_compat`` compares site keys FIRST and prints a site-level
  diff; a pure label renumbering (the measured 31-62 label churn one live
  edit causes on resnet18) is disclosure, never incompatibility.
- ``align_to`` falls back to site-key-first target lookup authorized by the
  SHIPPED guarded join's ``corroborated`` verdict when labels drift.
- THE NEIGHBOURHOOD TEST (B5): on a value-preserving same-cohort insertion,
  the naive key-membership oracle and payload equality both PASS (retained
  positive controls that must fail to DETECT), while the guarded join
  REFUSES exactly the affected cohort and the parent site-key signature
  change is visible -- the join's refusal is the primary assertion.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.postprocess._site_join import join_site_profiles, site_profile


class _Insertable(nn.Module):
    """One class, one forward source: ``extra`` inserts a value-preserving
    relu BEFORE the original relu line (the review T6 shape in miniature)."""

    def __init__(self, extra: bool) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)
        self.extra = extra

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        if self.extra:
            h = torch.relu(h)
        h = torch.relu(h)
        return self.fc2(h)


def _capture_pair() -> tuple[tl.Trace, tl.Trace]:
    # Standing rule: construct BOTH model variants before the first capture.
    base_model = _Insertable(extra=False)
    inserted_model = _Insertable(extra=True)
    inserted_model.load_state_dict(base_model.state_dict())
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    base = tl.trace(base_model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    inserted = tl.trace(
        inserted_model, x, capture=tl.options.CaptureOptions(intervention_ready=True)
    )
    return base, inserted


# ---------------------------------------------------------------------------
# The neighbourhood acceptance test (leverage B5)
# ---------------------------------------------------------------------------


def test_guarded_join_refuses_the_inserted_cohort() -> None:
    base, inserted = _capture_pair()
    rows = join_site_profiles(site_profile(base), site_profile(inserted))

    relu_key = base["relu_1_2"].site_key
    assert relu_key is not None

    # POSITIVE CONTROL that must FAIL to detect: naive key-set membership
    # claims the relu position is preserved (the same rendered key exists on
    # both sides -- but on the inserted side it names the INSERTED op).
    inserted_keys = {inserted.ops[label].site_key for label in inserted.op_labels}
    assert relu_key in inserted_keys, "naive membership passes (that is the trap)"

    # POSITIVE CONTROL that must FAIL to detect: payload equality. The
    # insertion is value-preserving (relu is idempotent), so the original
    # relu's values are bit-identical across the pair.
    assert torch.equal(base["relu_1_2"].out, inserted["relu_2_3"].out)

    # PRIMARY ASSERTION: the guarded join REFUSES the affected cohort.
    row = rows[relu_key]
    assert not row.joined
    assert row.verdict.value == "refused_cardinality"

    # Unaffected structural positions stay joined.
    fc1_key = base["linear_1_1"].site_key
    assert rows[fc1_key].joined


def test_parent_site_key_signature_change_is_visible() -> None:
    """The output-side consumer's parent SIGNATURE changes across the
    insertion (parent/child site-key agreement, review NEW-13's wording) --
    exactly what artifact-at-a-time oracles cannot see."""

    base, inserted = _capture_pair()

    def parent_keys(trace: tl.Trace, label: str) -> set[str | None]:
        op = trace.ops[label]
        return {trace.ops[parent].site_key for parent in (op.parents or ()) if parent in trace.ops}

    base_fc2_parents = parent_keys(base, "linear_2_3")
    inserted_fc2_parents = parent_keys(inserted, "linear_2_4")
    assert base_fc2_parents != inserted_fc2_parents


# ---------------------------------------------------------------------------
# check_spec_compat: site-key-first with the printed site-level diff
# ---------------------------------------------------------------------------


def test_manifest_carries_site_keys_and_compat_prints_site_diff(tmp_path) -> None:
    import os

    from torchlens.intervention.save import check_spec_compat, load_intervention_spec

    base_model = _Insertable(extra=False)
    inserted_model = _Insertable(extra=True)
    inserted_model.load_state_dict(base_model.state_dict())
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    base = tl.trace(base_model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    inserted = tl.trace(
        inserted_model, x, capture=tl.options.CaptureOptions(intervention_ready=True)
    )

    base.attach_hooks(tl.func("relu"), tl.scale(0.5), confirm_mutation=True)
    spec_path = os.path.join(tmp_path, "spec.tlspec")
    base.save_intervention(spec_path)
    spec = load_intervention_spec(spec_path)

    manifest = spec.metadata["target_manifest"]
    assert manifest, "manifest missing"
    assert all("resolved_site_keys" in entry for entry in manifest)
    assert manifest[0]["resolved_site_keys"] == [base["relu_1_2"].site_key]

    # Same trace: EXACT, empty drift.
    same = check_spec_compat(spec, base)
    assert same.outcome == "EXACT"

    # Inserted variant: the relu selector now fans differently; the compat
    # verdict must carry the printed site-level diff rather than a bare
    # label story. The fan-out disclosure warning is expected here.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        drifted = check_spec_compat(spec, inserted)
    assert drifted.outcome in ("COMPATIBLE_WITH_CONFIRMATION", "FAIL")
    assert "site-level diff" in drifted.site_diff
    assert drifted.diff.new_site_keys or drifted.diff.missing_site_keys


def test_spec_derived_disclosure_on_predicate_door_saves(tmp_path) -> None:
    """Leverage B7: a spec staged by the capture-time predicate door persists
    resolved labels only; the reload discloses spec_derived=True. Spec-door
    hooks carry their rule expression and are NOT derived."""

    import os

    from torchlens.intervention.save import load_intervention_spec

    model = _Insertable(extra=False)
    log = tl.trace(model, torch.randn(2, 4), intervene=tl.when(tl.func("relu"), tl.scale(0.5)))
    derived_path = os.path.join(tmp_path, "derived.tlspec")
    log.save_intervention(derived_path)
    derived = load_intervention_spec(derived_path)
    assert derived.metadata["spec_derived"] is True

    ready = tl.trace(
        model, torch.randn(2, 4), capture=tl.options.CaptureOptions(intervention_ready=True)
    )
    fork = ready.fork()
    fork.attach_hooks(tl.when(tl.func("relu"), tl.scale(0.5)), confirm_mutation=True)
    authored_path = os.path.join(tmp_path, "authored.tlspec")
    fork.save_intervention(authored_path)
    authored = load_intervention_spec(authored_path)
    assert authored.metadata["spec_derived"] is False
    hook_meta = dict(authored.hook_specs[0].metadata)
    assert "spec_where_repr" in hook_meta, "the rule expression must survive the save"


# ---------------------------------------------------------------------------
# align_to: site-key-first fallback under label drift (leverage B4)
# ---------------------------------------------------------------------------


def test_align_to_survives_label_renumbering_via_corroborated_join() -> None:
    """A structurally-intact site whose LABEL renumbered still aligns: the
    label leaves the join key, and the corroborated guarded-join verdict
    authorizes the rebind."""

    base, inserted = _capture_pair()
    selection = tl.units("linear_1_1", [(0, 0)]).resolve(base)
    # fc1's op renumbers relative labels? Same label here; the INSERTED trace
    # keeps linear_1_1, so exercise the drifted case with fc2 instead: its
    # label moved from linear_2_3 to linear_2_4 across the insertion.
    fc2_selection = tl.units("linear_2_3", [(0, 0)]).resolve(base)
    aligned = fc2_selection.align_to(inserted)
    (entry,) = tuple(aligned)
    assert entry.site_key[0] == "linear_2_4"
    # The unchanged-address path still aligns identically (fast path).
    same = selection.align_to(inserted)
    assert tuple(same)[0].site_key[0] == "linear_1_1"
