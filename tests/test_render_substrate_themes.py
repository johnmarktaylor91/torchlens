"""Themes substrate tests (lane C05; themes memo items 1, 2, 6).

Pins the lens registry-as-data (N12), defaults-not-overrides preset
resolution (N1), the N14 ``encodings`` renderer-capability bit, and the
N15 above-ceiling disclosure fix (an explicit ``collapse="max"`` above the
compute ceiling is never a SILENT byte-identical no-op).
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.visualization.theme_registry import (
    GRAPHVIZ_DRAW_SURFACE,
    LensPreset,
    describe_lens,
    get_lens,
    list_lenses,
    register_lens,
    resolve_lens_request,
)


@pytest.mark.smoke
def test_builtin_rows_are_registered_and_teach_their_settings() -> None:
    overview = get_lens("overview")
    blueprint = get_lens("blueprint")
    assert overview.question == "what is this model?"
    # The blueprint row pins TODAY'S bare-draw contract by name.
    assert blueprint.members["collapse"] == "none"
    assert blueprint.members["vis_mode"] == "unrolled"
    assert blueprint.members["fold_repeats"] is False
    description = describe_lens("overview")
    assert "collapse='auto'" in description
    assert "what is this model?" in description
    assert any(lens.name == "overview" for lens in list_lenses(GRAPHVIZ_DRAW_SURFACE))


def test_member_names_must_be_draw_parameters() -> None:
    with pytest.raises(InvalidArgumentError) as excinfo:
        register_lens(LensPreset(name="bogus", question="?", members={"not_a_param": 1}))
    assert excinfo.value.fields["code"] == "lens_member_unknown"


def test_secondary_members_must_be_declared() -> None:
    with pytest.raises(InvalidArgumentError) as excinfo:
        register_lens(
            LensPreset(
                name="bogus2",
                question="?",
                members={"collapse": "auto"},
                secondary_members=("show_legend",),
            )
        )
    assert excinfo.value.fields["code"] == "lens_secondary_undeclared"


def test_registry_key_collision_refuses() -> None:
    with pytest.raises(InvalidArgumentError) as excinfo:
        register_lens(LensPreset(name="overview", question="duplicate"))
    assert excinfo.value.fields["code"] == "lens_name_taken"


def test_unknown_lens_lookup_names_the_roster() -> None:
    with pytest.raises(InvalidArgumentError) as excinfo:
        get_lens("no_such_lens")
    assert excinfo.value.fields["code"] == "lens_unknown"
    assert "overview" in str(excinfo.value)


def test_resolution_is_defaults_not_overrides() -> None:
    # N1: lens members apply ONLY where the user did not explicitly speak.
    effective = resolve_lens_request("overview", {"collapse": "none"})
    assert effective["collapse"] == "none"  # explicit user kwarg wins
    assert effective["vis_mode"] == "rolled"  # lens member fills the gap


def test_resolution_rejects_unknown_user_kwargs() -> None:
    with pytest.raises(InvalidArgumentError) as excinfo:
        resolve_lens_request("overview", {"not_a_param": 1})
    assert excinfo.value.fields["code"] == "lens_user_kwarg_unknown"


def test_preset_spec_fn_rides_the_internal_slot() -> None:
    def spec_fn(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return None

    lens = register_lens(LensPreset(name="_c05_test_slot", question="?", preset_spec_fn=spec_fn))
    try:
        effective = resolve_lens_request(lens, {})
        assert effective["preset_spec_fn"] is spec_fn
    finally:
        from torchlens.visualization import theme_registry

        theme_registry._REGISTRY.pop((GRAPHVIZ_DRAW_SURFACE, "_c05_test_slot"), None)


def test_rows_are_plain_serialisable() -> None:
    import json

    payload = get_lens("overview").to_dict()
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["members"]["collapse"] == "auto"
    assert round_tripped["surface"] == GRAPHVIZ_DRAW_SURFACE


def test_graphviz_renderer_declares_encodings_capability() -> None:
    # N14: "refuse by capability, never draw an unencoded imitation" needs
    # the encodings bit to exist in the vocabulary.
    from torchlens.visualization.renderers.base import RendererCapabilities
    from torchlens.visualization.renderers.graphviz import GraphvizRenderer

    assert "encodings" in {f.name for f in RendererCapabilities.__dataclass_fields__.values()}
    assert GraphvizRenderer.capabilities.encodings is True
    assert RendererCapabilities().encodings is False


@pytest.mark.smoke
def test_explicit_max_above_ceiling_warns_every_call(monkeypatch) -> None:
    # N15 (themes item 1): explicit collapse="max" above the compute gate
    # was a SILENT byte-identical no-op on every call after the first
    # (once-per-trace warning dedupe). An explicit compaction request now
    # re-warns on every call. F11 (collapse memo D5): the over-constant
    # outcome is the deterministic compact fallback plan (coded
    # collapse_budget_fallback), never a decline -- the re-warn contract is
    # unchanged, including on cache hits.
    import torchlens.visualization.collapse_optimizer as co
    from torchlens.errors._base import TorchLensWarning
    from torchlens.visualization.collapse_plan import RenderContext

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    log = tl.trace(model, torch.randn(2, 4))
    monkeypatch.setattr(co, "COLLAPSE_OPTIMIZER_MAX_OPS", 1)
    with pytest.warns(TorchLensWarning, match="compact fallback plan"):
        first = co.select_collapse_plan(log, RenderContext(), mode="max")
    assert not first.declined
    assert first.planner == "linear_fallback"
    # The second explicit request must warn AGAIN, never silently no-op --
    # this one is a result-cache hit, which must still disclose for max.
    with pytest.warns(TorchLensWarning, match="compact fallback plan"):
        second = co.select_collapse_plan(log, RenderContext(), mode="max")
    assert second.planner == "linear_fallback"
