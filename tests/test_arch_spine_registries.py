"""Renderer + export-target door conformance kits (C01 items 16-17).

Seam-closure discipline (memo s10): one out-of-tree instance per seam
exercised on a real model, builtins visible through the same door, a planted
registration WITHOUT capability rows going RED, and the capability gate
refusing an incapable renderer typed.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._registry import RegistryError, kernel_universe_rows
from torchlens.visualization.renderer_registry import (
    RendererEntry,
    register_renderer,
    renderer_info,
    renderer_names,
    require_renderer_capabilities,
    unregister_renderer,
)
from torchlens.visualization.renderers.base import UnsupportedRendererCapabilityError

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def traced():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    trace = tl.trace(model, torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


class TestExportTargetDoor:
    def test_all_seventeen_builtins_registered_with_tiers(self) -> None:
        import torchlens.export as ex

        assert len(ex.export_targets()) >= 17
        assert ex.export_target_info("netron").capabilities["tier"] == "bridge"
        assert ex.export_target_info("model_explorer").capabilities["tier"] == "bridge"
        assert ex.export_target_info("tensorboard").capabilities["tier"] == "bridge"
        assert ex.export_target_info("csv").capabilities["tier"] == "present"
        assert ex.export_target_info("svg").capabilities["tier"] == "present"

    def test_out_of_tree_target_registers_and_runs(self, traced, tmp_path) -> None:
        import torchlens.export as ex
        from torchlens._registry import ProviderInfo

        def _write_labels(log, path):
            target = tmp_path / path
            target.write_text("\n".join(str(layer.layer_label) for layer in log.layer_list))
            return target

        ex.register_export_target(
            "test_labels",
            _write_labels,
            tier="present",
            capabilities={"output": "file", "requires_extra": "none"},
            provider=ProviderInfo(provider_id="testorg", version="1.0"),
        )
        try:
            assert "test_labels" in ex.export_targets()
            written = ex.resolve_export_target("test_labels")(traced, "labels.txt")
            assert written.read_text().strip()
        finally:
            ex.unregister_export_target("test_labels")

    def test_door_always_supplies_the_tier_capability_row(self) -> None:
        """The export door can never register capability-less: tier always rides."""

        import torchlens.export as ex

        ex.register_export_target("test_tier_row_only", lambda log: None, tier="present")
        try:
            rows = ex.export_target_info("test_tier_row_only").capabilities
            assert rows["tier"] == "present"
        finally:
            ex.unregister_export_target("test_tier_row_only")

    def test_non_callable_and_bad_tier_refuse(self) -> None:
        import torchlens.export as ex

        with pytest.raises(RegistryError) as excinfo:
            ex.register_export_target("bad", "not-a-function", tier="present")
        assert excinfo.value.fields["code"] == "export_target_not_callable"
        with pytest.raises(RegistryError) as excinfo:
            ex.register_export_target("bad", lambda log: None, tier="appliance")
        assert excinfo.value.fields["code"] == "export_target_tier_invalid"

    def test_builtin_exports_still_work_through_plain_spellings(self, traced, tmp_path) -> None:
        import torchlens.export as ex

        path = ex.svg(traced, tmp_path / "graph.svg")
        assert path.exists() and path.stat().st_size > 0
        payload = ex.netron(traced, tmp_path / "graph.onnx.json")
        assert payload.exists()


class TestRendererDoor:
    def test_builtins_registered_with_capability_rows(self) -> None:
        assert set(renderer_names()) >= {"graphviz", "dagua"}
        assert renderer_info("graphviz").capabilities["encoding_channels"] is True
        assert renderer_info("dagua").capabilities["encoding_channels"] is False

    def test_capability_gate_refuses_incapable_renderer_typed(self) -> None:
        with pytest.raises(UnsupportedRendererCapabilityError) as excinfo:
            require_renderer_capabilities("dagua", {"encoding_channels": True})
        assert excinfo.value.fields["code"] == "renderer_capability_unsupported"
        # graphviz passes the same requirement.
        require_renderer_capabilities("graphviz", {"encoding_channels": True})

    def test_unknown_renderer_name_refuses_teaching_at_draw(self, traced) -> None:
        from torchlens._errors import InvalidArgumentError

        with pytest.raises(InvalidArgumentError) as excinfo:
            traced.draw(vis_renderer="nonexistent", vis_save_only=True, vis_outpath="/tmp/x")
        assert excinfo.value.fields["code"] == "visualization_renderer_invalid"
        assert "graphviz" in str(excinfo.value)

    def test_planted_registration_without_capabilities_is_red(self) -> None:
        with pytest.raises(RegistryError) as excinfo:
            register_renderer(
                "planted",
                RendererEntry(name="planted", kind="render_ir"),
                capabilities={},
            )
        assert excinfo.value.fields["code"] == "registry_capabilities_missing"

    def test_out_of_tree_renderer_is_enumerable_then_gone(self) -> None:
        register_renderer(
            "test_third_party",
            RendererEntry(name="test_third_party", kind="render_ir"),
            capabilities={"encoding_channels": False, "formats": ("svg",)},
        )
        try:
            assert "test_third_party" in renderer_names()
        finally:
            unregister_renderer("test_third_party")
        assert "test_third_party" not in renderer_names()


class TestKernelInventoryCounts:
    def test_domains_are_countable(self) -> None:
        import torchlens.export  # noqa: F401 -- populates the export door
        import torchlens.io  # noqa: F401 -- imports the sidecar door

        rows = kernel_universe_rows()
        assert rows["export_targets"] >= 17
        assert rows["renderers"] >= 2
        assert "sidecar_families" in rows
