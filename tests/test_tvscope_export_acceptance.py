"""tvscope B7/B8: shaping-op contract, adapters, and export acceptance.

One recorded shaping operation shared by every consumer; the file route and
the in-memory route are numerically identical; row order is tied to
stimulus ids with typed refusals on disagreement; the exported artifact
reloads in a process that reads plain JSON + torch payloads (the
bring-your-own-alignment mechanics); adapters are stub-tested here because
rsatoolbox/xarray/brainio are optional.
"""

from __future__ import annotations

import importlib.machinery
import types

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.errors
from torchlens import features


class _SmallCNN(nn.Module):
    """Small CNN for cheap real captures."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conv + relu + pool."""

        return self.pool(torch.relu(self.conv(x)))


def _stub_module(monkeypatch, name: str) -> types.ModuleType:
    """Install an importable stub module with a real spec.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture.
    name:
        Dotted module name to stub.

    Returns
    -------
    types.ModuleType
        The installed stub.
    """

    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    monkeypatch.setitem(__import__("sys").modules, name, module)
    return module


@pytest.fixture()
def artifact(tmp_path):
    """One disk extraction artifact with ids + a matching in-memory trace."""

    torch.manual_seed(0)
    model = _SmallCNN().eval()
    stimuli = torch.rand(4, 3, 8, 8)
    out_dir = tmp_path / "run"
    tl.extract_dataset(
        model,
        stimuli,
        ["conv"],
        batch_size=2,
        path=out_dir,
        stimulus_ids=[f"img{i}" for i in range(4)],
        progress=False,
    )
    trace = tl.trace(model, stimuli, save=["conv"])
    return out_dir, trace


class TestShapingOp:
    """The ONE named recorded operation (memo D15)."""

    def test_scalar_input_refuses_typed(self) -> None:
        """A 0-dim tensor has no stimulus axis."""

        with pytest.raises(torchlens.errors.FeatureShapingError) as excinfo:
            features.as_matrix(torch.tensor(1.0))
        assert excinfo.value.fields["code"] == "feature_matrix_scalar_input"

    def test_row_id_disagreement_refuses_typed(self) -> None:
        """Row/id cardinality disagreement is the mislabeling tripwire."""

        with pytest.raises(torchlens.errors.FeatureShapingError) as excinfo:
            features.as_matrix(torch.randn(3, 4), row_ids=["a", "b"])
        assert excinfo.value.fields["code"] == "feature_rows_ids_mismatch"


class TestSiteMatrixRoutes:
    """File and in-memory routes share the op and agree numerically."""

    def test_payloadless_site_refuses_typed(self) -> None:
        """A metadata-only trace teaches how to get payloads."""

        model = _SmallCNN().eval()
        trace = tl.trace(model, torch.rand(2, 3, 8, 8), save="none")
        with pytest.raises(torchlens.errors.FeatureShapingError) as excinfo:
            features.site_matrix(trace, "conv")
        assert excinfo.value.fields["code"] == "feature_site_payload_unavailable"

    def test_unknown_extraction_key_refuses_typed(self, artifact) -> None:
        """A key outside the manifest refuses with the available set.

        The directory route refuses inside ``load_extraction`` (the v2 lazy
        reader's own typed door, ``extraction_reader_key_unknown``); the
        loaded-object route refuses in the shaping layer.
        """

        from torchlens._errors import InvalidArgumentError

        out_dir, _trace = artifact
        with pytest.raises(InvalidArgumentError) as dir_info:
            features.site_matrix(out_dir, "not_a_key")
        assert dir_info.value.fields["code"] == "extraction_reader_key_unknown"
        loaded = tl.load(out_dir)
        with pytest.raises(torchlens.errors.FeatureShapingError) as obj_info:
            features.site_matrix(loaded, "not_a_key")
        assert obj_info.value.fields["code"] == "feature_site_payload_unavailable"
        assert obj_info.value.fields["available"] == ["conv"]


class TestAdapters:
    """Stub-tested adapter wiring (optional libraries absent by design)."""

    def test_rsatoolbox_per_site_descriptors(self, artifact, monkeypatch) -> None:
        """The Dataset carries site, shaping, ids, and the verdict."""

        out_dir, trace = artifact
        rsa = _stub_module(monkeypatch, "rsatoolbox")
        rsa_data = _stub_module(monkeypatch, "rsatoolbox.data")

        class _Dataset:
            """Capture the constructor payload."""

            def __init__(self, measurements, obs_descriptors, channel_descriptors, descriptors):
                self.measurements = measurements
                self.obs_descriptors = obs_descriptors
                self.channel_descriptors = channel_descriptors
                self.descriptors = descriptors

        rsa_data.Dataset = _Dataset
        rsa.data = rsa_data
        from torchlens.bridge import rsatoolbox as tl_rsa

        file_side = tl_rsa.dataset(out_dir, site="conv")
        memory_side = tl_rsa.dataset(trace, site="conv")
        assert file_side.measurements.shape == memory_side.measurements.shape == (4, 256)
        assert (file_side.measurements == memory_side.measurements).all()
        assert file_side.descriptors["site"] == "conv"
        assert file_side.descriptors["shaping"] == features.SHAPING_OP
        assert list(file_side.obs_descriptors["stimulus_id"]) == [
            "img0",
            "img1",
            "img2",
            "img3",
        ]
        assert file_side.descriptors["input_preprocessing_verdict"] == "unknown"

    def test_xarray_data_array_routes_identical(self, artifact, monkeypatch) -> None:
        """DataArray wiring: dims, coords, attrs, route equality."""

        out_dir, trace = artifact
        xr = _stub_module(monkeypatch, "xarray")

        class _DataArray:
            """Capture the constructor payload."""

            def __init__(self, data, dims=None, coords=None, attrs=None, name=None):
                self.data, self.dims = data, dims
                self.coords, self.attrs, self.name = coords, attrs, name

        xr.DataArray = _DataArray
        from torchlens.bridge import xarray as tl_xr

        file_side = tl_xr.data_array(out_dir, "conv")
        memory_side = tl_xr.data_array(trace, "conv")
        assert file_side.dims == ("presentation", "neuroid")
        assert (file_side.data == memory_side.data).all()
        assert file_side.attrs["shaping"]["op"] == features.SHAPING_OP
        assert file_side.coords["stimulus_id"][0] == "presentation"

    @pytest.mark.parametrize(
        "home",
        ["brainscore_core.supported_data_standards.brainio.assemblies", "brainio.assemblies"],
    )
    def test_neuroid_assembly_finds_either_brainio_home(self, artifact, monkeypatch, home) -> None:
        """NeuroidAssembly resolves from brainscore_core's bundled brainio or standalone brainio.

        Current brainscore_vision installs ship no top-level ``brainio``; the
        assembly classes live under ``brainscore_core``. Both homes must work.
        """

        import sys

        homes = (
            "brainscore_core.supported_data_standards.brainio.assemblies",
            "brainio.assemblies",
        )
        for other in homes:
            monkeypatch.setitem(sys.modules, other, None)
        parts = home.split(".")
        for depth in range(1, len(parts) + 1):
            module = _stub_module(monkeypatch, ".".join(parts[:depth]))

        class _NeuroidAssembly:
            """Capture the constructor payload."""

            def __init__(self, data, dims=None, coords=None):
                self.data, self.dims, self.coords, self.attrs = data, dims, coords, {}

        module.NeuroidAssembly = _NeuroidAssembly
        out_dir, trace = artifact
        from torchlens.bridge import xarray as tl_xr

        assembly = tl_xr.neuroid_assembly(trace, "conv")
        assert isinstance(assembly, _NeuroidAssembly)
        assert assembly.dims == ("presentation", "neuroid")
        assert assembly.attrs["shaping"]["op"] == features.SHAPING_OP
        assert (assembly.data == tl_xr.neuroid_assembly(out_dir, "conv").data).all()

    def test_missing_optional_dependency_teaches(self, artifact, monkeypatch) -> None:
        """Absent adapters raise ImportError naming the install path."""

        import builtins
        import sys

        out_dir, _trace = artifact
        monkeypatch.delitem(sys.modules, "xarray", raising=False)
        real_import = builtins.__import__

        def _no_xarray(name, *args, **kwargs):
            """Refuse the optional import."""

            if name == "xarray":
                raise ImportError("absent")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_xarray)
        from torchlens.bridge import xarray as tl_xr

        with pytest.raises(ImportError, match="xarray"):
            tl_xr.data_array(out_dir, "conv")
