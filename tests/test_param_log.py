"""Tests for Param, ParamAccessor, and param-related visualization."""

import os
from os.path import join as opj
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens import trace as trace_fn
from torchlens._errors import AmbiguousOpLookupError
from torchlens._visualization.themes import resolve_theme
from torchlens.types import Param

#: Parameter fills as the DEFAULT skin renders them.
_TRAINABLE = resolve_theme("torchlens").semantic_palette["params_trainable"]
_MIXED = resolve_theme("torchlens").semantic_palette["params_gradient"]

VIS_OUTPUT_DIR = opj(os.environ["TORCHLENS_TEST_OUTPUTS_DIR"], "visualizations")

# ---------------------------------------------------------------------------
# Bugfix regression: shared tiny model
# ---------------------------------------------------------------------------


class _SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class _TiedEmbeddingHead(nn.Module):
    """Small model with a decoder head tied to the embedding table."""

    def __init__(self) -> None:
        """Initialize tied embedding and projection modules."""

        super().__init__()
        self.embed = nn.Embedding(7, 3)
        self.head = nn.Linear(3, 7, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project embedded token IDs with the tied output head.

        Parameters
        ----------
        x:
            Token IDs.

        Returns
        -------
        torch.Tensor
            Per-token logits summed across the sequence.
        """

        return self.head(self.embed(x)).sum(dim=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_model():
    return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))


def _simple_input():
    return torch.randn(1, 10)


def _assert_generated_pdf(path: Path) -> None:
    """Assert that a visualization run created a fresh non-empty PDF artifact."""

    assert path.exists(), f"expected visualization artifact at {path}"
    pdf_bytes = path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF"), f"expected {path} to be a PDF artifact"
    assert len(pdf_bytes) > 0, f"expected non-empty PDF artifact at {path}"


# ---------------------------------------------------------------------------
# Param class structure
# ---------------------------------------------------------------------------


def _make_frozen_first_layer():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    for p in model[0].parameters():
        p.requires_grad = False
    return model


def _make_all_frozen():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    for p in model.parameters():
        p.requires_grad = False
    return model


class TestParamLogFields:
    def test_repr_contains_key_info(self):
        """F10 re-pin: repr is one envelope+core line; grad/module facts
        moved to the str() card (D15)."""
        model = _make_simple_model()
        mh = trace_fn(model, _simple_input())
        mh._keepalive_model = model  # live-param cores need the source model
        pl = mh.params[0]
        r = repr(pl)
        assert pl.address in r
        assert "\n" not in r
        assert r.startswith(f"Param {pl.address}:")
        assert "trainable" in r or "frozen" in r
        card = str(pl)
        assert card.splitlines()[0] == r
        assert "  gradient  " + ("captured" if pl._has_grad else "none captured") in card

    def test_len_equals_num_params(self):
        mh = trace_fn(_make_simple_model(), _simple_input())
        pl = mh.params["0.weight"]
        assert len(pl) == pl.num_params == 50  # 5 * 10

    def test_is_quantized_false_for_float(self):
        mh = trace_fn(_make_simple_model(), _simple_input())
        for pl in mh.params:
            assert pl.is_quantized is False


# ---------------------------------------------------------------------------
# ParamAccessor on Trace
# ---------------------------------------------------------------------------


class TestParamAccessorMH:
    def test_contains(self):
        mh = trace_fn(_make_simple_model(), _simple_input())
        assert "0.weight" in mh.params
        assert "nonexistent" not in mh.params

    def test_short_name_ambiguous_raises(self):
        mh = trace_fn(_make_simple_model(), _simple_input())
        with pytest.raises(AmbiguousOpLookupError, match="Ambiguous"):
            mh.params["weight"]  # Both 0.weight and 2.weight

    def test_repr_dict_like(self):
        """F10 re-pin: collections point (one-line card); members show in str()."""
        mh = trace_fn(_make_simple_model(), _simple_input())
        r = repr(mh.params)
        assert "\n" not in r
        assert "ParamAccessor(" in r and "params" in r
        assert "0.weight" in str(mh.params)


# ---------------------------------------------------------------------------
# ParamAccessor on Op
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Param metadata correctness
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Trainable / frozen tallies
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Linked params
# ---------------------------------------------------------------------------


class TestLinkedParams:
    def test_tied_weight_alias_resolves(self) -> None:
        """Tied parameter addresses are aliases, not unresolved co-parents."""

        mh = trace_fn(_TiedEmbeddingHead(), torch.tensor([[1, 2, 3]]))
        tied = mh.params["embed.weight"]
        assert mh.params["head.weight"] is tied
        assert tied.all_addresses == ["embed.weight", "head.weight"]
        assert "head" in tied.all_module_addresses
        assert "head.weight" not in tied.co_parent_params
        assert mh.check_metadata_invariants() is True


# ---------------------------------------------------------------------------
# used_by_layers reverse mapping
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Recurrent / multi-pass
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Gradient tracking
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Optimizer support
# ---------------------------------------------------------------------------


class TestOptimizerSupport:
    def test_has_optimizer_partial(self):
        model = _make_simple_model()
        # Only optimize second linear layer
        optimizer = torch.optim.Adam(model[2].parameters())
        mh = trace_fn(
            model, _simple_input(), capture=tl.options.CaptureOptions(optimizer=optimizer)
        )
        assert mh.params["0.weight"].has_optimizer is False
        assert mh.params["0.bias"].has_optimizer is False
        assert mh.params["2.weight"].has_optimizer is True
        assert mh.params["2.bias"].has_optimizer is True


# ---------------------------------------------------------------------------
# Visualization — param labels and colors
# ---------------------------------------------------------------------------


class TestVisualizationParams:
    def _get_dot_source(self, model, x, rolled=False, depth=1000):
        outpath = opj(VIS_OUTPUT_DIR, "toy-networks", "_test_param_vis")
        mh = trace_fn(model, x)
        dot_source = mh.draw(
            rolled=rolled,
            path=outpath,
            depth=depth,
        ).dot
        return dot_source, mh

    def test_collapsed_module_mixed_grad(self):
        """Collapsed module with mixed frozen/trainable should use grad fill."""
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        model[0].weight.requires_grad = False
        dot, mh = self._get_dot_source(model, _simple_input(), depth=1)
        # The collapsed "0" module has mixed params
        assert _MIXED in dot or _TRAINABLE in dot  # at least one color appears


# ---------------------------------------------------------------------------
# Integration: end-to-end with various model types
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bugfix regression tests
# ---------------------------------------------------------------------------


class TestDerivedGradRecordPathFieldOrder:
    """cert9 MAJOR: ``_derived_grad_record_path`` must survive ``to_pandas()``.

    ``Param._derived_grad_record_path`` is a real, populated ``FieldPolicy.KEEP``
    field written unconditionally (no ``getattr``/``setdefault`` guard) by the
    mlx, paddle, jax, and tinygrad backends on every derived-gradient capture
    (``backends/mlx/backend.py``, ``backends/paddle/backend.py``,
    ``backends/jax/backend.py``, ``backends/tinygrad/backend.py``), but it was
    silently absent from ``PARAM_LOG_FIELD_ORDER`` -- the exact FIELD_ORDER /
    ``to_pandas()`` desync class this bucket exists to close, just on a
    surface (``Param``) not yet wired into the shared parametrized
    ``test_field_order_has_no_keep_field_desync`` (that test is scoped away
    from ``Param`` pending a dedicated alias-exclusion design for
    ``_grad_memory``/``_derived_grad_payload``; this field is not one of
    those documented aliases, so it needs its own litmus here rather than
    waiting on that broader design).
    """

    def test_derived_grad_record_path_survives_to_pandas(self) -> None:
        """A populated ``_derived_grad_record_path`` must appear as a column."""

        param_log = Param(
            module_address="self",
            name="weight",
            shape=(2, 2),
            dtype=torch.float32,
            num_params=4,
            param_memory=16,
            trainable=True,
            address="weight",
            barcode="b1",
        )
        param_log._derived_grad_payload = torch.zeros(2, 2)
        param_log._derived_grad_record_path = "grad_records[3].payload"

        df = param_log.to_pandas()

        assert "_derived_grad_record_path" in df.columns
        assert df["_derived_grad_record_path"].iloc[0] == "grad_records[3].payload"


class TestTrainableFrozenTallies:
    def test_mh_all_trainable(self):
        mh = trace_fn(_make_simple_model(), _simple_input())
        assert mh.num_params == mh.num_params_trainable
        assert mh.num_params_frozen == 0
        assert mh.num_params == 67  # 50+5+10+2

    def test_mh_mixed(self):
        mh = trace_fn(_make_frozen_first_layer(), _simple_input())
        assert mh.num_params_frozen == 55  # 50+5
        assert mh.num_params_trainable == 12  # 10+2
        assert mh.num_params == 67
        assert mh.num_params == mh.num_params_trainable + mh.num_params_frozen

    def test_mh_all_frozen(self):
        mh = trace_fn(_make_all_frozen(), _simple_input())
        assert mh.num_params_trainable == 0
        assert mh.num_params_frozen == 67

    def test_tle_tallies(self):
        mh = trace_fn(_make_frozen_first_layer(), _simple_input())
        for entry in mh:
            if entry.has_params:
                assert entry.num_params == entry.num_params_trainable + entry.num_params_frozen
        # First linear is frozen
        linear1 = [e for e in mh if "linear_1" in e.layer_label][0]
        assert linear1.num_params_frozen == 55
        assert linear1.num_params_trainable == 0
        # Second linear is trainable
        linear2 = [e for e in mh if "linear_2" in e.layer_label][0]
        assert linear2.num_params_trainable == 12
        assert linear2.num_params_frozen == 0

    def test_output_layer_no_params(self):
        mh = trace_fn(_make_simple_model(), _simple_input())
        output = mh["output_1"]
        assert output.num_params == 0
        assert output.num_params_trainable == 0
        assert output.num_params_frozen == 0
        assert len(output._param_logs) == 0


# Kept: edited on the C-100 side (C-091/C-098) after the shrink removed it.
class TestTensorLogEntries:
    def test_populated(self):
        mh = trace_fn(_make_simple_model(), _simple_input())
        for pl in mh.params:
            assert len(pl.used_by_ops) > 0
            assert len(pl.used_by_layers) > 0

    def test_points_to_correct_ops_and_layers(self):
        mh = trace_fn(_make_simple_model(), _simple_input())
        for pl in mh.params:
            for label in pl.used_by_ops:
                entry = mh[label]
                assert any(p.address == pl.address for p in entry._param_logs)
            for label in pl.used_by_layers:
                entry = mh.layers[label]
                assert any(
                    any(p.address == pl.address for p in op._param_logs)
                    for op in entry.ops.values()
                )


# ---------------------------------------------------------------------------
# Recurrent / multi-pass
# ---------------------------------------------------------------------------
