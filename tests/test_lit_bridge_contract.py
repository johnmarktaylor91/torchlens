"""LIT bridge contract tests that need NO lit-nlp install (lane F31).

Covers the import-inertness rule, the construction refusal family, the
pooling shape matrix (test-plan item 10 -- the one place toy tensors are the
right realism level: pure shape/mask contracts), blocks-preset discovery, and
the structural site resolver's refusals, all on tiny torch-native models.
The real-dependency contract tests live in ``test_lit_bridge_real_*.py``.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import (
    ArgumentTypeError,
    InvalidArgumentError,
    MissingDependencyError,
    RecordBindingError,
)
from torchlens.bridge.lit import _pooling, _refusals, _runtime, _sites


class _Block(nn.Module):
    """One tiny residual block for stack-discovery tests."""

    def __init__(self, width: int) -> None:
        """Initialize the block.

        Parameters
        ----------
        width:
            Feature width.
        """

        super().__init__()
        self.lin = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block.

        Parameters
        ----------
        x:
            Input activations.

        Returns
        -------
        torch.Tensor
            Residual output.
        """

        return x + torch.relu(self.lin(x))


class _StackModel(nn.Module):
    """Model with one homogeneous repeated-block stack (``h.0`` ... ``h.3``)."""

    def __init__(self) -> None:
        """Initialize four stacked blocks and a head."""

        super().__init__()
        self.h = nn.ModuleList([_Block(8) for _ in range(4)])
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stack.

        Parameters
        ----------
        x:
            Input activations.

        Returns
        -------
        torch.Tensor
            Head logits.
        """

        for block in self.h:
            x = block(x)
        return self.head(x)


class _ReusedModel(nn.Module):
    """Model whose one block fires twice per forward (multipass site)."""

    def __init__(self) -> None:
        """Initialize the reused block."""

        super().__init__()
        self.block = _Block(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block twice.

        Parameters
        ----------
        x:
            Input activations.

        Returns
        -------
        torch.Tensor
            Twice-processed activations.
        """

        return self.block(self.block(x))


def _hide_lit_nlp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import lit_nlp.api.model`` fail regardless of the install.

    Parameters
    ----------
    monkeypatch:
        Active monkeypatch fixture.
    """

    for name in list(sys.modules):
        if name == "lit_nlp" or name.startswith("lit_nlp."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "lit_nlp", None)


def _is_lit_module(name: str) -> bool:
    """Whether ``name`` is ``lit_nlp`` or one of its submodules."""

    return name == "lit_nlp" or name.startswith("lit_nlp.")


def test_import_inert_without_lit_nlp(monkeypatch: pytest.MonkeyPatch) -> None:
    """``import torchlens.bridge.lit`` never imports the foreign peer (L8).

    Order- and path-independent: a fresh import of the bridge runs with every
    ``lit_nlp`` module evicted (an earlier real-LIT test may have imported
    the peer legitimately), and the check is that the import put none back.
    ``monkeypatch`` restores the evicted modules and the original bridge
    module objects afterwards.
    """

    import importlib

    import torchlens.bridge as bridge_pkg

    assert tl.bridge.lit.__all__ == ["dataset", "layout", "model"]
    monkeypatch.setattr(bridge_pkg, "lit", sys.modules["torchlens.bridge.lit"])
    for name in list(sys.modules):
        if (
            _is_lit_module(name)
            or name == "torchlens.bridge.lit"
            or name.startswith("torchlens.bridge.lit.")
        ):
            monkeypatch.delitem(sys.modules, name)
    fresh = importlib.import_module("torchlens.bridge.lit")
    assert fresh.__all__ == ["dataset", "layout", "model"]
    leaked = sorted(name for name in sys.modules if _is_lit_module(name))
    assert leaked == [], f"importing torchlens.bridge.lit imported {leaked}"


def test_stub_surface_is_gone() -> None:
    """The mock-tested Trace-wrapping stub surface is deleted (M(lit) item 2)."""

    assert not hasattr(tl.bridge.lit, "TorchLensLitModel")


def test_attention_argument_refuses_before_anything() -> None:
    """``attention=`` refuses citing the upstream panel deletion (memo D12)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.bridge.lit.model(object(), object(), attention="heads")
    assert excinfo.value.fields["code"] == "lit_attention_unsupported"
    assert "2024-06-20" in str(excinfo.value)


def test_non_module_net_refuses_typed() -> None:
    """A non-module ``net`` refuses with the live-model teaching message."""

    with pytest.raises(ArgumentTypeError) as excinfo:
        tl.bridge.lit.model(object(), object())
    assert excinfo.value.fields["code"] == "lit_trace_not_executable"


def test_finished_trace_refuses_typed() -> None:
    """A finished Trace refuses: it cannot execute edited text (memo D2)."""

    log = tl.trace(_StackModel().eval(), torch.randn(2, 8))
    with pytest.raises(ArgumentTypeError) as excinfo:
        tl.bridge.lit.model(log, object())
    assert excinfo.value.fields["code"] == "lit_trace_not_executable"
    assert "edited text" in str(excinfo.value)


@pytest.mark.smoke
def test_missing_dependency_refuses_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without lit-nlp the factory refuses with the install command."""

    _hide_lit_nlp(monkeypatch)
    with pytest.raises(MissingDependencyError) as excinfo:
        tl.bridge.lit.model(_StackModel().eval(), object())
    assert excinfo.value.fields["code"] == "lit_dependency_missing"
    assert excinfo.value.fields["install"] == "pip install torchlens[lit]"
    with pytest.raises(MissingDependencyError):
        tl.bridge.lit.dataset(["x"])
    with pytest.raises(MissingDependencyError):
        tl.bridge.lit.layout()


@pytest.mark.smoke
def test_pooling_validation_refuses_unknown_names() -> None:
    """Unknown pooling spellings refuse with the closed vocabulary."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        _pooling.validate_pooling("mean")
    assert excinfo.value.fields["code"] == "lit_pooling_invalid"
    assert _pooling.validate_pooling("mean_masked") == "mean_masked"


@pytest.mark.smoke
def test_pooling_rank3_mask_awareness_exact() -> None:
    """Mask-aware pooling never averages PAD positions in (memo D10).

    The naive variants' failures are the reason the strategies exist: a
    right-padded batch mean over ALL positions dilutes real tokens with PAD
    vectors, and a naive last-position read returns a PAD vector outright.
    """

    value = torch.zeros(2, 4, 3)
    value[0, :2] = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    value[0, 2:] = 99.0  # PAD garbage that must never leak
    value[1, :4] = 1.0
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])

    mean = _pooling.pool("f", value, mask, "mean_masked")
    assert torch.equal(mean[0], torch.tensor([2.0, 3.0, 4.0]))
    first = _pooling.pool("f", value, mask, "first_token")
    assert torch.equal(first[0], value[0, 0])
    last = _pooling.pool("f", value, mask, "last_unmasked")
    assert torch.equal(last[0], value[0, 1])  # index L_i - 1, never the PAD tail
    assert torch.equal(last[1], value[1, 3])

    naive_mean = value.mean(dim=1)
    assert not torch.equal(mean[0], naive_mean[0])
    naive_last = value[:, -1]
    assert not torch.equal(last[0], naive_last[0])


@pytest.mark.smoke
def test_pooling_shape_matrix_refusals() -> None:
    """Rank 0/1/5, non-tensors, and geometry mismatches refuse typed."""

    mask = torch.ones(2, 4, dtype=torch.int64)
    for bad in (torch.tensor(1.0), torch.ones(3), torch.ones(2, 4, 3, 2, 2)):
        with pytest.raises(RecordBindingError) as excinfo:
            _pooling.pool("f", bad, mask, "mean_masked")
        assert excinfo.value.fields["code"] == "lit_pooling_unsupported"
    with pytest.raises(RecordBindingError):
        _pooling.pool("f", "not a tensor", mask, "mean_masked")
    with pytest.raises(RecordBindingError):
        _pooling.pool("f", torch.ones(2, 9, 3), mask, "mean_masked")  # wrong T
    assert _pooling.pool("f", torch.ones(2, 5), mask, "mean_masked").shape == (2, 5)
    assert _pooling.pool("f", torch.ones(2, 3, 4, 4), mask, "mean_masked").shape == (2, 3)


@pytest.mark.smoke
def test_pooling_custom_callable_validated() -> None:
    """A custom pooling callable's result geometry is validated."""

    mask = torch.ones(2, 4, dtype=torch.int64)
    value = torch.ones(2, 4, 3)

    def good(v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Pool by masked sum.

        Parameters
        ----------
        v:
            Payload.
        m:
            Mask.

        Returns
        -------
        torch.Tensor
            ``[batch, features]``.
        """

        return (v * m.unsqueeze(-1).to(v.dtype)).sum(dim=1)

    assert _pooling.pool("f", value, mask, good).shape == (2, 3)

    def bad(v: torch.Tensor, m: torch.Tensor) -> Any:
        """Return the wrong geometry.

        Parameters
        ----------
        v:
            Payload.
        m:
            Mask.

        Returns
        -------
        Any
            A non-[batch, features] value.
        """

        del m
        return v

    with pytest.raises(RecordBindingError) as excinfo:
        _pooling.pool("f", value, mask, bad)
    assert excinfo.value.fields["code"] == "lit_pooling_unsupported"


def test_blocks_preset_discovers_stack_and_last_pass() -> None:
    """Discovery finds ``h.0``-``h.3`` from module-call metadata (memo D7)."""

    log = tl.trace(_StackModel().eval(), torch.randn(2, 8))
    assert _sites.discover_block_stack(log) == ("h.0", "h.1", "h.2", "h.3")
    specs = _sites.pin_blocks(log)
    assert [spec.field_name for spec in specs] == [f"tl_block_{i}" for i in range(4)]
    assert all(spec.site_key.startswith("s1|") for spec in specs)


@pytest.mark.smoke
def test_blocks_preset_refuses_without_stack() -> None:
    """A model with no repeated stack refuses, naming candidates."""

    net = nn.Sequential(nn.Linear(8, 8))

    class _Flat(nn.Module):
        """Stackless wrapper."""

        def __init__(self) -> None:
            """Initialize one unrepeated layer."""

            super().__init__()
            self.only = net

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the layer.

            Parameters
            ----------
            x:
                Input activations.

            Returns
            -------
            torch.Tensor
                Output activations.
            """

            return self.only(x)

    log = tl.trace(_Flat().eval(), torch.randn(2, 8))
    with pytest.raises(InvalidArgumentError) as excinfo:
        _sites.discover_block_stack(log)
    assert excinfo.value.fields["code"] == "lit_blocks_preset_unavailable"


def test_explicit_site_unresolvable_names_candidates() -> None:
    """A bad explicit site refuses and names real addresses."""

    log = tl.trace(_StackModel().eval(), torch.randn(2, 8))
    with pytest.raises(InvalidArgumentError) as excinfo:
        _sites.pin_explicit(log, ("nonexistent.module",))
    assert excinfo.value.fields["code"] == "lit_site_unresolvable"
    assert "h.0" in str(excinfo.value)


@pytest.mark.smoke
def test_multipass_site_requires_pass_qualification() -> None:
    """An unqualified multipass site refuses; ``:N`` selects one pass."""

    log = tl.trace(_ReusedModel().eval(), torch.randn(2, 8))
    with pytest.raises(InvalidArgumentError) as excinfo:
        _sites.pin_explicit(log, ("block",))
    assert excinfo.value.fields["code"] == "lit_site_ambiguous"
    assert "'block:1'" in str(excinfo.value) and "'block:2'" in str(excinfo.value)
    specs = _sites.pin_explicit(log, ("block:2",))
    assert specs[0].pass_index == 2
    assert specs[0].field_name == "tl_block_pass2"


@pytest.mark.smoke
def test_resolver_refuses_cross_architecture_drift() -> None:
    """Pins from one wrapper class refuse on another's trace (memo D4).

    ``site_key`` is root-relative: rebinding across module trees would be a
    silent lie, so zero structural matches must land the drift refusal naming
    both the address and the key.
    """

    stack_log = tl.trace(_StackModel().eval(), torch.randn(2, 8))
    reused_log = tl.trace(_ReusedModel().eval(), torch.randn(2, 8))
    specs = _sites.pin_blocks(stack_log)
    resolved = _sites.resolve_pinned(stack_log, specs)
    assert set(resolved) == {f"tl_block_{i}" for i in range(4)}
    with pytest.raises(RecordBindingError) as excinfo:
        _sites.resolve_pinned(reused_log, specs)
    assert excinfo.value.fields["code"] == "lit_site_key_drift"
    assert excinfo.value.fields["address"] == "h.0"
    assert excinfo.value.fields["site_key"] == specs[0].site_key


@pytest.mark.smoke
def test_tokenizer_contract_refuses_typed() -> None:
    """Both tokenizer-contract failures land ``lit_tokenizer_invalid``."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        _runtime.validate_tokenizer(None)
    assert excinfo.value.fields["code"] == "lit_tokenizer_invalid"

    class _NoPad:
        """Callable tokenizer with neither pad nor eos token id."""

        def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            """Tokenize nothing.

            Parameters
            ----------
            *args:
                Unused.
            **kwargs:
                Unused.

            Returns
            -------
            dict[str, Any]
                Empty encoding.
            """

            return {}

    with pytest.raises(InvalidArgumentError) as excinfo:
        _runtime.validate_tokenizer(_NoPad())
    assert excinfo.value.fields["code"] == "lit_tokenizer_invalid"
    assert "pad" in str(excinfo.value)


@pytest.mark.smoke
def test_model_output_extraction_refuses_typed() -> None:
    """Non-tensor and wrong-rank logits land ``lit_model_output_unsupported``."""

    with pytest.raises(RecordBindingError) as excinfo:
        _runtime.extract_logits("classification", object())
    assert excinfo.value.fields["code"] == "lit_model_output_unsupported"
    with pytest.raises(RecordBindingError) as excinfo:
        _runtime.extract_logits("classification", torch.randn(2, 3, 4))
    assert excinfo.value.fields["code"] == "lit_model_output_unsupported"
    assert "rank 3" in str(excinfo.value)
    with pytest.raises(RecordBindingError) as excinfo:
        _runtime.extract_logits("causal_lm", torch.randn(2, 3))
    assert excinfo.value.fields["code"] == "lit_model_output_unsupported"


def test_capture_incomplete_refusal_is_typed() -> None:
    """The non-COMPLETE settle guard lands ``lit_capture_incomplete``.

    ``_runtime.capture`` consults this refusal on every predict-time trace;
    the settle statuses themselves are exercised by the capture-outcome
    suites, so the unit provocation pins the code and the no-fabrication
    teaching message.
    """

    with pytest.raises(RecordBindingError) as excinfo:
        _refusals.refuse_capture_incomplete("FAILED")
    assert excinfo.value.fields["code"] == "lit_capture_incomplete"
    assert "not be" in str(excinfo.value) and "fabricated" in str(excinfo.value)


@pytest.mark.smoke
def test_salience_unresolvable_embedding_refuses_typed() -> None:
    """A model without ``get_input_embeddings`` lands ``lit_salience_unavailable``."""

    from torchlens.bridge.lit import _resolve_salience_layer

    with pytest.raises(InvalidArgumentError) as excinfo:
        _resolve_salience_layer(nn.Linear(4, 4))
    assert excinfo.value.fields["code"] == "lit_salience_unavailable"
    assert "get_input_embeddings" in str(excinfo.value)
