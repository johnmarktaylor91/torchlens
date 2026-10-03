"""Bound-method root contract (lane F41; foldA MEMO s5 item 9 + D11).

The ruled root contract, pinned on toy models: ``tl.trace`` accepts an
``nn.Module`` OR a bound method of one; the owner resolves via
``method.__self__`` and registers as a submodule of the TL-authored
wrapper root; ``stepped_module`` defaults to the owner on bound-method
episode captures; the method is called exactly once; the synthetic root
discloses TL-authorship (``bound_method:`` root fact + the
``Op.tl_authored_root`` marker on the root op record) and reads
``type(owner).__name__``; and the entry-point fact joins the rerun
identity gate FAIL-CLOSED (interim posture: bound-method captures refuse
rerun/append).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.backends.torch.bound_root import TLBoundMethodRoot
from torchlens.errors import InvalidArgumentError
from torchlens.intervention.errors import ModelMismatchError


class _ToyLM(nn.Module):
    """Tiny LM-shaped fixture with a generate-like bound method."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16)
        self.generate_calls = 0

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.emb(ids).mean(1))

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, n_new: int = 3) -> torch.Tensor:
        self.generate_calls += 1
        for _ in range(n_new):
            next_token = self(ids).argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
        return ids


def _mlp() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def _ids() -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.randint(0, 16, (1, 4), generator=generator)


def test_bound_forward_capture_matches_module_capture() -> None:
    model = _mlp()
    x = torch.randn(2, 4, generator=torch.Generator().manual_seed(1))
    bound = tl.trace(model.forward, x)
    plain = tl.trace(model, x)
    assert torch.equal(bound[bound.output_layers[0]].out, plain[plain.output_layers[0]].out)
    # Same interior ops captured (labels match one-to-one).
    assert [op.label for op in bound.layer_list] == [op.label for op in plain.layer_list]


def test_synthetic_root_reads_owner_identity_never_method() -> None:
    model = _mlp()
    log = tl.trace(model.forward, torch.randn(2, 4))
    assert log.model_class_name == "Sequential"
    assert log.model_label == "Sequential"
    assert "method" not in (log.model_class_name, log.model_label)
    assert log.model_class_qualname == "torch.nn.modules.container.Sequential"
    assert log.root_entry_point == ("bound_method:torch.nn.modules.container.Sequential.forward")


def test_root_fact_stays_module_call_on_module_roots() -> None:
    model = _mlp()
    log = tl.trace(model, torch.randn(2, 4))
    assert log.root_entry_point == ("module_call:torch.nn.modules.container.Sequential.forward")
    for op in log.layer_list:
        assert op.tl_authored_root is None


def test_tl_authored_root_marker_lands_on_root_op_record() -> None:
    model = _mlp()
    log = tl.trace(model.forward, torch.randn(2, 4))
    for output_label in log.output_layers:
        for op in log[output_label].ops:
            assert op.tl_authored_root is True
    # The marker discloses the ROOT record only, never interior ops.
    interior = [op for op in log.layer_list if op.label.split(":")[0] not in log.output_layers]
    assert all(op.tl_authored_root is None for op in interior)


def test_bound_method_called_exactly_once() -> None:
    model = _ToyLM()
    log = tl.trace(model.generate, _ids())
    assert model.generate_calls == 1
    assert log.root_entry_point.endswith(".generate")


def test_owner_registers_as_submodule_of_wrapper() -> None:
    model = _ToyLM()
    wrapper = TLBoundMethodRoot(model.generate)
    assert wrapper.tl_owner is model
    assert dict(wrapper.named_children())["owner"] is model
    assert wrapper.tl_method_name == "generate"
    assert wrapper.tl_owner_class_name == "_ToyLM"


def test_wrapper_constructor_guards_non_bound_inputs() -> None:
    with pytest.raises(TypeError, match="bound method of an nn.Module"):
        TLBoundMethodRoot(torch.relu)


def test_closure_and_bare_function_teach_the_ruled_spelling() -> None:
    model = _mlp()
    x = torch.randn(2, 4)

    def closure(t: torch.Tensor) -> torch.Tensor:
        return model(t)

    for candidate in (closure, torch.relu):
        with pytest.raises(InvalidArgumentError) as excinfo:
            tl.trace(candidate, x)
        assert excinfo.value.fields["code"] == "model_type_unsupported"
        assert "bound method" in str(excinfo.value)
        assert "method.__self__" in excinfo.value.fields["remedy"]


def test_bound_method_of_non_module_still_refuses() -> None:
    class _Plain:
        def run(self, t: torch.Tensor) -> torch.Tensor:
            return torch.relu(t)

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.trace(_Plain().run, torch.randn(2, 4))
    assert excinfo.value.fields["code"] == "model_type_unsupported"


def test_episode_stepped_module_defaults_to_owner() -> None:
    model = _ToyLM()
    log = tl.trace(
        model.generate,
        _ids(),
        episode=tl.options.EpisodeSpec(n_steps=3),
    )
    episode = log.annotations["episode"]
    assert episode["header"]["stepped_module"] == "owner"
    assert len(episode["rows"]) == 3
    assert log.root_entry_point.endswith("._ToyLM.generate")


def test_episode_stepped_module_none_refuses_on_module_roots() -> None:
    from torchlens.errors.episode import EpisodeDeclarationError

    model = _ToyLM()
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        tl.trace(model, _ids(), episode=tl.options.EpisodeSpec(n_steps=1))
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"
    assert "bound-method root" in str(excinfo.value)


def test_rerun_identity_gate_refuses_bound_method_captures() -> None:
    model = _mlp()
    log = tl.trace(model.forward, torch.randn(2, 4))
    with pytest.raises(ModelMismatchError) as excinfo:
        log._validate_supplied_model_matches_capture(model)
    assert excinfo.value.fields["code"] == "rerun_entry_point_unsupported"
    assert "DIFFERENT entry point" in str(excinfo.value)


def test_rerun_identity_gate_fail_closed_on_absent_fact() -> None:
    model = _mlp()
    log = tl.trace(model, torch.randn(2, 4))
    log.root_entry_point = None  # simulate a legacy artifact
    with pytest.raises(ModelMismatchError) as excinfo:
        log._validate_supplied_model_matches_capture(model)
    assert excinfo.value.fields["code"] == "root_entry_point_unavailable"


def test_rerun_identity_gate_still_passes_module_roots() -> None:
    model = _mlp()
    log = tl.trace(model, torch.randn(2, 4))
    log._validate_supplied_model_matches_capture(model)


def test_append_preflight_refuses_bound_method_captures() -> None:
    from torchlens.intervention.rerun import _preflight_append

    model = _mlp()
    log = tl.trace(model.forward, torch.randn(2, 4))
    with pytest.raises(ModelMismatchError) as excinfo:
        _preflight_append(log, model)
    assert excinfo.value.fields["code"] == "rerun_entry_point_unsupported"


def test_bound_capture_save_load_persists_fact_and_marker(tmp_path) -> None:
    model = _mlp()
    log = tl.trace(model.forward, torch.randn(2, 4))
    target = tmp_path / "bound.tlspec"
    tl.save(log, target)
    loaded = tl.load(target)
    assert loaded.root_entry_point == log.root_entry_point
    out_op = loaded[loaded.output_layers[0]].ops[0]
    assert out_op.tl_authored_root is True
    with pytest.raises(ModelMismatchError) as excinfo:
        loaded._validate_supplied_model_matches_capture(model)
    assert excinfo.value.fields["code"] == "rerun_entry_point_unsupported"


def test_replay_engine_still_works_on_bound_captures() -> None:
    model = _mlp()
    log = tl.trace(
        model.forward,
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    fork = log.fork()
    relu_label = next(op.label for op in log.layer_list if op.label.startswith("relu"))
    fork.do(tl.units(relu_label, [(0, 0)]).resolve(fork), tl.zero_ablate())
    assert len(fork.intervention_audit) == 1


def test_validate_accepts_bound_method_roots() -> None:
    model = _mlp()
    assert tl.validate(model.forward, torch.randn(2, 4), scope="forward") is True


@pytest.mark.smoke
def test_validate_refuses_other_callables_typed() -> None:
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.validate(torch.relu, torch.randn(2, 4), scope="forward")
    assert excinfo.value.fields["code"] == "model_type_unsupported"


def test_input_rung_ladder_refuses_bound_method_roots_typed() -> None:
    model = _mlp()
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.trace(model.forward, input_size=(2, 4))
    assert excinfo.value.fields["code"] == "input_rung_requires_module_root"
    assert "gold rung" in excinfo.value.fields["remedy"]


class _ShapedReturns(nn.Module):
    """Bound methods returning None / tuple / dict (foldA s6 F41 arms)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)

    def emit_none(self, x: torch.Tensor) -> None:
        self.lin(x)
        return None

    def emit_tuple(self, x: torch.Tensor) -> tuple:
        y = self.lin(x)
        return (y, y.sum())

    def emit_dict(self, x: torch.Tensor) -> dict:
        return {"a": self.lin(x)}


@pytest.mark.smoke_cells("test_return_shapes_capture_instead_of_refusing[emit_none-0]")
@pytest.mark.parametrize(
    ("method_name", "expected_outputs"),
    [("emit_none", 0), ("emit_tuple", 2), ("emit_dict", 1)],
)
def test_return_shapes_capture_instead_of_refusing(method_name: str, expected_outputs: int) -> None:
    torch.manual_seed(0)
    model = _ShapedReturns()
    log = tl.trace(getattr(model, method_name), torch.randn(2, 4))
    assert log.root_entry_point.endswith(f"._ShapedReturns.{method_name}")
    assert len(log.output_layers) == expected_outputs
    # A None-returning root has no output op to carry the marker; the
    # Trace-level bound_method fact remains the TL-authorship disclosure.
    for output_label in log.output_layers:
        for op in log[output_label].ops:
            assert op.tl_authored_root is True
