"""Focused coverage for glossary-conformance phase B2c structural fields."""

import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl
import torchlens.receptive_field as receptive_field
from torchlens.options import CaptureOptions


class _B2cModel(nn.Module):
    """Model docstring captured on Trace.source.class_docstring."""

    def __init__(self) -> None:
        """Initialize B2c model modules."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Run the B2c forward pass."""

        return torch.relu(self.linear(x)) * scale


def _make_trace(capture: CaptureOptions | None = None, *, save: Any = "all") -> tl.Trace:
    """Create a small B2c trace."""

    model = _B2cModel()
    x = torch.randn(2, 3, requires_grad=True)
    return tl.trace(model, x, capture=capture, save=save)


def test_trace_parent_and_root_trace_resolve_fork_lineage() -> None:
    """Trace parent/root properties resolve the legacy parent_run weakrefs."""

    root = _make_trace(CaptureOptions())
    child = root.fork("child")
    grandchild = child.fork("grandchild")

    assert root.parent_trace is None
    assert root.root_trace is None
    assert child.parent_trace is root
    assert child.root_trace is root
    assert grandchild.parent_trace is child
    assert grandchild.root_trace is root


def test_grad_fn_calls_accessor_and_call_savedness() -> None:
    """GradFn.calls replaces ops and GradFnCall.is_saved mirrors saved grad payloads."""

    trace = _make_trace(CaptureOptions(save_grads=True))
    trace.backward(trace[trace.output_layers[0]].out.sum())
    grad_fn = next(record for record in trace.autograd.grad_fns if record.num_calls)
    grad_fn_call = grad_fn.calls[0]

    assert not hasattr(grad_fn, "ops")
    assert trace.autograd.grad_fn_calls[grad_fn_call.label] is grad_fn_call
    assert grad_fn_call.is_saved is (
        grad_fn_call.grad_inputs is not None or grad_fn_call.grad_outputs is not None
    )


class _Triple(nn.Module):
    """Multiply by a constant so the backward node's two gradient sides differ."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return three times the input."""

        return x * 3.0


def _glossary_row(name: str) -> str:
    """Return the glossary table row whose first cell is ``name``."""

    glossary = Path(__file__).resolve().parents[1] / "docs/reference/glossary.md"
    prefix = f"| `{name}` |"
    return next(line for line in glossary.read_text().splitlines() if line.startswith(prefix))


def test_grad_fn_call_gradient_sides_match_their_glossary_rows() -> None:
    """grad_outputs arrives at the node from later ops; grad_inputs leaves it toward earlier ops.

    The hook is torch's ``Node.register_hook(grad_inputs, grad_outputs)``: for
    ``y = 3x`` with upstream gradient ``g`` at ``y``, ``grad_outputs`` is ``g``
    and ``grad_inputs`` is ``3g``, and the glossary rows say which flows in.
    """

    x = torch.ones(2, requires_grad=True)
    trace = tl.trace(_Triple(), x, capture=CaptureOptions(save_grads=True))
    out = trace[trace.output_layers[0]].out
    upstream = torch.tensor([5.0, 7.0])
    trace.backward((out * upstream).sum())
    mul = next(g for g in trace.autograd.grad_fns if g.class_name.startswith("MulBackward"))
    call = mul.calls[0]

    assert torch.equal(call.grad_outputs[0], upstream)
    assert torch.equal(call.grad_inputs[0], 3.0 * upstream)
    assert "flowing into this grad-fn" in _glossary_row("GradFnCall.grad_outputs")
    assert "flowing out of this grad-fn" in _glossary_row("GradFnCall.grad_inputs")


def test_module_boundary_fields_are_populated_from_calls() -> None:
    """Module aggregate input/output fields are derived from ModuleCall boundaries."""

    trace = _make_trace(CaptureOptions())
    module = trace.modules["linear"]
    call = module.calls[0]

    assert module.input_ops == call.input_ops
    assert module.input_layers == call.input_layers
    assert module.output_ops == call.output_ops
    assert module.output_layers == call.output_layers
    assert module.output_structure == call.output_structure


def test_trace_layers_to_save_public_view() -> None:
    """Trace.options.save exposes saved Op labels instead of raw indexes."""

    all_trace = _make_trace(CaptureOptions())
    selected_label = all_trace.output_layers[0]
    selected_trace = _make_trace(save=[selected_label])

    assert all_trace.options.save == "all"
    assert selected_trace.options.save
    assert all(isinstance(label, str) for label in selected_trace.options.save)
    assert all(label in selected_trace.op_labels for label in selected_trace.options.save)


def test_receptive_field_docs_conform_to_public_surface() -> None:
    """Keep documented RF names in lockstep and reject stale shorthand names."""

    root = Path(__file__).resolve().parents[1]
    documented_paths = (
        root / "README.md",
        root / "CLAUDE.md",
        root / "AGENTS.md",
        root / "docs" / "receptive_projective_fields.md",
        root / "notebooks" / "audit" / "17_receptive_projective_fields.ipynb",
    )
    documented = "\n".join(path.read_text(encoding="utf-8") for path in documented_paths)

    required_names = (
        *receptive_field.__all__,
        "tl.receptive_field",
        "receptive_field",
        "projective_field",
        "receptive_fields",
        "projective_fields",
        ".at()",
        ".of()",
        ".probe_grad()",
        ".check()",
        ".show()",
        "source=",
        "direction=",
        "target=",
        "within=",
        "cross_validate",
        "verify",
        'scope="receptive_field"',
        "ReceptiveFieldStatus",
    )
    missing = [name for name in required_names if name not in documented]
    assert not missing, f"RF public names missing from docs: {missing}"

    stale_patterns = (
        r"\btl\.rf\b",
        r"\bget_receptive_field\b",
        r"\bget_projective_field\b",
        r"\bProjectiveField(?:Box|View|Gradient)?\b",
        r"\bRFStatus\b",
        r"\bop\.projective\b(?!_field)",
    )
    stale = [pattern for pattern in stale_patterns if re.search(pattern, documented)]
    assert not stale, f"stale or invented RF names documented: {stale}"
