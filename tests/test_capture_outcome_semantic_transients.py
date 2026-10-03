"""B1-02: the semantic-output scratch never survives onto a settled product.

Four session-time fields are written at capture entry (``user_funcs.py``) and
consumed ONLY by ``decode_outputs_for_trace``:

    _output_style  _output_head  _output_tokenizer  _semantic_output_metadata

Two of them pin LIVE USER OBJECTS -- an attached HF tokenizer
(``bridge/hf.py``) and a model-derived metadata key -- so a copy surviving onto
the escaping Trace is both a retention leak and a privacy leak: a plain
``pickle`` / ``torch.save`` of the Trace serializes the tokenizer's
vocab/merges into an artifact the user believes is a graph.

The pop block used to live ONLY on the normal forward-return arm, so the
HALTED axis (and the failure/interrupt axes) exited with all four still
pinned, and none of the four was declared in ``Trace.FIELD_POLICY`` /
``MODEL_LOG_FIELD_ORDER`` -- which is why the runtime-declaration lockstep gate
never saw them.
"""

from __future__ import annotations

import pickle

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.capture.trace import _SEMANTIC_OUTPUT_TRANSIENT_FIELDS
from torchlens.data_classes.field_policy import FieldPolicy


class _Tokenizer:
    """Stand-in for an attached HF tokenizer: a live, picklable user object."""

    def __init__(self) -> None:
        # A recognizable payload, standing in for vocab/merges.
        self.vocab = {f"tok{index}": index for index in range(64)}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:  # noqa: D102
        return " ".join(f"tok{int(index)}" for index in token_ids)


class _HaltableClassifier(nn.Module):
    """Two-stage model whose relu is a usable halt frontier."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return torch.relu(self.fc(x)) * 2.0


def _model_with_tokenizer() -> _HaltableClassifier:
    model = _HaltableClassifier().eval()
    model._torchlens_output_tokenizer = _Tokenizer()
    return model


def _leaked(trace: tl.Trace) -> list[str]:
    """Return the semantic-output scratch names still pinned to ``trace``."""

    return sorted(name for name in _SEMANTIC_OUTPUT_TRANSIENT_FIELDS if name in trace.__dict__)


# ---------------------------------------------------------------------------
# The four capture axes
# ---------------------------------------------------------------------------


def test_completed_capture_drops_the_semantic_output_scratch() -> None:
    """The historical normal-return arm keeps working."""

    trace = tl.trace(
        _model_with_tokenizer(),
        torch.ones(1, 3),
        capture=tl.options.CaptureOptions(output_style="classification"),
    )
    assert _leaked(trace) == []


def test_halted_capture_drops_the_semantic_output_scratch() -> None:
    """The HALTED axis: the defect's primary reproduction.

    A halt exits before the normal arm's pop block, so all four -- including
    the live tokenizer -- used to survive onto the returned partial Trace.
    """

    trace = tl.trace(
        _model_with_tokenizer(),
        torch.ones(1, 3),
        halt=tl.func("relu"),
        capture=tl.options.CaptureOptions(output_style="classification"),
    )
    assert trace.halted is True
    assert _leaked(trace) == []


def test_intervened_capture_drops_the_semantic_output_scratch() -> None:
    """The INTERVENED axis, the lockstep gate's other blind spot."""

    trace = tl.trace(
        _model_with_tokenizer(),
        torch.ones(1, 3),
        save=tl.func("relu"),
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
        capture=tl.options.CaptureOptions(output_style="classification"),
    )
    assert _leaked(trace) == []


def test_failed_capture_drops_the_semantic_output_scratch() -> None:
    """The FAILED axis: the partial attached to the escaping exception."""

    class Boom(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(3, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
            torch.relu(self.fc(x))
            raise RuntimeError("boom")

    model = Boom().eval()
    model._torchlens_output_tokenizer = _Tokenizer()
    with pytest.raises(RuntimeError) as excinfo:
        tl.trace(
            model,
            torch.ones(1, 3),
            capture=tl.options.CaptureOptions(output_style="classification"),
        )
    partial = getattr(excinfo.value, "partial_log", None)
    if partial is None:
        pytest.skip("this capture path attaches no partial product")
    assert _leaked(partial) == []


@pytest.mark.smoke
def test_halt_does_not_break_decoding_on_the_completed_path() -> None:
    """The drop never runs before the sole consumer.

    ``decode_outputs_for_trace`` is the only reader; if the drop moved ahead
    of it, decoding would silently stop working. This pins that it does not.
    """

    class Text(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # A registered buffer is a KNOWN provenance source; a closed-over
            # tensor would trip the unattributed-tensor-arg warning.
            self.register_buffer("logits", torch.tensor([[[0.0, 5.0], [6.0, 1.0]]]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
            return self.logits + x.sum() * 0.0

    model = Text().eval()
    model._torchlens_output_tokenizer = _Tokenizer()
    trace = tl.trace(
        model, torch.ones(1, 2), capture=tl.options.CaptureOptions(output_style="hf_text")
    )
    assert trace.output_postprocessor is not None
    assert trace.output_postprocessor.style == "hf_text"
    assert trace.decoded_output is not None
    assert _leaked(trace) == []


# ---------------------------------------------------------------------------
# Declaration + serialization boundaries
# ---------------------------------------------------------------------------


def test_all_four_fields_are_declared_drop_on_trace() -> None:
    """Declared, so the runtime-declaration lockstep gate can see them."""

    from torchlens.constants import MODEL_LOG_FIELD_ORDER

    policy = tl.Trace.FIELD_POLICY
    for name in _SEMANTIC_OUTPUT_TRANSIENT_FIELDS:
        assert name in policy, name
        assert policy[name].portable_policy is FieldPolicy.DROP, name
        # Session-time, never user-facing portable schema.
        assert name not in MODEL_LOG_FIELD_ORDER, name
        assert policy[name].user_facing is False, name


def test_declared_rows_are_the_single_scrub_authority() -> None:
    """The names left the pre-spec runtime-only allowance in `_io/scrub.py`.

    That allowance is consulted BEFORE ``PORTABLE_STATE_SPEC``, so an entry
    there would make the declared policy dead code.
    """

    from torchlens._io.scrub import _is_runtime_only_trace_field

    for name in _SEMANTIC_OUTPUT_TRANSIENT_FIELDS:
        assert not _is_runtime_only_trace_field(name), name


@pytest.mark.parametrize("halt", [False, True])
def test_plain_pickle_never_carries_the_live_tokenizer(halt: bool) -> None:
    """The pickle path, not just `.tlspec` (the fix's second half).

    ``Trace.__getstate__`` is a hand-maintained pop list, and plain pickle
    legitimately carries most session-time DROP state (``backward_ready``,
    ``save_budget``, ...). These four are different: two hold live user
    objects, so the serialization boundary drops them explicitly.
    """

    model = _model_with_tokenizer()
    kwargs = {"halt": tl.func("relu")} if halt else {}
    trace = tl.trace(
        model,
        torch.ones(1, 3),
        **kwargs,
        capture=tl.options.CaptureOptions(output_style="classification"),
    )
    state = trace.__getstate__()
    for name in _SEMANTIC_OUTPUT_TRANSIENT_FIELDS:
        assert name not in state, name
    restored = pickle.loads(pickle.dumps(trace))
    assert _leaked(restored) == []
    # No reconstructed tokenizer anywhere in the restored trace's state.
    assert not any(isinstance(value, _Tokenizer) for value in vars(restored).values())


def test_pickle_of_a_halted_trace_stays_tokenizer_free_by_size() -> None:
    """The retention leak has a measurable signature: pickle payload size.

    A pinned tokenizer reconstructs on load and inflates the artifact. Rather
    than pin an absolute byte count, compare the same capture with and without
    an attached tokenizer -- they must not differ by the tokenizer's payload.
    """

    plain = tl.trace(
        _HaltableClassifier().eval(),
        torch.ones(1, 3),
        halt=tl.func("relu"),
        capture=tl.options.CaptureOptions(output_style="classification"),
    )
    with_tok = tl.trace(
        _model_with_tokenizer(),
        torch.ones(1, 3),
        halt=tl.func("relu"),
        capture=tl.options.CaptureOptions(output_style="classification"),
    )
    tokenizer_bytes = len(pickle.dumps(_Tokenizer()))
    delta = abs(len(pickle.dumps(with_tok)) - len(pickle.dumps(plain)))
    assert delta < tokenizer_bytes, (
        f"halted pickle grew by {delta} bytes, at least one tokenizer "
        f"({tokenizer_bytes} bytes) worth of retained user state"
    )


def test_halted_trace_with_tokenizer_saves_and_reloads(tmp_path) -> None:
    """The `.tlspec` path still round-trips with the declared rows."""

    trace = tl.trace(
        _model_with_tokenizer(),
        torch.ones(1, 3),
        halt=tl.func("relu"),
        capture=tl.options.CaptureOptions(output_style="classification"),
    )
    path = tmp_path / "halted_semantic.tlspec"
    tl.save(trace, str(path))
    loaded = tl.load(str(path))
    assert _leaked(loaded) == []
