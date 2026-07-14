"""Round-17 write-class RETENTION dimension: a mutable zero-copy host alias of a captured
tensor that OUTLIVES the forward is a post-forward host-writable window the end-of-forward byte
watch cannot see.

The write-class guarantee is "no UNOBSERVED host write into captured tensor memory during the
forward." The r13-r16 byte watch (:func:`torchlens.backends.torch.completeness_witness._check_writeback_watch`)
can only witness a write that lands BEFORE the forward returns. A mutable zero-copy host alias
whose OBJECT is RETAINED past the forward (``self.alias = y.numpy()``) leaves a host-writable
window OPEN after the compare: a later ``self.alias[0] = 99`` mutates the captured tensor's bytes
with no aten dispatch, no version bump, and no byte the watch can re-inspect, so the sparse replay
recomputes the pre-write value and would falsely VERIFY+ATTEST.

The fix adds the RETENTION dimension: at forward end, if a mutable zero-copy host alias object
(a ``numpy`` / ``__array__`` ndarray or a ``storage`` handle) survives a single scoped
``gc.collect()``, it was RETAINED -> the tensor's post-forward memory is host-writable and
unobservable -> the descriptor is marked incomplete -> UNVERIFIABLE (+ NOT_APPLICABLE). A
transient LOCAL alias is collected by the ``gc.collect()`` -> the source stays VERIFIED. The
collect fires only when at least one such alias was handed out this forward, so escape-free
captures and real models pay nothing and are byte-unchanged.

This module proves the CRITICAL bug is now honestly refused AND the hard no-over-trigger contract
holds: a transient local alias, a retained read-only numpy COPY, a retained detached torch tensor,
a plain stateful counter, read-only exposures, and real models all stay VERIFIED.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness

_CAPTURE = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


def _save(model: nn.Module, capture_input: torch.Tensor, path: Path) -> Path:
    trace = tl.trace(model, capture_input, capture=_CAPTURE)
    trace.save(path, level="runnable", include_weights=True, include_activations=True)
    return path


def _run(path: Path, x: torch.Tensor):
    return tl.load(path).run(inputs=x.clone(), seed=0, on_divergence="return_diverged")


# --------------------------------------------------------------------------- #
# THE BUG (R17-C1): a mutable zero-copy numpy alias retained on ``self`` and written in a
# later forward. Capture must mark the tensor's post-forward memory unobservable -> the run is
# honestly refused (UNVERIFIABLE + NOT_APPLICABLE), never silently VERIFIED+ATTESTED.
# --------------------------------------------------------------------------- #
class RetainedMutableNumpyAlias(nn.Module):
    """Stashes a zero-copy WRITABLE numpy view on ``self`` and mutates it on the next forward."""

    def __init__(self) -> None:
        super().__init__()
        self.alias: np.ndarray | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.alias is None:
            y = x * 2
            self.alias = y.detach().numpy()  # zero-copy mutable numpy alias RETAINED on self
            return y
        self.alias[0] = 99  # post-forward host write no byte watch can see
        return torch.from_numpy(self.alias) * 3


@pytest.mark.smoke
def test_retained_mutable_numpy_alias_is_unverifiable(tmp_path: Path) -> None:
    model = RetainedMutableNumpyAlias()
    capture_x = torch.tensor([2.0, 4.0])
    path = _save(model, capture_x, tmp_path / "retained.tlspec")

    # The next TRUE live forward mutates the retained alias -> [297, 24].
    true_next = model(capture_x.clone())
    assert torch.allclose(true_next, torch.tensor([297.0, 24.0]))

    result = _run(path, capture_x)

    # The sparse replay recomputes the pre-write [4, 8]; the fix REFUSES to bless it.
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert result.report.poisoned
    assert not torch.allclose(result.output, true_next)


# --------------------------------------------------------------------------- #
# HARD NO-OVER-TRIGGER: each of these leaves the run VERIFIED.
# --------------------------------------------------------------------------- #
class TransientLocalNumpyAlias(nn.Module):
    """The make-or-break: a transient local numpy alias is collected at forward end."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        arr = y.detach().numpy()
        _ = arr.sum()  # read-only; ``arr`` dies when the forward returns
        return y


@pytest.mark.smoke
def test_transient_local_numpy_alias_stays_verified(tmp_path: Path) -> None:
    capture_x = torch.tensor([2.0, 4.0])
    path = _save(TransientLocalNumpyAlias(), capture_x, tmp_path / "transient.tlspec")

    result = _run(path, capture_x)

    # The gc.collect() at forward end reclaims the transient alias, so the source stays
    # VERIFIED + ATTESTED and value-correct on the original input.
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    assert not result.report.poisoned
    assert torch.allclose(result.output, torch.tensor([4.0, 8.0]))


class RetainedReadOnlyNumpyCopy(nn.Module):
    """A retained numpy COPY owns its OWN memory -- no window shared with the tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.cache: np.ndarray | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        self.cache = y.detach().numpy().copy()  # the shared VIEW is transient; only the copy lives
        return y


@pytest.mark.smoke
def test_retained_readonly_numpy_copy_stays_verified(tmp_path: Path) -> None:
    capture_x = torch.tensor([2.0, 4.0])
    path = _save(RetainedReadOnlyNumpyCopy(), capture_x, tmp_path / "copy.tlspec")

    result = _run(path, capture_x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned
    assert torch.allclose(result.output, torch.tensor([4.0, 8.0]))


class RetainedDetachedTorchTensor(nn.Module):
    """A retained DETACHED TORCH TENSOR is not a host alias -- ordinary retained tensor state."""

    def __init__(self) -> None:
        super().__init__()
        self.cache: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        self.cache = y.detach()  # a torch tensor, never a mutable host alias
        return y


@pytest.mark.smoke
def test_retained_detached_torch_tensor_stays_verified(tmp_path: Path) -> None:
    capture_x = torch.tensor([2.0, 4.0])
    path = _save(RetainedDetachedTorchTensor(), capture_x, tmp_path / "detached.tlspec")

    result = _run(path, capture_x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned
    assert torch.allclose(result.output, torch.tensor([4.0, 8.0]))


class PlainStatefulCounter(nn.Module):
    """A plain stateful model with no tensor alias whatsoever."""

    def __init__(self) -> None:
        super().__init__()
        self.counter = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.counter += 1
        return x * 2 if self.counter > 0 else x


@pytest.mark.smoke
def test_plain_stateful_counter_stays_verified(tmp_path: Path) -> None:
    capture_x = torch.tensor([2.0, 4.0])
    path = _save(PlainStatefulCounter(), capture_x, tmp_path / "counter.tlspec")

    result = _run(path, capture_x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned
    assert torch.allclose(result.output, torch.tensor([4.0, 8.0]))


class ReadOnlyExposures(nn.Module):
    """Read-only host exposures: none opens a retained mutable window."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        _ = y.tolist()  # copies into Python lists
        _ = int(y.sum())  # scalar escape, no alias
        _ = y.untyped_storage().nbytes()  # metadata on a transient handle
        _ = y.numpy().copy()  # the shared view is transient; the copy owns its memory
        return y


@pytest.mark.smoke
def test_readonly_exposures_stay_verified(tmp_path: Path) -> None:
    capture_x = torch.tensor([2.0, 4.0])
    path = _save(ReadOnlyExposures(), capture_x, tmp_path / "readonly.tlspec")

    result = _run(path, capture_x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned
    assert torch.allclose(result.output, torch.tensor([4.0, 8.0]))


class TransientStorageRead(nn.Module):
    """A transient ``storage()`` handle read is collected at forward end."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        _ = y.storage().size()  # TypedStorage handle dies when the forward returns
        return y


@pytest.mark.smoke
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_transient_storage_read_stays_verified(tmp_path: Path) -> None:
    capture_x = torch.tensor([2.0, 4.0])
    path = _save(TransientStorageRead(), capture_x, tmp_path / "storage.tlspec")

    result = _run(path, capture_x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned
    assert torch.allclose(result.output, torch.tensor([4.0, 8.0]))


# --------------------------------------------------------------------------- #
# Real models (no host alias): VERIFIED on any input, no perf regression, no hang.
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f1 = nn.Linear(4, 8)
        self.f2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f2(torch.relu(self.f1(x)))


class ConvReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c(x))


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("model_factory", "capture_x", "changed_x"),
    [
        (MLP, torch.randn(2, 4), torch.randn(2, 4)),
        (ConvReLU, torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8)),
    ],
    ids=["mlp", "conv_relu"],
)
def test_real_models_stay_verified_any_input(
    model_factory: type[nn.Module],
    capture_x: torch.Tensor,
    changed_x: torch.Tensor,
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    model = model_factory().eval()
    path = _save(model, capture_x, tmp_path / "real.tlspec")

    original = _run(path, capture_x)
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not original.report.poisoned
    assert torch.allclose(original.output, model(capture_x.clone()), atol=1e-5)

    changed = _run(path, changed_x)
    assert changed.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not changed.report.poisoned
    assert torch.allclose(changed.output, model(changed_x.clone()), atol=1e-5)
