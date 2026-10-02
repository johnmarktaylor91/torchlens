"""The RNG channel monitor warms torch's lazy dynamo import OUTSIDE its window.

The first wrapped op of a capture can trigger torch's own lazy
``import torch._dynamo`` (``torch/_compile.py``), whose import cascade draws
host entropy at module-exec time (``uuid.uuid4()`` in
``torch.distributed._composable.contract``, plus getrandbits / RNG-instance
draws). Fired INSIDE the host-nondeterminism monitor window, those draws
marked ``os.urandom``/getrandbits channels and permanently ceilinged the
FIRST selective runnable-capable capture of the process to UNVERIFIABLE on a
pure deterministic model -- a silent, order-dependent breach of the runnable
contract's "a plain deterministic capture records nothing" pin (hunt-b8 F1).

The poison is once-per-process (a second capture in the same process is
clean), so the authoritative gate runs in a fresh subprocess.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from torchlens.utils import _torch_compat
from torchlens.utils.rng import host_nondeterminism_monitor

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

_FIRST_SELECTIVE_CAPTURE_IS_CLEAN = """
import sys

import torch
from torch import nn

import torchlens as tl

assert "torch._dynamo" not in sys.modules, (
    "precondition broken: torch._dynamo was imported before the first capture, "
    "so this child cannot exercise the lazy-import-in-window path"
)

model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
x = torch.randn(1, 4)
log = tl.trace(
    model,
    x,
    save=tl.func("relu"),
    capture=tl.options.CaptureOptions(intervention_ready=True),
)
channels = tuple(log._runnable.host_rng_channels)
assert channels == (), (
    "torch's lazy torch._dynamo import cascade fired INSIDE the RNG monitor "
    f"window and poisoned a pure deterministic capture: channels={channels!r}"
)
assert log._runnable.host_rng_unreplayable is False, (
    "deterministic first selective capture settled unreplayable"
)
assert "torch._dynamo" in sys.modules, (
    "the capture never triggered (or pre-warmed) the dynamo import; "
    "this gate is vacuous on this torch build"
)
print("OK")
"""


@pytest.mark.heavy
def test_first_selective_runnable_capture_is_not_poisoned_by_lazy_dynamo_import() -> None:
    """Fresh process: first runnable-capable selective capture records nothing."""

    completed = subprocess.run(
        [sys.executable, "-c", _FIRST_SELECTIVE_CAPTURE_IS_CLEAN],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, (
        "first-selective-capture channel-cleanliness child failed:\n"
        f"STDOUT:{completed.stdout}\nSTDERR:{completed.stderr}"
    )
    assert "OK" in completed.stdout


@pytest.mark.smoke
def test_monitor_entry_warms_lazy_torch_imports() -> None:
    """Entering the monitor latches the compat warm flag before any patch."""

    import torch
    from torch import nn

    with host_nondeterminism_monitor(nn.Identity()):
        assert _torch_compat._LAZY_TORCH_IMPORTS_WARMED is True
        assert "torch._dynamo" in sys.modules
    assert torch is not None
