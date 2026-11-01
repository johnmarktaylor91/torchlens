"""Package-wide ``python -O`` verdict-identity gate (r7 b7-sol R24).

97 shipped ``assert`` statements survive in ``torchlens/`` (shrink-only
ceiling), several on verdict-adjacent paths (``validation/backward.py``
pre-comparator guards, postprocess invariants). ``-O`` strips every one of
them, and until now nothing proved that validation VERDICTS are identical in
optimized mode -- the matrix stop condition explicitly requires an
optimized-mode verdict-equivalence proof, not just the audit's armed-under--O
refusal.

The gate runs one ``-O`` child that must reproduce three verdict classes
byte-for-byte with normal mode:

* a clean forward validation stays ``True``;
* a clean backward validation stays ``True`` (the pre-comparator asserts at
  ``validation/backward.py`` are stripped here -- the verdict must not
  change);
* a planted spurious bool edge (frozen replay callable) STILL settles
  ``failed``/``perturbation_insensitive`` -- the tripwire direction: an
  assert-stripped check failing OPEN would bless corruption under ``-O``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.heavy

_CHILD = '\nimport sys\nif sys.flags.optimize < 1:\n    print("NOT_OPTIMIZED")\n    sys.exit(2)\nimport warnings\nwarnings.filterwarnings("ignore")\nimport torch\nimport torch.nn as nn\nimport torchlens as tl\nfrom torchlens.options import CaptureOptions\nfrom torchlens.validation.core import (\n    _check_whether_func_on_saved_parents_yields_saved_tensor,\n)\n\ntorch.manual_seed(0)\nmodel = nn.Sequential(nn.Linear(4, 4), nn.ReLU())\nx = torch.randn(2, 4)\nprint("FORWARD_VERDICT", tl.validate(model, x, scope="forward", seed=0))\nprint("BACKWARD_VERDICT", tl.validate(model, x, scope="backward", seed=0))\n\n\nclass Gate(nn.Module):\n    def forward(self, z):\n        return (z > 1.0e30).float()\n\n\ntrace = tl.trace(\n    Gate(),\n    torch.randn(3, 4),\n    capture=CaptureOptions(save_args=True, seed=0), save="all",\n)\nop = [o for o in trace._layer_list if o.func_name == "__gt__"][0]\nsaved = op.out.detach().clone()\nobject.__setattr__(op, "func", lambda *a, **k: saved.clone())\nresult = _check_whether_func_on_saved_parents_yields_saved_tensor(\n    trace, op.label, perturb=True, layers_to_perturb=[op.parents[0]]\n)\nprint("SPURIOUS_DECISION", result.decision, result.reason)\n'


def test_optimized_mode_verdicts_are_identical_and_tripwires_still_fire() -> None:
    """The three verdict classes must not change under ``python -O``."""

    completed = subprocess.run(
        [sys.executable, "-O", "-c", _CHILD],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.getcwd()},
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    lines = {
        line.split(" ", 1)[0]: line.split(" ", 1)[1]
        for line in completed.stdout.splitlines()
        if " " in line
    }
    assert lines.get("FORWARD_VERDICT") == "True", lines
    assert lines.get("BACKWARD_VERDICT") == "True", lines
    assert lines.get("SPURIOUS_DECISION") == "failed perturbation_insensitive", (
        "the perturbation tripwire changed its verdict under -O -- an "
        f"assert-stripped check is failing open: {lines}"
    )
