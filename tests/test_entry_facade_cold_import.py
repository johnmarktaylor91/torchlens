"""Cold-import subprocess matrix: the A10 row gate.

One fresh interpreter per matrix run proves, with no test-session state:

- ``import torchlens`` imports no appliance/integration namespace, no
  semantic package, and no foreign extra package;
- a ``hasattr``/``dir()`` probe sweep across the root and every facade
  namespace answers (never raises), imports nothing foreign, and -- with a
  REAL ``torchlens.recipes`` entry point planted via dist-info on the path --
  never imports or executes the provider (``EntryPoint.load()`` at probe
  time was the launch-blocking defect);
- the reachability rows (tl.bridge / tl.callbacks / tl.neuro / tl.notebook /
  tl.load_extraction) resolve cold, order-independent;
- explicit activation DOES import and run the planted provider, exactly
  once.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.heavy

_REPO_ROOT = Path(__file__).resolve().parent.parent

_MATRIX_SCRIPT = """
import json
import pickle
import sys

results = {}

import torchlens as tl

# Cell 1: import-inertness of the entry layer.
results["post_import_absent"] = [
    name for name in (
        "torchlens.bridge", "torchlens.callbacks", "torchlens.neuro",
        "torchlens.notebook", "torchlens.semantic",
        "rsatoolbox", "brainscore_core", "IPython", "jupyter_client",
        "fake_recipe_pkg",
    ) if name in sys.modules
]

# Cell 2: probe sweep -- hasattr answers, never raises, executes nothing.
probe_names = (
    "facets", "definitely_not_real", "_ipython_canary_method_should_not_exist_",
    "__wrapped__", "rdms", "datasets",
)
probe_failures = []
for namespace_name in ("tl", "tl.neuro", "tl.notebook", "tl.bridge", "tl.callbacks"):
    namespace = eval(namespace_name)
    for name in probe_names:
        try:
            hasattr(namespace, name)
            dir(namespace)
        except BaseException as exc:  # noqa: BLE001 - the assertion IS no-raise.
            probe_failures.append(f"{namespace_name}.{name}: {type(exc).__name__}: {exc}")
results["probe_failures"] = probe_failures
results["provider_imported_after_probes"] = "fake_recipe_pkg" in sys.modules
results["provider_ran_after_probes"] = bool(getattr(
    sys.modules.get("fake_recipe_pkg"), "RAN", False))

# Cell 3: reachability rows resolve cold (no capture side effect needed).
results["rows"] = {
    "bridge": tl.bridge.__name__,
    "callbacks": tl.callbacks.__name__,
    "neuro": tl.neuro.__name__,
    "notebook": tl.notebook.__name__,
    "load_extraction_callable": callable(tl.load_extraction),
}

# Cell 4: pickle-safety -- resolving a foreign reference into an appliance
# namespace answers AttributeError and imports nothing foreign.
try:
    stream = (
        pickle.PROTO + bytes([2])
        + pickle.GLOBAL + b"torchlens.neuro\\nnot_a_real_name\\n"
        + pickle.STOP
    )
    pickle.loads(stream)
    results["pickle_outcome"] = "resolved (BAD)"
except AttributeError:
    results["pickle_outcome"] = "attribute_error"
except BaseException as exc:  # noqa: BLE001 - disclose exact failure shape.
    results["pickle_outcome"] = f"{type(exc).__name__}: {exc}"
results["foreign_after_pickle"] = [
    name for name in ("rsatoolbox", "brainscore_core") if name in sys.modules
]

# Cell 5: explicit activation imports and runs the planted provider ONCE.
from torchlens.semantic import recipes
inventory = recipes.installed_recipe_providers()
results["inventory_has_plant"] = any(
    row["name"] == "planted" and row["activated"] == "false" for row in inventory
)
results["provider_imported_after_inventory"] = "fake_recipe_pkg" in sys.modules
activated = recipes.activate_entrypoint_recipes()
results["activated"] = sorted(activated)
plant = sys.modules.get("fake_recipe_pkg")
results["provider_ran_after_activation"] = bool(getattr(plant, "RAN", False))
results["provider_run_count"] = int(getattr(plant, "RUN_COUNT", 0))
recipes.activate_entrypoint_recipes()
results["provider_run_count_after_repeat"] = int(getattr(plant, "RUN_COUNT", 0))

print(json.dumps(results))
"""


def _plant_recipe_provider(root: Path) -> None:
    """Write a real installed-distribution recipe provider under ``root``."""

    package_dir = root / "fake_recipe_pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        textwrap.dedent(
            """
            RAN = False
            RUN_COUNT = 0


            def register():
                global RAN, RUN_COUNT
                RAN = True
                RUN_COUNT += 1


            register._torchlens_recipe_autoload = True
            """
        ),
        encoding="utf-8",
    )
    dist_info = root / "fake_recipe_dist-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: fake-recipe-dist\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[torchlens.recipes]\nplanted = fake_recipe_pkg:register\n",
        encoding="utf-8",
    )


def test_cold_import_subprocess_matrix(tmp_path: Path) -> None:
    """Run the whole matrix in one fresh interpreter and assert every cell."""

    _plant_recipe_provider(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-c", _MATRIX_SCRIPT],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(tmp_path),
        },
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, (
        f"cold-import matrix subprocess failed ({completed.returncode}):\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    results = json.loads(completed.stdout.splitlines()[-1])

    assert results["post_import_absent"] == [], results["post_import_absent"]
    assert results["probe_failures"] == [], results["probe_failures"]
    assert results["provider_imported_after_probes"] is False, (
        "a hasattr/dir probe imported the planted entry-point provider"
    )
    assert results["provider_ran_after_probes"] is False

    assert results["rows"] == {
        "bridge": "torchlens.bridge",
        "callbacks": "torchlens.callbacks",
        "neuro": "torchlens.neuro",
        "notebook": "torchlens.notebook",
        "load_extraction_callable": True,
    }

    assert results["pickle_outcome"] == "attribute_error", results["pickle_outcome"]
    assert results["foreign_after_pickle"] == []

    assert results["inventory_has_plant"] is True
    assert results["provider_imported_after_inventory"] is False, (
        "the metadata-only inventory imported the provider"
    )
    assert results["activated"] == ["planted"]
    assert results["provider_ran_after_activation"] is True
    assert results["provider_run_count"] == 1
    assert results["provider_run_count_after_repeat"] == 1, "activation must be idempotent"
