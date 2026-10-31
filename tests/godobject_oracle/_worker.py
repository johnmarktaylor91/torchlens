"""Subprocess entry point for isolated viz-identity DOT generation.

Renders every forward-DOT golden in ONE fresh interpreter, constructing both
models BEFORE the first capture so no model ctor ever runs on wrapped torch:
in-process generation inside the pytest session constructed models on
whatever wrap state earlier tests left behind, the SF-53 generator class
(b10 R78-1). Output is one JSON object mapping golden name -> DOT source.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def main() -> int:
    """Print the golden-name -> DOT-source mapping on stdout.

    Returns
    -------
    int
        Process exit status.
    """

    import torch

    import torchlens as tl
    from godobject_oracle.test_viz_identity import _SEED, VizCNN, VizRecurrent

    # Construct every model before the first capture wraps torch.
    models = {
        "viz_cnn": (VizCNN, torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4)),
        "viz_recurrent": (VizRecurrent, torch.linspace(-1.0, 1.0, 4).reshape(1, 4)),
    }
    built = {}
    for key, (model_class, model_input) in models.items():
        torch.manual_seed(_SEED)
        built[key] = (model_class(), model_input)

    renders: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for key, (model, model_input) in built.items():
            torch.manual_seed(_SEED)
            trace = tl.trace(model, model_input)
            for vis_mode in ("unrolled", "rolled"):
                source = trace.draw(
                    vis_save_only=True,
                    file_format="svg",
                    path=str(Path(tmp) / f"{key}_{vis_mode}"),
                    vis_mode=vis_mode,
                )
                renders[f"viz_{key}_{vis_mode}.gv"] = source
    print(json.dumps(renders))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
