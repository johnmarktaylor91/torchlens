"""Credit-required lint (workstream D01 gate).

The acknowledgments page (docs/acknowledgments.md) is the assembled credit
roster from every design memo's CREDIT section. The convert-memo maintenance
rule: any change landing a borrowed idea adds or extends a row on that page
in the same change. This lint is the mechanical floor for that rule:

- the page exists and is non-trivial;
- README.md links to it;
- every load-bearing credit entry demanded by a memo CREDIT section is
  present on the page (name match, case-insensitive).

Removing a name from REQUIRED_CREDITS requires removing the feature that
borrowed from it (or an explicit D-phase ruling) -- never edit the list to
make the test pass.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ACK_PATH = REPO_ROOT / "docs" / "acknowledgments.md"
README_PATH = REPO_ROOT / "README.md"

# One load-bearing name per memo CREDIT demand. Grouped by memo family for
# auditability; the test flattens the groups.
REQUIRED_CREDITS = {
    # mechinterp / edits / reads / ledger / mikit
    "TransformerLens",
    "nnsight",
    "pyvene",
    "baukit",
    "Redwood Research",
    "ACDC",
    "circuit-tracer",
    "SAELens",
    "nostalgebraist",
    "Belrose",
    "Elhage",
    "ARENA",
    # attribution
    "Captum",
    "zennit",
    "transformers-interpret",
    "inseq",
    "Grad-CAM",
    "Ancona",
    # summary / lovely / snoop / treescope / quickstart
    "torchinfo",
    "lovely-tensors",
    "treescope",
    "penzai",
    "torchsnooper",
    "pysnooper",
    "Keras",
    "flax",
    "rich",
    # visualization / export
    "Graphviz",
    "Netron",
    "Model Explorer",
    "hiddenlayer",
    "torchview",
    "TensorBoard",
    "BertViz",
    "CircuitsVis",
    "Ecco",
    "inspectus",
    "LIT",
    "Okabe-Ito",
    "ColorBrewer",
    "matplotlib",
    # checks / explorer / trackers / ledger
    "torcheck",
    "Karpathy",
    "torchexplorer",
    "Weights & Biases",
    "MLflow",
    "Neptune",
    "ClearML",
    "DDSketch",
    "Sacred",
    "Hydra",
    "DVC",
    "Lightning",
    # cost / torchnative / observe
    "Kineto",
    "CUPTI",
    "FlopCounterMode",
    "fvcore",
    "ptflops",
    "calflops",
    "thop",
    "PaLM",
    "roofline",
    "DeepSpeed",
    "py-spy",
    "speedscope",
    # neuro / brainpipe / tvscope / transforms
    "thingsvision",
    "rsatoolbox",
    "Brain-Score",
    "Net2Brain",
    "DeepJuice",
    "CORnet",
    "netrep",
    "himalaya",
    "Kriegeskorte",
    "Kornblith",
    "scikit-learn",
    "Achlioptas",
    "Johnson-Lindenstrauss",
    "COCO",
    "sentence-transformers",
    "timm",
    # extract / ecosystem / agent
    "safetensors",
    "Apache Arrow",
    "HDF5",
    "CRC-32",
    "StableHLO",
    "ONNX",
    "OpenTelemetry",
    "Model Context Protocol",
    "JSON Schema",
    "Hugging Face",
    # testing / compo / oracles
    "Hypothesis",
    "mutmut",
    "NIST ACTS",
    "nbclient",
    # pytorch neighborhood
    "torch.fx",
    "torchextractor",
    "surgeon-pytorch",
    "pytorch-grad-cam",
}


@pytest.mark.smoke
def test_acknowledgments_page_exists_and_is_substantial():
    assert ACK_PATH.is_file(), "docs/acknowledgments.md is missing"
    text = ACK_PATH.read_text(encoding="utf-8")
    assert len(text) > 5000, "acknowledgments page suspiciously short"
    assert "License hygiene" in text, "license-hygiene paragraph missing"
    assert "Maintenance rule" in text, "maintenance rule missing"


@pytest.mark.smoke
def test_readme_links_to_acknowledgments_page():
    readme = README_PATH.read_text(encoding="utf-8")
    assert re.search(r"docs/acknowledgments\.md", readme), (
        "README.md must link to docs/acknowledgments.md"
    )


@pytest.mark.smoke
def test_every_required_credit_is_present():
    text = ACK_PATH.read_text(encoding="utf-8").lower()
    missing = sorted(name for name in REQUIRED_CREDITS if name.lower() not in text)
    assert not missing, (
        "acknowledgments page is missing required credit entries "
        f"(add rows, never trim the list): {missing}"
    )
