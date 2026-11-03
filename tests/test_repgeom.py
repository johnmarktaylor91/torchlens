"""Tests for representation-geometry numeric helpers."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import repgeom
from torchlens._visualization.node_spec import NodeSpec

ANALYTIC_POINTS = np.array(
    [
        [-3.0, 0.5],
        [-1.7, -2.2],
        [-0.4, 1.8],
        [0.9, -0.8],
        [2.6, 2.4],
        [3.8, -1.5],
        [5.1, 0.7],
        [6.4, 3.1],
    ],
    dtype=np.float64,
)


class _MDSClassifier(nn.Module):
    """Small multi-layer model with batch activations for MDS tests."""

    def __init__(self) -> None:
        """Initialize deterministic linear layers."""

        super().__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 5)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.eye(5))
            self.fc1.bias.zero_()
            self.fc2.weight.copy_(
                torch.tensor(
                    [
                        [0.8, -0.6, 0.0, 0.0, 0.0],
                        [0.6, 0.8, 0.0, 0.0, 0.0],
                        [0.2, 0.1, 1.0, 0.0, 0.0],
                        [0.0, 0.3, 0.0, 1.0, 0.0],
                        [0.1, 0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=torch.float32,
                )
            )
            self.fc2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two saved linear layers."""

        return self.fc2(torch.tanh(self.fc1(x)))


class _RecurrentMDS(nn.Module):
    """Model that reuses one linear layer across multiple passes."""

    def __init__(self) -> None:
        """Initialize the recurrent layer."""

        super().__init__()
        self.attn = nn.Linear(5, 5)
        with torch.no_grad():
            self.attn.weight.copy_(torch.eye(5))
            self.attn.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the same module three times."""

        for _index in range(3):
            x = torch.tanh(self.attn(x))
        return x


def _pil_stimuli(n_items: int = 8) -> list[Any]:
    """Return deterministic PIL image stimuli for scatter-render tests.

    Parameters
    ----------
    n_items:
        Number of image stimuli to create.

    Returns
    -------
    list[Any]
        RGB PIL images with stable colors and simple index marks.
    """

    from PIL import Image as image_module, ImageDraw as draw_module

    images = []
    for index in range(n_items):
        image = image_module.new(
            "RGB",
            (28, 24),
            color=((37 * index) % 255, (91 + 23 * index) % 255, (170 - 11 * index) % 255),
        )
        draw = draw_module.Draw(image)
        draw.text((3, 4), str(index), fill=(255, 255, 255))
        images.append(image)
    return images


def _trace_with_mds_and_images() -> Any:
    """Return a trace with MDS coords and matching PIL raw stimuli.

    Returns
    -------
    Any
        TorchLens trace with two annotated linear layers.
    """

    trace = _mds_trace(_MDSClassifier(), tl.func("linear"))
    trace.input = _pil_stimuli(8)
    repgeom.mds_evolution(trace, save=tl.func("linear"), min_stimuli=8)
    return trace


def _mds_trace(model: nn.Module, save: Any) -> tl.Trace:
    """Capture a deterministic MDS test trace.

    Parameters
    ----------
    model:
        Model to trace.
    save:
        TorchLens save selector.

    Returns
    -------
    tl.Trace
        Captured trace with requested saved activations.
    """

    torch.manual_seed(1001)
    zeros = np.zeros((ANALYTIC_POINTS.shape[0], 3), dtype=np.float64)
    x = torch.tensor(np.concatenate([ANALYTIC_POINTS, zeros], axis=1), dtype=torch.float32)
    return tl.trace(model, x, save=save, capture=tl.options.CaptureOptions(seed=1001))


def test_classical_mds_rank_deficient_duplicate_gate_raises_clear_error() -> None:
    """MDS should reject duplicate stimuli as rank-deficient for display."""

    points = ANALYTIC_POINTS.copy()
    points[3] = points[0]

    with pytest.raises(ValueError, match="rank-deficient|duplicate"):
        repgeom.classical_mds(points, min_stimuli=8)


def test_rdm_degenerate_min_n_and_angular_metrics_raise_clear_errors() -> None:
    """RDM public paths should reject too-small and zero-norm cases clearly."""

    with pytest.raises(ValueError, match="min_stimuli must be at least 2"):
        repgeom.rdm_evolution(_mds_trace(_MDSClassifier(), tl.func("linear")), min_stimuli=1)

    trace = tl.trace(nn.Linear(3, 3), torch.ones(1, 3), save=tl.func("linear"))
    with pytest.raises(ValueError, match="too few stimuli"):
        repgeom.rdm_evolution(trace, min_stimuli=2)

    with pytest.raises(ValueError, match="cosine distance is undefined for zero-norm stimuli"):
        repgeom.rdm(np.array([[0.0, 0.0], [1.0, 0.0]]), metric="cosine")

    with pytest.raises(ValueError, match="correlation distance is undefined for zero-norm stimuli"):
        repgeom.rdm(np.array([[1.0, 1.0], [1.0, 2.0]]), metric="correlation")


def test_repgeom_import_clean_without_optional_neuro_or_plotting_deps() -> None:
    """Importing repgeom should not import optional plotting, sklearn, or neuro deps."""

    script = """
import sys
import torchlens.repgeom
blocked = {'scipy', 'sklearn', 'matplotlib', 'rsatoolbox', 'brainscore', 'brainscore_core'}
loaded = sorted(name for name in blocked if name in sys.modules)
if loaded:
    raise SystemExit('unexpected imports: ' + ', '.join(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout


def test_mds_evolution_single_pass_layers_annotates_and_round_trips(tmp_path: Path) -> None:
    """MDS evolution should compute, align, annotate, and persist coords."""

    trace = _mds_trace(_MDSClassifier(), tl.func("linear"))
    linear_layers = [layer for layer in trace.layers if layer.layer_type == "linear"]

    coords_by_key = repgeom.mds_evolution(trace, save=tl.func("linear"), min_stimuli=8)

    assert list(coords_by_key) == [f"layer:{layer.layer_label}" for layer in linear_layers]
    assert trace._annotation_blobs is not None
    first_key, second_key = list(coords_by_key)
    for key, coords in coords_by_key.items():
        assert coords.shape == (8, 2)
        # Derived MDS blobs live in the ``mds:`` namespace; bare layer:/op:
        # keys are reserved for user ``annotate(data=...)`` blobs.
        assert torch.equal(trace._annotation_blobs[f"mds:{key}"], torch.from_numpy(coords))

    raw_second_distances = repgeom.activation_distance_matrix(linear_layers[1].out)
    raw_second_coords, _info = repgeom.classical_mds(raw_second_distances, min_stimuli=8)
    expected_second = repgeom.procrustes_align(raw_second_coords, coords_by_key[first_key])
    assert np.allclose(coords_by_key[second_key], expected_second)

    bundle_path = tmp_path / "mds_evolution.tlspec"
    trace.save(bundle_path)
    loaded = tl.load(bundle_path)

    assert loaded._annotation_blobs is not None
    for key, coords in coords_by_key.items():
        assert torch.equal(loaded._annotation_blobs[f"mds:{key}"], torch.from_numpy(coords))


def test_effective_dimensionality_all_zero_fixture_is_safe() -> None:
    """All-zero representations should report rank zero without warnings."""

    activations = np.zeros((5, 4), dtype=np.float64)
    with np.errstate(all="raise"):
        info = repgeom.effective_dimensionality(activations, min_stimuli=3)

    assert np.array_equal(info["eigenvalues"], np.zeros(5))
    assert np.array_equal(info["variance_explained"], np.zeros(5))
    assert np.array_equal(info["cumulative_variance"], np.zeros(5))
    assert info["participation_ratio"] == 0.0
    assert info["n_components_for_threshold"] == 0
    assert info["total_positive_variance"] == 0.0
    assert info["effective_rank"] == 0


def test_scree_node_spec_sets_one_draw_time_image_for_annotated_layer() -> None:
    """The scree hook should set one bounded line-plot image for annotated layers."""

    trace = _mds_trace(_MDSClassifier(), tl.func("linear"))
    repgeom.scree_evolution(trace, save=tl.func("linear"), min_stimuli=8)
    layer = next(layer for layer in trace.layers if layer.layer_type == "linear")
    hook = tl.viz.nodes.scree(max_components=8)
    spec = NodeSpec(lines=[layer.layer_label])

    result = hook(layer, spec)

    assert result is not None
    assert result.image is not None
    assert Path(result.image).is_file()
    assert result.image.endswith(".png")
    from PIL import Image as image_module

    with image_module.open(result.image) as rendered:
        assert rendered.mode == "RGB"
        assert rendered.size == (320, 214)
    assert result.lines == [layer.layer_label]
    assert result.shape == "box"
    assert result.extra_attrs["imagescale"] == "true"
    assert result.extra_attrs["labelloc"] == "b"
    assert result.tooltip is not None
    assert "Scree plot" in result.tooltip
    assert "rank" in result.tooltip


def test_rdm_node_spec_draw_uses_one_contained_image_and_survives_save_load(
    tmp_path: Path,
) -> None:
    """RDM SVG output should inline one heatmap image per annotated node after reload."""

    trace = _mds_trace(_MDSClassifier(), tl.func("linear"))
    trace.input = _pil_stimuli(8)
    repgeom.rdm_evolution(trace, save=tl.func("linear"), min_stimuli=8)
    annotated_layers = [layer for layer in trace.layers if layer.layer_type == "linear"]
    bundle_path = tmp_path / "rdm_source.tlspec"
    trace.save(bundle_path)
    loaded = tl.load(bundle_path)
    output_path = tmp_path / "rdm.svg"

    loaded.draw(
        path=str(output_path),
        file_format="svg",
        node_spec_fn=tl.viz.nodes.rdm(max_stimuli=8),
    )

    svg_text = output_path.read_text(encoding="utf-8")
    image_tags = re.findall(r"<image\b[^>]+>", svg_text)
    assert len(image_tags) == len(annotated_layers) + 1
    for layer in annotated_layers:
        node_name = f"{layer.layer_label}pass1"
        block_match = re.search(
            rf"<title>{re.escape(node_name)}</title>(.*?)(?=<!--|</svg>)",
            svg_text,
            flags=re.DOTALL,
        )
        assert block_match is not None
        node_block = block_match.group(1)
        node_images = re.findall(r'<image\b[^>]+(?:xlink:href|href)="([^"]+)"', node_block)
        assert len(node_images) == 1
        assert node_images[0].startswith("data:image/png;base64,")
        assert "<polygon" in node_block
        assert f">{layer.layer_label}</text>" in node_block


def test_scree_node_spec_draw_uses_one_contained_image_and_survives_save_load(
    tmp_path: Path,
) -> None:
    """Scree SVG output should inline one plot image per annotated node after reload."""

    trace = _mds_trace(_MDSClassifier(), tl.func("linear"))
    repgeom.scree_evolution(trace, save=tl.func("linear"), min_stimuli=8)
    annotated_layers = [layer for layer in trace.layers if layer.layer_type == "linear"]
    bundle_path = tmp_path / "scree_source.tlspec"
    trace.save(bundle_path)
    loaded = tl.load(bundle_path)
    output_path = tmp_path / "scree.svg"

    loaded.draw(
        path=str(output_path),
        file_format="svg",
        node_spec_fn=tl.viz.nodes.scree(max_components=8),
    )

    svg_text = output_path.read_text(encoding="utf-8")
    image_tags = re.findall(r"<image\b[^>]+>", svg_text)
    assert len(image_tags) == len(annotated_layers)
    assert "PR=" in svg_text
    assert "comps for 90% var" in svg_text
    for layer in annotated_layers:
        node_name = f"{layer.layer_label}pass1"
        block_match = re.search(
            rf"<title>{re.escape(node_name)}</title>(.*?)(?=<!--|</svg>)",
            svg_text,
            flags=re.DOTALL,
        )
        assert block_match is not None
        node_block = block_match.group(1)
        node_images = re.findall(r'<image\b[^>]+(?:xlink:href|href)="([^"]+)"', node_block)
        assert len(node_images) == 1
        assert node_images[0].startswith("data:image/png;base64,")
        assert "<polygon" in node_block
        assert f">{layer.layer_label}</text>" in node_block


def test_mds_scatter_cap_overlap_and_more_indicator(tmp_path: Path) -> None:
    """Close coordinates should render deterministically with a visible cap label."""

    trace = _trace_with_mds_and_images()
    layer = next(layer for layer in trace.layers if layer.layer_type == "linear")
    assert trace._annotation_blobs is not None
    trace._annotation_blobs[f"layer:{layer.layer_label}"] = torch.zeros(8, 2)
    hook = tl.viz.nodes.mds_scatter(max_thumbnails=3, thumbnail_size=28)

    first = hook(layer, NodeSpec(lines=[layer.layer_label]))
    second = hook(layer, NodeSpec(lines=[layer.layer_label]))

    assert first is not None
    assert second is not None
    assert first.tooltip is not None
    assert "+5 more" in first.tooltip
    assert first.image is not None
    assert second.image is not None
    assert Path(first.image).read_bytes() == Path(second.image).read_bytes()

    output_path = tmp_path / "scatter_cap.svg"
    trace.draw(
        path=str(output_path),
        file_format="svg",
        node_spec_fn=hook,
    )
    assert "+5 more" in output_path.read_text(encoding="utf-8")


def test_mds_scatter_fallback_renders_points_without_raw_images() -> None:
    """Coords without matching raw PIL images should render a points fallback."""

    trace = _trace_with_mds_and_images()
    trace.input = None
    layer = next(layer for layer in trace.layers if layer.layer_type == "linear")
    hook = tl.viz.nodes.mds_scatter(max_thumbnails=8)

    result = hook(layer, NodeSpec(lines=[layer.layer_label]))

    assert result is not None
    assert result.image is not None
    assert Path(result.image).is_file()
    from PIL import Image as image_module

    with image_module.open(result.image) as rendered:
        assert rendered.mode == "RGB"
        assert rendered.size == (420, 420)
    assert result.tooltip is not None
    assert "points fallback" in result.tooltip


def test_mds_scatter_import_does_not_load_matplotlib() -> None:
    """The scatter renderer must stay PIL-only and avoid matplotlib imports."""

    script = """
import sys
import torchlens.viz.nodes as viz_nodes
viz_nodes.mds_scatter()
print('matplotlib' in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"


def test_mds_evolution_recurrent_aggregate_requires_pass_selection() -> None:
    """Aggregate recurrent layers should raise a clear pass-selection error."""

    trace = _mds_trace(_RecurrentMDS(), tl.func("linear"))

    with pytest.raises(ValueError, match="select a pass \\(layer is recurrent\\)"):
        repgeom.mds_evolution(trace, save=tl.func("linear"), min_stimuli=8)


def test_rdm_evolution_recurrent_expansion_leaves_mds_refusal() -> None:
    """U24.2: expansion leaves mds_evolution's recurrent refusal unchanged."""

    trace = _mds_trace(_RecurrentMDS(), tl.func("linear"))

    with pytest.raises(ValueError, match="select a pass \\(layer is recurrent\\)"):
        repgeom.mds_evolution(trace, min_stimuli=8)
    assert trace._annotation_blobs is None
    assert len(repgeom.rdm_evolution(trace, min_stimuli=8)) == 3


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("euclidean", "rdm/*/euclidean"),
        ("cosine", "rdm/*/cosine"),
        ("correlation", "rdm/*/correlation"),
        ("manhattan", None),
    ],
)
def test_rdm_evolution_records_similarity_repository_id(metric: str, expected: Any) -> None:
    """U56.3: the result names its exact RDM variant in the catalogue, or None."""

    trace = _mds_trace(_MDSClassifier(), tl.func("linear"))

    result = repgeom.rdm_evolution(trace, metric=metric, min_stimuli=8)

    assert result.similarity_repository_id == expected
    assert repgeom.mds_evolution(trace, min_stimuli=8).similarity_repository_id is None


def test_scree_evolution_recurrent_pass_qualified_selector_uses_op_key(tmp_path: Path) -> None:
    """A single recurrent pass selection should read op.out and store op scree."""

    trace = _mds_trace(_RecurrentMDS(), tl.func("linear"))
    linear_op = next(op for op in trace._layer_list if op.type == "linear" and op.pass_number == 2)

    spectra_by_key = repgeom.scree_evolution(trace, save=tl.label(linear_op.label), min_stimuli=8)

    key = f"op:{linear_op.label}"
    assert list(spectra_by_key) == [key]
    assert spectra_by_key[key].shape == (8,)
    assert trace._annotation_blobs is not None
    assert torch.equal(
        trace._annotation_blobs[f"scree:{key}"], torch.from_numpy(spectra_by_key[key])
    )

    bundle_path = tmp_path / "scree_recurrent.tlspec"
    trace.save(bundle_path)
    loaded = tl.load(bundle_path)
    assert loaded._annotation_blobs is not None
    assert torch.equal(
        loaded._annotation_blobs[f"scree:{key}"], torch.from_numpy(spectra_by_key[key])
    )


def test_scree_evolution_default_recurrent_saved_layer_requires_pass_selection() -> None:
    """Default recurrent scree selection should not silently omit saved passes."""

    trace = _mds_trace(_RecurrentMDS(), tl.func("linear"))

    with pytest.raises(ValueError, match="select a pass \\(layer is recurrent\\)"):
        repgeom.scree_evolution(trace, min_stimuli=8)


def test_mds_evolution_unsaved_activation_raises_capture_guidance() -> None:
    """Selected layers without saved outs should raise save= guidance."""

    trace = _mds_trace(_MDSClassifier(), tl.func("tanh"))

    with pytest.raises(ValueError, match="capture with save=.*MDS layers"):
        repgeom.mds_evolution(trace, save=tl.func("linear"), min_stimuli=8)


def test_mds_evolution_min_n_gate_is_honored() -> None:
    """MDS evolution should forward the minimum stimulus gate."""

    trace = _mds_trace(_MDSClassifier(), tl.func("linear"))

    with pytest.raises(ValueError, match="too few stimuli"):
        repgeom.mds_evolution(trace, save=tl.func("linear"), min_stimuli=9)
