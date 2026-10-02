"""Brute-force oracle hardening for receptive/projective-field geometry (r22).

Every geometry claim here is pinned against an independent ground truth that
never consults the TorchLens geometry engine: potential support is measured by
perturbing one element at a time and observing which outputs change (reverse
for receptive fields). Exact boxes must equal the true hull; upper bounds must
contain it. The suites were mutation-proven against the r21 audit defects
(max-pool dilation dropped, positional antialias missed, strided-slice
projective lattice loss, the line-637 bare assert, empty-box ``slices()``).
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
from torchlens.postprocess.graph_traversal import output_payload_aliases_parent
from torchlens.receptive_field import ReceptiveFieldStatus, ReceptiveFieldValidationStatus

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Independent brute-force oracles (no TorchLens geometry involved)
# ---------------------------------------------------------------------------


def _forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x).detach().clone()


def true_receptive_support(
    model: nn.Module,
    x: torch.Tensor,
    out_index: tuple[int, ...],
    deltas: tuple[float, ...] = (1000.0, -1000.0, 0.5),
) -> list[tuple[int, ...]]:
    """Input elements whose perturbation changes ``out[out_index]``."""

    base = _forward(model, x)
    shape = tuple(x.shape)
    hits: list[tuple[int, ...]] = []
    for flat in range(x.numel()):
        index = []
        remaining = flat
        for extent in reversed(shape):
            index.append(remaining % extent)
            remaining //= extent
        index_tuple = tuple(reversed(index))
        for delta in deltas:
            perturbed = x.detach().clone()
            perturbed[index_tuple] += delta
            out = _forward(model, perturbed)
            if not torch.allclose(out[out_index], base[out_index]):
                hits.append(index_tuple)
                break
    return hits


def true_projective_support(
    model: nn.Module,
    x: torch.Tensor,
    source_index: tuple[int, ...],
    deltas: tuple[float, ...] = (1000.0, -1000.0, 0.5),
) -> list[tuple[int, ...]]:
    """Output elements whose value changes when ``x[source_index]`` moves."""

    base = _forward(model, x)
    hits: set[tuple[int, ...]] = set()
    for delta in deltas:
        perturbed = x.detach().clone()
        perturbed[source_index] += delta
        out = _forward(model, perturbed)
        for row in (out != base).nonzero(as_tuple=False).tolist():
            hits.add(tuple(int(value) for value in row))
    return sorted(hits)


def hull(indices: list[tuple[int, ...]], axis: int) -> tuple[int, int] | None:
    """Half-open hull of one axis over a support set, or ``None`` when empty."""

    if not indices:
        return None
    values = [index[axis] for index in indices]
    return (min(values), max(values) + 1)


def box_bounds(box: object, spatial_rank: int) -> list[tuple[int | None, int | None]]:
    """Clipped bounds of the trailing ``spatial_rank`` axes of a box."""

    return [(axis.clipped_start, axis.clipped_stop) for axis in box.axes[-spatial_rank:]]


def capture(model: nn.Module, x: torch.Tensor) -> object:
    """Capture with the full gradient-verification triple armed."""

    return tl.trace(
        model,
        x.detach().clone().requires_grad_(True),
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )


def sole_input(trace: object) -> object:
    return next(op for op in trace.layer_list if op.is_input)


def op_named(trace: object, fragment: str) -> object:
    matches = [op for op in trace.layer_list if fragment in op.func_name]
    assert matches, f"no op matching {fragment!r}"
    return matches[-1]


def assert_box_against_truth(
    box: object,
    truth: list[tuple[int, ...]],
    spatial_axes: tuple[int, ...],
    *,
    context: str,
) -> None:
    """Exact boxes equal the true hull; upper bounds contain it; empty is empty."""

    bounds = [(axis.clipped_start, axis.clipped_stop) for axis in box.axes]
    if not truth:
        if box.exact:
            assert box.empty, f"{context}: exact box must be empty (true support empty)"
        return
    assert not box.empty, f"{context}: box empty but true support {truth}"
    for axis in spatial_axes:
        true_hull = hull(truth, axis)
        assert true_hull is not None
        start, stop = bounds[axis]
        assert start is not None and stop is not None, f"{context}: axis {axis} unbounded"
        if box.exact:
            assert (start, stop) == true_hull, (
                f"{context}: exact axis {axis} reported {(start, stop)} != true {true_hull}"
            )
        else:
            assert start <= true_hull[0] and stop >= true_hull[1], (
                f"{context}: axis {axis} bound {(start, stop)} does not contain {true_hull}"
            )


# ---------------------------------------------------------------------------
# Scalar control ancestry at spatial merges
# ---------------------------------------------------------------------------


class _TensorDerivedShapeMerge(nn.Module):
    """Two convolutions followed by a zero factory with input-derived dimensions."""

    def __init__(self) -> None:
        """Create deterministic positive convolutions for perturbation support."""

        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, bias=False)
        self.conv2 = nn.Conv2d(2, 2, 3, bias=False)
        with torch.no_grad():
            self.conv1.weight.fill_(1.0)
            self.conv2.weight.fill_(1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Merge spatial activations with zeros whose shape uses a runtime scalar."""

        spatial = self.conv2(torch.relu(self.conv1(inputs)))
        extent = (inputs.sum() * 0).long() + 8
        return spatial + torch.zeros(1, 2, extent, extent)


class _TwoSpatialWindowMerge(nn.Module):
    """Merge aligned spatial branches with nested three- and five-pixel windows."""

    def __init__(self) -> None:
        """Create positive convolutions whose outputs share a ten-pixel grid."""

        super().__init__()
        self.narrow = nn.Conv2d(1, 1, 3, bias=False)
        self.wide = nn.Conv2d(1, 1, 5, padding=1, bias=False)
        with torch.no_grad():
            self.narrow.weight.fill_(1.0)
            self.wide.weight.fill_(1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the elementwise union of two genuine spatial data branches."""

        return self.narrow(inputs) + self.wide(inputs)


class _ScalarValueSpatialMerge(nn.Module):
    """Merge a local convolution with a broadcast scalar derived from input data."""

    def __init__(self) -> None:
        """Create a deterministic local spatial branch."""

        super().__init__()
        self.spatial = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        with torch.no_grad():
            self.spatial.weight.fill_(1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a merge whose scalar-value branch genuinely depends on every pixel."""

        spatial = self.spatial(inputs)
        return spatial + inputs.sum()


def test_tensor_derived_shape_control_branch_preserves_exact_rf() -> None:
    """Keep scalar size provenance from erasing a brute-force exact spatial field."""

    model = _TensorDerivedShapeMerge().eval()
    inputs = torch.ones(1, 1, 12, 12)
    truth = true_receptive_support(model, inputs, (0, 0, 4, 4), deltas=(1.0,))
    assert len(truth) == 25
    assert hull(truth, 2) == (4, 9)
    assert hull(truth, 3) == (4, 9)

    trace = capture(model, inputs)
    target = op_named(trace, "add")
    assert target.receptive_field.center_unit(batch_index=0) == (0, 1, 4, 4)
    box = target.receptive_field.at((4, 4))
    assert box.exact
    assert_box_against_truth(box, truth, (2, 3), context="tensor-derived zero shape")
    checked = target.receptive_field.check((0, 0, 4, 4))
    assert checked.status is ReceptiveFieldValidationStatus.PASS
    assert checked.n_violations == 0

    verify_trace = capture(model, inputs)
    verified = tl.receptive_field.verify(verify_trace, units="center")
    verify_target = op_named(verify_trace, "add")
    target_results = [
        result for result in verified.containment if result.op_label == verify_target.label
    ]
    assert target_results
    # FINDING B1-19b: verify() emits one row per DIRECTION for the same
    # (op, unit), and the two directions are not interchangeable here. The
    # receptive direction is the property this test exists for and must PASS.
    # The projective direction depends on whether this op's only child, the
    # structural OUTPUT MARKER, rides the SAME payload as its parent: HONESTY
    # 13-R1 (postprocess.graph_traversal.output_payload_aliases_parent)
    # deliberately shares one payload when the returned value IS the
    # producer's retained value bit-for-bit, in which case there IS a real
    # autograd edge (they are the same tensor) and PASS is the honest answer.
    # Only when the two payloads are genuinely independent copies is there no
    # VJP to traverse, so that case stays INDETERMINATE -- fail-closed, never
    # upgraded to a PASS it cannot prove -- and its message must name the real
    # cause instead of prescribing the save_mode the trace is ALREADY using
    # (the SF-04 circular-remedy class).
    receptive_results = [
        result
        for result in target_results
        if str(getattr(result, "direction", "")).endswith("RECEPTIVE")
    ]
    projective_results = [
        result
        for result in target_results
        if str(getattr(result, "direction", "")).endswith("PROJECTIVE")
    ]
    assert receptive_results
    assert all(result.status is ReceptiveFieldValidationStatus.PASS for result in receptive_results)
    output_children = [
        label for label in verify_target.children if label in verify_trace.output_layers
    ]
    aliases_parent = bool(output_children) and output_payload_aliases_parent(
        verify_trace[output_children[0]].out, verify_target.out
    )
    for result in projective_results:
        if aliases_parent:
            assert result.status is ReceptiveFieldValidationStatus.PASS
        else:
            assert result.status is ReceptiveFieldValidationStatus.INDETERMINATE
            assert "structural output marker" in result.message
            assert "save_mode" not in result.message, (
                "an INDETERMINATE remedy must not prescribe the save mode already in force"
            )
    assert verified.verdict is not ReceptiveFieldValidationStatus.FAIL


def test_genuine_spatial_branch_union_stays_exact_against_oracle() -> None:
    """Continue merging real spatial branches instead of discarding either branch."""

    model = _TwoSpatialWindowMerge().eval()
    inputs = torch.ones(1, 1, 12, 12)
    truth = true_receptive_support(model, inputs, (0, 0, 4, 4), deltas=(1.0,))
    assert len(truth) == 25
    assert hull(truth, 2) == (3, 8)
    assert hull(truth, 3) == (3, 8)

    trace = capture(model, inputs)
    target = op_named(trace, "add")
    box = target.receptive_field.at((4, 4))
    assert box.exact
    assert_box_against_truth(box, truth, (2, 3), context="two spatial branches")


def test_scalar_value_spatial_branch_is_not_discarded_as_control() -> None:
    """Keep a scalar value edge as whole-input geometry because it carries global data."""

    model = _ScalarValueSpatialMerge().eval()
    inputs = torch.ones(1, 1, 8, 8)
    truth = true_receptive_support(model, inputs, (0, 0, 4, 4), deltas=(1.0,))
    assert len(truth) == inputs.numel()
    assert hull(truth, 2) == (0, 8)
    assert hull(truth, 3) == (0, 8)

    trace = capture(model, inputs)
    target = op_named(trace, "add")
    assert target.receptive_field.status is ReceptiveFieldStatus.WHOLE_INPUT
    assert all(axis.kind == "full" for axis in target.receptive_field.axes)


# ---------------------------------------------------------------------------
# G1 — max-pool dilation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", [2, 3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_maxpool1d_dilation_rf_pf_exact(kernel: int, stride: int, dilation: int) -> None:
    """Exact RF/PF hulls across the max-pool dilation matrix vs perturbation truth."""

    extent = 12
    model = nn.MaxPool1d(kernel_size=kernel, stride=stride, dilation=dilation)
    x = torch.linspace(0.0, 1.0, extent, dtype=torch.float64).reshape(1, 1, extent)
    trace = capture(model, x)
    pool = op_named(trace, "max_pool")
    source = sole_input(trace)
    n_out = int(pool.shape[-1])
    for out_pos in range(n_out):
        truth = true_receptive_support(model, x, (0, 0, out_pos))
        box = pool.receptive_field.at((out_pos,))
        assert box.exact, f"dilated max-pool RF must stay exact (out {out_pos})"
        assert_box_against_truth(
            box, truth, (2,), context=f"RF k{kernel}s{stride}d{dilation} out{out_pos}"
        )
    for src in range(extent):
        truth = true_projective_support(model, x, (0, 0, src))
        box = source.projective_field.at((src,))
        assert_box_against_truth(
            box, truth, (2,), context=f"PF k{kernel}s{stride}d{dilation} src{src}"
        )


def test_maxpool_dilation_check_and_verify_pass() -> None:
    """The r21 dilated-pool repro now passes containment and full verify()."""

    model = nn.MaxPool1d(kernel_size=3, stride=1, dilation=2)
    x = torch.arange(8, dtype=torch.float64).reshape(1, 1, 8) * 1.0
    trace = capture(model, x)
    pool = op_named(trace, "max_pool")
    box = pool.receptive_field.at((1,))
    axis = box.axes[-1]
    assert (axis.clipped_start, axis.clipped_stop) == (1, 6)
    verification = tl.receptive_field.verify(trace, units="center")
    assert verification.passed
    assert all(
        result.status is ReceptiveFieldValidationStatus.PASS for result in verification.containment
    )


def test_pool_ceil_mode_positional_and_kwarg_downgrade_exactness() -> None:
    """ceil_mode reaches the rule in both spellings and stays an honest envelope."""

    for model in (
        nn.MaxPool2d(3, stride=2, ceil_mode=True),  # forwarded as a keyword
        nn.AvgPool2d(3, stride=2, ceil_mode=True),  # forwarded positionally
    ):
        trace = capture(model, torch.randn(1, 1, 10, 10, dtype=torch.float64))
        pool = op_named(trace, "pool")
        last = (int(pool.shape[-2]) - 1, int(pool.shape[-1]) - 1)
        box = pool.receptive_field.at(last)
        assert not box.exact, f"{type(model).__name__} ceil_mode window must not claim exact"


def test_plain_maxpool_stays_exact() -> None:
    """Dilation/ceil handling must not disturb the default pooling geometry."""

    model = nn.MaxPool2d(3, stride=2)
    x = torch.randn(1, 1, 10, 10, dtype=torch.float64)
    trace = capture(model, x)
    pool = op_named(trace, "max_pool")
    truth = true_receptive_support(model, x, (0, 0, 1, 1))
    box = pool.receptive_field.at((1, 1))
    assert box.exact
    assert_box_against_truth(box, truth, (2, 3), context="plain maxpool RF")


# ---------------------------------------------------------------------------
# G2 — antialiased interpolation
# ---------------------------------------------------------------------------


class _Interp(nn.Module):
    """Interpolate wrapper covering keyword and positional argument spellings."""

    def __init__(self, *, positional: bool = False, **kwargs: object) -> None:
        super().__init__()
        self.positional = positional
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.positional:
            return F.interpolate(
                x,
                self.kwargs.get("size"),
                self.kwargs.get("scale_factor"),
                self.kwargs.get("mode", "nearest"),
                self.kwargs.get("align_corners"),
                self.kwargs.get("recompute_scale_factor"),
                self.kwargs.get("antialias", False),
            )
        return F.interpolate(x, **self.kwargs)


@pytest.mark.parametrize(
    ("mode", "in_size", "kwargs"),
    [
        ("bilinear", 4, {"scale_factor": (0.5, 0.5)}),
        ("bilinear", 6, {"size": (2, 2)}),
        ("bilinear", 13, {"size": (7, 7)}),  # odd/odd scale: float-boundary taps
        ("bilinear", 8, {"size": (3, 3)}),
        ("bicubic", 9, {"size": (3, 3)}),  # interior filter zeros (holes)
        ("bicubic", 4, {"scale_factor": (0.5, 0.5)}),
        ("bilinear", 4, {"size": (6, 6)}),  # antialiased upsampling
    ],
)
def test_antialias_interpolate_rf_pf_against_truth(mode: str, in_size: int, kwargs: dict) -> None:
    """Exact AA boxes equal, and inexact ones contain, the perturbation truth."""

    model = _Interp(mode=mode, align_corners=False, antialias=True, **kwargs)
    x = torch.randn(1, 1, in_size, in_size, dtype=torch.float64)
    trace = capture(model, x)
    interp = op_named(trace, "interpolate")
    source = sole_input(trace)
    out_extent = int(interp.shape[-1])
    for out_pos in ((0, 0), (out_extent - 1, out_extent - 1), (0, out_extent // 2)):
        truth = true_receptive_support(model, x, (0, 0, *out_pos), deltas=(0.5, -0.5))
        box = interp.receptive_field.at(out_pos)
        assert_box_against_truth(
            box, truth, (2, 3), context=f"AA RF {mode} in{in_size} out{out_pos}"
        )
    for src in ((0, 1), (in_size - 1, in_size - 1), (in_size // 2, 0)):
        truth = true_projective_support(model, x, (0, 0, *src), deltas=(0.5, -0.5))
        box = source.projective_field.at(src)
        assert_box_against_truth(box, truth, (2, 3), context=f"AA PF {mode} in{in_size} src{src}")


def test_antialias_positional_and_keyword_spellings_agree() -> None:
    """The r21 positional-antialias repro: both spellings give the same exact box."""

    x = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    boxes = []
    for positional in (False, True):
        model = _Interp(
            positional=positional,
            scale_factor=(0.5, 0.5),
            mode="bilinear",
            align_corners=False if positional else None,
            antialias=True,
        )
        trace = capture(model, x)
        interp = op_named(trace, "interpolate")
        box = interp.receptive_field.at((0, 0))
        boxes.append([(axis.clipped_start, axis.clipped_stop) for axis in box.axes[-2:]])
        assert box.exact
        verification = tl.receptive_field.verify(trace, units="center")
        assert verification.passed
    assert boxes[0] == boxes[1] == [(0, 3), (0, 3)]


def test_antialias_align_corners_true_fails_closed() -> None:
    """Uncertified AA configurations must refuse rather than claim geometry."""

    model = _Interp(size=(3, 3), mode="bicubic", align_corners=True, antialias=True)
    trace = capture(model, torch.randn(1, 1, 7, 7, dtype=torch.float64))
    interp = op_named(trace, "interpolate")
    with pytest.raises(Exception, match="(?i)geometry|gradient"):
        interp.receptive_field.at((0, 0))


class _GetItem(nn.Module):
    """Basic-indexing wrapper for slice-geometry probes."""

    def __init__(self, key: tuple) -> None:
        super().__init__()
        self.key = key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.key]


@pytest.mark.parametrize("start", [0, 1, 2])
@pytest.mark.parametrize("step", [1, 2, 3])
def test_strided_slice_projective_lattice(start: int, step: int) -> None:
    """PF of every source under a strided slice equals perturbation truth.

    Off-lattice sources must be exactly EMPTY; on-lattice sources must map to
    their single surviving output. Guards the forward window-edge transpose's
    integer membership proof.
    """

    if (start, step) == (0, 1):
        pytest.skip("identity slice is a passthrough with no windowed axes")
    extent = 9
    key = (slice(None), slice(None), slice(start, None, step))
    model = _GetItem(key)
    x = torch.arange(extent, dtype=torch.float64).reshape(1, 1, extent) * 1.0
    trace = capture(model, x)
    source = sole_input(trace)
    for src in range(extent):
        truth = true_projective_support(model, x, (0, 0, src))
        box = source.projective_field.at((src,))
        assert box.exact, f"slice start={start} step={step} src={src} must stay exact"
        assert_box_against_truth(
            box, truth, (2,), context=f"slice PF start={start} step={step} src={src}"
        )
    getitem = op_named(trace, "getitem")
    for out_pos in range(int(getitem.shape[-1])):
        truth = true_receptive_support(model, x, (0, 0, out_pos))
        box = getitem.receptive_field.at((out_pos,))
        assert_box_against_truth(
            box, truth, (2,), context=f"slice RF start={start} step={step} out={out_pos}"
        )


def test_strided_slice_spurious_nonempty_pin() -> None:
    """The r21 repro: source 1 of ``x[:, :, ::2]`` must report an EMPTY box."""

    model = _GetItem((slice(None), slice(None), slice(None, None, 2)))
    x = torch.arange(5, dtype=torch.float64).reshape(1, 1, 5) * 1.0
    trace = capture(model, x)
    source = sole_input(trace)
    off_lattice = source.projective_field.at((1,))
    assert off_lattice.empty, "off-lattice source must have an empty projective field"
    assert off_lattice.exact
    on_lattice = source.projective_field.at((2,))
    axis = on_lattice.axes[-1]
    assert (axis.clipped_start, axis.clipped_stop) == (1, 2)


# ---------------------------------------------------------------------------
# Metamorphic laws: adjoint duality, composition, translation equivariance
# ---------------------------------------------------------------------------


def _windowed_hull(box: object) -> tuple[tuple[int, int] | None, ...]:
    """Clipped hull over the trailing windowed axes, ``None`` for empty."""

    if box.empty:
        return (None,)
    return tuple(
        (axis.clipped_start, axis.clipped_stop) for axis in box.axes if axis.kind == "windowed"
    )


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: _GetItem((slice(None), slice(None), slice(1, None, 2))),
        lambda: nn.Conv1d(1, 1, 3, stride=2, bias=False),
        lambda: nn.AvgPool1d(2, stride=2),
        lambda: _Interp(scale_factor=(0.5,), mode="linear", align_corners=False),
    ],
)
def test_adjoint_duality_law(model_factory) -> None:
    """For dense exact windows: ``u in PF(p)`` iff ``p in RF(u)`` (no autograd)."""

    extent = 8
    model = model_factory()
    x = torch.randn(1, 1, extent)
    trace = capture(model, x)
    source = sole_input(trace)
    target = [op for op in trace.layer_list if not op.is_input and not op.is_output][-1]
    out_extent = int(target.shape[-1])
    for src in range(extent):
        pf_box = source.projective_field.at((src,))
        for out_pos in range(out_extent):
            rf_box = target.receptive_field.at((out_pos,))
            if not (pf_box.exact and rf_box.exact):
                continue
            pf_axis = pf_box.axes[-1]
            rf_axis = rf_box.axes[-1]
            in_pf = (
                not pf_box.empty
                and pf_axis.clipped_start is not None
                and pf_axis.clipped_start <= out_pos < pf_axis.clipped_stop
            )
            in_rf = (
                not rf_box.empty
                and rf_axis.clipped_start is not None
                and rf_axis.clipped_start <= src < rf_axis.clipped_stop
            )
            assert in_pf == in_rf, (
                f"{type(model).__name__}: adjoint duality broken at src={src} "
                f"out={out_pos}: u-in-PF={in_pf} p-in-RF={in_rf}"
            )


def test_layer_to_layer_composition_law() -> None:
    """box_input(u) equals the bbox of box_input over box_mid(u) members."""

    model = nn.Sequential(
        nn.Conv1d(1, 1, 3, stride=2, bias=False),
        nn.Conv1d(1, 1, 3, bias=False),
    )
    x = torch.randn(1, 1, 17)
    trace = capture(model, x)
    convs = [op for op in trace.layer_list if "conv" in op.func_name]
    mid, final = convs[0], convs[1]
    source = sole_input(trace)
    for out_pos in range(int(final.shape[-1])):
        full_box = final.receptive_field.at((out_pos,), input=source)
        mid_box = final.receptive_field.at((out_pos,), source=mid)
        mid_axis = mid_box.axes[-1]
        assert mid_axis.clipped_start is not None
        starts, stops = [], []
        for mid_pos in range(mid_axis.clipped_start, mid_axis.clipped_stop):
            inner = mid.receptive_field.at((mid_pos,), input=source)
            inner_axis = inner.axes[-1]
            starts.append(inner_axis.clipped_start)
            stops.append(inner_axis.clipped_stop)
        composed = (min(starts), max(stops))
        full_axis = full_box.axes[-1]
        assert (full_axis.clipped_start, full_axis.clipped_stop) == composed, (
            f"composition broken at out={out_pos}: "
            f"full={(full_axis.clipped_start, full_axis.clipped_stop)} composed={composed}"
        )


def test_translation_equivariance() -> None:
    """Interior units of a pure conv stack shift RFs by exactly jump*d."""

    model = nn.Sequential(
        nn.Conv1d(1, 1, 3, stride=2, bias=False),
        nn.Conv1d(1, 1, 3, dilation=2, bias=False),
    )
    x = torch.randn(1, 1, 33)
    trace = capture(model, x)
    final = [op for op in trace.layer_list if "conv" in op.func_name][-1]
    view = final.receptive_field
    jump = view.jump[-1]
    base = view.at((5,), clip=False)
    base_axis = base.axes[-1]
    for shift in (1, 2, 3):
        shifted = view.at((5 + shift,), clip=False)
        axis = shifted.axes[-1]
        assert axis.index_start - base_axis.index_start == jump * shift
        assert axis.index_stop - base_axis.index_stop == jump * shift


# ---------------------------------------------------------------------------
# Crash-class pins — rank-changing partial-full transposes (line-637 assert)
# ---------------------------------------------------------------------------


class _HyperLinear(nn.Module):
    """F.linear with a computed weight: the audited line-637 crash model."""

    def __init__(self) -> None:
        super().__init__()
        self.raw = nn.Parameter(torch.randn(4, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, torch.tanh(self.raw))


class _ComputedParent(nn.Module):
    """Partial-axes full rules (softmax/cumsum) fed by a rank-mismatched parent."""

    def __init__(self, op: str) -> None:
        super().__init__()
        self.op = op

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.op == "softmax":
            return F.softmax(x + y.mean(), dim=-1)
        return torch.cumsum(x * y.mean(), dim=-1)


def test_hypernetwork_projective_solve_degrades_typed() -> None:
    """The r21 4-op hypernetwork must pass invariants with an UNKNOWN branch."""

    from torchlens.receptive_field._validation import check_geometric_metadata_invariants

    trace = tl.trace(_HyperLinear(), torch.randn(2, 5, 3))
    assert check_geometric_metadata_invariants(trace) is True
    tanh = op_named(trace, "tanh")
    descriptors = tanh.projective_field.per_input
    assert descriptors, "weight branch must still produce a projective descriptor"
    for descriptor in descriptors.values():
        assert descriptor.status.value == "unknown"
        assert descriptor.status.value != "exact"


def test_hypernetwork_input_weight_receptive_degrades_typed() -> None:
    """Input-fed computed weights hit the same class in the receptive engine."""

    from torchlens.receptive_field._validation import check_geometric_metadata_invariants

    class HyperInput(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return F.linear(x, torch.tanh(y))

    trace = tl.trace(HyperInput(), (torch.randn(2, 5, 3), torch.randn(4, 3)))
    assert check_geometric_metadata_invariants(trace) is True
    linear = op_named(trace, "linear")
    statuses = {
        role: descriptor.status.value
        for role, descriptor in linear.receptive_field.per_input.items()
    }
    assert statuses.get("input.y") == "unknown"
    assert len(trace.projective_fields().to_pandas()) > 0


@pytest.mark.parametrize("op", ["softmax", "cumsum"])
def test_partial_full_rule_computed_parent_never_crashes(op: str) -> None:
    """softmax/cumsum with rank-mismatched parents solve without assertions."""

    from torchlens.receptive_field._validation import check_geometric_metadata_invariants

    trace = tl.trace(_ComputedParent(op), (torch.randn(2, 5, 3), torch.randn(4, 3)))
    assert check_geometric_metadata_invariants(trace) is True


# ---------------------------------------------------------------------------
# Empty-box slices()
# ---------------------------------------------------------------------------


def test_empty_box_slices_select_nothing() -> None:
    """An empty RF box's slices() must select zero elements, never the whole input."""

    model = nn.ConvTranspose2d(1, 1, 3, stride=3, padding=1, output_padding=2, bias=False)
    x = torch.randn(1, 1, 6, 6)
    trace = capture(model, x)
    conv = op_named(trace, "conv_transpose")
    last = int(conv.shape[-1]) - 1
    truth = true_receptive_support(model, x, (0, 0, last, last), deltas=(0.5, -0.5))
    assert truth == [], "the output_padding artifact unit must have no true support"
    box = conv.receptive_field.at((last, last))
    assert box.empty
    selected = x[box.slices()]
    assert selected.numel() == 0, f"empty box selected {tuple(selected.shape)}"
    # Pointwise batch/channel axes keep their same-index full-slice semantics.
    assert selected.shape[:2] == (1, 1)


def test_nonempty_box_slices_match_support_hull() -> None:
    """Non-empty boxes still slice exactly their clipped spatial hull."""

    model = nn.ConvTranspose2d(1, 1, 3, stride=3, padding=1, output_padding=2, bias=False)
    x = torch.randn(1, 1, 6, 6)
    trace = capture(model, x)
    conv = op_named(trace, "conv_transpose")
    box = conv.receptive_field.at((4, 4))
    assert not box.empty
    truth = true_receptive_support(model, x, (0, 0, 4, 4), deltas=(0.5, -0.5))
    selected = x[box.slices()]
    rows = hull(truth, 2)
    cols = hull(truth, 3)
    assert rows is not None and cols is not None
    assert selected.shape[-2:] == (rows[1] - rows[0], cols[1] - cols[0])


# ---------------------------------------------------------------------------
# Projective ambiguity vocabulary
# ---------------------------------------------------------------------------


def test_multi_target_projective_raises_target_error() -> None:
    """Projective convenience properties speak target vocabulary, not input."""

    from torchlens.receptive_field import AmbiguousInputError, AmbiguousTargetError

    class TwoOut(nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x * 2.0, x + 1.0

    trace = tl.trace(TwoOut(), torch.randn(1, 3))
    source = sole_input(trace)
    for accessor in ("status", "axes", "size", "jump", "center0", "layout"):
        with pytest.raises(AmbiguousTargetError, match="target") as exc_info:
            getattr(source.projective_field, accessor)
        assert not isinstance(exc_info.value, AmbiguousInputError)
        message = str(exc_info.value)
        assert "target=" in message and "input=" not in message

    class TwoIn(nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

    trace_in = tl.trace(TwoIn(), (torch.randn(1, 3), torch.randn(1, 3)))
    add = op_named(trace_in, "add")
    with pytest.raises(AmbiguousInputError, match="input="):
        _ = add.receptive_field.status


# ---------------------------------------------------------------------------
# verify() verdict tri-state, remediation arming, and the adjoint tripwire
# ---------------------------------------------------------------------------


class _PlainCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_verify_verdict_tri_state() -> None:
    """Unarmed is INDETERMINATE, armed honest is PASS, a violation is FAIL."""

    from torchlens.receptive_field import _rules

    plain = tl.trace(_PlainCNN(), torch.randn(1, 1, 6, 6))
    unarmed = tl.receptive_field.verify(plain, units="center")
    assert unarmed.verdict is ReceptiveFieldValidationStatus.INDETERMINATE
    assert unarmed.passed is False

    armed = capture(_PlainCNN(), torch.randn(1, 1, 6, 6))
    verified = tl.receptive_field.verify(armed, units="center")
    assert verified.verdict is ReceptiveFieldValidationStatus.PASS
    assert verified.passed is True

    original_rules = dict(_rules._RF_RULES)
    original_epoch = _rules._RF_RULES_EPOCH
    try:

        @_rules.register_rf_rule("conv2d", replace=True)
        def undersized(context):  # type: ignore[no-untyped-def]
            """Deliberately claim a 1x1 window for a real 3x3 convolution."""

            return context.window(kernel=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1))

        model = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        trace = capture(model, torch.ones(1, 1, 7, 7))
        violated = tl.receptive_field.verify(trace, units="center")
        assert violated.verdict is ReceptiveFieldValidationStatus.FAIL
        assert violated.passed is False
        assert violated.verdict is not unarmed.verdict, "FAIL must be distinct from INDETERMINATE"
    finally:
        _rules._RF_RULES.clear()
        _rules._RF_RULES.update(original_rules)
        _rules._RF_RULES_EPOCH = original_epoch


def test_verify_remediation_message_arms_tripwire() -> None:
    """Executing the indeterminate message's own recipe must arm verification."""

    model = _PlainCNN()
    x = torch.randn(1, 1, 6, 6)
    plain = tl.trace(model, x)
    unarmed = tl.receptive_field.verify(plain, units="center")
    assert unarmed.verdict is ReceptiveFieldValidationStatus.INDETERMINATE
    message = next(
        result.message
        for result in unarmed.containment
        if result.status is ReceptiveFieldValidationStatus.INDETERMINATE
    )
    assert 'save_mode="reference"' in message, message
    marker = "Recapture with "
    assert marker in message, message
    recipe = message.split(marker, 1)[1].strip()
    recipe = recipe[: recipe.rfind(")") + 1]
    rearmed_trace = eval(recipe, {"tl": tl, "model": model, "x": x, "torch": torch})
    rearmed = tl.receptive_field.verify(rearmed_trace, units="center")
    assert rearmed.verdict is not ReceptiveFieldValidationStatus.INDETERMINATE
    assert rearmed.verdict is ReceptiveFieldValidationStatus.PASS


def test_verify_fails_on_inconsistent_forward_claim() -> None:
    """A forward accelerator contradicting its backward oracle must FAIL verify().

    This exercises the exact-box corner cross-check: the bogus rule claims
    every source influences output ``s + 1`` while its authoritative backward
    relation is the identity, i.e. a spurious-nonempty exact projective claim.
    Containment alone can never see it (an empty true support is contained in
    any box); the opposite-direction membership check fails it without
    gradients.
    """

    from fractions import Fraction

    from torchlens.receptive_field import _rules
    from torchlens.receptive_field._query import _IndexSet

    original_rules = dict(_rules._RF_RULES)
    original_epoch = _rules._RF_RULES_EPOCH
    try:

        @_rules.register_rf_rule("softplus", replace=True)
        def inconsistent(context):  # type: ignore[no-untyped-def]
            """Identity backward relation with a shifted (lying) forward claim."""

            extent = int(context.out_shape[-1])

            def backward(axis: int, output_set):  # type: ignore[no-untyped-def]
                return output_set, True

            def forward(axis: int, source_set):  # type: ignore[no-untyped-def]
                shifted = [value + 1 for value in source_set.values() if value + 1 < extent]
                return _IndexSet.from_values(shifted, exact=True), True

            edges = (((Fraction(1), Fraction(0)), (Fraction(1), Fraction(0))),)
            return context.window_edges(
                edges, exact=True, map_index_set=backward, map_index_set_forward=forward
            )

        model = nn.Sequential(nn.Softplus())
        trace = capture(model, torch.randn(1, 1, 8))
        verification = tl.receptive_field.verify(trace, units="center")
        assert verification.verdict is ReceptiveFieldValidationStatus.FAIL
        assert any(
            "opposite-direction membership" in violation.reason
            for result in verification.containment
            for violation in result.violations
        )
    finally:
        _rules._RF_RULES.clear()
        _rules._RF_RULES.update(original_rules)
        _rules._RF_RULES_EPOCH = original_epoch


def test_non_antialiased_interpolate_regression() -> None:
    """The AA branch must not disturb ordinary interpolation geometry."""

    for mode, align in (("bilinear", False), ("bilinear", True), ("nearest", None)):
        kwargs = {"size": (3, 3), "mode": mode}
        if align is not None:
            kwargs["align_corners"] = align
        model = _Interp(**kwargs)
        x = torch.randn(1, 1, 7, 7, dtype=torch.float64)
        trace = capture(model, x)
        interp = op_named(trace, "interpolate")
        truth = true_receptive_support(model, x, (0, 0, 1, 1), deltas=(0.5, -0.5))
        box = interp.receptive_field.at((1, 1))
        assert_box_against_truth(box, truth, (2, 3), context=f"non-AA {mode} ac={align}")


def test_float_ambiguity_margin_scales_with_magnitude() -> None:
    """The interpolation ambiguity margin grows with the compared quantity.

    ATen evaluates tap centers and window bounds in float64, whose rounding
    error is RELATIVE (~ULP of the value), while the historical margin was a
    fixed absolute 2^-40. Beyond extent ~4k a real float64 rounding could land
    outside that margin and the ``exact=True`` containment stamp would lie.
    The margin is now max(2^-40, |value| * 2^-48).
    """

    from fractions import Fraction

    from torchlens.receptive_field.rules.interpolation import (
        _FLOAT_AMBIGUITY_MARGIN,
        _ambiguity_margin,
        _antialias_filter_verdict,
        _floor_with_margin,
    )

    # Small quantities keep the historical absolute floor exactly.
    assert _ambiguity_margin(Fraction(3, 2)) == _FLOAT_AMBIGUITY_MARGIN

    # Large quantities get a proportionally larger margin.
    big = Fraction(2**20)
    assert _ambiguity_margin(big) == big * Fraction(1, 2**48)

    # A window bound 2^-30 away from an integer at magnitude 2^20 is within
    # float64 ambiguity there; the fixed absolute margin called it exact.
    near_boundary = big + Fraction(1, 2**30)
    value, exact = _floor_with_margin(near_boundary, prefer_low=True)
    assert exact is False
    assert value == 2**20 - 1  # containment: the widened (lower) bound.

    # The same offset at small magnitude is genuinely unambiguous.
    small_value, small_exact = _floor_with_margin(Fraction(5) + Fraction(1, 2**30), prefer_low=True)
    assert small_exact is True
    assert small_value == 5

    # Filter verdicts near a zero use the center-scaled margin: with a
    # large-extent margin the tap is kept and exactness downgraded
    # (containment), never silently dropped.
    t = Fraction(1) + Fraction(1, 2**30)
    assert _antialias_filter_verdict("bilinear", t, _ambiguity_margin(big)) == "ambiguous"
    assert _antialias_filter_verdict("bilinear", t, _ambiguity_margin(Fraction(1))) == "zero"


# ---------------------------------------------------------------------------
# b6 next-round attack-list oracle rows (fixwave-2, opus attack list):
# grid_sample, unfold/fold, pixel_shuffle, dilated pooling, batch-axis units.
# Contract for every row: an exact box equals the brute-force hull, an upper
# bound contains it, and geometry the engine cannot derive REFUSES typed --
# never a silently wrong exact claim.
# ---------------------------------------------------------------------------


class _GridSampleConv(nn.Module):
    """Identity affine grid_sample feeding a convolution."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        theta = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        grid = F.affine_grid(theta, (1, 1, 8, 8), align_corners=False)
        return self.conv(F.grid_sample(inputs, grid, align_corners=False))


class _UnfoldFold(nn.Module):
    """unfold -> fold round trip (overlap-add) over a 3x3 window."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        unfolded = F.unfold(inputs, kernel_size=3, padding=1)
        return F.fold(unfolded, output_size=(8, 8), kernel_size=3, padding=1)


class _PixelShuffleConv(nn.Module):
    """Convolution feeding a 2x pixel_shuffle channel-to-space rearrangement."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.conv(inputs), 2)


class _BatchMeanMix(nn.Module):
    """Convolution merged with a batch-mean branch coupling every sample."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs) - inputs.mean(dim=0, keepdim=True)


@pytest.mark.parametrize(
    ("model_type", "fragment", "unit"),
    [
        (_GridSampleConv, "conv2d", (2, 2)),
        (_UnfoldFold, "fold", (2, 2)),
        (_PixelShuffleConv, "pixel_shuffle", (4, 4)),
    ],
)
def test_geometry_refuses_typed_for_underivable_rearrangements(
    model_type: type[nn.Module], fragment: str, unit: tuple[int, ...]
) -> None:
    """grid_sample / unfold+fold / pixel_shuffle refuse per-unit geometry typed."""

    from torchlens.receptive_field import ReceptiveFieldError

    trace = capture(model_type().eval(), torch.randn(1, 1, 8, 8))
    target = op_named(trace, fragment)
    with pytest.raises(ReceptiveFieldError, match="use .gradient"):
        target.receptive_field.at(unit)


def test_grid_sample_gradient_fallback_reports_support() -> None:
    """The advertised gradient() remedy actually works where geometry refuses."""

    trace = capture(_GridSampleConv().eval(), torch.randn(1, 1, 8, 8))
    target = op_named(trace, "conv2d")
    results = target.receptive_field.gradient((0, 0, 2, 2))
    gradient = next(iter(results.values())) if isinstance(results, Mapping) else results
    assert gradient.support_mask is not None
    assert bool(gradient.support_mask.any())


def test_dilated_max_pool_exact_box_matches_bruteforce() -> None:
    """Dilated max pooling: exact claim pinned against perturbation truth."""

    model = nn.MaxPool2d(3, stride=2, dilation=2).eval()
    inputs = torch.randn(1, 1, 12, 12)
    truth = true_receptive_support(model, inputs, (0, 0, 1, 1))
    assert hull(truth, 2) == (2, 7)
    assert hull(truth, 3) == (2, 7)

    trace = capture(model, inputs)
    target = op_named(trace, "max_pool2d")
    box = target.receptive_field.at((1, 1))
    assert box.exact
    assert_box_against_truth(box, truth, (2, 3), context="dilated max pool")
    checked = target.receptive_field.check((0, 0, 1, 1))
    assert checked.status is ReceptiveFieldValidationStatus.PASS
    assert checked.n_violations == 0


def test_dilated_box_axes_disclose_sparse_possible() -> None:
    """The box view must carry the axis view's ``sparse_possible`` disclosure.

    A dilated kernel keeps the hull exact while provably skipping interior
    positions; the axis view disclosed that, but the box axes did not, so
    ``rf.at(unit)`` presented a dense window under an unqualified
    ``exact=True`` (b6 R20).
    """

    model = nn.Conv2d(1, 1, 3, dilation=3, bias=False).eval()
    inputs = torch.randn(1, 1, 14, 14)
    trace = capture(model, inputs)
    box = op_named(trace, "conv2d").receptive_field.at((2, 2))
    assert box.exact
    windowed = [axis for axis in box.axes if axis.kind == "windowed"]
    assert windowed, "expected windowed spatial axes on a conv box"
    assert all(axis.sparse_possible for axis in windowed)
    assert box.sparse_possible

    dense = nn.Conv2d(1, 1, 3, bias=False).eval()
    dense_trace = capture(dense, inputs)
    dense_box = op_named(dense_trace, "conv2d").receptive_field.at((2, 2))
    assert dense_box.exact
    assert not dense_box.sparse_possible
    assert all(not axis.sparse_possible for axis in dense_box.axes)


def test_batch_mean_mix_claims_full_batch_axis() -> None:
    """Batch-axis units: batch mixing must surface as a full batch axis."""

    model = _BatchMeanMix().eval()
    inputs = torch.randn(3, 1, 8, 8)

    # Independent truth: another sample's pixel influences this sample's output.
    base = _forward(model, inputs)
    perturbed = inputs.detach().clone()
    perturbed[2, 0, 2, 2] += 1000.0
    assert not torch.allclose(_forward(model, perturbed)[1, 0, 2, 2], base[1, 0, 2, 2])

    trace = capture(model, inputs)
    target = op_named(trace, "sub")
    box = target.receptive_field.at((2, 2))
    assert box.exact
    kinds = {axis.input_axis: axis.kind for axis in box.axes}
    assert kinds[0] == "full", "batch mixing must not claim a pointwise batch axis"
    batch_axis = next(axis for axis in box.axes if axis.input_axis == 0)
    assert (batch_axis.clipped_start, batch_axis.clipped_stop) == (0, 3)
    truth = true_receptive_support(model, inputs, (1, 0, 2, 2), deltas=(1000.0,))
    assert hull(truth, 0) == (0, 3)
    for axis in (2, 3):
        spatial = next(item for item in box.axes if item.input_axis == axis)
        assert (spatial.clipped_start, spatial.clipped_stop) == hull(truth, axis)
    checked = target.receptive_field.check((1, 0, 2, 2))
    assert checked.status is ReceptiveFieldValidationStatus.PASS
    assert checked.n_violations == 0


def test_check_exposes_retain_graph() -> None:
    """``rf.check`` exposes ``retain_graph`` like its sibling ``gradient``.

    b3 R14-N1 (4th round): ``check()`` hardcoded ``retain_graph=False`` and
    disclosed nothing, so one check on an armed capture silently freed the
    graph and a later ``gradient(..., retain_graph=True)`` surfaced torch's
    raw second-backward RuntimeError mid-workflow.
    """

    model = nn.Conv2d(1, 1, 3).eval()
    trace = capture(model, torch.randn(1, 1, 8, 8))
    target = op_named(trace, "conv2d")
    unit = target.receptive_field.center_unit(batch_index=0)

    checked = target.receptive_field.check(unit, retain_graph=True)
    assert checked.status is ReceptiveFieldValidationStatus.PASS
    # The graph survived the check, so a later gradient works.
    assert target.receptive_field.gradient(unit, retain_graph=True)

    # The default still frees the graph (unchanged behavior, now disclosed).
    fresh = capture(model, torch.randn(1, 1, 8, 8))
    fresh_target = op_named(fresh, "conv2d")
    fresh_unit = fresh_target.receptive_field.center_unit(batch_index=0)
    fresh_target.receptive_field.check(fresh_unit)
    with pytest.raises(RuntimeError, match="backward through the graph"):
        fresh_target.receptive_field.gradient(fresh_unit, retain_graph=True)
