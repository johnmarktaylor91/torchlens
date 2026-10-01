"""tvscope row gate: the CLIP round-trip invariant (config-built, zero network).

The audience's mandated CLIP case on the R0 config-built CLIP fixture: the
site inventory answers ``visual_projection`` with a working spelling, EVERY
emitted module-origin selector extracts keyed by the requested string, the
manual projection over the ``post_layernorm`` site equals the forward's
``image_embeds`` exactly (version-adaptively: transformers 5.x returns them
L2-normalized), and the captured projection op output is bit-equal to the
manual projection. Includes the escrow-spill regression (op-label selector
at batch 1) that used to crash this exact workflow.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

import torchlens as tl  # noqa: E402
import torchlens.inventory as inv  # noqa: E402
from tests.real_model.r0.families import FAMILY_BY_NAME  # noqa: E402

pytestmark = [pytest.mark.smoke, pytest.mark.real_model]

_PROJECTED_SITE = "vision_model.post_layernorm"


@pytest.fixture(scope="module")
def clip_setup():
    """One config-built CLIP + seeded inputs shared by this module."""

    model = FAMILY_BY_NAME["clip"].build("eager")
    model.eval()
    torch.manual_seed(0)
    input_ids = torch.randint(0, 100, (2, 8))
    pixel_values = torch.randn(2, 3, 32, 32)
    inventory = inv.list_sites(
        model, input_kwargs={"input_ids": input_ids, "pixel_values": pixel_values.clone()}
    )
    return model, input_ids, pixel_values, inventory


def test_inventory_answers_visual_projection(clip_setup) -> None:
    """The first question resolves to a working, shape-bearing row.

    On the full CLIPModel the projection's output op doubles as a model
    output boundary, so the bare address is engine-ambiguous; the inventory
    must answer with a selector that actually WORKS (the invariant), never
    the pretty-but-rejected spelling.
    """

    model, input_ids, pixel_values, inventory = clip_setup
    row = inventory.resolve("visual_projection")
    assert row.module_address == "visual_projection"
    assert row.origin == "module_output"
    assert row.shape is not None and len(row.shape) == 2
    out = tl.extract(model, [input_ids.clone(), pixel_values.clone()], [row.selector])
    assert row.selector in out
    leaf = inventory.resolve("post_layernorm")
    assert leaf.selector == _PROJECTED_SITE


def test_every_module_origin_selector_round_trips(clip_setup) -> None:
    """THE ROW GATE: emitted selectors extract keyed by the requested string."""

    model, input_ids, pixel_values, inventory = clip_setup
    selectors = sorted({r.selector for r in inventory if r.origin == "module_output"})
    assert len(selectors) > 20
    out = tl.extract(model, [input_ids.clone(), pixel_values.clone()], selectors)
    missing = [s for s in selectors if s not in out]
    assert missing == [], missing


def test_manual_projection_matches_forward_image_embeds(clip_setup) -> None:
    """post_layernorm -> visual_projection == image_embeds, exactly.

    Version-adaptive: transformers 5.x L2-normalizes ``image_embeds`` on the
    forward output (and ``get_image_features`` no longer returns the raw
    tensor), so the comparison normalizes iff the reference is normalized.
    """

    model, input_ids, pixel_values, _inventory = clip_setup
    out = tl.extract(model, [input_ids.clone(), pixel_values.clone()], [_PROJECTED_SITE])
    with torch.no_grad():
        manual = model.visual_projection(out[_PROJECTED_SITE])
        reference = model(
            input_ids=input_ids.clone(), pixel_values=pixel_values.clone()
        ).image_embeds
    norms = reference.norm(p=2, dim=-1)
    if torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
        manual = manual / manual.norm(p=2, dim=-1, keepdim=True)
    assert (manual - reference).abs().max().item() == 0.0


def test_captured_projection_op_is_bit_equal_to_manual(clip_setup) -> None:
    """The captured visual_projection output == the user-applied projection."""

    model, input_ids, pixel_values, inventory = clip_setup
    projection_selector = inventory.resolve("visual_projection").selector
    out = tl.extract(
        model,
        [input_ids.clone(), pixel_values.clone()],
        [_PROJECTED_SITE, projection_selector],
    )
    with torch.no_grad():
        manual = model.visual_projection(out[_PROJECTED_SITE])
    assert torch.equal(out[projection_selector], manual)


def test_op_label_selector_at_batch_one(clip_setup) -> None:
    """Escrow-spill regression: op-label selective capture at batch size 1."""

    model, input_ids, pixel_values, inventory = clip_setup
    projection_row = inventory.resolve("visual_projection")
    label = str(projection_row.layer_label)
    out = tl.extract(model, [input_ids[:1].clone(), pixel_values[:1].clone()], [label])
    assert label in out
    assert out[label].shape[0] == 1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "extract/reads seam residue (tvscope memo B13/DIS-4, reported-by-one-lab): "
        "tl.extract(model, x, 'all') silently omits the declared output alias "
        "'output_1' that a save-all trace retains; owned by the extract/reads "
        "seam -- this tripwire flips loudly when the seam fix lands"
    ),
)
def test_extract_all_retains_declared_output_alias(clip_setup) -> None:
    """extract-'all' should keep the output alias a save-all trace retains."""

    model, input_ids, pixel_values, _inventory = clip_setup
    out = tl.extract(model, [input_ids.clone(), pixel_values.clone()], "all")
    assert any(key.startswith("output") for key in out)
