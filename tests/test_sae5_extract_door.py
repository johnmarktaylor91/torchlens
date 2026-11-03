"""``tl.extract`` with facet selectors (ships with Lead's extract-door patch).

``tl.extract(model, x, [tl.head(0, "q")])`` returns the head slice keyed by the
selector's spelling; a mapping keys it by the user label; a mixed request reads
plain layers from the same trace. Values equal ``trace[selector]`` and the
closed forms. Tolerances, fixed before the first run: ``atol=1e-6, rtol=0``.
"""

from __future__ import annotations

import torch

import torchlens as tl
from tests.test_sae4_latent_facets import TOL, X, _HandSAE, _latent_column, _Model
from tests.test_sae5_read_doors import _AttnModel, _q_head


def test_extract_reads_a_head_slice_keyed_by_its_spelling() -> None:
    torch.manual_seed(0)
    model, x = _AttnModel().eval(), torch.randn(2, 3, 8)
    selector = tl.head(0, "q")
    out = tl.extract(model, x, [selector])
    assert list(out) == [repr(selector)]
    torch.testing.assert_close(out[repr(selector)], _q_head(model, x, 0), **TOL)


def test_extract_reads_a_latent_under_a_user_label_beside_a_plain_layer() -> None:
    model, sae = _Model().eval(), _HandSAE()
    tl.bridge.sae.register("res", sae, site="fc1")
    try:
        x = torch.tensor(X)
        out = tl.extract(model, x, {"z1": tl.facet("res").latent(1), "site": "fc1"})
        torch.testing.assert_close(out["z1"], _latent_column(1), **TOL)
        torch.testing.assert_close(out["site"], x, **TOL)
    finally:
        tl.bridge.sae.unregister("res")
