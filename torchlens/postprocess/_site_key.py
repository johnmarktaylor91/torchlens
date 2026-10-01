"""Re-export shim: the site-key minter moved to :mod:`torchlens.data_classes._site_key`.

Ratified relocation (architecture memo 3.3 / Rule V3, C01 item 6):
``SiteKeyMinter`` reads type-shaped but MINTS site keys, so it fails V3's
no-smuggling test and moves to L1 with the coordinates it mints. Historical
spellings stay importable here; lower-layer consumers import the L1 home.
"""

from ..data_classes._site_key import (
    ROOT_CALL_INSTANCE,
    SITE_KEY_PREFIX,
    SiteKeyMinter,
    call_instance_id,
    escape_site_component,
    operation_witness,
    parse_site_key,
    render_site_key,
    site_axis,
    unescape_site_component,
)

__tl_layer__ = "FACADE"

__all__ = [
    "ROOT_CALL_INSTANCE",
    "SITE_KEY_PREFIX",
    "SiteKeyMinter",
    "call_instance_id",
    "escape_site_component",
    "operation_witness",
    "parse_site_key",
    "render_site_key",
    "site_axis",
    "unescape_site_component",
]
