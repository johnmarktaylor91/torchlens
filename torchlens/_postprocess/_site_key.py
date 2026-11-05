"""Re-export shim: the site-key minter moved to :mod:`torchlens._data_classes._site_key`.

``SiteKeyMinter`` reads type-shaped but MINTS site keys, so it fails the
layering no-smuggling test and lives in L1 with the coordinates it mints. Historical
spellings stay importable here; lower-layer consumers import the L1 home.
"""

from .._data_classes._site_key import (
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
