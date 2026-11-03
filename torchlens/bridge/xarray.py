"""xarray / NeuroidAssembly adapters over TorchLens sites (tvscope B8).

Per-site "presentation x neuroid" arrays for the bring-your-own-alignment
workflow: both the in-memory route (a saved ``Trace`` site) and the file
route (a ``LoadedExtraction`` / artifact directory) run the ONE recorded
shaping operation (:mod:`torchlens.features`), so the two are numerically
identical for the same capture. Coordinates carry recorded stimulus ids
when the artifact has them, and provenance (site, shaping op,
input-preprocessing verdict) rides ``attrs``.

``data_array`` needs only ``xarray``; ``neuroid_assembly`` additionally
needs Brain-Score's brainio assembly classes (bundled in ``brainscore_core``,
which ``brainscore_vision`` installs, or the older standalone ``brainio``
package) and degrades to a typed ImportError naming the extra when absent.

Every spelling is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__tl_layer__ = "L8"


def _site_matrix(source: Any, site: str) -> Any:
    """Shape one site through the shared operation (B7)."""

    from .. import features as _features

    return _features.site_matrix(source, site)


def _assembly_pieces(source: Any, site: str) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Build the array, coordinate, and attr pieces both adapters share.

    Parameters
    ----------
    source:
        Trace, ``LoadedExtraction``, or artifact directory path.
    site:
        Site selector or extraction output key.

    Returns
    -------
    tuple[numpy.ndarray, dict[str, Any], dict[str, Any]]
        The 2-D array, the coords mapping, and the attrs mapping.
    """

    site_matrix = _site_matrix(source, site)
    array = site_matrix.matrix.numpy()
    coords: dict[str, Any] = {
        "presentation": np.arange(array.shape[0]),
        "neuroid": np.arange(array.shape[1]),
    }
    if site_matrix.row_ids is not None:
        coords["stimulus_id"] = ("presentation", np.asarray(site_matrix.row_ids))
    attrs: dict[str, Any] = {
        "source": "torchlens",
        "site": site_matrix.site,
        "shaping": site_matrix.shaping.to_json(),
    }
    if site_matrix.input_preprocessing is not None:
        attrs["input_preprocessing_verdict"] = site_matrix.input_preprocessing.get(
            "verdict", "unknown"
        )
    return array, coords, attrs


def data_array(source: Any, site: str) -> Any:
    """Convert one TorchLens site into an ``xarray.DataArray`` (B8).

    Parameters
    ----------
    source:
        A TorchLens ``Trace`` with saved activations, a
        ``LoadedExtraction``, or an extraction-artifact directory path.
    site:
        Site selector (qualified module address, pass-qualified label) or
        extraction output key.

    Returns
    -------
    Any
        ``xarray.DataArray`` with ``presentation`` and ``neuroid`` dims,
        stimulus-id coordinates when recorded, and provenance attrs.

    Raises
    ------
    ImportError
        If xarray is unavailable.

    Examples
    --------
    >>> # requires: xarray
    >>> trace = tl.trace(model, x)
    >>> da = tl.bridge.xarray.data_array(trace, "2")
    >>> da.dims
    ('presentation', 'neuroid')
    >>> da.shape
    (1, 128)
    """

    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "xarray adapter requires xarray: pip install xarray (or the torchlens[neuro] extra)."
        ) from exc

    array, coords, attrs = _assembly_pieces(source, site)
    return xr.DataArray(
        array,
        dims=("presentation", "neuroid"),
        coords=coords,
        attrs=attrs,
        name=attrs["site"],
    )


def _neuroid_assembly_class() -> Any:
    """Return Brain-Score's ``NeuroidAssembly`` class from wherever it is installed.

    Current ``brainscore_vision`` releases ship the brainio assembly classes inside
    ``brainscore_core.supported_data_standards.brainio``; the standalone ``brainio``
    package is their older home.

    Returns
    -------
    Any
        The ``NeuroidAssembly`` class.

    Raises
    ------
    ImportError
        If neither home is importable.
    """

    try:
        from brainscore_core.supported_data_standards.brainio.assemblies import (
            NeuroidAssembly,
        )
    except ImportError:
        try:
            from brainio.assemblies import NeuroidAssembly
        except ImportError as exc:
            raise ImportError(
                "neuroid_assembly requires Brain-Score's brainio assembly classes "
                "(installed with brainscore_vision); use data_array() for a "
                "plain xarray.DataArray."
            ) from exc
    return NeuroidAssembly


def neuroid_assembly(source: Any, site: str) -> Any:
    """Convert one TorchLens site into a Brain-Score ``NeuroidAssembly`` (B8).

    Parameters
    ----------
    source:
        A TorchLens ``Trace`` with saved activations, a
        ``LoadedExtraction``, or an extraction-artifact directory path.
    site:
        Site selector or extraction output key.

    Returns
    -------
    Any
        Brain-Score's ``NeuroidAssembly`` over the same "presentation x
        neuroid" array as :func:`data_array`.

    Raises
    ------
    ImportError
        If neither ``brainscore_core`` nor the standalone ``brainio``
        package provides Brain-Score's assembly classes.

    Examples
    --------
    >>> # requires: brainscore_core
    >>> trace = tl.trace(model, x)
    >>> assembly = tl.bridge.xarray.neuroid_assembly(trace, "2")
    >>> assembly.dims
    ('presentation', 'neuroid')
    """

    neuroid_assembly_class = _neuroid_assembly_class()
    array, coords, attrs = _assembly_pieces(source, site)
    assembly = neuroid_assembly_class(
        array,
        dims=("presentation", "neuroid"),
        coords=coords,
    )
    assembly.attrs.update(attrs)
    return assembly


__all__ = ["data_array", "neuroid_assembly"]
