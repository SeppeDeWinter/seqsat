"""Extract the subsampling curves embedded in a Cell Ranger ``web_summary.html``.

The page holds one inline script with ``const data = {...};``. Plots in it are
Plotly figures (``{"data": [traces], "layout": {...}}``), stored under fixed keys
whose nesting depends on the pipeline and version:

- cellranger-arc 2.x: top-level ``gex_umis_per_cell_plot``, ``gex_genes_per_cell_plot``,
  ``atac_seq_saturation_plot``.
- cellranger count 6.x: ``summary.analysis_tab.{seq_saturation_plot,median_gene_plot}.plot``.
- cellranger-atac 2.x: top-level ``assay_efficiency_plot``.

Older reports (e.g. cellranger 3.x) store LZString-compressed data in
``var compressed_data``; these are not supported.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_MARKER = "const data = "
COMPRESSED_MARKER = "var compressed_data = "

#: web-summary plot key -> (modality, metric)
CURVE_PLOTS: dict[str, tuple[str, str]] = {
    "gex_umis_per_cell_plot": ("gex", "umis_per_cell"),
    "gex_genes_per_cell_plot": ("gex", "genes_per_cell"),
    "atac_seq_saturation_plot": ("atac", "fragments_per_cell"),
    "median_gene_plot": ("gex", "genes_per_cell"),
    "seq_saturation_plot": ("gex", "sequencing_saturation"),
    "assay_efficiency_plot": ("atac", "fragments_per_cell"),
}

CURVE_COLUMNS: tuple[str, ...] = ("modality", "metric", "trace", "mean_reads_per_cell", "value")


def load_data(path: Path) -> dict[str, Any]:
    """Parse the ``const data = {...}`` object of a web summary."""
    html = Path(path).read_text(encoding="utf-8")
    if DATA_MARKER not in html:
        if COMPRESSED_MARKER in html:
            raise ValueError(
                f"{path}: web summary stores LZString-compressed data (older Cell Ranger); "
                "not supported"
            )
        raise ValueError(f"{path}: no '{DATA_MARKER.strip()}' object found")
    start = html.index(DATA_MARKER) + len(DATA_MARKER)
    data, _ = json.JSONDecoder().raw_decode(html, start)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: embedded data is not a JSON object")
    return data


def _as_figure(value: Any) -> dict[str, Any] | None:
    """The Plotly figure stored under a plot key, either directly or under ``"plot"``."""
    if isinstance(value, dict):
        if isinstance(value.get("data"), list):
            return value
        inner = value.get("plot")
        if isinstance(inner, dict) and isinstance(inner.get("data"), list):
            return inner
    return None


def _find_plots(obj: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(key, figure)`` for every known curve key, at any nesting depth."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            figure = _as_figure(value) if key in CURVE_PLOTS else None
            if figure is not None:
                yield key, figure
            else:
                yield from _find_plots(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _find_plots(value)


def curves_from_data(data: dict[str, Any]) -> pd.DataFrame:
    """Tidy table of the subsampling curves found in parsed web-summary data.

    Returns
    -------
    pandas.DataFrame
        Columns ``modality, metric, trace, mean_reads_per_cell, value``; ``trace`` is
        the Plotly trace name (the genome for GEX plots). Each metric is taken from
        the first plot found for it.
    """
    frames = []
    seen: set[tuple[str, str]] = set()
    for key, figure in _find_plots(data):
        modality, metric = CURVE_PLOTS[key]
        if (modality, metric) in seen:
            continue
        seen.add((modality, metric))
        for trace in figure["data"]:
            x = np.asarray(trace.get("x", []), dtype=np.float64)
            y = np.asarray(trace.get("y", []), dtype=np.float64)
            if x.shape != y.shape:
                raise ValueError(f"{key}: x and y have different lengths")
            frames.append(
                pd.DataFrame(
                    {
                        "modality": modality,
                        "metric": metric,
                        "trace": str(trace.get("name", "")),
                        "mean_reads_per_cell": x,
                        "value": y,
                    },
                    columns=list(CURVE_COLUMNS),
                )
            )
    if not frames:
        return pd.DataFrame(columns=list(CURVE_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def extract_curves(path: Path) -> pd.DataFrame:
    """Subsampling curves of a ``web_summary.html``; see :func:`curves_from_data`."""
    return curves_from_data(load_data(path))
