"""Extract the subsampling curves embedded in a Cell Ranger ``web_summary.html``.

The page holds one inline script with ``const data = {...};``. Plots in it are
Plotly figures (``{"data": [traces], "layout": {...}}``) stored under fixed keys.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_MARKER = "const data = "

#: web-summary key -> (modality, metric)
CURVE_PLOTS: dict[str, tuple[str, str]] = {
    "gex_umis_per_cell_plot": ("gex", "umis_per_cell"),
    "gex_genes_per_cell_plot": ("gex", "genes_per_cell"),
    "atac_seq_saturation_plot": ("atac", "fragments_per_cell"),
}

CURVE_COLUMNS: tuple[str, ...] = ("modality", "metric", "trace", "mean_reads_per_cell", "value")


def load_data(path: Path) -> dict[str, Any]:
    """Parse the ``const data = {...}`` object of a web summary."""
    html = Path(path).read_text(encoding="utf-8")
    try:
        start = html.index(DATA_MARKER) + len(DATA_MARKER)
    except ValueError:
        raise ValueError(f"{path}: no '{DATA_MARKER.strip()}' object found") from None
    data, _ = json.JSONDecoder().raw_decode(html, start)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: embedded data is not a JSON object")
    return data


def curves_from_data(data: dict[str, Any]) -> pd.DataFrame:
    """Tidy table of the subsampling curves found in parsed web-summary data.

    Returns
    -------
    pandas.DataFrame
        Columns ``modality, metric, trace, mean_reads_per_cell, value``; ``trace`` is
        the Plotly trace name (the genome for GEX plots). Plots that are absent are
        skipped.
    """
    frames = []
    for key, (modality, metric) in CURVE_PLOTS.items():
        plot = data.get(key)
        if not isinstance(plot, dict) or "data" not in plot:
            continue
        for trace in plot["data"]:
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
