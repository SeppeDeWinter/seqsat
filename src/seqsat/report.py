"""Write curve tables, fit results and figures."""

import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from seqsat import __version__
from seqsat.curves import SATURATION_METRICS
from seqsat.fit import MetricResult, TargetDepth

CURVES_FILE = "curves.csv"
FITS_FILE = "fits.json"
FIGURE_FILE = "saturation.png"

METRIC_LABELS = {
    "umis_per_cell": "Median UMIs per cell",
    "genes_per_cell": "Median genes per cell",
    "sequencing_saturation": "Sequencing saturation",
    "fragments_per_cell": "Median unique fragments per cell",
    "duplication": "Duplication",
}
MODALITY_LABELS = {"gex": "GEX", "atac": "ATAC"}

# Reference palette (light mode): one data series per panel, neutral ink for the rest.
SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
SERIES_2 = "#eb6834"

#: Maximum number of Cell Ranger markers per panel in :func:`plot_curve_comparison`.
COMPARISON_MAX_MARKERS = 20
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8985"
GRID = "#e6e5e1"


def write_curves(curves: pd.DataFrame, out: Path) -> Path:
    """Write the tidy curve table to ``out/curves.csv``."""
    path = Path(out) / CURVES_FILE
    curves.to_csv(path, index=False)
    return path


def _finite(v: float) -> float | None:
    return v if math.isfinite(v) else None


def result_to_dict(r: MetricResult) -> dict[str, Any]:
    """JSON-serialisable view of a :class:`MetricResult`."""
    return {
        "modality": r.modality,
        "metric": r.metric,
        "kind": r.kind,
        "model": "michaelis_menten",
        "params": {
            "vmax": {"value": r.fit.vmax, "se": _finite(r.fit.vmax_se)},
            "km": {"value": r.fit.km, "se": _finite(r.fit.km_se)},
        },
        "r2": _finite(r.fit.r2),
        "n_points": r.fit.n_points,
        "n_cells": r.n_cells,
        "current_reads_per_cell": r.x_now,
        "current_value": r.current_value,
        "current_saturation": r.current_saturation,
        "marginal_gain_per_1000_reads_per_cell": r.marginal_gain_per_1000,
        "targets": [asdict(t) for t in r.targets],
    }


def write_fits(results: Sequence[MetricResult], inputs: dict[str, Any], out: Path) -> Path:
    """Write ``out/fits.json`` with the inputs used and one entry per metric."""
    path = Path(out) / FITS_FILE
    doc = {
        "seqsat_version": __version__,
        "inputs": inputs,
        "metrics": [result_to_dict(r) for r in results],
    }
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def _fmt_count(v: float) -> str:
    if v >= 1e9:
        return f"{v / 1e9:.2f}G"
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.1f}k"
    return f"{v:.0f}"


def _target_label(t: TargetDepth) -> str:
    """E.g. ``"70% · 73.3k reads/cell · +19.5k/cell, +199.4M total"``."""
    label = f"{t.target:.0%} · {_fmt_count(t.reads_per_cell)} reads/cell"
    if t.extra_reads_per_cell <= 0:
        return f"{label} · reached"
    extra = f"+{_fmt_count(t.extra_reads_per_cell)}/cell, +{_fmt_count(t.extra_total_reads)} total"
    return f"{label} · {extra}"


def _metric_title(r: MetricResult) -> str:
    modality = MODALITY_LABELS.get(r.modality, r.modality)
    return f"{modality} – {METRIC_LABELS.get(r.metric, r.metric)}"


def plot_saturation(curves: pd.DataFrame, results: Sequence[MetricResult], out: Path) -> Path:
    """One panel per metric: curve points, fitted curve, targets and the current depth."""
    path = Path(out) / FIGURE_FILE
    n = len(results)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig = Figure(figsize=(4.8 * ncols, 3.8 * nrows), facecolor=SURFACE, layout="tight")
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for ax, r in zip(axes.ravel(), results, strict=False):
        c = curves[(curves.modality == r.modality) & (curves.metric == r.metric)]
        x_max = max([1.15 * r.x_now, *(1.1 * t.reads_per_cell for t in r.targets)])
        x_fit = np.linspace(0, x_max, 300)

        _style_axes(ax)
        ax.plot(x_fit, r.fit.predict(x_fit), color=TEXT_SECONDARY, linewidth=1.5, zorder=2)
        ax.plot(
            c.mean_reads_per_cell, c.value, "o", color=SERIES, markersize=3.5,
            markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=3,
        )  # fmt: skip

        ax.axvline(r.x_now, color=TEXT_PRIMARY, linewidth=1.0, linestyle="-", zorder=1)
        ax.annotate(
            f"current {_fmt_count(r.x_now)}", (r.x_now, 0), xytext=(4, 4),
            textcoords="offset points", xycoords=("data", "axes fraction"),
            color=TEXT_PRIMARY, fontsize=7.5,
        )  # fmt: skip

        for t in r.targets:
            y_t = t.target * r.fit.vmax if r.kind == "count" else t.target
            ax.axhline(y_t, color=TEXT_MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
            ax.annotate(
                _target_label(t), (0, y_t),
                xytext=(3, 2),
                textcoords="offset points", xycoords=("axes fraction", "data"),
                color=TEXT_SECONDARY, fontsize=7,
                bbox={"boxstyle": "square,pad=0.15", "facecolor": SURFACE, "edgecolor": "none",
                      "alpha": 0.85},
                zorder=4,
            )  # fmt: skip

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, None if r.kind == "count" else 1.0)
        _label_axes(ax, _metric_title(r), r.metric)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    return path


def _style_axes(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def _label_axes(ax: Any, title: str, metric: str) -> None:
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=10, loc="left")
    ax.set_xlabel("Mean read pairs per cell", color=TEXT_SECONDARY, fontsize=8.5)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), color=TEXT_SECONDARY, fontsize=8.5)
    ax.xaxis.set_major_formatter(lambda v, _: _fmt_count(v))
    if metric in SATURATION_METRICS:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    else:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")


def plot_curve_comparison(
    panels: Sequence[tuple[str, str, pd.DataFrame, pd.DataFrame]], path: Path
) -> Path:
    """Compare seqsat curves with Cell Ranger's web-summary curves, one panel per curve.

    Cell Ranger's points are shown as markers, thinned to at most
    ``COMPARISON_MAX_MARKERS`` evenly spaced points per panel so the seqsat line stays
    visible; the last point (current depth) is always shown.

    Parameters
    ----------
    panels
        ``(title, metric, ours, cellranger)``; ``ours`` and ``cellranger`` have columns
        ``mean_reads_per_cell`` and ``value``.
    path
        Output image file.
    """
    n = len(panels)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig = Figure(figsize=(4.8 * ncols, 3.6 * nrows + 0.4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for ax, (title, metric, ours, cr) in zip(axes.ravel(), panels, strict=False):
        if len(cr) > COMPARISON_MAX_MARKERS:
            keep = np.unique(
                np.linspace(0, len(cr) - 1, COMPARISON_MAX_MARKERS).round().astype(int)
            )
            cr = cr.iloc[keep]
        _style_axes(ax)
        ax.plot(
            ours.mean_reads_per_cell, ours.value, color=SERIES, linewidth=2,
            label="seqsat (exact expectation)", zorder=2,
        )  # fmt: skip
        ax.plot(
            cr.mean_reads_per_cell, cr.value, "o", color=SERIES_2, markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=0.8, label="Cell Ranger web summary",
            zorder=3,
        )  # fmt: skip
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        _label_axes(ax, title, metric)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncols=2, frameon=False,
        labelcolor=TEXT_PRIMARY, fontsize=9,
    )  # fmt: skip
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    return Path(path)


def format_summary(results: Sequence[MetricResult]) -> str:
    """Plain-text summary: one row per metric and target."""
    rows = []
    for r in results:
        for t in r.targets:
            rows.append(
                {
                    "metric": _metric_title(r),
                    "now": (
                        f"{r.current_value:.0f}" if r.kind == "count" else f"{r.current_value:.1%}"
                    ),
                    "saturation": f"{r.current_saturation:.1%}",
                    "target": f"{t.target:.0%}",
                    "reads/cell": _fmt_count(t.reads_per_cell),
                    "extra reads": _fmt_count(t.extra_total_reads),
                }
            )
    df = pd.DataFrame(rows)
    return df.to_string(index=False)
