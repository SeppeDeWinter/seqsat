"""Run the analyses behind the ``seqsat`` subcommands: read inputs, build curves, fit, report."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from seqsat.atac import AtacHistograms, read_fragments
from seqsat.cli import ArcArgs, AtacArgs, GexArgs
from seqsat.curves import SATURATION_METRICS, modality_curves, rate_grid
from seqsat.fit import MetricResult, analyze_curve
from seqsat.gex import GexHistograms, raw_read_pairs, read_molecule_info
from seqsat.inputs import (
    ArcOutputs,
    AtacOutputs,
    GexOutputs,
    arc_cell_barcodes,
    atac_cell_barcodes,
    read_barcode_file,
    read_summary_csv,
    summary_metric,
)
from seqsat.report import format_summary, plot_saturation, write_curves, write_fits

#: Relative difference between read counts from two sources that triggers a warning.
READ_COUNT_TOLERANCE = 1e-3


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _total_reads(override: int | None, summary: dict[str, str], metric: str) -> int:
    """``override`` if given, else ``metric`` from the summary CSV; must be > 0."""
    total = override if override is not None else round(summary_metric(summary, metric))
    if total <= 0:
        raise SystemExit(f"error: total reads must be > 0, got {total}")
    return total


def _check_read_count(name: str, used: float, other: float | None, source: str) -> None:
    if other is not None and abs(used - other) > READ_COUNT_TOLERANCE * used:
        _log(f"warning: {name} = {used:,.0f}, but {source} reports {other:,.0f}")


def _gex_analysis(
    h: GexHistograms, total_reads: float, targets: tuple[float, ...], n_points: int
) -> tuple[pd.DataFrame, list[MetricResult]]:
    rates = rate_grid(total_reads, h.n_cells, n_points)
    curves = modality_curves(
        "gex",
        {"umis_per_cell": h.umi, "genes_per_cell": h.gene},
        "sequencing_saturation",
        h.all_barcode_umi,
        total_reads,
        h.n_cells,
        rates,
    )
    return curves, _analyze(curves, h.n_cells, targets)


def _atac_analysis(
    h: AtacHistograms, total_reads: float, targets: tuple[float, ...], n_points: int
) -> tuple[pd.DataFrame, list[MetricResult]]:
    rates = rate_grid(total_reads, h.n_cells, n_points)
    curves = modality_curves(
        "atac",
        {"fragments_per_cell": h.cell},
        "duplication",
        h.all_barcode,
        total_reads,
        h.n_cells,
        rates,
    )
    return curves, _analyze(curves, h.n_cells, targets)


def _analyze(curves: pd.DataFrame, n_cells: int, targets: tuple[float, ...]) -> list[MetricResult]:
    results = []
    pairs = curves[["modality", "metric"]].drop_duplicates()
    for modality, metric in zip(pairs["modality"], pairs["metric"], strict=True):
        c = curves[(curves["modality"] == modality) & (curves["metric"] == metric)]
        kind = "saturation" if metric in SATURATION_METRICS else "count"
        results.append(
            analyze_curve(
                str(modality), str(metric), kind, c.mean_reads_per_cell, c.value, n_cells, targets
            )
        )
    return results


def _cells(override: Path | None) -> NDArray[np.str_] | None:
    return read_barcode_file(override) if override is not None else None


def _finish(
    out: Path, curves: pd.DataFrame, results: list[MetricResult], inputs: dict[str, Any]
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_curves(curves, out)
    write_fits(results, inputs, out)
    plot_saturation(curves, results, out)
    print(format_summary(results))
    _log(f"wrote {out}/curves.csv, fits.json, saturation.png")


def gex(args: GexArgs) -> None:
    """GEX analysis of a `cellranger count` outs directory."""
    o = GexOutputs.from_dir(args.outs)
    summary = read_summary_csv(o.metrics_summary)
    total = _total_reads(args.total_reads, summary, "Number of Reads")
    if args.total_reads is None:
        _check_read_count(
            "Number of Reads", total, raw_read_pairs(o.molecule_info), "molecule_info"
        )
    _log(f"reading {o.molecule_info}")
    h = read_molecule_info(o.molecule_info, _cells(args.cells))
    curves, results = _gex_analysis(h, total, args.targets, args.n_points)
    inputs = {
        "pipeline": "cellranger count",
        "outs": str(o.outs),
        "gex": {"n_cells": h.n_cells, "total_raw_reads": total},
    }
    _finish(args.out, curves, results, inputs)


def atac(args: AtacArgs) -> None:
    """ATAC analysis of a `cellranger-atac count` outs directory."""
    o = AtacOutputs.from_dir(args.outs)
    summary = read_summary_csv(o.summary)
    total = _total_reads(args.total_reads, summary, "Sequenced read pairs")
    cells = _cells(args.cells)
    if cells is None:
        cells = atac_cell_barcodes(o.singlecell)
    _log(f"reading {o.fragments}")
    h = read_fragments(o.fragments, cells)
    curves, results = _atac_analysis(h, total, args.targets, args.n_points)
    inputs = {
        "pipeline": "cellranger-atac count",
        "outs": str(o.outs),
        "atac": {"n_cells": h.n_cells, "total_raw_reads": total},
    }
    _finish(args.out, curves, results, inputs)


def arc(args: ArcArgs) -> None:
    """GEX and ATAC analyses of a `cellranger-arc count` outs directory."""
    o = ArcOutputs.from_dir(args.outs)
    summary = read_summary_csv(o.summary)
    gex_total = _total_reads(args.gex_total_reads, summary, "GEX Sequenced read pairs")
    atac_total = _total_reads(args.atac_total_reads, summary, "ATAC Sequenced read pairs")
    if args.gex_total_reads is None:
        _check_read_count(
            "GEX Sequenced read pairs", gex_total, raw_read_pairs(o.gex_molecule_info),
            "gex_molecule_info.h5",
        )  # fmt: skip
    cells = _cells(args.cells)
    if cells is None:
        cells = arc_cell_barcodes(o.per_barcode_metrics)

    _log(f"reading {o.gex_molecule_info}")
    gex_h = read_molecule_info(o.gex_molecule_info, cells)
    gex_curves, gex_results = _gex_analysis(gex_h, gex_total, args.targets, args.n_points)
    _log(f"reading {o.atac_fragments}")
    atac_h = read_fragments(o.atac_fragments, cells)
    atac_curves, atac_results = _atac_analysis(atac_h, atac_total, args.targets, args.n_points)

    inputs = {
        "pipeline": "cellranger-arc count",
        "outs": str(o.outs),
        "gex": {"n_cells": gex_h.n_cells, "total_raw_reads": gex_total},
        "atac": {"n_cells": atac_h.n_cells, "total_raw_reads": atac_total},
    }
    curves = pd.concat([gex_curves, atac_curves], ignore_index=True)
    _finish(args.out, curves, gex_results + atac_results, inputs)
