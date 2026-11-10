"""Command-line interface for ``seqsat``."""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tyro
from numpy.typing import NDArray

from seqsat import __version__
from seqsat.atac import AtacHistograms, read_fragments
from seqsat.curves import modality_curves, rate_grid
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

DEFAULT_TARGETS = (0.5, 0.7, 0.9)
#: Relative difference between read counts from two sources that triggers a warning.
READ_COUNT_TOLERANCE = 1e-3


@dataclass(frozen=True)
class GexArgs:
    """Sequencing saturation of a `cellranger count` run (Gene Expression)."""

    outs: tyro.conf.Positional[Path]
    """Cell Ranger `outs/` directory."""
    out: Path = Path("seqsat_out")
    """Output directory for curves.csv, fits.json and saturation.png."""
    targets: tuple[float, ...] = DEFAULT_TARGETS
    """Targets: fractions of the fitted maximum (UMIs, genes) and saturation values."""
    total_reads: int | None = None
    """Sequenced reads; default: "Number of Reads" in metrics_summary.csv."""
    cells: Path | None = None
    """File of cell barcodes ("SEQ-1"), one per line; default: cells in molecule_info.h5."""
    n_points: int = 100
    """Number of evenly spaced depths in each curve."""


@dataclass(frozen=True)
class AtacArgs:
    """Library complexity of a `cellranger-atac count` run."""

    outs: tyro.conf.Positional[Path]
    """Cell Ranger ATAC `outs/` directory."""
    out: Path = Path("seqsat_out")
    """Output directory for curves.csv, fits.json and saturation.png."""
    targets: tuple[float, ...] = DEFAULT_TARGETS
    """Targets: fractions of the fitted maximum (fragments) and duplication values."""
    total_reads: int | None = None
    """Sequenced read pairs; default: "Sequenced read pairs" in summary.csv."""
    cells: Path | None = None
    """File of cell barcodes, one per line; default: is__cell_barcode in singlecell.csv."""
    n_points: int = 100
    """Number of evenly spaced depths in each curve."""


@dataclass(frozen=True)
class ArcArgs:
    """Sequencing saturation of both modalities of a `cellranger-arc count` run."""

    outs: tyro.conf.Positional[Path]
    """Cell Ranger ARC `outs/` directory."""
    out: Path = Path("seqsat_out")
    """Output directory for curves.csv, fits.json and saturation.png."""
    targets: tuple[float, ...] = DEFAULT_TARGETS
    """Targets: fractions of the fitted maximum (counts) and saturation/duplication values."""
    gex_total_reads: int | None = None
    """GEX sequenced read pairs; default: "GEX Sequenced read pairs" in summary.csv."""
    atac_total_reads: int | None = None
    """ATAC sequenced read pairs; default: "ATAC Sequenced read pairs" in summary.csv."""
    cells: Path | None = None
    """File of cell barcodes, one per line; default: is_cell in per_barcode_metrics.csv."""
    n_points: int = 100
    """Number of evenly spaced depths in each curve."""


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


SATURATION_METRICS = {"sequencing_saturation", "duplication"}


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


def _cells(override: Path | None, default: NDArray[np.str_] | None) -> NDArray[np.str_] | None:
    return read_barcode_file(override) if override is not None else default


def _finish(
    out: Path, curves: pd.DataFrame, results: list[MetricResult], inputs: dict[str, Any]
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_curves(curves, out)
    write_fits(results, inputs, out)
    plot_saturation(curves, results, out)
    print(format_summary(results))
    _log(f"wrote {out}/curves.csv, fits.json, saturation.png")


def run_gex(args: GexArgs) -> None:
    """Run the GEX analysis for a `cellranger count` outs directory."""
    o = GexOutputs.from_dir(args.outs)
    summary = read_summary_csv(o.metrics_summary)
    total = _total_reads(args.total_reads, summary, "Number of Reads")
    if args.total_reads is None:
        _check_read_count(
            "Number of Reads", total, raw_read_pairs(o.molecule_info), "molecule_info"
        )
    _log(f"reading {o.molecule_info}")
    h = read_molecule_info(o.molecule_info, _cells(args.cells, None))
    curves, results = _gex_analysis(h, total, args.targets, args.n_points)
    inputs = {
        "pipeline": "cellranger count",
        "outs": str(o.outs),
        "gex": {"n_cells": h.n_cells, "total_raw_reads": total},
    }
    _finish(args.out, curves, results, inputs)


def run_atac(args: AtacArgs) -> None:
    """Run the ATAC analysis for a `cellranger-atac count` outs directory."""
    o = AtacOutputs.from_dir(args.outs)
    summary = read_summary_csv(o.summary)
    total = _total_reads(args.total_reads, summary, "Sequenced read pairs")
    cells = _cells(args.cells, None)
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


def run_arc(args: ArcArgs) -> None:
    """Run the GEX and ATAC analyses for a `cellranger-arc count` outs directory."""
    o = ArcOutputs.from_dir(args.outs)
    summary = read_summary_csv(o.summary)
    gex_total = _total_reads(args.gex_total_reads, summary, "GEX Sequenced read pairs")
    atac_total = _total_reads(args.atac_total_reads, summary, "ATAC Sequenced read pairs")
    if args.gex_total_reads is None:
        _check_read_count(
            "GEX Sequenced read pairs", gex_total, raw_read_pairs(o.gex_molecule_info),
            "gex_molecule_info.h5",
        )  # fmt: skip
    cells = _cells(args.cells, None)
    if cells is None:
        cells = arc_cell_barcodes(o.per_barcode_metrics)

    _log(f"reading {o.gex_molecule_info}")
    gex = read_molecule_info(o.gex_molecule_info, cells)
    gex_curves, gex_results = _gex_analysis(gex, gex_total, args.targets, args.n_points)
    _log(f"reading {o.atac_fragments}")
    atac = read_fragments(o.atac_fragments, cells)
    atac_curves, atac_results = _atac_analysis(atac, atac_total, args.targets, args.n_points)

    inputs = {
        "pipeline": "cellranger-arc count",
        "outs": str(o.outs),
        "gex": {"n_cells": gex.n_cells, "total_raw_reads": gex_total},
        "atac": {"n_cells": atac.n_cells, "total_raw_reads": atac_total},
    }
    curves = pd.concat([gex_curves, atac_curves], ignore_index=True)
    _finish(args.out, curves, gex_results + atac_results, inputs)


SUBCOMMANDS: dict[str, Callable[..., None]] = {"gex": run_gex, "atac": run_atac, "arc": run_arc}


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``seqsat`` command."""
    tyro.extras.subcommand_cli_from_dict(
        SUBCOMMANDS,
        prog="seqsat",
        description=f"seqsat {__version__}: sequencing saturation from Cell Ranger outputs.",
        args=argv,
        config=(tyro.conf.OmitArgPrefixes,),
    )
