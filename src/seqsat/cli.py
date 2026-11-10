"""Command-line interface for ``seqsat``.

Only the argument dataclasses and tyro are imported at startup. Each subcommand
imports :mod:`seqsat.pipeline` (numpy, pandas, scipy, h5py, pyarrow, matplotlib)
when it runs, so ``seqsat --help`` stays fast.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import tyro

from seqsat import __version__

DEFAULT_TARGETS = (0.5, 0.7, 0.9)


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


def run_gex(args: GexArgs) -> None:
    """Run the GEX analysis for a `cellranger count` outs directory."""
    from seqsat import pipeline

    pipeline.gex(args)


def run_atac(args: AtacArgs) -> None:
    """Run the ATAC analysis for a `cellranger-atac count` outs directory."""
    from seqsat import pipeline

    pipeline.atac(args)


def run_arc(args: ArcArgs) -> None:
    """Run the GEX and ATAC analyses for a `cellranger-arc count` outs directory."""
    from seqsat import pipeline

    pipeline.arc(args)


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
