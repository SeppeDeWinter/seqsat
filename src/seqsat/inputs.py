"""Locate Cell Ranger output files, parse summary CSVs and read cell barcodes."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def parse_number(value: str) -> float:
    """Parse a Cell Ranger metric value such as ``"552,342,334"``, ``"63.3%"`` or ``"0.6330"``.

    Thousands separators are removed. A trailing ``%`` divides the value by 100.

    Raises
    ------
    ValueError
        If the value is not numeric (``"nan"`` is returned as ``nan``).
    """
    s = value.strip().replace(",", "")
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    return float(s)


def read_summary_csv(path: Path) -> dict[str, str]:
    """Read a one-row Cell Ranger summary CSV (header row + value row) into a dict."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    rows = [r for r in rows if r]
    if len(rows) != 2:
        raise ValueError(f"{path}: expected a header row and one value row, got {len(rows)} rows")
    header, values = rows
    if len(header) != len(values):
        raise ValueError(f"{path}: header has {len(header)} fields, values have {len(values)}")
    return dict(zip(header, values, strict=True))


def summary_metric(summary: dict[str, str], name: str) -> float:
    """Look up ``name`` in a parsed summary and convert it with :func:`parse_number`."""
    try:
        raw = summary[name]
    except KeyError:
        raise KeyError(f"metric {name!r} not in summary; available: {sorted(summary)}") from None
    return parse_number(raw)


def _find(outs: Path, names: list[str], required: bool = True) -> Path | None:
    """Return the first ``outs/<name>`` that exists."""
    for name in names:
        p = outs / name
        if p.exists():
            return p
    if required:
        raise FileNotFoundError(f"none of {names} found in {outs}")
    return None


@dataclass(frozen=True)
class ArcOutputs:
    """Files of a ``cellranger-arc count`` outs directory."""

    outs: Path
    summary: Path
    per_barcode_metrics: Path
    gex_molecule_info: Path
    atac_fragments: Path
    web_summary: Path | None

    @classmethod
    def from_dir(cls, outs: Path) -> "ArcOutputs":
        outs = Path(outs)
        summary = _find(outs, ["summary.csv"])
        per_barcode = _find(outs, ["per_barcode_metrics.csv"])
        mol = _find(outs, ["gex_molecule_info.h5"])
        frags = _find(outs, ["atac_fragments.tsv.gz"])
        assert summary and per_barcode and mol and frags
        return cls(
            outs=outs,
            summary=summary,
            per_barcode_metrics=per_barcode,
            gex_molecule_info=mol,
            atac_fragments=frags,
            web_summary=_find(outs, ["web_summary.html"], required=False),
        )


@dataclass(frozen=True)
class GexOutputs:
    """Files of a ``cellranger count`` outs directory."""

    outs: Path
    metrics_summary: Path
    molecule_info: Path
    web_summary: Path | None

    @classmethod
    def from_dir(cls, outs: Path) -> "GexOutputs":
        outs = Path(outs)
        metrics = _find(outs, ["metrics_summary.csv"])
        mol = _find(outs, ["molecule_info.h5"])
        assert metrics and mol
        return cls(
            outs=outs,
            metrics_summary=metrics,
            molecule_info=mol,
            web_summary=_find(outs, ["web_summary.html"], required=False),
        )


def arc_cell_barcodes(per_barcode_metrics: Path) -> NDArray[np.str_]:
    """Cell barcodes (``is_cell == 1``) from an ARC ``per_barcode_metrics.csv``.

    Returns the joint ``barcode`` column, e.g. ``"AAACAGCCAAACAACA-1"``. This is the
    barcode used in ``atac_fragments.tsv.gz``; for GEX it is the molecule_info barcode
    plus a ``-<gem_group>`` suffix.
    """
    df = pd.read_csv(per_barcode_metrics, usecols=["barcode", "is_cell"])
    return df.loc[df["is_cell"] == 1, "barcode"].to_numpy(dtype=np.str_)


def split_gem_group(barcodes: NDArray[np.str_]) -> tuple[NDArray[np.bytes_], NDArray[np.int64]]:
    """Split ``"SEQ-G"`` barcodes into sequences (bytes) and gem groups.

    Barcodes without a ``-G`` suffix get gem group 1.
    """
    seqs = np.empty(len(barcodes), dtype=object)
    groups = np.empty(len(barcodes), dtype=np.int64)
    for i, bc in enumerate(barcodes):
        seq, sep, grp = str(bc).rpartition("-")
        if sep and grp.isdigit():
            seqs[i], groups[i] = seq.encode(), int(grp)
        else:
            seqs[i], groups[i] = str(bc).encode(), 1
    return seqs.astype(np.bytes_), groups


@dataclass(frozen=True)
class AtacOutputs:
    """Files of a ``cellranger-atac count`` outs directory."""

    outs: Path
    summary: Path
    singlecell: Path
    fragments: Path
    web_summary: Path | None

    @classmethod
    def from_dir(cls, outs: Path) -> "AtacOutputs":
        outs = Path(outs)
        summary = _find(outs, ["summary.csv"])
        singlecell = _find(outs, ["singlecell.csv"])
        frags = _find(outs, ["fragments.tsv.gz"])
        assert summary and singlecell and frags
        return cls(
            outs=outs,
            summary=summary,
            singlecell=singlecell,
            fragments=frags,
            web_summary=_find(outs, ["web_summary.html"], required=False),
        )


def atac_cell_barcodes(singlecell: Path) -> NDArray[np.str_]:
    """Cell barcodes (``is__cell_barcode == 1``) from a cellranger-atac ``singlecell.csv``."""
    df = pd.read_csv(singlecell, usecols=["barcode", "is__cell_barcode"])
    return df.loc[df["is__cell_barcode"] == 1, "barcode"].to_numpy(dtype=np.str_)


def read_barcode_file(path: Path) -> NDArray[np.str_]:
    """Barcodes from a text file (optionally gzipped), one per line; blank lines skipped.

    Only the first comma- or tab-separated field of each line is used, so a
    ``barcodes.tsv.gz`` or a one-column CSV works too.
    """
    df = pd.read_csv(path, header=None, sep=r"[,\t]", engine="python", usecols=[0], dtype=str)
    barcodes = df[0].dropna().str.strip()
    barcodes = barcodes[barcodes != ""]
    if barcodes.empty:
        raise ValueError(f"{path}: no barcodes")
    return barcodes.to_numpy(dtype=np.str_)
