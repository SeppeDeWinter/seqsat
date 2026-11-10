"""Read ``fragments.tsv.gz`` into read-support histograms for the ATAC curves.

One item per fragment; ``k`` is its read support (column 5). The file is streamed
in blocks with pyarrow, reading only the contig, barcode and read-support columns.
Contig and barcode columns are dictionary-encoded, so each block maps only its
distinct barcodes to cell rows.
"""

import gzip
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
from numpy.typing import NDArray
from scipy import sparse

from seqsat.curves import add_hists, build_hist

FRAGMENT_COLUMNS = ("chrom", "start", "end", "barcode", "readSupport", "strand")
DEFAULT_BLOCK_SIZE = 64 << 20  # bytes of uncompressed text per block


@dataclass(frozen=True)
class FragmentsHeader:
    """Leading ``#`` lines of a fragments file."""

    n_lines: int
    n_bytes: int
    n_columns: int
    primary_contigs: tuple[str, ...]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AtacHistograms:
    """Read-support histograms from a fragments file.

    Attributes
    ----------
    cell
        ``(n_cells, kmax + 1)`` histogram: fragments per cell with read support ``k``.
    all_barcode
        1-D histogram over all barcodes: ``all_barcode[k]`` = fragments with support ``k``.
    cell_barcodes
        Cell barcodes in row order of ``cell``.
    contig_fragments
        Fragments per contig over all barcodes (after exclusion, excluded contigs are 0).
    """

    cell: sparse.csr_array
    all_barcode: NDArray[np.int64]
    cell_barcodes: NDArray[np.str_]
    contig_fragments: dict[str, int]

    @property
    def n_cells(self) -> int:
        return int(self.cell.shape[0])


def read_header(path: Path) -> FragmentsHeader:
    """Parse the ``#`` header lines and count the columns of the first data line."""
    n_lines = 0
    n_bytes = 0
    contigs: list[str] = []
    metadata: dict[str, str] = {}
    n_columns = 0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                n_columns = len(line.rstrip("\n").split("\t"))
                break
            n_lines += 1
            n_bytes += len(line.encode())
            key, sep, value = line[1:].strip().partition("=")
            if not sep:
                continue
            if key == "primary_contig":
                contigs.append(value)
            else:
                metadata[key] = value
    if n_columns < 5:
        raise ValueError(f"{path}: expected at least 5 columns, got {n_columns}")
    return FragmentsHeader(n_lines, n_bytes, n_columns, tuple(contigs), metadata)


def _accumulate(total: NDArray[np.int64], add: NDArray[np.int64]) -> NDArray[np.int64]:
    """Add two 1-D histograms of possibly different length."""
    if add.size > total.size:
        add = add.copy()
        add[: total.size] += total
        return add
    total[: add.size] += add
    return total


def read_fragments(
    path: Path,
    cell_barcodes: NDArray[np.str_],
    exclude_contigs: Collection[str] = (),
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> AtacHistograms:
    """Build ATAC histograms from a fragments file.

    Parameters
    ----------
    path
        ``fragments.tsv.gz`` (``atac_fragments.tsv.gz`` for ARC).
    cell_barcodes
        Cell barcodes as written in the fragments file (e.g. ``"AAACAGCCAAACAACA-1"``).
    exclude_contigs
        Contigs whose fragments are dropped, for cells and for all barcodes.
    block_size
        Bytes of uncompressed text parsed per block. Raised if needed so the header
        fits in the first block (pyarrow skips header rows within that block).

    Returns
    -------
    AtacHistograms
        Rows follow ``cell_barcodes``.
    """
    header = read_header(path)
    names = list(FRAGMENT_COLUMNS[: header.n_columns])
    names += [f"extra{i}" for i in range(header.n_columns - len(names))]
    cell_index = pd.Index(np.asarray(cell_barcodes, dtype=np.str_))
    if not cell_index.is_unique:
        raise ValueError("duplicate cell barcodes")
    n_cells = len(cell_index)
    excluded = set(exclude_contigs)
    block_size = max(block_size, header.n_bytes + (1 << 16))

    reader = pv.open_csv(
        pa.input_stream(str(path), compression="gzip"),
        read_options=pv.ReadOptions(
            skip_rows=header.n_lines, column_names=names, block_size=block_size
        ),
        parse_options=pv.ParseOptions(delimiter="\t"),
        convert_options=pv.ConvertOptions(
            include_columns=["chrom", "barcode", "readSupport"],
            column_types={
                "chrom": pa.dictionary(pa.int32(), pa.string()),
                "barcode": pa.dictionary(pa.int32(), pa.string()),
                "readSupport": pa.int64(),
            },
        ),
    )

    cell_hist = build_hist([], [], n_cells)
    all_hist = np.zeros(1, dtype=np.int64)
    contig_fragments: dict[str, int] = {}

    for batch in reader:
        chrom = batch.column("chrom")
        barcode = batch.column("barcode")
        k = batch.column("readSupport").to_numpy()
        if k.size and k.min() < 1:
            raise ValueError(f"{path}: readSupport must be >= 1")

        chrom_names = chrom.dictionary.to_pylist()
        chrom_idx = chrom.indices.to_numpy()
        per_chrom = np.bincount(chrom_idx, minlength=len(chrom_names))
        keep_chrom = np.array([c not in excluded for c in chrom_names], dtype=bool)
        for name, n, keep in zip(chrom_names, per_chrom, keep_chrom, strict=True):
            contig_fragments[name] = contig_fragments.get(name, 0) + (int(n) if keep else 0)

        rows = cell_index.get_indexer(barcode.dictionary.to_pandas())[barcode.indices.to_numpy()]
        if excluded:
            keep = keep_chrom[chrom_idx]
            rows, k = rows[keep], k[keep]

        all_hist = _accumulate(all_hist, np.bincount(k))
        in_cell = rows >= 0
        if np.any(in_cell):
            cell_hist = add_hists(cell_hist, build_hist(rows[in_cell], k[in_cell], n_cells))

    return AtacHistograms(
        cell=cell_hist,
        all_barcode=all_hist,
        cell_barcodes=np.asarray(cell_barcodes, dtype=np.str_),
        contig_fragments=contig_fragments,
    )
