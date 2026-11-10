"""Read ``molecule_info.h5`` into read-count histograms for the GEX curves.

Items and their read counts ``k``:

- UMIs per cell: one item per molecule (barcode, UMI, gene, library); ``k = count``.
- Genes per cell: one item per (cell, gene); ``k`` = sum of ``count`` over the
  molecules of that gene in that cell.
- Sequencing saturation: one item per molecule over all barcodes; only the 1-D
  histogram of ``count`` is kept.

Only ``Gene Expression`` libraries and features are used. Molecules of all
``umi_type`` values (exonic and intronic) are kept.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from seqsat.curves import add_hists, build_hist
from seqsat.inputs import split_gem_group

GENE_EXPRESSION = "Gene Expression"
DEFAULT_CHUNK_SIZE = 1_000_000


@dataclass(frozen=True)
class GexHistograms:
    """Read-count histograms from a molecule_info file.

    Attributes
    ----------
    umi
        ``(n_cells, kmax + 1)`` histogram: molecules per cell with ``k`` reads.
    gene
        ``(n_cells, kmax + 1)`` histogram: genes per cell with ``k`` reads.
    all_barcode_umi
        1-D histogram over all barcodes: ``all_barcode_umi[k]`` = molecules with ``k`` reads.
    cell_barcodes
        Cell barcodes as ``"SEQ-G"`` strings, in row order of ``umi`` and ``gene``.
    """

    umi: sparse.csr_array
    gene: sparse.csr_array
    all_barcode_umi: NDArray[np.int64]
    cell_barcodes: NDArray[np.str_]

    @property
    def n_cells(self) -> int:
        return int(self.umi.shape[0])

    @property
    def total_molecule_reads(self) -> int:
        """Reads summed over all GEX molecules of all barcodes."""
        k = np.arange(self.all_barcode_umi.size, dtype=np.int64)
        return int(k @ self.all_barcode_umi)


def _read_json(ds: h5py.Dataset) -> Any:
    """Decode a JSON string stored as a scalar or length-1 dataset."""
    raw = ds[()]
    if isinstance(raw, np.ndarray):
        raw = raw.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode()
    return json.loads(raw)


def library_info(f: h5py.File) -> list[dict[str, Any]]:
    """Parsed ``library_info`` JSON (one dict per library)."""
    return _read_json(f["library_info"])


def gex_library_ids(f: h5py.File) -> NDArray[np.int64]:
    """Library indices with ``library_type == "Gene Expression"``."""
    ids = [lib["library_id"] for lib in library_info(f) if lib["library_type"] == GENE_EXPRESSION]
    if not ids:
        raise ValueError("molecule_info has no Gene Expression library")
    return np.asarray(sorted(ids), dtype=np.int64)


def gex_feature_mask(f: h5py.File) -> NDArray[np.bool_]:
    """Boolean mask over features: ``feature_type == "Gene Expression"``."""
    ft = f["features/feature_type"][:]
    return ft == GENE_EXPRESSION.encode()


def raw_read_pairs(path: Path) -> int | None:
    """Sum of ``raw_read_pairs`` over GEX libraries in ``metrics_json``, if present.

    ``metrics_json`` is undocumented; use this as a cross-check only.
    """
    with h5py.File(path, "r") as f:
        if "metrics_json" not in f:
            return None
        libs = _read_json(f["metrics_json"]).get("libraries", {})
        total = 0
        for lib_id in gex_library_ids(f):
            entry = libs.get(str(lib_id))
            if entry is None or "raw_read_pairs" not in entry:
                return None
            total += int(entry["raw_read_pairs"])
        return total


def _cell_keys_from_pass_filter(f: h5py.File, n_groups: int) -> NDArray[np.int64]:
    """Cell keys ``barcode_idx * n_groups + gem_group`` from ``barcode_info/pass_filter``."""
    pf = f["barcode_info/pass_filter"][:].astype(np.int64)
    libs = {lib["library_id"]: lib for lib in library_info(f)}
    gex_ids = set(gex_library_ids(f).tolist())
    keep = np.array([int(lib) in gex_ids for lib in pf[:, 1]], dtype=bool)
    pf = pf[keep]
    groups = np.array([libs[int(lib)]["gem_group"] for lib in pf[:, 1]], dtype=np.int64)
    return np.unique(pf[:, 0] * n_groups + groups)


def _cell_keys_from_barcodes(
    f: h5py.File, cell_barcodes: NDArray[np.str_], n_groups: int
) -> NDArray[np.int64]:
    """Cell keys for ``"SEQ-G"`` barcodes, looked up in the h5 ``barcodes`` list."""
    seqs, groups = split_gem_group(cell_barcodes)
    h5_barcodes = f["barcodes"][:]
    order = np.argsort(h5_barcodes, kind="stable")
    sorted_bc = h5_barcodes[order]
    pos = np.searchsorted(sorted_bc, seqs)
    pos_c = np.minimum(pos, sorted_bc.size - 1)
    found = (pos < sorted_bc.size) & (sorted_bc[pos_c] == seqs)
    if not np.all(found):
        missing = np.asarray(cell_barcodes)[~found][:5]
        raise ValueError(f"{(~found).sum()} cell barcodes not in molecule_info, e.g. {missing}")
    if groups.max() >= n_groups:
        raise ValueError(f"gem group {groups.max()} not in molecule_info")
    keys = order[pos_c].astype(np.int64) * n_groups + groups
    if np.unique(keys).size != keys.size:
        raise ValueError("duplicate cell barcodes")
    return keys


def _chunks(n: int, chunk_size: int) -> Iterator[slice]:
    for start in range(0, n, chunk_size):
        yield slice(start, min(start + chunk_size, n))


def read_molecule_info(
    path: Path,
    cell_barcodes: NDArray[np.str_] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> GexHistograms:
    """Build GEX histograms from a molecule_info file, reading it in chunks.

    Parameters
    ----------
    path
        ``molecule_info.h5`` (``gex_molecule_info.h5`` for ARC).
    cell_barcodes
        Cell barcodes as ``"SEQ-G"`` strings (e.g. from ``per_barcode_metrics.csv``).
        Default: the cells in ``barcode_info/pass_filter`` of GEX libraries.
    chunk_size
        Number of molecules read per chunk.

    Returns
    -------
    GexHistograms
        Rows follow ``cell_barcodes`` if given, else sorted (barcode, gem group) order.
    """
    with h5py.File(path, "r") as f:
        n_mol = f["count"].shape[0]
        gex_libs = gex_library_ids(f)
        feat_ok = gex_feature_mask(f)
        n_features = feat_ok.size
        n_groups = int(max(lib["gem_group"] for lib in library_info(f))) + 1

        if cell_barcodes is None:
            keys = _cell_keys_from_pass_filter(f, n_groups)
            h5_barcodes = f["barcodes"][:]
            cell_barcodes = np.asarray(
                [f"{h5_barcodes[k // n_groups].decode()}-{k % n_groups}" for k in keys],
                dtype=np.str_,
            )
        else:
            cell_barcodes = np.asarray(cell_barcodes, dtype=np.str_)
            keys = _cell_keys_from_barcodes(f, cell_barcodes, n_groups)
        n_cells = keys.size
        key_order = np.argsort(keys, kind="stable")
        sorted_keys = keys[key_order]
        streaming = is_sorted_by_barcode(f, chunk_size)

        umi_parts: list[sparse.csr_array] = []
        gene_parts: list[sparse.csr_array] = []
        gene_reads = sparse.csr_array((n_cells, n_features), dtype=np.int64)
        all_hist = np.zeros(1, dtype=np.int64)
        empty = np.zeros(0, dtype=np.int64)
        carry_row, carry_feat, carry_k, carry_bc = empty, empty, empty, empty

        for sl in _chunks(n_mol, chunk_size):
            barcode_raw = f["barcode_idx"][sl].astype(np.int64)
            count = f["count"][sl].astype(np.int64)
            mask = (count > 0) & np.isin(f["library_idx"][sl], gex_libs)
            feature = f["feature_idx"][sl]
            mask &= feat_ok[feature]
            count, feature = count[mask], feature[mask].astype(np.int64)
            barcode = barcode_raw[mask]
            group = f["gem_group"][sl][mask].astype(np.int64)

            hc = np.bincount(count)
            if hc.size > all_hist.size:
                hc[: all_hist.size] += all_hist
                all_hist = hc
            else:
                all_hist[: hc.size] += hc

            key = barcode * n_groups + group
            pos = np.searchsorted(sorted_keys, key)
            pos_c = np.minimum(pos, max(n_cells - 1, 0))
            in_cell = (pos < n_cells) & (sorted_keys[pos_c] == key) if n_cells else pos < 0
            row = key_order[pos_c[in_cell]].astype(np.int64)
            k_cell = count[in_cell]
            feat_cell = feature[in_cell]
            umi_parts.append(build_hist(row, k_cell, n_cells))

            if streaming:
                # Molecules of one barcode are contiguous. Hold back the chunk's last
                # barcode, which may continue in the next chunk; all other cells are
                # complete, so their (cell, gene) read sums are final.
                row = np.concatenate([carry_row, row])
                feat_cell = np.concatenate([carry_feat, feat_cell])
                k_cell = np.concatenate([carry_k, k_cell])
                bc_cell = np.concatenate([carry_bc, barcode[in_cell]])
                last_chunk = sl.stop == n_mol
                hold = bc_cell == barcode_raw[-1] if not last_chunk else np.zeros(row.size, bool)
                carry_row, carry_feat = row[hold], feat_cell[hold]
                carry_k, carry_bc = k_cell[hold], bc_cell[hold]
                g_rows, g_k = _sum_by_item(row[~hold], feat_cell[~hold], k_cell[~hold], n_features)
                gene_parts.append(build_hist(g_rows, g_k, n_cells))
            else:
                chunk_genes = sparse.coo_array(
                    (k_cell, (row, feat_cell)), shape=(n_cells, n_features)
                ).tocsr()
                gene_reads = sparse.csr_array(gene_reads + chunk_genes)

    umi = add_hists(*umi_parts) if umi_parts else build_hist([], [], n_cells)
    if streaming:
        gene = add_hists(*gene_parts) if gene_parts else build_hist([], [], n_cells)
    else:
        gene_reads.sum_duplicates()
        gene_reads.eliminate_zeros()
        gene_rows = np.repeat(np.arange(n_cells), np.diff(gene_reads.indptr))
        gene = build_hist(gene_rows, gene_reads.data, n_cells)
    return GexHistograms(umi=umi, gene=gene, all_barcode_umi=all_hist, cell_barcodes=cell_barcodes)


def is_sorted_by_barcode(f: h5py.File, chunk_size: int = DEFAULT_CHUNK_SIZE) -> bool:
    """Whether ``barcode_idx`` is non-decreasing over the whole file."""
    ds = f["barcode_idx"]
    prev = -1
    for sl in _chunks(ds.shape[0], chunk_size):
        x = ds[sl].astype(np.int64)
        if x.size == 0:
            continue
        if x[0] < prev or np.any(x[1:] < x[:-1]):
            return False
        prev = int(x[-1])
    return True


def _sum_by_item(
    row: NDArray[np.int64], feature: NDArray[np.int64], k: NDArray[np.int64], n_features: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Sum ``k`` per (row, feature) pair; return the pair rows and their sums."""
    if row.size == 0:
        return row, k
    key = row * n_features + feature
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    return ks[starts] // n_features, np.add.reduceat(k[order], starts)
