"""Exact expected subsampling curves under binomial read thinning.

Each read is kept independently with probability ``p``. An item (molecule, gene or
fragment) supported by ``k`` reads is still detected with probability
``s(k) = 1 - (1 - p) ** k``. By linearity of expectation, the expected number of
detected items in a cell is the sum of ``s(k)`` over its items, so per cell it only
depends on the histogram of ``k``.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

#: Fixed depths (mean read pairs per cell) that Cell Ranger uses for its GEX curves.
CELLRANGER_GEX_DEPTHS: tuple[int, ...] = (1000, 3000, 5000, 10000, 20000, 30000, 50000)

CURVE_COLUMNS: tuple[str, ...] = ("modality", "metric", "rate", "mean_reads_per_cell", "value")


def survival(p: float, kmax: int) -> NDArray[np.float64]:
    """Probability that an item with ``k`` reads survives thinning, for ``k = 0..kmax``.

    Parameters
    ----------
    p
        Probability of keeping each read, ``0 <= p <= 1``.
    kmax
        Largest read count to evaluate.

    Returns
    -------
    numpy.ndarray
        Array of length ``kmax + 1`` with ``s[k] = 1 - (1 - p) ** k``; ``s[0] = 0``.

    Notes
    -----
    Computed as ``-expm1(k * log1p(-p))`` so it stays accurate for very small ``p``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if kmax < 0:
        raise ValueError(f"kmax must be >= 0, got {kmax}")
    k = np.arange(kmax + 1, dtype=np.float64)
    if p == 1.0:
        s = np.ones(kmax + 1, dtype=np.float64)
    else:
        s = -np.expm1(k * np.log1p(-p))
    s[0] = 0.0
    return s


def survival_matrix(rates: ArrayLike, kmax: int) -> NDArray[np.float64]:
    """Stack :func:`survival` for several rates into a ``(kmax + 1, n_rates)`` matrix."""
    r = np.asarray(rates, dtype=np.float64).ravel()
    return np.column_stack([survival(float(p), kmax) for p in r]).reshape(kmax + 1, r.size)


def build_hist(
    rows: ArrayLike, k: ArrayLike, n_rows: int, kmax: int | None = None
) -> sparse.csr_array:
    """Build a sparse histogram: ``H[r, k]`` = number of items in row ``r`` with ``k`` reads.

    Parameters
    ----------
    rows
        Row (cell) index of each item, in ``0..n_rows - 1``.
    k
        Read count of each item, ``>= 1``.
    n_rows
        Number of rows (cells).
    kmax
        Number of columns minus one. Defaults to ``max(k)``. Use a common value to make
        histograms from different chunks addable.

    Returns
    -------
    scipy.sparse.csr_array
        ``(n_rows, kmax + 1)`` int64 matrix with duplicates summed.
    """
    r = np.asarray(rows, dtype=np.int64).ravel()
    kk = np.asarray(k, dtype=np.int64).ravel()
    if r.shape != kk.shape:
        raise ValueError(f"rows and k must have the same length, got {r.size} and {kk.size}")
    if kk.size and kk.min() < 1:
        raise ValueError("read counts k must be >= 1")
    if r.size and (r.min() < 0 or r.max() >= n_rows):
        raise ValueError("row indices out of range")
    if kmax is None:
        kmax = int(kk.max()) if kk.size else 0
    elif kk.size and kk.max() > kmax:
        raise ValueError(f"max(k) = {kk.max()} exceeds kmax = {kmax}")
    data = np.ones(r.size, dtype=np.int64)
    h = sparse.coo_array((data, (r, kk)), shape=(n_rows, kmax + 1)).tocsr()
    h.sum_duplicates()
    return h


def add_hists(*hists: sparse.csr_array) -> sparse.csr_array:
    """Sum histograms that may have different numbers of k columns (padding with zeros)."""
    if not hists:
        raise ValueError("need at least one histogram")
    n_rows = hists[0].shape[0]
    if any(h.shape[0] != n_rows for h in hists):
        raise ValueError("all histograms must have the same number of rows")
    ncol = max(h.shape[1] for h in hists)
    total = sparse.csr_array((n_rows, ncol), dtype=np.int64)
    for h in hists:
        if h.shape[1] < ncol:
            h = sparse.hstack(
                [h, sparse.csr_array((n_rows, ncol - h.shape[1]), dtype=h.dtype)], format="csr"
            )
        total = total + h
    return sparse.csr_array(total)


def expected_counts(
    h: sparse.csr_array | sparse.csr_matrix, rates: ArrayLike
) -> NDArray[np.float64]:
    """Expected number of detected items per row at each rate.

    Parameters
    ----------
    h
        ``(n_rows, kmax + 1)`` histogram from :func:`build_hist`.
    rates
        Read-keeping probabilities.

    Returns
    -------
    numpy.ndarray
        ``(n_rows, n_rates)`` array; column ``j`` is ``h @ survival(rates[j])``.
    """
    s = survival_matrix(rates, h.shape[1] - 1)
    return np.asarray(h @ s, dtype=np.float64)


def expected_curve(
    h: sparse.csr_array | sparse.csr_matrix, rates: ArrayLike
) -> NDArray[np.float64]:
    """Median over rows (cells) of the expected detected items, at each rate."""
    e = expected_counts(h, rates)
    if e.shape[0] == 0:
        raise ValueError("histogram has no rows")
    return np.median(e, axis=0)


def expected_saturation(hist: ArrayLike, rates: ArrayLike) -> NDArray[np.float64]:
    """Expected sequencing saturation (duplication rate) at each rate.

    ``S(p) = 1 - E[items](p) / (p * total_reads)``, with ``total_reads = sum_k k * hist[k]``.

    Parameters
    ----------
    hist
        1-D histogram: ``hist[k]`` = number of items (over all barcodes) with ``k`` reads.
    rates
        Read-keeping probabilities, each ``> 0``.

    Returns
    -------
    numpy.ndarray
        Saturation at each rate.
    """
    h = np.asarray(hist, dtype=np.float64).ravel()
    r = np.asarray(rates, dtype=np.float64).ravel()
    if np.any(r <= 0):
        raise ValueError("rates must be > 0 for saturation")
    k = np.arange(h.size, dtype=np.float64)
    total_reads = float(k @ h)
    if total_reads <= 0:
        raise ValueError("histogram has no reads")
    items = h @ survival_matrix(r, h.size - 1)
    return 1.0 - items / (r * total_reads)


def mean_reads_per_cell(
    rates: ArrayLike, total_raw_reads: float, n_cells: int
) -> NDArray[np.float64]:
    """X-axis value: ``p * total_raw_reads / n_cells``."""
    return np.asarray(rates, dtype=np.float64) * total_raw_reads / n_cells


def rate_grid(
    total_raw_reads: float,
    n_cells: int,
    n_points: int = 100,
    fixed_depths: Sequence[float] = CELLRANGER_GEX_DEPTHS,
) -> NDArray[np.float64]:
    """Read-keeping rates to evaluate the curves at.

    The union of: the ``fixed_depths`` at or below the current depth, ``n_points``
    evenly spaced depths in ``(0, current]`` and the current depth itself (``p = 1``).
    Because the evenly spaced depths are ``i / n_points`` of the current depth, with
    the default ``n_points = 100`` they include Cell Ranger's 0.1-step points.

    Parameters
    ----------
    total_raw_reads
        All sequenced reads (read pairs) of the library.
    n_cells
        Number of cell barcodes.
    n_points
        Number of evenly spaced depths.
    fixed_depths
        Extra depths (mean reads per cell) to include when reachable.

    Returns
    -------
    numpy.ndarray
        Sorted, unique rates with ``0 < p <= 1``, ending at exactly 1.
    """
    if total_raw_reads <= 0 or n_cells <= 0:
        raise ValueError("total_raw_reads and n_cells must be > 0")
    if n_points < 1:
        raise ValueError("n_points must be >= 1")
    x_now = total_raw_reads / n_cells
    even = np.arange(1, n_points + 1, dtype=np.float64) / n_points
    fixed = np.asarray([d for d in fixed_depths if 0 < d <= x_now], dtype=np.float64) / x_now
    rates = np.unique(np.concatenate([even, fixed, [1.0]]))
    return rates[(rates > 0) & (rates <= 1.0)]


def curve_frame(
    modality: str,
    metric: str,
    rates: ArrayLike,
    values: ArrayLike,
    total_raw_reads: float,
    n_cells: int,
) -> pd.DataFrame:
    """Tidy curve table with columns ``modality, metric, rate, mean_reads_per_cell, value``."""
    r = np.asarray(rates, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64).ravel()
    if r.shape != v.shape:
        raise ValueError("rates and values must have the same length")
    return pd.DataFrame(
        {
            "modality": modality,
            "metric": metric,
            "rate": r,
            "mean_reads_per_cell": mean_reads_per_cell(r, total_raw_reads, n_cells),
            "value": v,
        },
        columns=list(CURVE_COLUMNS),
    )
