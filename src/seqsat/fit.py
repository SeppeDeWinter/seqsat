"""Michaelis–Menten fits of saturation curves and the sequencing depth needed for a target.

Model: ``y = vmax * x / (km + x)`` with ``x`` the mean reads per cell. At depth ``x``
the metric has reached the fraction ``x / (km + x)`` of its asymptote ``vmax``, so
fraction ``f`` is reached at ``x_f = km * f / (1 - f)``.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit


def michaelis_menten(x: ArrayLike, vmax: float, km: float) -> NDArray[np.float64]:
    """``vmax * x / (km + x)``."""
    xx = np.asarray(x, dtype=np.float64)
    return vmax * xx / (km + xx)


@dataclass(frozen=True)
class MMFit:
    """Fitted Michaelis–Menten parameters.

    Attributes
    ----------
    vmax, km
        Asymptote and half-saturation depth (mean reads per cell).
    vmax_se, km_se
        Standard errors from the covariance matrix of the fit.
    r2
        Coefficient of determination on the fitted points.
    n_points
        Number of points used (those with ``x > 0``).
    """

    vmax: float
    km: float
    vmax_se: float
    km_se: float
    r2: float
    n_points: int

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        return michaelis_menten(x, self.vmax, self.km)

    def saturation(self, x: float) -> float:
        """Fraction of ``vmax`` reached at depth ``x``: ``x / (km + x)``."""
        return float(x / (self.km + x))

    def depth_for_fraction(self, f: float) -> float:
        """Depth ``x_f = km * f / (1 - f)`` at which fraction ``f`` of ``vmax`` is reached."""
        _check_fraction(f)
        return float(self.km * f / (1.0 - f))

    def marginal_gain(self, x: float) -> float:
        """Slope ``dy/dx = vmax * km / (km + x) ** 2`` at depth ``x``."""
        return float(self.vmax * self.km / (self.km + x) ** 2)


def _check_fraction(f: float) -> None:
    if not 0.0 < f < 1.0:
        raise ValueError(f"target fraction must be in (0, 1), got {f}")


def fit_mm(x: ArrayLike, y: ArrayLike) -> MMFit:
    """Fit ``y = vmax * x / (km + x)`` by least squares on the points with ``x > 0``.

    Starts from ``vmax = 2 * max(y)``, ``km = max(x)`` with both parameters bounded
    to ``(0, inf)``.

    Raises
    ------
    ValueError
        With fewer than 3 points with ``x > 0``.
    """
    xx = np.asarray(x, dtype=np.float64).ravel()
    yy = np.asarray(y, dtype=np.float64).ravel()
    if xx.shape != yy.shape:
        raise ValueError("x and y must have the same length")
    keep = xx > 0
    xx, yy = xx[keep], yy[keep]
    if xx.size < 3:
        raise ValueError(f"need at least 3 points with x > 0, got {xx.size}")
    p0 = [2.0 * float(yy.max()), float(xx.max())]
    popt, pcov = curve_fit(michaelis_menten, xx, yy, p0=p0, bounds=(0.0, np.inf))
    vmax, km = (float(v) for v in popt)
    se = np.sqrt(np.diag(pcov))
    resid = yy - michaelis_menten(xx, vmax, km)
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return MMFit(vmax, km, float(se[0]), float(se[1]), r2, int(xx.size))


@dataclass(frozen=True)
class TargetDepth:
    """Sequencing depth needed for one target.

    Attributes
    ----------
    target
        Target fraction (of ``vmax``, or the sequencing saturation itself).
    reads_per_cell
        Mean raw reads per cell at which the target is reached.
    extra_reads_per_cell
        ``max(reads_per_cell - current, 0)``.
    extra_total_reads
        ``extra_reads_per_cell * n_cells``.
    """

    target: float
    reads_per_cell: float
    extra_reads_per_cell: float
    extra_total_reads: float


def target_depth(target: float, reads_per_cell: float, x_now: float, n_cells: int) -> TargetDepth:
    """Wrap a required depth into a :class:`TargetDepth` relative to the current depth."""
    extra = max(reads_per_cell - x_now, 0.0)
    return TargetDepth(
        target=target,
        reads_per_cell=reads_per_cell,
        extra_reads_per_cell=extra,
        extra_total_reads=extra * n_cells,
    )


def reads_needed(
    fit: MMFit, targets: Sequence[float], x_now: float, n_cells: int
) -> list[TargetDepth]:
    """Depth needed to reach each target fraction of ``vmax``."""
    return [target_depth(f, fit.depth_for_fraction(f), x_now, n_cells) for f in targets]


def saturation_reads_needed(s_now: float, x_now: float, s_target: float) -> float:
    """Depth at which sequencing saturation reaches ``s_target``.

    ``R_target = R_now * (1 - S_now) / S_now * S_t / (1 - S_t)``: under a
    Michaelis–Menten curve with unit slope at zero, saturation is ``S = x / (km + x)``,
    so ``km = R_now * (1 - S_now) / S_now``.
    """
    _check_fraction(s_target)
    if not 0.0 < s_now < 1.0:
        raise ValueError(f"current saturation must be in (0, 1), got {s_now}")
    return float(x_now * (1.0 - s_now) / s_now * s_target / (1.0 - s_target))


def saturation_targets(
    s_now: float, x_now: float, targets: Sequence[float], n_cells: int
) -> list[TargetDepth]:
    """Depth needed to reach each target sequencing saturation."""
    return [
        target_depth(s, saturation_reads_needed(s_now, x_now, s), x_now, n_cells) for s in targets
    ]


@dataclass(frozen=True)
class MetricResult:
    """Fit and target depths for one curve.

    Attributes
    ----------
    modality, metric
        Curve identity.
    kind
        ``"count"``: targets are fractions of the fitted ``vmax``.
        ``"saturation"``: targets are sequencing-saturation values, solved in closed
        form from the current saturation (:func:`saturation_reads_needed`).
    fit
        Michaelis–Menten fit of the curve.
    x_now
        Current mean raw reads per cell.
    n_cells
        Number of cell barcodes.
    current_value
        Observed curve value at the current depth.
    current_saturation
        ``x_now / (km + x_now)`` for counts; the observed saturation for ``"saturation"``.
    marginal_gain_per_1000
        Fitted gain in the metric per 1,000 extra reads per cell at the current depth.
    targets
        Required depth for each target.
    """

    modality: str
    metric: str
    kind: str
    fit: MMFit
    x_now: float
    n_cells: int
    current_value: float
    current_saturation: float
    marginal_gain_per_1000: float
    targets: list[TargetDepth]


def analyze_curve(
    modality: str,
    metric: str,
    kind: str,
    x: ArrayLike,
    y: ArrayLike,
    n_cells: int,
    targets: Sequence[float],
) -> MetricResult:
    """Fit one curve and compute the depth needed for each target.

    The current depth is the largest ``x`` (rate 1).
    """
    if kind not in ("count", "saturation"):
        raise ValueError(f"kind must be 'count' or 'saturation', got {kind!r}")
    xx = np.asarray(x, dtype=np.float64).ravel()
    yy = np.asarray(y, dtype=np.float64).ravel()
    i_now = int(np.argmax(xx))
    x_now, y_now = float(xx[i_now]), float(yy[i_now])
    fit = fit_mm(xx, yy)
    if kind == "count":
        current = fit.saturation(x_now)
        depths = reads_needed(fit, targets, x_now, n_cells)
    else:
        current = y_now
        depths = saturation_targets(y_now, x_now, targets, n_cells)
    return MetricResult(
        modality=modality,
        metric=metric,
        kind=kind,
        fit=fit,
        x_now=x_now,
        n_cells=n_cells,
        current_value=y_now,
        current_saturation=current,
        marginal_gain_per_1000=1000.0 * fit.marginal_gain(x_now),
        targets=depths,
    )
