import numpy as np
import pytest

from seqsat.fit import (
    MMFit,
    fit_mm,
    michaelis_menten,
    reads_needed,
    saturation_reads_needed,
    saturation_targets,
)

VMAX, KM = 12000.0, 25000.0


def synthetic(noise: float = 0.01, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 60000, 101)
    y = michaelis_menten(x, VMAX, KM)
    return x, y * (1 + noise * rng.standard_normal(x.size))


def test_fit_recovers_parameters():
    fit = fit_mm(*synthetic())
    assert fit.vmax == pytest.approx(VMAX, rel=0.02)
    assert fit.km == pytest.approx(KM, rel=0.02)
    assert fit.r2 > 0.99
    assert fit.n_points == 100  # x = 0 is excluded
    assert 0 < fit.vmax_se < 0.02 * VMAX
    assert 0 < fit.km_se < 0.02 * KM


def test_fit_is_exact_without_noise():
    fit = fit_mm(*synthetic(noise=0.0))
    assert fit.vmax == pytest.approx(VMAX, rel=1e-6)
    assert fit.km == pytest.approx(KM, rel=1e-6)
    assert fit.r2 == pytest.approx(1.0)


def test_fit_needs_three_points():
    with pytest.raises(ValueError):
        fit_mm([0, 1, 2], [0, 1, 2])


def fixed_fit() -> MMFit:
    return MMFit(vmax=VMAX, km=KM, vmax_se=0, km_se=0, r2=1, n_points=10)


def test_saturation_and_depth_are_inverse():
    fit = fixed_fit()
    for f in [0.1, 0.5, 0.9]:
        assert fit.saturation(fit.depth_for_fraction(f)) == pytest.approx(f)
    assert fit.depth_for_fraction(0.5) == pytest.approx(KM)


def test_marginal_gain_matches_numerical_derivative():
    fit = fixed_fit()
    x, h = 30000.0, 1e-3
    numeric = (fit.predict(x + h) - fit.predict(x - h)) / (2 * h)
    assert fit.marginal_gain(x) == pytest.approx(float(numeric), rel=1e-6)


def test_reads_needed_closed_form():
    fit = fixed_fit()
    x_now, n_cells = 20000.0, 1000
    res = reads_needed(fit, [0.5, 0.8, 0.95], x_now, n_cells)
    for r, f in zip(res, [0.5, 0.8, 0.95], strict=True):
        assert r.reads_per_cell == pytest.approx(KM * f / (1 - f))
        assert r.extra_reads_per_cell == pytest.approx(max(KM * f / (1 - f) - x_now, 0))
        assert r.extra_total_reads == pytest.approx(r.extra_reads_per_cell * n_cells)


def test_target_already_reached_needs_no_extra_reads():
    res = reads_needed(fixed_fit(), [0.1], x_now=50000.0, n_cells=10)[0]
    assert res.extra_reads_per_cell == 0 and res.extra_total_reads == 0


@pytest.mark.parametrize("f", [0.0, 1.0, -0.5])
def test_invalid_target(f):
    with pytest.raises(ValueError):
        fixed_fit().depth_for_fraction(f)


def test_saturation_shortcut_matches_mm_with_unit_slope():
    # Saturation S(x) = x / (km + x) for an MM curve with vmax = km (slope 1 at zero).
    km, x_now = 30000.0, 50000.0
    s_now = x_now / (km + x_now)
    for s_t in [0.5, 0.7, 0.9]:
        assert saturation_reads_needed(s_now, x_now, s_t) == pytest.approx(km * s_t / (1 - s_t))
    # Target equal to current saturation gives the current depth.
    assert saturation_reads_needed(s_now, x_now, s_now) == pytest.approx(x_now)


def test_saturation_targets():
    res = saturation_targets(s_now=0.6, x_now=50000.0, targets=[0.5, 0.9], n_cells=100)
    assert res[0].extra_reads_per_cell == 0  # already above 0.5
    assert res[1].reads_per_cell == pytest.approx(50000 * 0.4 / 0.6 * 9)
    assert res[1].extra_total_reads == pytest.approx((300000 - 50000) * 100)
