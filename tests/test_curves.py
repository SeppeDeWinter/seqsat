import numpy as np
import pandas as pd
import pytest

from seqsat.curves import (
    CELLRANGER_GEX_DEPTHS,
    CURVE_COLUMNS,
    add_hists,
    build_hist,
    curve_frame,
    expected_counts,
    expected_curve,
    expected_saturation,
    mean_reads_per_cell,
    rate_grid,
    survival,
)

# ---------------------------------------------------------------- helpers


def random_molecules(rng: np.random.Generator, n_cells: int = 40, n_genes: int = 30):
    """Synthetic molecules: (cell, gene, reads) with heterogeneous depth per cell."""
    cells, genes, reads = [], [], []
    for c in range(n_cells):
        n_mol = int(rng.integers(20, 200))
        cells.append(np.full(n_mol, c))
        genes.append(rng.integers(0, n_genes, n_mol))
        reads.append(rng.geometric(rng.uniform(0.2, 0.8), n_mol))
    return np.concatenate(cells), np.concatenate(genes), np.concatenate(reads)


def gene_level(cells, genes, reads, n_genes):
    """Collapse molecules to (cell, gene) items with summed reads."""
    key = cells * n_genes + genes
    uniq, inv = np.unique(key, return_inverse=True)
    k = np.bincount(inv, weights=reads).astype(np.int64)
    return uniq // n_genes, k


# ---------------------------------------------------------------- survival


def test_survival_basic():
    s = survival(0.5, 3)
    np.testing.assert_allclose(s, [0.0, 0.5, 0.75, 0.875])


def test_survival_endpoints():
    np.testing.assert_array_equal(survival(1.0, 5), [0, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(survival(0.0, 5), np.zeros(6))


def test_survival_small_p_is_accurate():
    p, k = 1e-12, 7
    assert survival(p, k)[k] == pytest.approx(k * p, rel=1e-9)


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_survival_rejects_bad_p(p):
    with pytest.raises(ValueError):
        survival(p, 3)


# ---------------------------------------------------------------- histograms


def test_build_hist_counts_items():
    h = build_hist(rows=[0, 0, 0, 1, 1], k=[1, 1, 3, 2, 2], n_rows=3)
    assert h.shape == (3, 4)
    np.testing.assert_array_equal(h.toarray(), [[0, 2, 0, 1], [0, 0, 2, 0], [0, 0, 0, 0]])
    assert h.dtype == np.int64


def test_build_hist_validation():
    with pytest.raises(ValueError):
        build_hist([0], [0], n_rows=1)  # k must be >= 1
    with pytest.raises(ValueError):
        build_hist([1], [1], n_rows=1)  # row out of range
    with pytest.raises(ValueError):
        build_hist([0], [5], n_rows=1, kmax=3)


def test_add_hists_pads_columns():
    a = build_hist([0, 1], [1, 2], n_rows=2)
    b = build_hist([0, 1], [4, 1], n_rows=2)
    total = add_hists(a, b)
    whole = build_hist([0, 1, 0, 1], [1, 2, 4, 1], n_rows=2)
    np.testing.assert_array_equal(total.toarray(), whole.toarray())


# ---------------------------------------------------------------- expected curves


def test_hand_example():
    # One cell with read counts [1, 1, 1, 2, 5] at p = 0.5:
    # 3 * 0.5 + 0.75 + (1 - 0.5**5) = 3.21875
    h = build_hist([0] * 5, [1, 1, 1, 2, 5], n_rows=1)
    assert expected_curve(h, [0.5])[0] == pytest.approx(3.21875)


def test_p1_gives_observed_counts():
    rng = np.random.default_rng(0)
    cells, _, reads = random_molecules(rng)
    n_cells = int(cells.max()) + 1
    h = build_hist(cells, reads, n_cells)
    observed = np.bincount(cells, minlength=n_cells)
    np.testing.assert_array_equal(expected_counts(h, [1.0])[:, 0], observed)
    assert expected_curve(h, [1.0])[0] == np.median(observed)


def test_p_to_zero_gives_zero():
    rng = np.random.default_rng(1)
    cells, _, reads = random_molecules(rng)
    h = build_hist(cells, reads, int(cells.max()) + 1)
    vals = expected_curve(h, [1e-9, 1e-6])
    assert vals[0] < vals[1] < 1e-3


def test_closed_form_matches_monte_carlo():
    """Mean over binomial thinning draws matches the closed form within 3 SE, per cell."""
    rng = np.random.default_rng(42)
    cells, _, reads = random_molecules(rng)
    n_cells = int(cells.max()) + 1
    h = build_hist(cells, reads, n_cells)
    rates = [0.05, 0.3, 0.7]
    expected = expected_counts(h, rates)

    n_draws = 200
    for j, p in enumerate(rates):
        draws = np.empty((n_draws, n_cells))
        for d in range(n_draws):
            detected = rng.binomial(reads, p) > 0
            draws[d] = np.bincount(cells, weights=detected, minlength=n_cells)
        mean = draws.mean(axis=0)
        se = draws.std(axis=0, ddof=1) / np.sqrt(n_draws)
        z = np.abs(mean - expected[:, j]) / np.maximum(se, 1e-12)
        # 3 SE per cell; allow the ~0.3% of cells expected to exceed it by chance.
        assert np.mean(z > 3) <= 0.02, f"p={p}: {np.sum(z > 3)} cells outside 3 SE"
        # The total over cells is a much sharper test.
        tot = draws.sum(axis=1)
        assert abs(tot.mean() - expected[:, j].sum()) < 3 * tot.std(ddof=1) / np.sqrt(n_draws)


def test_gene_never_above_umi_and_both_increase():
    rng = np.random.default_rng(7)
    n_genes = 30
    cells, genes, reads = random_molecules(rng, n_genes=n_genes)
    n_cells = int(cells.max()) + 1
    h_umi = build_hist(cells, reads, n_cells)
    g_cells, g_k = gene_level(cells, genes, reads, n_genes)
    h_gene = build_hist(g_cells, g_k, n_cells)

    rates = rate_grid(total_raw_reads=1e6, n_cells=n_cells)
    umi = expected_curve(h_umi, rates)
    gene = expected_curve(h_gene, rates)
    assert np.all(gene <= umi + 1e-9)
    assert np.all(np.diff(umi) > 0)
    assert np.all(np.diff(gene) > 0)
    # Per cell too, not only for the median.
    assert np.all(expected_counts(h_gene, rates) <= expected_counts(h_umi, rates) + 1e-9)


# ---------------------------------------------------------------- saturation


def test_saturation_hand_example():
    # Items with reads [1, 1, 1, 2, 5]: 5 items, 10 reads.
    hist = np.bincount([1, 1, 1, 2, 5])
    sat = expected_saturation(hist, [1.0, 0.5])
    assert sat[0] == pytest.approx(1 - 5 / 10)
    assert sat[1] == pytest.approx(1 - 3.21875 / (0.5 * 10))


def test_saturation_limits_and_monotone():
    rng = np.random.default_rng(3)
    _, _, reads = random_molecules(rng)
    hist = np.bincount(reads)
    sat = expected_saturation(hist, [1e-8, 0.1, 0.5, 1.0])
    assert sat[0] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(sat) > 0)
    assert sat[-1] == pytest.approx(1 - reads.size / reads.sum())


def test_saturation_matches_monte_carlo():
    rng = np.random.default_rng(11)
    _, _, reads = random_molecules(rng)
    p = 0.3
    draws = []
    for _ in range(200):
        kept = rng.binomial(reads, p)
        draws.append(1 - np.count_nonzero(kept) / kept.sum())
    draws = np.asarray(draws)
    expected = expected_saturation(np.bincount(reads), [p])[0]
    # The ratio estimator has O(1/n) bias; it is far below 3 SE at this size.
    assert abs(draws.mean() - expected) < 3 * draws.std(ddof=1) / np.sqrt(draws.size)


def test_saturation_rejects_zero_rate():
    with pytest.raises(ValueError):
        expected_saturation(np.bincount([1, 2]), [0.0])


# ---------------------------------------------------------------- rate grid and table


def test_rate_grid_contents():
    total, n = 53829 * 10261, 10261
    rates = rate_grid(total, n)
    x = mean_reads_per_cell(rates, total, n)
    assert rates[-1] == 1.0
    assert np.all((rates > 0) & (rates <= 1))
    assert np.all(np.diff(rates) > 0)
    for d in CELLRANGER_GEX_DEPTHS:  # all fixed depths are below 53829
        assert np.any(np.isclose(x, d))
    for frac in np.arange(1, 11) / 10:  # Cell Ranger's 0.1-step points
        assert np.any(np.isclose(rates, frac))
    assert len(rates) == 100 + len(CELLRANGER_GEX_DEPTHS)


def test_rate_grid_drops_unreachable_fixed_depths():
    rates = rate_grid(total_raw_reads=4000 * 100, n_cells=100, n_points=10)
    x = mean_reads_per_cell(rates, 4000 * 100, 100)
    assert x.max() == pytest.approx(4000)
    assert np.any(np.isclose(x, 1000)) and np.any(np.isclose(x, 3000))
    assert not np.any(np.isclose(x, 5000))


def test_curve_frame():
    df = curve_frame("gex", "umis_per_cell", [0.5, 1.0], [10.0, 15.0], 2000, 10)
    assert tuple(df.columns) == CURVE_COLUMNS
    np.testing.assert_allclose(df["mean_reads_per_cell"], [100, 200])
    assert (df["modality"] == "gex").all()
    assert isinstance(df, pd.DataFrame)
