"""End-to-end checks on the 10k Human Brain Multiome dataset (Cell Ranger ARC 2.2.0)."""

import resource

import numpy as np
import pandas as pd
import pytest

from seqsat.curves import expected_counts, expected_saturation
from seqsat.gex import GexHistograms, raw_read_pairs, read_molecule_info
from seqsat.inputs import ArcOutputs, arc_cell_barcodes, read_summary_csv, summary_metric

pytestmark = pytest.mark.slow


def peak_rss_gb() -> float:
    """Peak resident memory of this process (ru_maxrss is in kB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


@pytest.fixture(scope="module")
def arc(brain10k):
    return ArcOutputs.from_dir(brain10k)


@pytest.fixture(scope="module")
def summary(arc):
    return read_summary_csv(arc.summary)


@pytest.fixture(scope="module")
def gex(arc) -> GexHistograms:
    h = read_molecule_info(arc.gex_molecule_info, arc_cell_barcodes(arc.per_barcode_metrics))
    print(f"\nGEX histograms built; peak RSS {peak_rss_gb():.2f} GB")
    return h


@pytest.fixture(scope="module")
def per_barcode(arc, gex):
    cols = ["barcode", "is_cell", "gex_umis_count", "gex_genes_count"]
    pb = pd.read_csv(arc.per_barcode_metrics, usecols=cols)
    return pb[pb["is_cell"] == 1].set_index("barcode").loc[gex.cell_barcodes]


def test_gex_n_cells(gex, summary):
    assert gex.n_cells == summary_metric(summary, "Estimated number of cells")


def test_gex_median_umis_exact(gex, summary):
    median = np.median(expected_counts(gex.umi, [1.0])[:, 0])
    assert median == summary_metric(summary, "GEX Median UMI counts per cell")


def test_gex_median_genes_exact(gex, summary):
    median = np.median(expected_counts(gex.gene, [1.0])[:, 0])
    assert median == summary_metric(summary, "GEX Median genes per cell")


def test_gex_per_cell_counts_exact(gex, per_barcode):
    np.testing.assert_array_equal(
        expected_counts(gex.umi, [1.0])[:, 0], per_barcode["gex_umis_count"]
    )
    np.testing.assert_array_equal(
        expected_counts(gex.gene, [1.0])[:, 0], per_barcode["gex_genes_count"]
    )


def test_gex_saturation(gex, summary):
    # ARC has no "Sequencing saturation" metric; "GEX Percent duplicates" is the same quantity.
    sat = expected_saturation(gex.all_barcode_umi, [1.0])[0]
    assert sat == pytest.approx(summary_metric(summary, "GEX Percent duplicates"), abs=1e-3)


def test_gex_pass_filter_cells_match_arc_cells(arc, gex):
    from_h5 = read_molecule_info(arc.gex_molecule_info)
    assert set(from_h5.cell_barcodes) == set(gex.cell_barcodes)


def test_gex_metrics_json_read_pairs(arc, summary):
    assert raw_read_pairs(arc.gex_molecule_info) == summary_metric(
        summary, "GEX Sequenced read pairs"
    )


def test_gex_peak_memory(gex):
    print(f"\npeak RSS {peak_rss_gb():.2f} GB")
    assert peak_rss_gb() < 2.0
