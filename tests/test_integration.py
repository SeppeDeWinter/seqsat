"""End-to-end checks on a Cell Ranger ARC outs directory (see the ``brain10k`` fixture)."""

import json
import resource

import numpy as np
import pandas as pd
import pytest

from seqsat import cli
from seqsat.atac import AtacHistograms, read_fragments, read_header
from seqsat.curves import expected_counts, expected_curve, expected_saturation
from seqsat.gex import GexHistograms, raw_read_pairs, read_molecule_info
from seqsat.inputs import ArcOutputs, arc_cell_barcodes, read_summary_csv, summary_metric
from seqsat.websummary import extract_curves

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


# ---------------------------------------------------------------- ATAC


@pytest.fixture(scope="module")
def atac(arc) -> AtacHistograms:
    h = read_fragments(arc.atac_fragments, arc_cell_barcodes(arc.per_barcode_metrics))
    print(f"\nATAC histograms built; peak RSS {peak_rss_gb():.2f} GB")
    return h


@pytest.fixture(scope="module")
def atac_deficit(arc, atac):
    """Cell Ranger's per-cell ``atac_fragments`` minus our fragments-file count."""
    pb = pd.read_csv(arc.per_barcode_metrics, usecols=["barcode", "atac_fragments"])
    cr = pb.set_index("barcode").loc[atac.cell_barcodes, "atac_fragments"].to_numpy()
    return cr - expected_counts(atac.cell, [1.0])[:, 0]


def test_atac_fragments_only_on_primary_contigs(arc, atac):
    assert set(atac.contig_fragments) <= set(read_header(arc.atac_fragments).primary_contigs)


# Cell Ranger's per-barcode fragment counts include a small number of fragments that are
# not written to the fragments file. The file is the ground truth for the curves, so the
# endpoint can be slightly below Cell Ranger's median but never above it.
MAX_MISSING_FRACTION = 1e-4


def test_atac_fragments_never_above_cellranger(atac_deficit):
    assert np.all(atac_deficit >= 0)


def test_atac_missing_fragments_are_rare(atac, atac_deficit):
    ours = expected_counts(atac.cell, [1.0])[:, 0]
    missing = atac_deficit.sum() / (ours.sum() + atac_deficit.sum())
    print(f"\nfragments counted by Cell Ranger but not in the file: {missing:.2e}")
    assert missing < MAX_MISSING_FRACTION


def test_atac_median_fragments(atac, summary):
    median = np.median(expected_counts(atac.cell, [1.0])[:, 0])
    reported = summary_metric(summary, "ATAC Median high-quality fragments per cell")
    print(f"\nmedian fragments per cell: {median:.0f} (Cell Ranger: {reported:.0f})")
    assert reported * (1 - MAX_MISSING_FRACTION) <= median <= reported


def test_atac_duplication(atac, summary):
    dup = expected_saturation(atac.all_barcode, [1.0])[0]
    assert dup == pytest.approx(summary_metric(summary, "ATAC Percent duplicates"), abs=1e-3)


def test_atac_peak_memory(atac):
    assert peak_rss_gb() < 2.0


# ---------------------------------------------------------------- web-summary curves


@pytest.fixture(scope="module")
def web_curves(arc):
    assert arc.web_summary is not None
    return extract_curves(arc.web_summary)


# metric -> (histogram fixture, histogram attribute, total read pairs metric, max relative diff)
# GEX: Cell Ranger plots one random binomial draw per point, so its values scatter around
# the expectation: 1%.
# ATAC: Cell Ranger's curve is an expectation as well; the remaining difference comes from
# the fragments it counts that are not in the fragments file: 0.1%.
CURVES = {
    "umis_per_cell": ("gex", "umi", "GEX Sequenced read pairs", 0.01),
    "genes_per_cell": ("gex", "gene", "GEX Sequenced read pairs", 0.01),
    "fragments_per_cell": ("atac", "cell", "ATAC Sequenced read pairs", 0.001),
}
MIN_CURVE_POINTS = 10


@pytest.mark.parametrize("metric", list(CURVES))
def test_curve_matches_web_summary(request, metric, web_curves, summary):
    fixture, attr, total_metric, tol = CURVES[metric]
    hists = request.getfixturevalue(fixture)
    ws = web_curves[web_curves.metric == metric]
    assert len(ws) >= MIN_CURVE_POINTS

    ws = ws[ws.mean_reads_per_cell > 0]
    total = summary_metric(summary, total_metric)
    rates = np.minimum(ws.mean_reads_per_cell.to_numpy() * hists.n_cells / total, 1.0)
    ours = expected_curve(getattr(hists, attr), rates)
    rel = np.abs(ours - ws.value.to_numpy()) / ws.value.to_numpy()
    print(f"\n{metric}: max relative difference {rel.max():.4%} over {len(rel)} points")
    assert rel.max() < tol


# ---------------------------------------------------------------- CLI end to end


def test_arc_cli_end_to_end(brain10k, summary, tmp_path):
    cli.main(["arc", str(brain10k), "--out", str(tmp_path)])
    curves = pd.read_csv(tmp_path / "curves.csv")
    fits = {m["metric"]: m for m in json.loads((tmp_path / "fits.json").read_text())["metrics"]}
    assert (tmp_path / "saturation.png").stat().st_size > 0

    # The p = 1 point of each curve is the observed value.
    at_one = curves[curves.rate == 1.0].set_index("metric").value
    assert at_one["umis_per_cell"] == summary_metric(summary, "GEX Median UMI counts per cell")
    assert at_one["genes_per_cell"] == summary_metric(summary, "GEX Median genes per cell")
    for metric in at_one.index:
        assert fits[metric]["current_value"] == pytest.approx(at_one[metric])
        assert fits[metric]["r2"] > 0.99
