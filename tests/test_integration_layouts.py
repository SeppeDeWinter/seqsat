"""End-to-end checks on cellranger count and cellranger-atac count outs directories."""

import numpy as np
import pytest

from seqsat.atac import AtacHistograms, read_fragments
from seqsat.curves import expected_counts, expected_curve, expected_saturation
from seqsat.gex import read_molecule_info
from seqsat.inputs import (
    AtacOutputs,
    GexOutputs,
    atac_cell_barcodes,
    read_summary_csv,
    summary_metric,
)
from seqsat.websummary import extract_curves

pytestmark = pytest.mark.slow

# Summary CSVs report medians as integers, so a median of an even number of cells
# (x.5) is shown rounded.
MEDIAN_ROUNDING = 0.5
# Saturation and duplication are reported with 3-4 significant digits.
SATURATION_TOL = 1e-3
# Cell Ranger plots one random binomial draw per GEX point; ATAC curves are expectations.
GEX_CURVE_TOL = 0.01
ATAC_CURVE_TOL = 0.001
# Cell Ranger may count a small number of fragments that are not in the fragments file.
MAX_MISSING_FRACTION = 1e-4


def max_rel_diff(ws, ours_at) -> float:
    """Largest relative difference between web-summary points (x > 0) and ``ours_at(x)``."""
    ws = ws[ws.mean_reads_per_cell > 0]
    ours = ours_at(ws.mean_reads_per_cell.to_numpy())
    return float(np.max(np.abs(ours - ws.value.to_numpy()) / ws.value.to_numpy()))


# ---------------------------------------------------------------- cellranger count


@pytest.fixture(scope="module", params=["pbmc10k_gex", "pbmc1k_gex"])
def gex_run(request):
    outs = GexOutputs.from_dir(request.getfixturevalue(request.param))
    summary = read_summary_csv(outs.metrics_summary)
    h = read_molecule_info(outs.molecule_info)
    return outs, summary, h


def test_count_n_cells(gex_run):
    _, summary, h = gex_run
    assert h.n_cells == summary_metric(summary, "Estimated Number of Cells")


@pytest.mark.parametrize(
    ("attr", "metric"),
    [("umi", "Median UMI Counts per Cell"), ("gene", "Median Genes per Cell")],
)
def test_count_medians(gex_run, attr, metric):
    _, summary, h = gex_run
    median = np.median(expected_counts(getattr(h, attr), [1.0])[:, 0])
    assert abs(median - summary_metric(summary, metric)) <= MEDIAN_ROUNDING


def test_count_sequencing_saturation(gex_run):
    _, summary, h = gex_run
    sat = expected_saturation(h.all_barcode_umi, [1.0])[0]
    assert sat == pytest.approx(
        summary_metric(summary, "Sequencing Saturation"), abs=SATURATION_TOL
    )


def test_count_curves_match_web_summary(gex_run):
    outs, summary, h = gex_run
    try:
        ws = extract_curves(outs.web_summary)
    except ValueError as e:
        pytest.skip(str(e))
    total = summary_metric(summary, "Number of Reads")

    def rates(x):
        return np.minimum(x * h.n_cells / total, 1.0)

    curves = {
        "umis_per_cell": lambda x: expected_curve(h.umi, rates(x)),
        "genes_per_cell": lambda x: expected_curve(h.gene, rates(x)),
        "sequencing_saturation": lambda x: expected_saturation(h.all_barcode_umi, rates(x)),
    }
    found = set(ws.metric)
    assert found, "no curves found in web summary"
    for metric in found:
        diff = max_rel_diff(ws[ws.metric == metric], curves[metric])
        print(f"\n{outs.outs.name} {metric}: max relative difference {diff:.4%}")
        assert diff < GEX_CURVE_TOL


# ---------------------------------------------------------------- cellranger-atac count


@pytest.fixture(scope="module")
def atac_run(pbmc1k_atac) -> tuple[AtacOutputs, dict[str, str], AtacHistograms]:
    outs = AtacOutputs.from_dir(pbmc1k_atac)
    summary = read_summary_csv(outs.summary)
    h = read_fragments(outs.fragments, atac_cell_barcodes(outs.singlecell))
    return outs, summary, h


def test_atac_n_cells(atac_run):
    _, summary, h = atac_run
    assert h.n_cells == summary_metric(summary, "Estimated number of cells")


def test_atac_median_fragments(atac_run):
    _, summary, h = atac_run
    median = np.median(expected_counts(h.cell, [1.0])[:, 0])
    reported = summary_metric(summary, "Median high-quality fragments per cell")
    assert reported * (1 - MAX_MISSING_FRACTION) <= median <= reported


def test_atac_duplication(atac_run):
    _, summary, h = atac_run
    dup = expected_saturation(h.all_barcode, [1.0])[0]
    assert dup == pytest.approx(summary_metric(summary, "Percent duplicates"), abs=SATURATION_TOL)


def test_atac_curve_matches_web_summary(atac_run):
    outs, summary, h = atac_run
    assert outs.web_summary is not None
    ws = extract_curves(outs.web_summary)
    ws = ws[ws.metric == "fragments_per_cell"]
    assert len(ws) > 0
    total = summary_metric(summary, "Sequenced read pairs")
    diff = max_rel_diff(
        ws, lambda x: expected_curve(h.cell, np.minimum(x * h.n_cells / total, 1.0))
    )
    print(f"\nfragments_per_cell: max relative difference {diff:.4%}")
    assert diff < ATAC_CURVE_TOL
