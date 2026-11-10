import math

import numpy as np
import pytest

from seqsat.inputs import (
    ArcOutputs,
    arc_cell_barcodes,
    parse_number,
    read_summary_csv,
    split_gem_group,
    summary_metric,
)


@pytest.mark.parametrize(
    ("raw", "value"),
    [("552,342,334", 552342334.0), ("63.3%", 0.633), ("0.6330", 0.633), (" 7403.0000 ", 7403.0)],
)
def test_parse_number(raw, value):
    assert parse_number(raw) == pytest.approx(value)


def test_parse_number_nan():
    assert math.isnan(parse_number("nan"))


def test_read_summary_csv(tmp_path):
    p = tmp_path / "summary.csv"
    p.write_text('Sample ID,GEX Sequenced read pairs,ATAC Percent duplicates\nx,"1,000",0.42\n')
    s = read_summary_csv(p)
    assert summary_metric(s, "GEX Sequenced read pairs") == 1000
    assert summary_metric(s, "ATAC Percent duplicates") == pytest.approx(0.42)
    with pytest.raises(KeyError, match="not in summary"):
        summary_metric(s, "Sequencing saturation")


def test_read_summary_csv_rejects_multiple_rows(tmp_path):
    p = tmp_path / "summary.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    with pytest.raises(ValueError):
        read_summary_csv(p)


def test_split_gem_group():
    seqs, groups = split_gem_group(np.array(["AAAC-1", "GGTT-2", "CCCC"]))
    assert list(seqs) == [b"AAAC", b"GGTT", b"CCCC"]
    assert list(groups) == [1, 2, 1]


def test_arc_cell_barcodes(tmp_path):
    p = tmp_path / "per_barcode_metrics.csv"
    p.write_text(
        "barcode,gex_barcode,atac_barcode,is_cell\nA-1,A-1,X-1,1\nC-1,C-1,Y-1,0\nG-1,G-1,Z-1,1\n"
    )
    assert list(arc_cell_barcodes(p)) == ["A-1", "G-1"]


def test_arc_outputs_missing_file(tmp_path):
    (tmp_path / "summary.csv").write_text("a\n1\n")
    with pytest.raises(FileNotFoundError, match="per_barcode_metrics.csv"):
        ArcOutputs.from_dir(tmp_path)
