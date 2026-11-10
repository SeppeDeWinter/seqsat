import json

import numpy as np
import pytest

from seqsat.websummary import CURVE_COLUMNS, extract_curves, load_data

DATA = {
    "sample": {"id": "test"},
    "gex_umis_per_cell_plot": {
        "data": [{"name": "GRCh38", "x": [0, 1000, 2000], "y": [0, 10.0, 15.0]}],
        "layout": {"title": "Median UMI Counts per Cell"},
    },
    "atac_seq_saturation_plot": {
        "data": [{"name": "Observed", "type": "scatter", "x": [0.5, 100.0], "y": [0.3, 50.5]}],
        "layout": {},
    },
    "gex_barcode_rank_plot": {"data": [{"x": [1, 2], "y": [5, 4]}], "layout": {}},
}


def write_html(path, data=DATA):
    path.write_text(
        "<html><body><div id='root'></div>\n"
        '<script type="text/javascript">\n'
        f"      const data = {json.dumps(data)};\n"
        "      render(data);\n</script>\n"
        '<script>var other = {"x": 1};</script></body></html>'
    )
    return path


def test_load_data(tmp_path):
    assert load_data(write_html(tmp_path / "ws.html"))["sample"]["id"] == "test"


def test_load_data_without_blob(tmp_path):
    p = tmp_path / "ws.html"
    p.write_text("<html></html>")
    with pytest.raises(ValueError, match="const data"):
        load_data(p)


def test_extract_curves(tmp_path):
    df = extract_curves(write_html(tmp_path / "ws.html"))
    assert tuple(df.columns) == CURVE_COLUMNS
    assert set(zip(df.modality, df.metric, strict=True)) == {
        ("gex", "umis_per_cell"),
        ("atac", "fragments_per_cell"),
    }
    gex = df[df.metric == "umis_per_cell"]
    assert (gex.trace == "GRCh38").all()
    np.testing.assert_array_equal(gex.mean_reads_per_cell, [0, 1000, 2000])
    np.testing.assert_array_equal(gex.value, [0, 10, 15])


def test_extract_curves_none_present(tmp_path):
    df = extract_curves(write_html(tmp_path / "ws.html", data={"sample": {}}))
    assert df.empty and tuple(df.columns) == CURVE_COLUMNS
