import numpy as np
import pandas as pd

from seqsat.report import COMPARISON_MAX_MARKERS, plot_curve_comparison


def curve(x, y) -> pd.DataFrame:
    return pd.DataFrame({"mean_reads_per_cell": x, "value": y})


def test_plot_curve_comparison_writes_image(tmp_path):
    x = np.linspace(0, 1000, 101)
    dense = curve(x, x / (300 + x))  # more points than COMPARISON_MAX_MARKERS
    assert len(dense) > COMPARISON_MAX_MARKERS
    panels = [
        ("A – saturation", "sequencing_saturation", dense, dense),
        ("B – UMIs", "umis_per_cell", curve(x, 5 * x), curve(x[::10], 5 * x[::10])),
    ]
    out = plot_curve_comparison(panels, tmp_path / "cmp.png")
    assert out.stat().st_size > 0
