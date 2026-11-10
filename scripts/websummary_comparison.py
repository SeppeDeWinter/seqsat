"""Plot seqsat curves against the curves in Cell Ranger's web_summary.html.

Each argument pairs a Cell Ranger outs directory with a seqsat results directory
(the ``--out`` of a ``seqsat`` run on it), plus a label for the panel titles:

    uv run python scripts/websummary_comparison.py docs/websummary_comparison.png \
        "brain10k (ARC 2.2)=data/brain10k:results/brain10k" \
        "pbmc10k (count 6.1)=data/pbmc10k_gex:results/pbmc10k_gex" \
        "pbmc1k (ATAC 2.0)=data/pbmc1k_atac:results/pbmc1k_atac"
"""

import argparse
from pathlib import Path

import pandas as pd

from seqsat.report import CURVES_FILE, METRIC_LABELS, plot_curve_comparison
from seqsat.websummary import extract_curves


def parse_spec(spec: str) -> tuple[str, Path, Path]:
    label, _, dirs = spec.partition("=")
    outs, _, results = dirs.partition(":")
    if not (label and outs and results):
        raise argparse.ArgumentTypeError(f"expected LABEL=OUTS:RESULTS, got {spec!r}")
    return label, Path(outs), Path(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="output image")
    parser.add_argument("datasets", nargs="+", type=parse_spec, help="LABEL=OUTS:RESULTS")
    args = parser.parse_args()

    panels = []
    for label, outs, results in args.datasets:
        ws = extract_curves(outs / "web_summary.html")
        ours = pd.read_csv(results / CURVES_FILE)
        for (modality, metric), cr in ws.groupby(["modality", "metric"], sort=False):
            mine = ours[(ours.modality == modality) & (ours.metric == metric)]
            if mine.empty:
                continue
            # Prepend the origin so the line starts at (0, 0) like Cell Ranger's curve.
            mine = pd.concat(
                [pd.DataFrame({"mean_reads_per_cell": [0.0], "value": [0.0]}), mine],
                ignore_index=True,
            )
            title = f"{label} – {METRIC_LABELS.get(str(metric), str(metric))}"
            panels.append((title, str(metric), mine, cr))
    # Same metric order as the seqsat figure; datasets keep their command-line order.
    order = list(METRIC_LABELS)
    panels.sort(key=lambda p: order.index(p[1]) if p[1] in order else len(order))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_curve_comparison(panels, args.output)
    print(f"wrote {args.output} ({len(panels)} panels)")


if __name__ == "__main__":
    main()
