import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from seqsat import cli
from seqsat.report import CURVES_FILE, FIGURE_FILE, FITS_FILE

from .conftest import write_molecule_info
from .test_atac import write_fragments


@pytest.fixture
def captured(monkeypatch):
    """Replace the subcommand functions with ones that record their arguments."""
    calls: list = []

    def fake_gex(args: cli.GexArgs) -> None:
        calls.append(args)

    def fake_atac(args: cli.AtacArgs) -> None:
        calls.append(args)

    def fake_arc(args: cli.ArcArgs) -> None:
        calls.append(args)

    monkeypatch.setitem(cli.SUBCOMMANDS, "gex", fake_gex)
    monkeypatch.setitem(cli.SUBCOMMANDS, "atac", fake_atac)
    monkeypatch.setitem(cli.SUBCOMMANDS, "arc", fake_arc)
    return calls


def test_parse_defaults(captured):
    cli.main(["gex", "some/outs"])
    (args,) = captured
    assert args == cli.GexArgs(outs=Path("some/outs"))
    assert args.targets == cli.DEFAULT_TARGETS


def test_parse_targets_and_overrides(captured):
    cli.main(
        ["arc", "outs", "--targets", "0.5", "0.8", "--out", "res", "--gex-total-reads", "1000"]
    )
    (args,) = captured
    assert isinstance(args, cli.ArcArgs)
    assert args.targets == (0.5, 0.8)
    assert args.out == Path("res")
    assert args.gex_total_reads == 1000
    assert args.atac_total_reads is None


def test_parse_atac_cells(captured):
    cli.main(["atac", "outs", "--cells", "cells.txt", "--total-reads", "42"])
    (args,) = captured
    assert args == cli.AtacArgs(outs=Path("outs"), cells=Path("cells.txt"), total_reads=42)


def test_outs_is_required(captured):
    with pytest.raises(SystemExit):
        cli.main(["gex"])


def test_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["arc", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--targets" in out and "--gex-total-reads" in out


# ---------------------------------------------------------------- end to end (synthetic)


def check_outputs(out: Path, metrics: set[str]) -> dict:
    curves = pd.read_csv(out / CURVES_FILE)
    assert set(curves.metric) == metrics
    assert (out / FIGURE_FILE).stat().st_size > 0
    fits = json.loads((out / FITS_FILE).read_text())
    assert {m["metric"] for m in fits["metrics"]} == metrics
    for m in fits["metrics"]:
        assert [t["target"] for t in m["targets"]] == [0.5, 0.9]
    return fits


def test_gex_end_to_end(tmp_path, capsys):
    outs = tmp_path / "outs"
    outs.mkdir()
    write_molecule_info(outs / "molecule_info.h5")
    (outs / "metrics_summary.csv").write_text(
        'Estimated Number of Cells,Number of Reads\n2,"1,234"\n'
    )
    out = tmp_path / "res"
    cli.main(["gex", str(outs), "--out", str(out), "--targets", "0.5", "0.9"])
    fits = check_outputs(out, {"umis_per_cell", "genes_per_cell", "sequencing_saturation"})
    assert fits["inputs"]["gex"] == {"n_cells": 2, "total_raw_reads": 1234}
    umi = next(m for m in fits["metrics"] if m["metric"] == "umis_per_cell")
    assert umi["current_value"] == 3.0  # median of 3 and 3 molecules
    assert umi["current_reads_per_cell"] == pytest.approx(617)
    assert "umis" in capsys.readouterr().out.lower()


def test_atac_end_to_end(tmp_path):
    outs = tmp_path / "outs"
    outs.mkdir()
    write_fragments(outs / "fragments.tsv.gz")
    (outs / "singlecell.csv").write_text(
        "barcode,is__cell_barcode\nNO_BARCODE,0\nAAAA-1,1\nCCCC-1,0\nGGGG-1,1\n"
    )
    (outs / "summary.csv").write_text("Sequenced read pairs,Percent duplicates\n500,0.4\n")
    out = tmp_path / "res"
    cli.main(["atac", str(outs), "--out", str(out), "--targets", "0.5", "0.9"])
    fits = check_outputs(out, {"fragments_per_cell", "duplication"})
    assert fits["inputs"]["atac"] == {"n_cells": 2, "total_raw_reads": 500}


def test_cli_import_does_not_load_analysis_stack():
    # The analysis stack is imported by the subcommands, keeping `seqsat --help` fast.
    code = (
        "import sys, seqsat.cli; "
        "print(','.join(m for m in ('numpy', 'pandas', 'scipy', 'h5py', 'pyarrow', 'matplotlib') "
        "if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""
