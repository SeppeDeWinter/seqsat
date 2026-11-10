import gzip
from pathlib import Path

import numpy as np
import pytest

from seqsat.atac import read_fragments, read_header
from seqsat.curves import expected_counts, expected_saturation

HEADER = """\
# id=test
# pipeline_name=cellranger-arc
#
# primary_contig=chr1
# primary_contig=chr2
"""

# (chrom, start, end, barcode, readSupport)
FRAGMENTS = [
    ("chr1", 100, 300, "AAAA-1", 1),
    ("chr1", 150, 400, "CCCC-1", 2),  # not a cell
    ("chr1", 200, 350, "AAAA-1", 3),
    ("chr1", 500, 700, "GGGG-1", 1),
    ("chr2", 100, 200, "GGGG-1", 2),
    ("chr2", 300, 500, "AAAA-1", 1),
    ("chrM", 10, 200, "AAAA-1", 4),  # mitochondrial
    ("chrM", 20, 300, "GGGG-1", 1),
]
CELLS = np.array(["AAAA-1", "GGGG-1"])


def write_fragments(path: Path, strand: bool = True, header: str = HEADER) -> Path:
    lines = []
    for i, (c, s, e, b, k) in enumerate(FRAGMENTS):
        fields = [c, str(s), str(e), b, str(k)] + (["+-"[i % 2]] if strand else [])
        lines.append("\t".join(fields))
    with gzip.open(path, "wt") as fh:
        fh.write(header + "\n".join(lines) + "\n")
    return path


@pytest.fixture
def fragments(tmp_path) -> Path:
    return write_fragments(tmp_path / "fragments.tsv.gz")


def test_read_header(fragments):
    h = read_header(fragments)
    assert h.n_lines == 5
    assert h.n_bytes == len(HEADER)
    assert h.n_columns == 6
    assert h.primary_contigs == ("chr1", "chr2")
    assert h.metadata["pipeline_name"] == "cellranger-arc"


def test_cell_histogram(fragments):
    h = read_fragments(fragments, CELLS)
    u = h.cell.toarray()
    # AAAA: k = 1, 3, 1, 4 ; GGGG: k = 1, 2, 1
    np.testing.assert_array_equal(u[0], [0, 2, 0, 1, 1])
    np.testing.assert_array_equal(u[1], [0, 2, 1, 0, 0])
    np.testing.assert_array_equal(expected_counts(h.cell, [1.0])[:, 0], [4, 3])


def test_all_barcode_histogram(fragments):
    h = read_fragments(fragments, CELLS)
    np.testing.assert_array_equal(h.all_barcode, [0, 4, 2, 1, 1])
    assert expected_saturation(h.all_barcode, [1.0])[0] == pytest.approx(1 - 8 / 15)


def test_exclude_contigs(fragments):
    h = read_fragments(fragments, CELLS, exclude_contigs=["chrM"])
    np.testing.assert_array_equal(expected_counts(h.cell, [1.0])[:, 0], [3, 2])
    assert h.all_barcode.sum() == 6
    assert h.contig_fragments == {"chr1": 4, "chr2": 2, "chrM": 0}


def test_five_columns(tmp_path):
    path = write_fragments(tmp_path / "f.tsv.gz", strand=False)
    assert read_header(path).n_columns == 5
    h = read_fragments(path, CELLS)
    np.testing.assert_array_equal(expected_counts(h.cell, [1.0])[:, 0], [4, 3])


def test_no_header(tmp_path):
    path = write_fragments(tmp_path / "f.tsv.gz", header="")
    assert read_header(path).n_lines == 0
    h = read_fragments(path, CELLS)
    np.testing.assert_array_equal(expected_counts(h.cell, [1.0])[:, 0], [4, 3])


@pytest.mark.parametrize("block_size", [64, 100, 1 << 20])
def test_block_size_does_not_matter(fragments, block_size):
    h = read_fragments(fragments, CELLS, block_size=block_size)
    np.testing.assert_array_equal(expected_counts(h.cell, [1.0])[:, 0], [4, 3])
    np.testing.assert_array_equal(h.all_barcode, [0, 4, 2, 1, 1])


def test_cell_without_fragments(fragments):
    h = read_fragments(fragments, np.array(["TTTT-1", "AAAA-1"]))
    np.testing.assert_array_equal(expected_counts(h.cell, [1.0])[:, 0], [0, 4])
