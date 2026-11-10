import h5py
import numpy as np
import pytest

from seqsat.curves import expected_counts, expected_saturation
from seqsat.gex import is_sorted_by_barcode, raw_read_pairs, read_molecule_info

from .conftest import SYN_MOLECULES, write_molecule_info

# Expected histograms for the synthetic molecule_info (see conftest.py):
#   AAAA: molecules k = 1, 3, 2      genes: gene0 k = 4, gene1 k = 2
#   GGGG: molecules k = 1, 1, 5      genes: gene1 k = 1, gene2 k = 6
UMI_AAAA = [0, 1, 1, 1]
UMI_GGGG = [0, 2, 0, 0, 0, 1]
GENE_AAAA = [0, 0, 1, 0, 1]
GENE_GGGG = [0, 1, 0, 0, 0, 0, 1]
# All barcodes, GEX library and features only: k = 1, 3, 2, 4, 1, 1, 5, 1
ALL_BARCODE = [0, 4, 1, 1, 1, 1]


def padded(row, n):
    return np.pad(np.asarray(row), (0, n - len(row)))


def check_synthetic(h):
    assert list(h.cell_barcodes) == ["AAAA-1", "GGGG-1"]
    u, g = h.umi.toarray(), h.gene.toarray()
    np.testing.assert_array_equal(u[0], padded(UMI_AAAA, u.shape[1]))
    np.testing.assert_array_equal(u[1], padded(UMI_GGGG, u.shape[1]))
    np.testing.assert_array_equal(g[0], padded(GENE_AAAA, g.shape[1]))
    np.testing.assert_array_equal(g[1], padded(GENE_GGGG, g.shape[1]))
    np.testing.assert_array_equal(h.all_barcode_umi, ALL_BARCODE)


def test_synthetic_histograms(molecule_info):
    check_synthetic(read_molecule_info(molecule_info))


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 7, 100])
def test_chunk_size_does_not_matter(molecule_info, chunk_size):
    # Chunk boundaries fall inside cells for most sizes; the carry-over must handle it.
    check_synthetic(read_molecule_info(molecule_info, chunk_size=chunk_size))


def test_unsorted_file_uses_fallback(tmp_path):
    shuffled = [SYN_MOLECULES[i] for i in np.random.default_rng(0).permutation(len(SYN_MOLECULES))]
    path = write_molecule_info(tmp_path / "unsorted.h5", molecules=shuffled)
    with h5py.File(path) as f:
        assert not is_sorted_by_barcode(f)
    for chunk_size in [1, 3, 100]:
        check_synthetic(read_molecule_info(path, chunk_size=chunk_size))


def test_cell_barcodes_override_sets_row_order(molecule_info):
    h = read_molecule_info(molecule_info, cell_barcodes=np.array(["GGGG-1", "CCCC-1"]))
    assert list(h.cell_barcodes) == ["GGGG-1", "CCCC-1"]
    np.testing.assert_array_equal(expected_counts(h.umi, [1.0])[:, 0], [3, 1])
    np.testing.assert_array_equal(expected_counts(h.gene, [1.0])[:, 0], [2, 1])


def test_cell_barcode_without_suffix_is_gem_group_1(molecule_info):
    h = read_molecule_info(molecule_info, cell_barcodes=np.array(["AAAA"]))
    np.testing.assert_array_equal(expected_counts(h.umi, [1.0])[:, 0], [3])


def test_unknown_cell_barcode_raises(molecule_info):
    with pytest.raises(ValueError, match="not in molecule_info"):
        read_molecule_info(molecule_info, cell_barcodes=np.array(["ACGT-1"]))


def test_pass_filter_of_other_library_is_ignored(tmp_path):
    # CCCC is a cell only in the antibody library; it must not become a GEX cell.
    path = write_molecule_info(tmp_path / "m.h5", pass_filter=[(0, 0, 0), (1, 1, 0), (2, 0, 0)])
    assert list(read_molecule_info(path).cell_barcodes) == ["AAAA-1", "GGGG-1"]


def test_saturation_from_all_barcodes(molecule_info):
    h = read_molecule_info(molecule_info)
    # 8 GEX molecules, 18 GEX reads
    assert h.total_molecule_reads == 18
    assert expected_saturation(h.all_barcode_umi, [1.0])[0] == pytest.approx(1 - 8 / 18)


def test_raw_read_pairs_sums_gex_libraries(molecule_info):
    assert raw_read_pairs(molecule_info) == 1234


def test_library_idx_is_position_in_library_info(tmp_path):
    # file_version 3 stores library_id as a string; library_idx indexes the list.
    libs = [
        {"library_id": "0", "library_type": "Gene Expression", "gem_group": 1},
        {"library_id": "1", "library_type": "Antibody Capture", "gem_group": 1},
    ]
    check_synthetic(read_molecule_info(write_molecule_info(tmp_path / "m.h5", libraries=libs)))
