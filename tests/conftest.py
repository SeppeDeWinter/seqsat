import json
from pathlib import Path

import h5py
import numpy as np
import pytest

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# Synthetic molecule_info content.
#   features: 0..2 Gene Expression, 3 Antibody Capture
#   libraries: 0 Gene Expression, 1 Antibody Capture (both gem group 1)
#   barcodes: 0 AAAA, 1 CCCC, 2 GGGG, 3 TTTT ; cells = AAAA, GGGG (via pass_filter)
SYN_BARCODES = [b"AAAA", b"CCCC", b"GGGG", b"TTTT"]
SYN_FEATURE_TYPES = [b"Gene Expression"] * 3 + [b"Antibody Capture"]
SYN_LIBRARIES = [
    {"library_id": 0, "library_type": "Gene Expression", "gem_group": 1, "target_set_name": None},
    {"library_id": 1, "library_type": "Antibody Capture", "gem_group": 1, "target_set_name": None},
]
# (barcode_idx, feature_idx, library_idx, count, umi_type), sorted by barcode_idx
SYN_MOLECULES = [
    # AAAA (cell): gene0 x2 molecules (1, 3 reads), gene1 (2), antibody molecule (dropped)
    (0, 0, 0, 1, 1),
    (0, 0, 0, 3, 0),
    (0, 1, 0, 2, 1),
    (0, 3, 1, 9, 1),
    # CCCC (not a cell): gene2 (4 reads)
    (1, 2, 0, 4, 1),
    # GGGG (cell): gene1 (1), gene2 (1), gene2 (5); GEX feature in antibody library (dropped)
    (2, 1, 0, 1, 0),
    (2, 2, 0, 1, 1),
    (2, 2, 0, 5, 1),
    (2, 0, 1, 7, 1),
    # TTTT (not a cell): gene0 (1)
    (3, 0, 0, 1, 1),
]


def write_molecule_info(
    path: Path,
    molecules: list[tuple[int, int, int, int, int]] = SYN_MOLECULES,
    pass_filter: list[tuple[int, int, int]] | None = None,
    libraries: list[dict] = SYN_LIBRARIES,
) -> Path:
    """Write a molecule_info.h5 with the same layout as Cell Ranger ARC 2.2.0."""
    m = np.asarray(molecules, dtype=np.int64)
    if pass_filter is None:
        pass_filter = [(0, 0, 0), (2, 0, 0)]
    with h5py.File(path, "w") as f:
        f.attrs["file_version"] = 6
        f.attrs["filetype"] = "molecule"
        f.create_dataset("barcode_idx", data=m[:, 0].astype(np.uint64))
        f.create_dataset("feature_idx", data=m[:, 1].astype(np.uint32))
        f.create_dataset("library_idx", data=m[:, 2].astype(np.uint16))
        f.create_dataset("count", data=m[:, 3].astype(np.uint32))
        f.create_dataset("umi_type", data=m[:, 4].astype(np.uint32))
        f.create_dataset("umi", data=np.arange(len(m), dtype=np.uint32))
        f.create_dataset("gem_group", data=np.ones(len(m), dtype=np.uint16))
        f.create_dataset("barcodes", data=np.asarray(SYN_BARCODES, dtype="S4"))
        f.create_dataset("barcode_info/pass_filter", data=np.asarray(pass_filter, dtype=np.uint64))
        f.create_dataset("barcode_info/genomes", data=np.asarray([b"GRCh38"]))
        n_feat = len(SYN_FEATURE_TYPES)
        f.create_dataset("features/feature_type", data=np.asarray(SYN_FEATURE_TYPES, dtype="S256"))
        f.create_dataset("features/id", data=np.asarray([f"F{i}".encode() for i in range(n_feat)]))
        f.create_dataset(
            "features/name", data=np.asarray([f"g{i}".encode() for i in range(n_feat)])
        )
        f.create_dataset("features/genome", data=np.asarray([b"GRCh38"] * n_feat))
        f.create_dataset(
            "library_info", data=np.asarray([json.dumps(libraries).encode()], dtype="S262144")
        )
        metrics = {"libraries": {"0": {"raw_read_pairs": 1234}, "1": {"raw_read_pairs": 99}}}
        f.create_dataset("metrics_json", data=json.dumps(metrics), dtype=h5py.string_dtype())
    return path


@pytest.fixture
def molecule_info(tmp_path: Path) -> Path:
    return write_molecule_info(tmp_path / "molecule_info.h5")


def _dataset(name: str) -> Path:
    path = DATA_ROOT / name
    if not path.is_dir() or not any(path.iterdir()):
        pytest.skip(f"test data not found in {path}; run scripts/download_test_data.sh {name}")
    return path


@pytest.fixture(scope="session")
def brain10k() -> Path:
    """cellranger-arc 2.2.0 outs."""
    return _dataset("brain10k")


@pytest.fixture(scope="session")
def pbmc10k_gex() -> Path:
    """cellranger 6.1.0 count outs."""
    return _dataset("pbmc10k_gex")


@pytest.fixture(scope="session")
def pbmc1k_gex() -> Path:
    """cellranger 3.0.0 count outs (molecule_info file_version 3)."""
    return _dataset("pbmc1k_gex")


@pytest.fixture(scope="session")
def pbmc1k_atac() -> Path:
    """cellranger-atac 2.0.0 count outs."""
    return _dataset("pbmc1k_atac")
