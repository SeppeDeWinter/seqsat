#!/usr/bin/env bash
# Download public 10x Genomics datasets used for testing, into data/<name>/.
# Files are saved without the sample prefix, so each directory looks like an outs/ dir.
# Re-running resumes partial downloads (curl -C -) and skips completed files.
#
# Usage: scripts/download_test_data.sh [NAME ...]   (default: brain10k)
#
# Datasets:
#   brain10k     cellranger-arc 2.2.0, 10k Human Brain Nuclei GEM-X Multiome   (~3.5 GB)
#   pbmc10k_gex  cellranger 6.1.0, 10k PBMC 3' NextGEM, Chromium X             (~0.4 GB)
#   pbmc1k_gex   cellranger 3.0.0, 1k PBMC 3' v3                               (~70 MB)
#   pbmc1k_atac  cellranger-atac 2.0.0, 1k PBMC NextGEM                        (~0.25 GB)
#   all          every dataset above

set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"
S=https://cf.10xgenomics.com/samples

dataset() {
    # Prints: base URL, then the file names.
    case "$1" in
        brain10k)
            echo "$S/cell-arc/2.2.0/10k_Human_Brain_MO_gemx/10k_Human_Brain_MO_gemx"
            echo web_summary.html summary.csv per_barcode_metrics.csv gex_molecule_info.h5 \
                atac_fragments.tsv.gz atac_fragments.tsv.gz.tbi
            ;;
        pbmc10k_gex)
            echo "$S/cell-exp/6.1.0/10k_PBMC_3p_nextgem_Chromium_X/10k_PBMC_3p_nextgem_Chromium_X"
            echo web_summary.html metrics_summary.csv molecule_info.h5
            ;;
        pbmc1k_gex)
            echo "$S/cell-exp/3.0.0/pbmc_1k_v3/pbmc_1k_v3"
            echo web_summary.html metrics_summary.csv molecule_info.h5
            ;;
        pbmc1k_atac)
            echo "$S/cell-atac/2.0.0/atac_pbmc_1k_nextgem/atac_pbmc_1k_nextgem"
            echo web_summary.html summary.csv singlecell.csv fragments.tsv.gz
            ;;
        *)
            echo "unknown dataset: $1" >&2
            return 1
            ;;
    esac
}

fetch() {
    local url="$1" out="$2" f
    f="$(basename "$out")"
    # Some files (web_summary.html) are served without Content-Length; tolerate that.
    local remote_size
    remote_size=$(curl -sIL "$url" | { grep -i '^content-length' || true; } | tail -1 | tr -dc '0-9')
    if [[ -f "$out" && -n "$remote_size" && "$(stat -c %s "$out")" == "$remote_size" ]]; then
        echo "ok      $f"
        return
    fi
    # Size unknown: resuming a complete file gives HTTP 416, so keep a non-empty local copy.
    # Delete the file to force a fresh download.
    if [[ -s "$out" && -z "$remote_size" ]]; then
        echo "ok      $f (remote size unknown; keeping local copy)"
        return
    fi
    echo "fetch   $f"
    curl -fL --retry 5 --retry-delay 10 -C - -o "$out" "$url"
}

names=("$@")
[[ ${#names[@]} -eq 0 ]] && names=(brain10k)
[[ "${names[0]}" == all ]] && names=(brain10k pbmc10k_gex pbmc1k_gex pbmc1k_atac)

for name in "${names[@]}"; do
    spec="$(dataset "$name")"
    base="$(sed -n 1p <<<"$spec")"
    read -ra files <<<"$(sed -n 2p <<<"$spec")"
    dest="$DATA_DIR/$name"
    mkdir -p "$dest"
    echo "== $name -> $dest"
    for f in "${files[@]}"; do
        fetch "${base}_${f}" "$dest/$f"
    done
done
