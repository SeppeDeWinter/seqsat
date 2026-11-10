#!/usr/bin/env bash
# Download the 10k Human Brain Nuclei GEM-X Epi Multiome dataset (Cell Ranger ARC 2.2.0).
# https://www.10xgenomics.com/datasets/multiome-gemx-10k-human-brain
#
# Total size ~3.5 GB (atac_fragments.tsv.gz alone is ~2.9 GB).
# Re-running resumes partial downloads (curl -C -) and skips completed files.
#
# Usage: scripts/download_test_data.sh [DEST_DIR]   (default: data/brain10k)

set -euo pipefail

DEST="${1:-$(dirname "$0")/../data/brain10k}"
BASE="https://cf.10xgenomics.com/samples/cell-arc/2.2.0/10k_Human_Brain_MO_gemx/10k_Human_Brain_MO_gemx"

# Files without a sample prefix locally, so the directory looks like an ARC outs/ dir.
FILES=(
    web_summary.html
    summary.csv
    per_barcode_metrics.csv
    gex_molecule_info.h5
    atac_fragments.tsv.gz
    atac_fragments.tsv.gz.tbi
)

mkdir -p "$DEST"

for f in "${FILES[@]}"; do
    url="${BASE}_${f}"
    out="${DEST}/${f}"
    # Some files (web_summary.html) are served without Content-Length; tolerate that.
    remote_size=$(curl -sIL "$url" | { grep -i '^content-length' || true; } | tail -1 | tr -dc '0-9')
    if [[ -f "$out" && -n "$remote_size" && "$(stat -c %s "$out")" == "$remote_size" ]]; then
        echo "ok      $f"
        continue
    fi
    # Size unknown: resuming a complete file gives HTTP 416, so keep a non-empty local copy.
    # Delete the file to force a fresh download.
    if [[ -s "$out" && -z "$remote_size" ]]; then
        echo "ok      $f (remote size unknown; keeping local copy)"
        continue
    fi
    echo "fetch   $f"
    # -C - fails with 416 if the file is already complete; the size check above avoids that.
    curl -fL --retry 5 --retry-delay 10 -C - -o "$out" "$url"
done

echo "Done: $DEST"
ls -l "$DEST"
