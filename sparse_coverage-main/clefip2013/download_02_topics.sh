#!/usr/bin/env bash
# Download and extract CLEF-IP 2013 topics & qrels (02_topics).
#
# Source: TU Wien Research Data repository (open, CC BY-NC-SA 3.0)
#   https://researchdata.tuwien.at/records/nw2xc-41j75
#
# What this script downloads (15.8 MiB total):
#   02_topics.tgz  — outer archive containing:
#     topics_clef-ip-2013-clms-psg.tgz   — 149 test topics (EN/DE/FR XML claim files + meta .txt)
#     relass_clef-ip-2013-clsm-to-psg.zip — qrels for claims-to-passages task
#     training_clef-ip-2013-clms-psg.zip  — training topics (not used for evaluation)
#
# After extraction the layout expected by load_clefip.py is:
#   clefip2013/02_topics/
#     clef-ip-2013-clms-psg-TEST/
#       clef-ip-2013-clms-psg-TEST.txt         ← topic meta list
#       tfiles/                                 ← per-topic XML claim files
#     2013-clef-ip-clsm-to-psg-qrels/
#       clef-ip-2013-clms-psg-TEST.en.qrels    ← EN relevance assessments
#
# Usage:
#   cd /path/to/sparse_retriever && bash clefip2013/download_02_topics.sh
#   # or
#   cd /path/to/sparse_retriever/clefip2013 && bash download_02_topics.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPICS_DIR="$SCRIPT_DIR/02_topics"
OUTER_TGZ="$SCRIPT_DIR/02_topics.tgz"
TU_URL="https://researchdata.tuwien.at/records/nw2xc-41j75/files/02_topics.tgz"

# ── 1. Download outer archive if not already present ──────────────────────────
if [[ -f "$OUTER_TGZ" ]]; then
    echo "[skip] $OUTER_TGZ already exists — skipping download."
else
    echo "Downloading 02_topics.tgz (~15.8 MiB) from TU Wien Research Data..."
    if command -v wget &>/dev/null; then
        wget --show-progress -O "$OUTER_TGZ" "$TU_URL"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$OUTER_TGZ" "$TU_URL"
    else
        echo "Error: neither wget nor curl found. Install one and retry." >&2
        exit 1
    fi
fi

# ── 2. Verify MD5 (from TU Wien) ──────────────────────────────────────────────
EXPECTED_MD5="85fb3bfbafbd55824e79b1a3759abc08"
if command -v md5sum &>/dev/null; then
    ACTUAL_MD5=$(md5sum "$OUTER_TGZ" | awk '{print $1}')
    if [[ "$ACTUAL_MD5" == "$EXPECTED_MD5" ]]; then
        echo "[ok] MD5 checksum verified."
    else
        echo "[warn] MD5 mismatch: expected $EXPECTED_MD5, got $ACTUAL_MD5"
        echo "       The file may be corrupt; consider re-downloading."
    fi
fi

# ── 3. Extract outer archive into clefip2013/ ──────────────────────────────────
echo "Extracting $OUTER_TGZ → $SCRIPT_DIR ..."
tar -xzf "$OUTER_TGZ" -C "$SCRIPT_DIR"

# ── 4. Extract test topics (XML claim files + meta) ───────────────────────────
INNER_TOPICS_TGZ="$TOPICS_DIR/topics_clef-ip-2013-clms-psg.tgz"
if [[ -f "$INNER_TOPICS_TGZ" ]]; then
    echo "Extracting topics (claim XML files + meta)..."
    tar -xzf "$INNER_TOPICS_TGZ" -C "$TOPICS_DIR"
else
    echo "Error: $INNER_TOPICS_TGZ not found after outer extraction." >&2
    exit 1
fi

# ── 5. Extract qrels ──────────────────────────────────────────────────────────
QRELS_ZIP="$TOPICS_DIR/relass_clef-ip-2013-clsm-to-psg.zip"
if [[ -f "$QRELS_ZIP" ]]; then
    echo "Extracting qrels..."
    if command -v unzip &>/dev/null; then
        unzip -q -o "$QRELS_ZIP" -d "$TOPICS_DIR"
    else
        echo "Error: unzip not found. Install unzip and re-run." >&2
        exit 1
    fi
else
    echo "Error: $QRELS_ZIP not found after outer extraction." >&2
    exit 1
fi

# ── 6. Verify expected structure ──────────────────────────────────────────────
echo ""
echo "Verification:"
OK=1
META_FILE="$TOPICS_DIR/clef-ip-2013-clms-psg-TEST/clef-ip-2013-clms-psg-TEST.txt"
TFILES_DIR="$TOPICS_DIR/clef-ip-2013-clms-psg-TEST/tfiles"
QRELS_EN="$TOPICS_DIR/2013-clef-ip-clsm-to-psg-qrels/2013-clef-ip-QRELS-EN-claims-to-passages.txt"

for path in "$META_FILE" "$TFILES_DIR" "$QRELS_EN"; do
    if [[ -e "$path" ]]; then
        echo "  [OK] $path"
    else
        echo "  [MISSING] $path"
        OK=0
    fi
done

TOPIC_COUNT=$(wc -l < "$META_FILE" 2>/dev/null | tr -d ' ')
TFILE_COUNT=$(find "$TFILES_DIR" -name "*.xml" 2>/dev/null | wc -l)
QREL_LINES=$(wc -l < "$QRELS_EN" 2>/dev/null | tr -d ' ')
echo "  Topics in meta file:   $TOPIC_COUNT"
echo "  Topic XML files:       $TFILE_COUNT"
echo "  EN qrel lines:         $QREL_LINES"

if [[ $OK -eq 1 ]]; then
    echo "  All required files present."
else
    echo "  Some files missing — check extraction errors above."
    exit 1
fi

echo ""
echo "Use for evaluation:"
echo "  python evaluate.py --model_name bm25 \\"
echo "      --clefip_root $SCRIPT_DIR \\"
echo "      --clefip_sample_size 25000"
