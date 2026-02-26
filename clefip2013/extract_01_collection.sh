#!/usr/bin/env bash
# Extract CLEF-IP 2013 document collection (01) from 7z split archives.
#
# --- Download (do this first if you don't have the 7z files) ---
#   cd /path/to/sparse_retriever/clefip2013
#   wget -O 01_document_collection.tgz "https://researchdata.tuwien.at/records/nw2xc-41j75/files/01_document_collection.tgz?download=1"
#   tar -xf 01_document_collection.tgz
#   # Then run this script (from repo root or clefip2013/).
#
# --- Extract 7z (this script) ---
# Run from repo root or from clefip2013/. Requires: p7zip-full (7z).
# Disk: ensure ~15GB+ free for 01_extracted/ (collection is large).
#
# Usage:
#   cd /path/to/sparse_retriever && bash clefip2013/extract_01_collection.sh
#   # or
#   cd /path/to/sparse_retriever/clefip2013 && bash extract_01_collection.sh
#
# Output: clefip2013/01_document_collection/01_extracted/
# Then: python evaluate.py --model_name bm25 --clefip_doc_root /path/to/clefip2013/01_document_collection/01_extracted

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTION_DIR="$(cd "$SCRIPT_DIR/01_document_collection" && pwd)"
OUT_DIR="$COLLECTION_DIR/01_extracted"

if ! command -v 7z &>/dev/null; then
  echo "Error: 7z not found. Install p7zip-full, e.g.:"
  echo "  Ubuntu/Debian: sudo apt-get install p7zip-full"
  echo "  CentOS/RHEL:   sudo yum install p7zip p7zip-plugins"
  echo "  conda:         conda install -c conda-forge p7zip"
  exit 1
fi

if [[ ! -f "$COLLECTION_DIR/document_collection_clef-ip-2012-ep0.7z.001" ]]; then
  echo "Error: 01_document_collection not found or incomplete. Expected:"
  echo "  $COLLECTION_DIR/document_collection_clef-ip-2012-ep0.7z.001 (and .002–.008)"
  echo "  $COLLECTION_DIR/document_collection_clef-ip-2012-ep1.7z.001 (and .002–.003)"
  echo "  $COLLECTION_DIR/document_collection_clef-ip-2012-wo.7z.001 (and .002–.004)"
  exit 1
fi

mkdir -p "$OUT_DIR"
cd "$COLLECTION_DIR"

echo "Extracting ep0 (8 parts)..."
7z x -y "document_collection_clef-ip-2012-ep0.7z.001" -o"$OUT_DIR"

echo "Extracting ep1 (3 parts)..."
7z x -y "document_collection_clef-ip-2012-ep1.7z.001" -o"$OUT_DIR"

echo "Extracting wo (4 parts)..."
7z x -y "document_collection_clef-ip-2012-wo.7z.001" -o"$OUT_DIR"

echo "Extracting dtds.7z..."
7z x -y "dtds.7z" -o"$OUT_DIR"

echo "Done. Extracted to: $OUT_DIR"
echo "Directory layout:"
ls -la "$OUT_DIR" 2>/dev/null | head -20

# Verify: 01 collection has EP (EPO) and WO (WIPO); dtds may be inside or alongside
echo ""
echo "Verification:"
OK=1
for sub in EP WO ep0 ep1 wo; do
  if [[ -d "$OUT_DIR/$sub" ]]; then
    echo "  [OK] $OUT_DIR/$sub exists"
  else
    if [[ "$sub" == "EP" ]] || [[ "$sub" == "WO" ]]; then
      echo "  [MISSING] $OUT_DIR/$sub (expected from ep0/ep1 or wo)"
      OK=0
    fi
  fi
done
XML_COUNT=$(find "$OUT_DIR" -type f -name "*.xml" 2>/dev/null | wc -l)
echo "  XML files under $OUT_DIR: $XML_COUNT"
if [[ "$XML_COUNT" -lt 1000 ]]; then
  echo "  [WARN] Expected hundreds of thousands of XMLs; re-run if extraction was interrupted."
  OK=0
fi
if [[ $OK -eq 1 ]]; then
  echo "  All expected top-level dirs present."
else
  echo "  Some content missing; re-run script and check for 7z errors (ep0, ep1, wo, dtds)."
fi
echo ""
echo "Use for evaluation:"
echo "  python evaluate.py --model_name bm25 --clefip_doc_root $OUT_DIR"
