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
echo ""
echo "Use for evaluation:"
echo "  python evaluate.py --model_name bm25 --dataset clefip --clefip_doc_root $OUT_DIR"
