#!/usr/bin/env python3
"""
Phase 1: Pre-compute spaCy spans for all doc-sections and unit types.

Run ONCE per corpus.  All model-specific embedding jobs (Phase 2) then reuse the cached spans, avoiding repeated spaCy processing.

For each (doc, section, unit), the raw section text (not yet formatted with model-specific prefixes) is:
  1. Normalized (semicolons, reference numerals)
  2. Truncated to --max_visible_chars (default 3000, conservatively larger than any model's 512-token visible window)
  3. Processed with spaCy → spans are extracted per unit type
  4. Saved to {cache_dir}/{unit}_spans.jsonl.gz

For the abstract section, title and body are processed SEPARATELY so that Phase 2 can adjust char offsets based on each model's formatting (title{sep}[abstract] body  vs  title{sep}body).

Usage:
    # Build cache for all 3 units using cached corpus:
    python 0cache_spacy_spans.py --data_dir ./cache_epo --cache_dir ./span_cache

    # Build cache directly from EPO:
    python 0cache_spacy_spans.py --data_dir /path/to/EPO/epo_en \\
        --epo_year_min 2000 --epo_year_max 2020 --max_docs 300000 --epo_sample_seed 42 \\
        --cache_dir ./span_cache

    # Then run Phase 2 (embedding creation) pointing to this cache:
    python 1create_N_embeddings.py --span_cache_dir ./span_cache ...
"""

from typing import Literal  # noqa: F401  (kept for downstream type imports)


import os
import argparse
import time
import json
import spacy
from tqdm import tqdm

import utils
from utils import (
    load_corpus,
    load_corpus_epo,
    normalize_text_for_pipeline,
    extract_char_spans_from_spacy,
    save_span_cache,
)

SECTIONS = ("abstract", "claim", "invention")
UNITS = ("spacy_token", "spacy_sentence", "noun_chunk")


def main():
    ap = argparse.ArgumentParser(description="Phase 1: build spaCy span cache for all units.")
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Corpus directory (content/documents.json for JSONL, or EPO year subdirs)")
    ap.add_argument("--cache_dir", type=str, default="./span_cache",
                    help="Output directory for cached spans")
    ap.add_argument("--max_visible_chars", type=int, default=3000,
                    help="Truncate section text to this many chars before spaCy. "
                         "Must be >= any model's visible window (~1500-2500 chars for 512 tokens). Default: 3000.")
    ap.add_argument("--max_docs", type=int, default=300000)
    ap.add_argument("--epo_year_min", type=int, default=None)
    ap.add_argument("--epo_year_max", type=int, default=None)
    ap.add_argument("--epo_sample_seed", type=int, default=42)
    ap.add_argument("--spacy_model", type=str, default="sci_lg",
                    choices=["sm", "md", "lg", "sci_sm", "sci_md", "sci_lg"])
    ap.add_argument("--spacy_batch_size", type=int, default=512,
                    help="Batch size for nlp.pipe(). Larger is faster if memory allows. Default: 512.")
    ap.add_argument("--spacy_n_process", type=int, default=1,
                    help="Number of processes for nlp.pipe() (multi-CPU). Default: 1. Try 4 or 8 on a many-core node.")
    ap.add_argument("--units", type=str, nargs="+", default=list(UNITS),
                    choices=list(UNITS),
                    help="Which unit types to cache. Default: all three.")
    ap.add_argument("--save_corpus_to", type=str, default=None,
                    help="When loading from EPO, save the loaded corpus to DIR/content/documents.json so future runs can use DATA_DIR=DIR for fast load. Ignored when using cached corpus.")
    args = ap.parse_args()

    # Load spaCy
    if args.spacy_model.startswith("sci_"):
        spacy_name = f"en_core_sci_{args.spacy_model[4:]}"
    else:
        spacy_name = f"en_core_web_{args.spacy_model}"
    disable = ["ner", "textcat", "lemmatizer"]
    print(f"Loading spaCy model: {spacy_name} ...")
    nlp = spacy.load(spacy_name, disable=disable)
    nlp.max_length = args.max_visible_chars + 10000
    utils.NLP = nlp
    print(f"✓ spaCy loaded: {spacy_name}")

    # Load corpus
    print("\nLoading corpus...")
    t_load_start = time.time()
    documents_path = os.path.join(args.data_dir, "content/documents.json")
    if os.path.isfile(documents_path):
        documents = load_corpus(documents_path)
        print(f"Loaded {len(documents):,} documents from {documents_path}")
    else:
        documents = load_corpus_epo(
            args.data_dir,
            max_docs=args.max_docs,
            year_min=args.epo_year_min,
            year_max=args.epo_year_max,
            sample_seed=args.epo_sample_seed,
        )
        if not documents:
            raise FileNotFoundError(f"No corpus found at {args.data_dir}")
        print(f"Loaded {len(documents):,} documents from EPO dir {args.data_dir}")
        # Optional: save corpus for future fast load
        if args.save_corpus_to:
            out_dir = os.path.join(args.save_corpus_to, "content")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "documents.json")
            with open(out_path, "w") as f:
                for doc_id, doc in documents.items():
                    line = json.dumps({
                        "dnum": doc_id,
                        "Content": {
                            "title": doc.get("title", ""),
                            "pa01": doc.get("abstract", ""),
                            "c-en-001": doc.get("claim", ""),
                            "p001": doc.get("invention", ""),
                        },
                    }, ensure_ascii=False) + "\n"
                    f.write(line)
            print(f"Saved corpus to {out_path} for future fast load (use --data_dir {args.save_corpus_to})")
    t_load_elapsed = time.time() - t_load_start
    print(f"  Load time: {t_load_elapsed:.1f}s")

    if args.max_docs and len(documents) > args.max_docs:
        doc_items = list(documents.items())[:args.max_docs]
        documents = dict(doc_items)
        print(f"Truncated to {len(documents):,} documents")

    # Build list of (doc_id, section, sub_part, normalized_truncated_text)
    print("\nPreparing texts for spaCy...")
    t_prep_start = time.time()
    text_items = []
    max_c = args.max_visible_chars
    for doc_id, doc in documents.items():
        title = normalize_text_for_pipeline((doc.get("title", "") or "").strip())
        abstract = normalize_text_for_pipeline((doc.get("abstract", "") or "").strip())
        claim = normalize_text_for_pipeline((doc.get("claim", "") or "").strip())
        invention = normalize_text_for_pipeline((doc.get("invention", "") or "").strip())

        if abstract or title:
            if title:
                text_items.append((doc_id, "abstract", "title", title[:max_c]))
            if abstract:
                text_items.append((doc_id, "abstract", "body", abstract[:max_c]))
        if claim:
            text_items.append((doc_id, "claim", "body", claim[:max_c]))
        if invention:
            text_items.append((doc_id, "invention", "body", invention[:max_c]))

    print(f"Total text items: {len(text_items):,}")
    t_prep_elapsed = time.time() - t_prep_start
    print(f"  Prepare time: {t_prep_elapsed:.1f}s")

    # Stream spaCy + extract spans in a single pass (avoids materializing all Doc objects).
    # For 300K docs × 3 sections ≈ 900K Docs, holding them all in RAM can hit tens of GB.
    n_process = max(1, int(getattr(args, "spacy_n_process", 1)))
    print(f"\nRunning spaCy + extracting spans (batch_size={args.spacy_batch_size}, n_process={n_process})...")
    pipe_kw = {"batch_size": args.spacy_batch_size, "as_tuples": True}
    if n_process > 1:
        pipe_kw["n_process"] = n_process

    entries_by_unit = {u: [] for u in args.units}
    t0 = time.time()
    gen = nlp.pipe(((item[3], idx) for idx, item in enumerate(text_items)), **pipe_kw)
    for doc_spacy, idx in tqdm(gen, total=len(text_items), desc="  spacy+extract"):
        doc_id, section, sub_part, _ = text_items[idx]
        is_title = (sub_part == "title")
        is_body = (sub_part == "body" and section == "abstract")
        for unit in args.units:
            char_spans = extract_char_spans_from_spacy(
                doc_spacy, section, unit,
                is_abstract_title=is_title,
                is_abstract_body=is_body,
            )
            entries_by_unit[unit].append({
                "d": doc_id,
                "s": section,
                "p": sub_part,
                "sp": char_spans,
            })
        # doc_spacy goes out of scope on next iteration → eligible for GC
    t_extract = time.time() - t0
    print(f"✓ spaCy + extract done in {t_extract:.1f}s ({len(text_items) / max(t_extract, 0.01):.0f} texts/s)")
    for unit in args.units:
        total_spans = sum(len(e["sp"]) for e in entries_by_unit[unit])
        print(f"  {unit}: {total_spans:,} spans")
        save_span_cache(args.cache_dir, unit, entries_by_unit[unit])
        del entries_by_unit[unit]

    print(f"\n{'='*60}")
    print(f"✓ Span cache saved to: {os.path.abspath(args.cache_dir)}")
    print(f"  Use with: python 1create_N_embeddings.py --span_cache_dir {args.cache_dir} ...")


if __name__ == "__main__":
    main()
