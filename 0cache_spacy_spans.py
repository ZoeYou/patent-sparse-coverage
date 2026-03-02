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

from typing import Literal


import os
import argparse
import time
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
    ap.add_argument("--spacy_batch_size", type=int, default=256,
                    help="Batch size for nlp.pipe(). Default: 256.")
    ap.add_argument("--units", type=str, nargs="+", default=list[Literal['spacy_token', 'spacy_sentence', 'noun_chunk']](UNITS),
                    choices=list[Literal['spacy_token', 'spacy_sentence', 'noun_chunk']](UNITS),
                    help="Which unit types to cache. Default: all three.")
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

    if args.max_docs and len(documents) > args.max_docs:
        doc_items = list(documents.items())[:args.max_docs]
        documents = dict(doc_items)
        print(f"Truncated to {len(documents):,} documents")

    # Build list of (doc_id, section, sub_part, normalized_truncated_text)
    print("\nPreparing texts for spaCy...")
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

    # Run spaCy on all texts (once)
    print(f"\nRunning spaCy (batch_size={args.spacy_batch_size})...")
    texts_only = [item[3] for item in text_items]
    t0 = time.time()
    spacy_docs = list(nlp.pipe(texts_only, batch_size=args.spacy_batch_size))
    t_spacy = time.time() - t0
    print(f"✓ spaCy done in {t_spacy:.1f}s ({len(texts_only) / max(t_spacy, 0.01):.0f} texts/s)")
    del texts_only

    # Extract spans per unit
    for unit in args.units:
        print(f"\n{'='*60}")
        print(f"Extracting spans for unit: {unit}")
        t0 = time.time()
        cache_entries = []
        total_spans = 0
        for idx, (doc_id, section, sub_part, text) in enumerate(tqdm(text_items, desc=f"  {unit}")):
            doc_spacy = spacy_docs[idx]
            is_title = (sub_part == "title")
            is_body = (sub_part == "body" and section == "abstract")
            char_spans = extract_char_spans_from_spacy(
                doc_spacy, section, unit,
                is_abstract_title=is_title,
                is_abstract_body=is_body,
            )
            cache_entries.append({
                "d": doc_id,
                "s": section,
                "p": sub_part,
                "sp": char_spans,
            })
            total_spans += len(char_spans)
        t_extract = time.time() - t0
        print(f"  Total spans: {total_spans:,} ({t_extract:.1f}s)")
        save_span_cache(args.cache_dir, unit, cache_entries)
        del cache_entries

    print(f"\n{'='*60}")
    print(f"✓ Span cache saved to: {os.path.abspath(args.cache_dir)}")
    print(f"  Use with: python 1create_N_embeddings.py --span_cache_dir {args.cache_dir} ...")


if __name__ == "__main__":
    main()
