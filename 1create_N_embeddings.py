"""
Create contextual span embeddings for patent documents.

Default mode: claim2all (uses all three sections: title+abstract, claim, invention).
Input format is model-dependent (see utils.get_encoder_format_scheme, get_encoder_sep_for_model):
- section_tokens: abstract = "title {sep} [abstract] {abstract}", claim = "[claim] {claim}", invention = "[invention] {invention}"
- title_sep_only: abstract = "title {sep} {abstract}", claim/invention = plain text (no section tokens)

For each section:
- Step 1: Encode full text with model
- Step 2: Extract spans based on unit type (spacy_token, spacy_sentence, or noun_chunk)
- Step 3: Pool token embeddings for each span (mean or max)

Unit types (only these are supported):
- spacy_token: one embedding per spaCy token, with noun_chunks merged into one embedding per chunk; tokens inside a noun_chunk are not output separately.
- spacy_sentence: one embedding per sentence.
- noun_chunk: only noun-chunk spans (one embedding per noun phrase); tokens not in any noun_chunk are not included.

Output directory name includes: model name, unit, keep_cls, layer, and optionally _meanpool when keep_doc_mean=1.

Storage: embeddings are saved as .npy (uncompressed) for fast I/O and memmap-friendly loading in center-building;
metadata as .jsonl. For compression use system-level tools (e.g. zstd) or chunked files.

Use cases:
  1. Center-building only (e.g. EPO): default --save_metadata 0, no doc_id mapping needed.
  2. When this output will be used as the document corpus in evaluate (--embeddings_dir): use
     --save_metadata 1 so evaluate can build span->doc_id aggregation. Prior-art/clefip runs need this.

EPO epo_en: If data_dir does not contain content/documents.json, the script tries EPO layout:
  data_dir/<year>/*.txt. Each .txt is "FIELD ::: value" format (TITLE, ABSTR, DESCR, CLAIM1).
  Use --epo_year_min/--epo_year_max to restrict to a year range (e.g. 2000-2020) and --max_docs
  to sample documents (proportional per year); --epo_sample_seed for reproducibility.

Usage:
    python 1create_N_embeddings.py --layer last
    python 1create_N_embeddings.py --unit noun_chunk --keep_doc_mean 1
    python 1create_N_embeddings.py --data_dir /path/to/EPO/epo_en --save_metadata 0  # centers only
"""

import os
import json
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModel
import utils
from utils import (
    ensure_section_tokens,
    get_encoder_format_scheme,
    get_encoder_sep_for_model,
    load_corpus,
    load_corpus_epo,
    create_contextual_span_embeddings,
    load_span_cache,
    DEVICE,
)


SECTIONS = ("abstract", "claim", "invention")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./downstream/perf200/",
                       help="Directory containing the corpus (content/documents.json)")

    parser.add_argument("--model_path", type=str, default="ZoeYou/PatentMap-V0-SecPair-Claim",
                       help="Path to pretrained model or HuggingFace model ID")
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Batch size for encoding documents")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length for tokenizer (max_length parameter)")
    parser.add_argument("--max_section_chars", type=int, default=32768,
                       help="Max character length per section; text beyond this is truncated before tokenization and spaCy. "
                            "Encoder still sees only first max_length (512) tokens. Default: 32768.")

    parser.add_argument("--max_docs", type=int, default=None,
                       help="Maximum number of documents to process (for testing / EPO sampling)")
    parser.add_argument("--epo_year_min", type=int, default=None,
                       help="[EPO only] Minimum year (inclusive), e.g. 2000. Only year subdirs >= this are loaded.")
    parser.add_argument("--epo_year_max", type=int, default=None,
                       help="[EPO only] Maximum year (inclusive), e.g. 2020. Only year subdirs <= this are loaded.")
    parser.add_argument("--epo_sample_seed", type=int, default=None,
                       help="[EPO only] Random seed when sampling to max_docs (proportional per year). For reproducibility.")
    parser.add_argument("--max_spans", type=int, default=5000000,
                       help="Maximum number of spans (embeddings) to extract")

    parser.add_argument("--embed_dtype", type=str, default="float32", choices=["float32", "float16"],
                       help="Storage dtype for embeddings: float32 (default, full precision), float16 (half size, minimal quality loss for retrieval).")
    parser.add_argument("--save_metadata", type=int, default=0, choices=[0, 1],
                       help="Save metadata (0=no, 1=yes). Default: 0. Use 1 when this output will be used as the document corpus in evaluate "
                            "(--embeddings_dir), so span->doc_id aggregation can be built. Saved AFTER sampling.")
    parser.add_argument("--unit", type=str, default="spacy_token",
                       choices=["spacy_token", "spacy_sentence", "noun_chunk"],
                       help="Semantic unit: spacy_token (tokens + merged noun_chunks), "
                            "spacy_sentence (one per sentence), "
                            "noun_chunk (only noun-chunk spans, no standalone tokens).")
    parser.add_argument("--keep_cls", type=int, default=1, choices=[0, 1],
                       help="Whether to keep [CLS] token in output (1=yes, 0=no). Default: 1.")
    parser.add_argument("--keep_doc_mean", type=int, default=1, choices=[0, 1],
                       help="If 1, keep one extra span per doc-section = mean of all token embeddings (sequence-level vector). Default: 1. Both cls and doc_mean are then in metadata as span_kind.")
    parser.add_argument("--layer", type=str, default="last", choices=["last", "second_last"],
                       help="Which layer to use for embeddings: 'last' (default) for last layer, 'second_last' for second-to-last layer")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for embeddings. If not specified, will be auto-generated based on model and parameters.")
    parser.add_argument("--span_cache_dir", type=str, default=None,
                       help="Directory with pre-computed spaCy spans (from 0cache_spacy_spans.py). "
                            "When set, spaCy is NOT loaded or run; spans come from the cache. "
                            "Saves significant time when creating embeddings for multiple models.")
    
    # spaCy model parameters
    parser.add_argument("--spacy_model", type=str, default="sci_lg",
                        choices=["sm", "md", "lg", "sci_sm", "sci_md", "sci_lg"],
                        help="spaCy model: sm/md/lg (en_core_web_*), sci_sm/sci_md/sci_lg (en_core_sci_*). Default: sci_lg.")
    parser.add_argument("--spacy_batch_size", type=int, default=128,
                       help="Batch size for spaCy nlp.pipe(). Larger = faster, more memory. Default: 128.")
    parser.add_argument("--spacy_disable_extra", type=int, default=1, choices=[0, 1],
                       help="If 1 (default), disable lemmatizer for speed (attribute_ruler is kept for POS tags / noun_chunks). Set 0 to keep full pipeline.")
    args = parser.parse_args()

    # Load span cache if provided (Phase 2: skip spaCy entirely)
    span_cache = None
    if args.span_cache_dir:
        print(f"Loading span cache from {args.span_cache_dir} (unit={args.unit})...")
        span_cache = load_span_cache(args.span_cache_dir, args.unit)
        print(f"✓ Span cache loaded: {len(span_cache):,} entries — spaCy will NOT be loaded")
    else:
        # Initialize global spaCy model (utils.NLP) so process_doc_batch can use it
        import spacy
        print("Loading spaCy model...")
        if args.spacy_model.startswith("sci_"):
            spacy_model_name = f"en_core_sci_{args.spacy_model[4:]}"
        else:
            spacy_model_name = f"en_core_web_{args.spacy_model}"
        disable = ["ner", "textcat"]
        if args.spacy_disable_extra:
            disable.append("lemmatizer")
        try:
            utils.NLP = spacy.load(spacy_model_name, disable=disable)
        except OSError:
            print(f"   Run: python -m spacy download {spacy_model_name}")
            raise
        utils.NLP.max_length = args.max_section_chars + 10000
        utils.SPACY_PIPE_BATCH_SIZE = getattr(args, "spacy_batch_size", 128)
        assert utils.NLP.has_pipe("parser"), "spaCy parser is required for noun_chunks extraction"
        print(f"✓ spaCy loaded: {spacy_model_name}, pipe batch_size={utils.SPACY_PIPE_BATCH_SIZE}, disabled={disable}")
    
    # Extract model name from model_path (handle both paths and HuggingFace IDs)
    model_name = args.model_path.strip("/").split("/")[-1].replace("/", "_").replace("\\", "_")
    
    # Load model and tokenizer
    print("\n[1/4] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Build output directory name with all key parameters
    keep_cls = bool(args.keep_cls)
    cls_suffix = "cls" if keep_cls else "nocls"
    layer_suffix = args.layer
    unit_suffix = args.unit  # spacy_token, spacy_sentence, noun_chunk
    keep_doc_mean = bool(args.keep_doc_mean)

    if args.output_dir is None:
        # Format: embeddings_{model_name}_{unit}_{cls}_{layer}[_meanpool][_fp16]
        output_dir = f"./embeddings_{model_name}_{unit_suffix}_{cls_suffix}_{layer_suffix}"
        if keep_doc_mean:
            output_dir += "_meanpool"
        if args.embed_dtype == "float16":
            output_dir += "_fp16"
    else:
        output_dir = args.output_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # .npy = fast write/read and memmap-friendly for center-building; metadata stays .jsonl
    # For compression use system-level (e.g. zstd) or chunked files; avoid np.savez_compressed at scale
    file_ext = "npy"
    save_dtype = np.float32 if args.embed_dtype == "float32" else np.float16

    # Build output filenames per section
    output_files = {}
    metadata_files = {}
    for section in SECTIONS:
        output_files[section] = os.path.join(output_dir, f"{section}_{args.unit}.{file_ext}")
        metadata_files[section] = os.path.join(output_dir, f"{section}_{args.unit}_metadata.jsonl")
    
    print(f"=" * 80)
    print(f"Creating Contextual Span Embeddings")
    print(f"Device: {DEVICE}")
    print(f"Model: {args.model_path}")
    print(f"Unit: {args.unit}")
    print(f"Layer: {args.layer}")
    print(f"Batch size: {args.batch_size}, Max length: {args.max_length}, Max section chars: {args.max_section_chars}")
    print(f"Keep [CLS]: {keep_cls}")
    print(f"Keep doc mean: {keep_doc_mean}")
    print(f"Span cache: {args.span_cache_dir or 'none (spaCy at runtime)'}")
    print(f"Output directory: {output_dir}")
    print(f"=" * 80)
    
    # Ensure section tokens are in vocabulary (for section_tokens format; no-op for title_sep_only if model has no section tokens)
    ensure_section_tokens(tokenizer, model)

    format_scheme = get_encoder_format_scheme(args.model_path)
    sep = get_encoder_sep_for_model(args.model_path, tokenizer)
    print(f"   Encoder format scheme: {format_scheme}, sep: {repr(sep)} (from model {args.model_path})")

    model.to(DEVICE)
    model.eval()

    # Load corpus
    print("\n[2/4] Loading corpus...")
    documents_path = os.path.join(args.data_dir, "content/documents.json")
    if os.path.isfile(documents_path):
        documents = load_corpus(documents_path)
        print(f"   Loaded {len(documents):,} documents from {documents_path}")
    else:
        # EPO epo_en: year subdirs with .txt files (FIELD ::: value format)
        documents = load_corpus_epo(
            args.data_dir,
            max_docs=args.max_docs,
            year_min=getattr(args, "epo_year_min", None),
            year_max=getattr(args, "epo_year_max", None),
            sample_seed=getattr(args, "epo_sample_seed", None),
        )
        if not documents:
            raise FileNotFoundError(
                f"No corpus found. Expected either:\n  - {documents_path} (JSONL)\n  - or EPO layout: {args.data_dir}/<year>/*.txt"
            )
        print(f"   Loaded {len(documents):,} documents from EPO dir {args.data_dir}")
        if getattr(args, "epo_year_min", None) is not None or getattr(args, "epo_year_max", None) is not None:
            print(f"   Year range: [{getattr(args, 'epo_year_min', 'any')}, {getattr(args, 'epo_year_max', 'any')}]")
        if args.max_docs and getattr(args, "epo_sample_seed", None) is not None:
            print(f"   Sampled to max_docs={args.max_docs} (seed={args.epo_sample_seed})")
    print(f"Loaded {len(documents):,} documents")

    # Create embeddings (same format_scheme and sep as query time in baselines for coherence)
    print(f"\n[3/4] Creating contextual span embeddings...")
    embeddings_by_section, metadata = create_contextual_span_embeddings(
        documents, model, tokenizer, unit=args.unit,
        max_docs=args.max_docs, batch_size=args.batch_size, max_length=args.max_length,
        keep_cls=keep_cls, layer=args.layer, format_scheme=format_scheme, sep=sep,
        keep_doc_mean=keep_doc_mean,
        max_section_chars=args.max_section_chars,
        max_spans=args.max_spans,
        span_cache=span_cache,
    )
    
    # Separate metadata by section for sampling and saving.
    # Order within each section must match embeddings_by_section row order (asserted in utils.create_contextual_span_embeddings).
    metadata_by_section = {s: [] for s in SECTIONS}
    for meta in metadata:
        section = meta["section"]
        if section in metadata_by_section:
            metadata_by_section[section].append(meta)
    # Rebuild flat metadata in section order (abstract, claim, invention) so sample display and downstream use are consistent
    metadata = [m for sec in SECTIONS for m in metadata_by_section.get(sec, [])]

    # Truncate/sample per section if needed
    total_spans = sum(len(emb) if emb is not None else 0 for emb in embeddings_by_section.values())
    if total_spans > args.max_spans:
        print(f"\n{'='*80}")
        print(f"Sampling {args.max_spans:,} spans from {total_spans:,} total spans (across all sections)")
        print(f"Strategy: random sampling (proportional per section)")
        
        # Calculate per-section sampling budget (proportional to section size)
        section_sizes = {s: len(emb) if emb is not None else 0 for s, emb in embeddings_by_section.items()}
        total_size = sum(section_sizes.values())
        
        # Allocate budget proportionally, but ensure at least some from each non-empty section
        section_budgets = {}
        remaining_budget = args.max_spans
        for section in SECTIONS:
            if section_sizes[section] > 0:
                # Proportional allocation
                budget = max(1, int(args.max_spans * section_sizes[section] / total_size))
                section_budgets[section] = min(budget, section_sizes[section])
                remaining_budget -= section_budgets[section]
        
        # Distribute remaining budget to largest sections
        if remaining_budget > 0:
            sorted_sections = sorted(section_budgets.items(), key=lambda x: section_sizes[x[0]], reverse=True)
            for section, _ in sorted_sections:
                if remaining_budget <= 0:
                    break
                additional = min(remaining_budget, section_sizes[section] - section_budgets[section])
                section_budgets[section] += additional
                remaining_budget -= additional
        
        print(f"Per-section budgets: {section_budgets}")
        
        # Sample per section
        sampled_embeddings_by_section = {}
        sampled_metadata_by_section = {}
        
        for section in SECTIONS:
            if embeddings_by_section[section] is None or len(embeddings_by_section[section]) == 0:
                sampled_embeddings_by_section[section] = None
                sampled_metadata_by_section[section] = []
                continue
            
            section_embeddings = embeddings_by_section[section]
            section_metadata = metadata_by_section[section]
            section_budget = section_budgets.get(section, 0)
            
            if len(section_embeddings) <= section_budget:
                # No sampling needed
                sampled_embeddings_by_section[section] = section_embeddings
                sampled_metadata_by_section[section] = section_metadata
                continue
            
            print(f"\nSampling {section_budget:,} spans from {len(section_embeddings):,} in section '{section}'...")
            # Random sampling without replacement
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(section_embeddings), size=section_budget, replace=False)
            indices = np.sort(indices)  # Sort to maintain some locality
            
            # Select embeddings and metadata
            sampled_embeddings_by_section[section] = section_embeddings[indices].copy()
            sampled_metadata_by_section[section] = [section_metadata[i] for i in indices]
            del indices
            print(f"✓ Sampled {len(sampled_embeddings_by_section[section]):,} spans for section '{section}'")

        import gc
        gc.collect()
        embeddings_by_section = sampled_embeddings_by_section
        metadata_by_section = sampled_metadata_by_section
        metadata = []
        for section in SECTIONS:
            metadata.extend(metadata_by_section.get(section, []))
        
        print(f"✓ Total sampled spans: {sum(len(emb) if emb is not None else 0 for emb in embeddings_by_section.values()):,}")
        print(f"{'='*80}\n")
    
    if args.embed_dtype == "float16":
        print(f"Storage dtype: float16 (half size; downstream will cast to float32 on load)")

    # Assert alignment before saving (each section: embedding rows == metadata rows)
    for sec in SECTIONS:
        emb = embeddings_by_section.get(sec)
        if emb is not None:
            n_meta = len(metadata_by_section.get(sec, []))
            assert n_meta == len(emb), f"Section '{sec}': metadata count {n_meta} != embedding rows {len(emb)}"

    print(f"\n[4/4] Saving results per section...")
    
    # Save embeddings and metadata per section
    total_spans = 0
    for section in SECTIONS:
        if embeddings_by_section[section] is None:
            print(f"Skipping section '{section}' (no embeddings)")
            continue

        section_embeddings = embeddings_by_section[section]
        section_metadata = metadata_by_section.get(section, [])
        # Cast to storage dtype (float16 halves size with minimal retrieval impact)
        to_save = section_embeddings.astype(save_dtype, copy=True)

        # Estimate file size
        dtype_size = to_save.itemsize
        estimated_size_mb = (to_save.size * dtype_size) / (1024 ** 2)
        estimated_size_gb = estimated_size_mb / 1024

        print(f"\nSection '{section}':")
        print(f"  Embeddings shape: {to_save.shape}")
        print(f"  Embedding dimension: {to_save.shape[1]}")
        print(f"  Data type: {to_save.dtype}")
        if estimated_size_gb >= 1.0:
            print(f"  Size: {estimated_size_gb:.2f} GB")
        else:
            print(f"  Size: {estimated_size_mb:.1f} MB")

        # Save embeddings as .npy (fast I/O, memmap-friendly for center-building; compress with zstd if needed)
        np.save(output_files[section], to_save, allow_pickle=False)
        print(f"  ✓ Saved embeddings to: {output_files[section]}")
        actual_size_mb = os.path.getsize(output_files[section]) / (1024 ** 2)
        if actual_size_mb >= 1024:
            print(f"  Actual size: {actual_size_mb / 1024:.2f} GB")
        else:
            print(f"  Actual size: {actual_size_mb:.1f} MB")
        
        # Save metadata
        if args.save_metadata:
            with open(metadata_files[section], 'w') as f:
                for meta in section_metadata:
                    compact = {
                        'd': meta['doc_id'],
                        's': meta['section'],
                        'i': meta.get('i', -1),  # 0-based row index within section (embedding row index)
                        'k': meta.get('span_kind', 'content'),  # content | cls | doc_mean (retrievable for filtering)
                        'r': meta['span_text_raw'],
                        'u': meta.get('unit', '')
                    }
                    f.write(json.dumps(compact, ensure_ascii=False) + '\n')
            meta_size_mb = os.path.getsize(metadata_files[section]) / (1024 ** 2)
            print(f"  ✓ Saved metadata to: {metadata_files[section]} ({meta_size_mb:.1f} MB)")
        
        total_spans += len(section_embeddings)
    
    # Print overall statistics
    print(f"\n" + "=" * 80)
    print(f"Overall Statistics:")
    print(f"  - Total spans: {total_spans}")
    for section in SECTIONS:
        if embeddings_by_section[section] is not None:
            emb = embeddings_by_section[section]
            print(f"  - {section}: {len(emb):,} spans, shape {emb.shape}, mean={emb.mean():.4f}, std={emb.std():.4f}")
    
    # Show sample spans: 2–3 content spans per section (skip cls/doc_mean so we see real spaCy output)
    print(f"\n  Sample spans (raw, content only):")
    shown_per_section = {s: 0 for s in SECTIONS}
    max_per_section = 3
    for meta in metadata:
        if all(shown_per_section[s] >= max_per_section for s in SECTIONS):
            break
        if meta.get("span_kind") in ("cls", "doc_mean"):
            continue
        sec = meta.get("section")
        if sec not in shown_per_section or shown_per_section[sec] >= max_per_section:
            continue
        raw = (meta.get("span_text_raw") or "")[:50]
        print(f"    - [{sec}] {raw}")
        shown_per_section[sec] += 1
    # No content spans in metadata despite having embeddings => pipeline bug (e.g. only cls/doc_mean)
    if total_spans > 0 and sum(shown_per_section.values()) == 0:
        raise RuntimeError(
            "No content spans in metadata although total_spans > 0. "
            "All spans are cls/doc_mean or metadata is misaligned (pipeline bug)."
        )

    print(f"=" * 80)
    print(f"✓ Done! Embeddings saved to: {output_dir}")


if __name__ == "__main__":
    main()