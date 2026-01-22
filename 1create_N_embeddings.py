"""
Create contextual span embeddings for patent documents.

Default mode: claim2all (uses all three sections: title+abstract, claim, invention)
- For abstract section: uses "title [SEP] [abstract] {abstract}" format
- For claim section: uses "[claim] {claim}" format  
- For invention section: uses "[invention] {invention}" format

For each section:
- Step 1: Encode full text with model
- Step 2: Extract spans (noun phrases, technical phrases) based on unit type
- Step 3: Pool token embeddings for each semantic unit

Outputs are saved separately per section in an output directory that includes:
- Encoder model name
- Unit type (encoder_token, spacy_token, spacy_sentence, doc, noun_chunk)
- Whether CLS token is kept
- Which layer is used (last or second_last)

Usage:
    python create_N_embeddings.py --layer last
    python create_N_embeddings.py --layer second_last --output_dir ./my_embeddings
"""

import os
import json
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModel
import spacy
import utils
from utils import (
    ensure_section_tokens,
    load_corpus,
    create_contextual_span_embeddings,
    DEVICE,
)


def main():
    parser = argparse.ArgumentParser()
    # Mode is now fixed to claim2all
    mode = "claim2all"
    parser.add_argument("--data_dir", type=str, default="./downstream/perf200/",
                       help="Directory containing the corpus (content/documents.json)")

    parser.add_argument("--model_path", type=str, default="ZoeYou/PatentMap-V0-SecPair-Claim",
                       help="Path to pretrained model or HuggingFace model ID")
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Batch size for encoding documents")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length for tokenizer (max_length parameter)")
    parser.add_argument("--max_text_length", type=int, default=900000,
                       help="Maximum text length for spaCy processing (stays under spaCy's 1M limit)")

    parser.add_argument("--max_docs", type=int, default=None,
                       help="Maximum number of documents to process (for testing)")
    parser.add_argument("--max_spans", type=int, default=5000000,
                       help="Maximum number of spans (embeddings) to extract")

    parser.add_argument("--compress", action="store_true",
                       help="Save as compressed .npz file (gzip) to reduce file size without precision loss")
    parser.add_argument("--save_metadata", type=int, default=1, choices=[0, 1],
                       help="Whether to save metadata (0=no, 1=yes). Default: 1. Metadata is saved AFTER sampling.")
    parser.add_argument("--unit", type=str, default="spacy_token",
                       choices=["spacy_token", "spacy_sentence", "doc", "noun_chunk", "encoder_token"],
                       help="Semantic unit to pool over within the 512-token visible text. "
                            "spacy_token pools per spaCy token (many embeddings). "
                            "spacy_sentence pools per sentence (fewer). "
                            "doc pools one embedding per doc-section. "
                            "noun_chunk pools per noun chunk (subset). "
                            "encoder_token outputs one embedding per tokenizer token (model last_hidden_state) excluding special/pad.")
    parser.add_argument("--keep_cls", type=int, default=1, choices=[0, 1],
                       help="Whether to keep [CLS] token in output (1=yes, 0=no). Default: 1. [CLS] contains sequence-level information useful for retrieval.")
    parser.add_argument("--layer", type=str, default="last", choices=["last", "second_last"],
                       help="Which layer to use for embeddings: 'last' (default) for last layer, 'second_last' for second-to-last layer")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for embeddings. If not specified, will be auto-generated based on model and parameters.")
    args = parser.parse_args()
    
    # Initialize global spaCy model (utils.NLP) so process_doc_batch can use it
    print("Loading spaCy model...")
    utils.NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    utils.NLP.max_length = args.max_text_length + 10000  # Small buffer
    assert utils.NLP.has_pipe("parser"), "spaCy parser is required for noun_chunks extraction"
    print("✓ spaCy model loaded")
    
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
    unit_suffix = args.unit  # encoder_token, spacy_token, spacy_sentence, doc, noun_chunk
    
    if args.output_dir is None:
        # Auto-generate output directory: encoder_unit_cls_layer
        # Format: embeddings_{model_name}_{unit}_{cls}_{layer}
        output_dir = f"./embeddings_{model_name}_{unit_suffix}_{cls_suffix}_{layer_suffix}"
    else:
        output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    file_ext = "npz" if args.compress else "npy"
    
    # Build output filenames per section
    output_files = {}
    metadata_files = {}
    for section in ['abstract', 'claim', 'invention']:
        # Format: {section}_{unit}.{ext}
        output_files[section] = os.path.join(output_dir, f"{section}_{args.unit}.{file_ext}")
        metadata_files[section] = os.path.join(output_dir, f"{section}_{args.unit}_metadata.jsonl")
    
    print(f"=" * 80)
    print(f"Creating Contextual Span Embeddings")
    print(f"Mode: {mode} (fixed)")
    print(f"Device: {DEVICE}")
    print(f"Model: {args.model_path}")
    print(f"Unit: {args.unit}")
    print(f"Layer: {args.layer}")
    print(f"Batch size: {args.batch_size}, Max length: {args.max_length}")
    print(f"Keep [CLS]: {keep_cls}")
    print(f"Output directory: {output_dir}")
    print(f"=" * 80)
    
    # Ensure section tokens are in vocabulary
    ensure_section_tokens(tokenizer, model)
    
    model.to(DEVICE)
    model.eval()
    
    # Load corpus
    print("\n[2/4] Loading corpus...")
    documents_path = os.path.join(args.data_dir, "content/documents.json")
    documents = load_corpus(documents_path)
    print(f"Loaded {len(documents)} documents")
    
    # Create embeddings
    print(f"\n[3/4] Creating contextual span embeddings...")
    embeddings_by_section, metadata = create_contextual_span_embeddings(
        documents, model, tokenizer, unit=args.unit, 
        max_docs=args.max_docs, batch_size=args.batch_size, max_length=args.max_length,
        keep_cls=keep_cls, layer=args.layer
    )
    
    # Separate metadata by section for sampling
    metadata_by_section = {
        'abstract': [],
        'claim': [],
        'invention': []
    }
    for meta in metadata:
        section = meta['section']
        if section in metadata_by_section:
            metadata_by_section[section].append(meta)
    
    # Truncate/sample per section if needed
    total_spans = sum(len(emb) if emb is not None else 0 for emb in embeddings_by_section.values())
    if total_spans > args.max_spans:
        print(f"\n{'='*80}")
        print(f"Sampling {args.max_spans:,} spans from {total_spans:,} total spans (across all sections)")
        print(f"Strategy: {args.sampling}")
        
        # Calculate per-section sampling budget (proportional to section size)
        section_sizes = {s: len(emb) if emb is not None else 0 for s, emb in embeddings_by_section.items()}
        total_size = sum(section_sizes.values())
        
        # Allocate budget proportionally, but ensure at least some from each non-empty section
        section_budgets = {}
        remaining_budget = args.max_spans
        for section in ['abstract', 'claim', 'invention']:
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
        
        for section in ['abstract', 'claim', 'invention']:
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
            import gc
            gc.collect()
            
            print(f"✓ Sampled {len(sampled_embeddings_by_section[section]):,} spans for section '{section}'")
        
        # Update embeddings_by_section with sampled results
        embeddings_by_section = sampled_embeddings_by_section
        # Reconstruct full metadata list from sampled sections
        metadata = []
        for section in ['abstract', 'claim', 'invention']:
            metadata.extend(sampled_metadata_by_section.get(section, []))
        
        print(f"✓ Total sampled spans: {sum(len(emb) if emb is not None else 0 for emb in embeddings_by_section.values()):,}")
        print(f"{'='*80}\n")
    
    if args.compress:
        print(f"Compression enabled: files will be saved as .npz (gzip compressed)")
        print(f"Expected compression ratio: 30-50% (depending on data)")
    
    print(f"\n[4/4] Saving results per section...")
    
    # Save embeddings and metadata per section
    total_spans = 0
    for section in ['abstract', 'claim', 'invention']:
        if embeddings_by_section[section] is None:
            print(f"Skipping section '{section}' (no embeddings)")
            continue
        
        section_embeddings = embeddings_by_section[section]
        section_metadata = [m for m in metadata if m['section'] == section]
        
        # Estimate file size
        dtype_size = section_embeddings.itemsize
        estimated_size_mb = (section_embeddings.size * dtype_size) / (1024 ** 2)
        estimated_size_gb = estimated_size_mb / 1024
        
        print(f"\nSection '{section}':")
        print(f"  Embeddings shape: {section_embeddings.shape}")
        print(f"  Embedding dimension: {section_embeddings.shape[1]}")
        print(f"  Data type: {section_embeddings.dtype}")
        if estimated_size_gb >= 1.0:
            print(f"  Uncompressed size: {estimated_size_gb:.2f} GB")
        else:
            print(f"  Uncompressed size: {estimated_size_mb:.1f} MB")
        
        # Save embeddings
        if args.compress:
            np.savez_compressed(output_files[section], embeddings=section_embeddings)
            print(f"  ✓ Saved compressed embeddings to: {output_files[section]}")
            actual_size_mb = os.path.getsize(output_files[section]) / (1024 ** 2)
            actual_size_gb = actual_size_mb / 1024
            if actual_size_gb >= 1.0:
                print(f"  Actual compressed size: {actual_size_gb:.2f} GB")
                compression_ratio = (1 - actual_size_gb / estimated_size_gb) * 100
            else:
                print(f"  Actual compressed size: {actual_size_mb:.1f} MB")
                compression_ratio = (1 - actual_size_mb / estimated_size_mb) * 100
            print(f"  Compression ratio: {compression_ratio:.1f}%")
        else:
            np.save(output_files[section], section_embeddings)
            print(f"  ✓ Saved embeddings to: {output_files[section]}")
        
        # Save metadata
        if args.save_metadata:
            with open(metadata_files[section], 'w') as f:
                for meta in section_metadata:
                    compact = {
                        'd': meta['doc_id'],
                        's': meta['section'],
                        't': meta['span_text'],
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
    for section in ['abstract', 'claim', 'invention']:
        if embeddings_by_section[section] is not None:
            emb = embeddings_by_section[section]
            print(f"  - {section}: {len(emb):,} spans, shape {emb.shape}, mean={emb.mean():.4f}, std={emb.std():.4f}")
    
    # Show sample spans
    print(f"\n  Sample spans (raw → canonical):")
    sample_count = 0
    for meta in metadata:
        if sample_count >= 10:
            break
        raw = meta['span_text_raw'][:40]
        canonical = meta['span_text'][:40]
        print(f"    - {raw} → {canonical} ({meta['section']})")
        sample_count += 1
    
    print(f"=" * 80)
    print(f"✓ Done! Embeddings saved to: {output_dir}")


if __name__ == "__main__":
    main()