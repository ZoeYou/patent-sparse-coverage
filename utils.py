"""
Utility functions for patent document processing.
"""

import json
import os
import re
from typing import List, Tuple, Optional
import torch
import numpy as np
from tqdm import tqdm




# Global device (set in main)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global spaCy model (initialized in main)
NLP = None
# Batch size for NLP.pipe(); 1create_N_embeddings sets this from --spacy_batch_size
SPACY_PIPE_BATCH_SIZE = 64



def ensure_section_tokens(tokenizer, model):
    """
    Ensure section tokens [abstract], [claim], [invention] are in tokenizer vocabulary.
    If not, add them and resize model embeddings.
    
    Args:
        tokenizer: The tokenizer to check and modify
        model: The model whose embeddings need to be resized
    """
    section_tokens = ['[abstract]', '[claim]', '[invention]']
    tokens_to_add = []
    
    for token in section_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            tokens_to_add.append(token)
    
    if tokens_to_add:
        print(f"⚠️  Section tokens {tokens_to_add} not found in tokenizer vocabulary.")
        print(f"   Adding them and resizing model embeddings...")
        tokenizer.add_tokens(tokens_to_add)
        model.resize_token_embeddings(len(tokenizer))
        print(f"✓ Added {len(tokens_to_add)} tokens. New vocab size: {len(tokenizer)}")
    else:
        print(f"✓ All section tokens already in vocabulary")
    
    # Verify all tokens are now valid
    for token in section_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert token_id != tokenizer.unk_token_id, f"Failed to add token {token}"


# -----------------------------------------------------------------------------
# Encoder input format by model (single source of truth for embeddings + query)
# -----------------------------------------------------------------------------
ENCODER_FORMAT_SECTION_TOKENS = "section_tokens"   # title SEP [abstract] abstract, [claim] claim, [invention] invention
ENCODER_FORMAT_TITLE_SEP_ONLY = "title_sep_only"   # title SEP abstract (no section tokens)

DEFAULT_SEP = " [SEP] "  # fallback when tokenizer not available


# Models that use title + sep + text only (no [abstract]/[claim]/[invention]). Aligned with baselines.py dense retrieval.
_TITLE_SEP_ONLY_MODEL_IDS = (
    "allenai/specter2_base",
    "patentbert",
    "mpi-inno-comp/paecter",
    "datalyes/patembed-large",
    "patembed-large",
)


def get_encoder_format_scheme(model_id: str) -> str:
    """
    Return the encoder input format scheme for a given model.
    Aligned with baselines.py: specter2/patentbert use title+sep+text; paecter same; patembed same;
    bert-for-patents uses [SEP] [abstract] / [claim] / [invention]; PatentMap (sparse_coverage default) uses section_tokens.
    """
    if not model_id:
        return ENCODER_FORMAT_SECTION_TOKENS
    mid = (model_id or "").strip().lower().replace("\\", "/")
    for candidate in _TITLE_SEP_ONLY_MODEL_IDS:
        if candidate.lower() in mid or mid.endswith(candidate.lower().split("/")[-1]):
            return ENCODER_FORMAT_TITLE_SEP_ONLY
    return ENCODER_FORMAT_SECTION_TOKENS


def get_encoder_sep_for_model(model_id: str, tokenizer=None) -> str:
    """
    Return the separator string between title and abstract for this model.
    When tokenizer is provided, uses tokenizer.sep_token so that models with
    different separators (e.g. not "[SEP]") stay coherent. When tokenizer is
    None, returns DEFAULT_SEP.
    """
    if tokenizer is None:
        return DEFAULT_SEP
    sep_token = getattr(tokenizer, "sep_token", None) or "[SEP]"
    s = str(sep_token).strip()
    return f" {s} " if s else DEFAULT_SEP


def format_abstract_for_encoder(scheme: str, title: str, abstract: str, sep: str = DEFAULT_SEP) -> str:
    """Format title+abstract for encoder input. sep is the model's separator (see get_encoder_sep_for_model)."""
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if scheme == ENCODER_FORMAT_TITLE_SEP_ONLY:
        return f"{title}{sep}{abstract}".strip() if (title or abstract) else ""
    return f"{title}{sep}[abstract] {abstract}".strip()


def format_claim_for_encoder(scheme: str, claim: str) -> str:
    """Format claim for encoder input."""
    claim = (claim or "").strip()
    if scheme == ENCODER_FORMAT_TITLE_SEP_ONLY:
        return claim
    return f"[claim] {claim}".strip()


def format_invention_for_encoder(scheme: str, invention: str) -> str:
    """Format invention for encoder input."""
    invention = (invention or "").strip()
    if scheme == ENCODER_FORMAT_TITLE_SEP_ONLY:
        return invention
    return f"[invention] {invention}".strip()


def get_chunk_sep_marker(scheme: str, sep: str = DEFAULT_SEP) -> str:
    """Separator used to split title vs abstract when chunking. Must match format_abstract_for_encoder."""
    if scheme == ENCODER_FORMAT_TITLE_SEP_ONLY:
        return sep
    return sep + "[abstract] "


def remove_escape_and_decode(text: str) -> str:
    """Clean escape sequences from text."""
    if not text:
        return ""
    text = re.sub(r'\\[^nrtbfav"\'\\]', '', text)
    return text.replace('\/', '/').replace('\"', '"').replace(" -->", "")





def load_corpus(file_path: str) -> dict:
    """
    Load JSON corpus file (JSONL format) and normalize to unified structure.
    
    The raw data has:
    - title: Content['title']
    - abstract: Content['pa01']
    - claims: all Content['c-en-XXXX'] fields joined
    - invention (description): all Content['p0XXX'] fields joined
    
    Returns dict with doc_id -> {'title', 'abstract', 'claim', 'invention'}
    """
    corpus = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                doc_id = str(doc.get('dnum') or doc.get('Application_Number', f'doc_{len(corpus)}'))
                content = doc.get('Content', doc)
                
                # Extract title
                title = content.get('title', '').strip()
                
                # Extract abstract from 'pa01' field (NOT 'abstract')
                abstract = content.get('pa01', '').strip()
                abstract = remove_escape_and_decode(abstract)
                
                # Extract claims (all c-en-XXXX fields)
                claims = []
                for key in sorted(content.keys()):
                    if key.startswith('c-en-'):
                        claims.append(content[key])
                claim_text = '\n'.join(claims)
                claim_text = remove_escape_and_decode(claim_text)
                
                # Extract invention/description (all p0XXX fields)
                description = []
                for key in sorted(content.keys()):
                    if key.startswith('p0'):
                        description.append(content[key])
                invention_text = ' '.join(description)
                invention_text = remove_escape_and_decode(invention_text)
                
                # Store normalized structure
                corpus[doc_id] = {
                    'title': title,
                    'abstract': abstract,
                    'claim': claim_text,
                    'invention': invention_text
                }
            except json.JSONDecodeError:
                continue
    return corpus


def collect_doc_texts(
    documents: dict,
    max_docs: int = None,
    format_scheme: Optional[str] = None,
    sep: Optional[str] = None,
    max_section_chars: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    """
    Build ONE encoder input per document section (doc-level encoding).
    format_scheme and sep (from get_encoder_sep_for_model) keep indexing and retrieval coherent.
    max_section_chars: if set, truncate each section text to this many characters before encoding
                       (encoder still sees only first max_length tokens; this bounds spaCy and tokenizer input).
    """
    if format_scheme is None:
        format_scheme = ENCODER_FORMAT_SECTION_TOKENS
    if sep is None:
        sep = DEFAULT_SEP
    items = []
    doc_items = list(documents.items())

    if max_docs:
        doc_items = doc_items[:max_docs]
        print(f"Processing only first {max_docs} documents")

    def _trunc(s: str) -> str:
        if max_section_chars is not None and len(s) > max_section_chars:
            return s[:max_section_chars]
        return s

    for doc_id, doc in doc_items:
        title = (doc.get('title', '') or '').strip()
        abstract = (doc.get('abstract', '') or '').strip()
        claim = (doc.get('claim', '') or '').strip()
        invention = (doc.get('invention', '') or '').strip()

        if abstract:
            formatted_abs = format_abstract_for_encoder(format_scheme, title, abstract, sep=sep)
            if formatted_abs:
                items.append((doc_id, "abstract", _trunc(formatted_abs)))
        if claim:
            formatted_claim = format_claim_for_encoder(format_scheme, claim)
            if formatted_claim:
                items.append((doc_id, "claim", _trunc(formatted_claim)))
        if invention:
            formatted_inv = format_invention_for_encoder(format_scheme, invention)
            if formatted_inv:
                items.append((doc_id, "invention", _trunc(formatted_inv)))

    return items





def create_contextual_span_embeddings(
    documents: dict,
    model,
    tokenizer,
    unit: str,
    max_docs: int = None,
    batch_size: int = 64,
    max_length: int = 512,
    keep_cls: bool = True,
    layer: str = "last",
    span_pooling: str = "mean",
    format_scheme: Optional[str] = None,
    sep: Optional[str] = None,
    keep_doc_mean: bool = False,
    max_section_chars: Optional[int] = None,
) -> Tuple[dict, List[dict]]:
    """
    Create contextual span embeddings for all documents using batch processing.
    format_scheme and sep keep format coherent with retrieval. keep_doc_mean: one span per section = mean of tokens.
    max_section_chars: if set, truncate each section text to this many chars before encoding (bounds spaCy/tokenizer input).
    """
    if format_scheme is None:
        format_scheme = ENCODER_FORMAT_SECTION_TOKENS
    if sep is None:
        sep = DEFAULT_SEP
    model.eval()
    all_embeddings_by_section = {
        'abstract': [],
        'claim': [],
        'invention': []
    }
    temp_embeddings_by_section = {
        'abstract': [],
        'claim': [],
        'invention': []
    }
    all_metadata = []

    doc_data = collect_doc_texts(documents, max_docs=max_docs, format_scheme=format_scheme, sep=sep, max_section_chars=max_section_chars)
    from collections import Counter
    section_counts = Counter(item[1] for item in doc_data)
    n_abs = section_counts.get("abstract", 0)
    n_clm = section_counts.get("claim", 0)
    n_inv = section_counts.get("invention", 0)
    print(f"Collected {len(doc_data)} doc-sections (abstract: {n_abs}, claim: {n_clm}, invention: {n_inv})")
    if n_abs == 0 and max_docs is not None:
        print("⚠️  No abstract sections: the first max_docs documents have no 'pa01' (abstract). "
              "Use a larger --max_docs or run without --max_docs to include abstract embeddings.")
    
    # Process in batches
    print(f"\nExtracting contextual span embeddings (batch size={batch_size})...")
    num_batches = (len(doc_data) + batch_size - 1) // batch_size
    
    # Chunk size: convert to numpy arrays every N batches to reduce peak memory
    # For memory-intensive units, chunk more frequently
    chunk_frequency = 20 if unit == "encoder_token" else 50
    
    def _chunk_embeddings():
        """Convert accumulated embeddings to numpy arrays and clear temp lists."""
        for section in ['abstract', 'claim', 'invention']:
            if len(temp_embeddings_by_section[section]) > 0:
                chunk_array = np.vstack(temp_embeddings_by_section[section])
                all_embeddings_by_section[section].append(chunk_array)
                temp_embeddings_by_section[section] = []
                import gc
                gc.collect()
    
    # Note: We'll process embeddings per section after all batches are done
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(doc_data))
        batch_data = doc_data[start_idx:end_idx]
        
        try:
            doc_ids = [item[0] for item in batch_data]
            sections = [item[1] for item in batch_data]
            doc_texts = [item[2] for item in batch_data]

            # Process doc batch - returns (doc_id, section, doc_text, span_text_raw, span_text_canonical, span_emb)
            batch_results = process_doc_batch(
                doc_texts=doc_texts,
                doc_ids=doc_ids,
                sections=sections,
                unit=unit,
                model=model,
                tokenizer=tokenizer,
                device=DEVICE,
                max_length=max_length,
                keep_cls=keep_cls,
                layer=layer,
                span_pooling=span_pooling,
                keep_doc_mean=keep_doc_mean,
            )
            
            # Store results with proper metadata tracking, separated by section
            for doc_id, section, doc_text, span_text_raw, span_text_canonical, span_emb in batch_results:
                if section in temp_embeddings_by_section:
                    temp_embeddings_by_section[section].append(span_emb)
                all_metadata.append({
                    'doc_id': doc_id,
                    'section': section,
                    'sentence': doc_text[:100],  # Keep key name for backwards compatibility (first 100 chars)
                    'unit': unit,
                    'span_text_raw': span_text_raw,  # Original span text
                    'span_text': span_text_canonical  # Canonical version for dedup/stats
                })
            
            # Clear batch results immediately to free memory
            del batch_results
            del doc_ids, sections, doc_texts
        
        except Exception as e:
            print(f"\nError processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Periodically chunk embeddings to reduce peak memory usage
        if (batch_idx + 1) % chunk_frequency == 0:
            _chunk_embeddings()
        
        # Clear GPU cache more frequently for memory-intensive units
        cache_frequency = 5 if unit == "encoder_token" else 10
        if (batch_idx + 1) % cache_frequency == 0:
            torch.cuda.empty_cache()
            import gc
            gc.collect()
    
    # Final chunking for any remaining embeddings
    _chunk_embeddings()
    
    # Convert embeddings per section to numpy arrays (concatenate chunks)
    embeddings_by_section = {}
    total_embeddings = 0
    
    for section in ['abstract', 'claim', 'invention']:
        if len(all_embeddings_by_section[section]) > 0:
            print(f"Converting {sum(len(chunk) for chunk in all_embeddings_by_section[section])} embeddings for section '{section}'...")
            # Concatenate all chunks
            embeddings_by_section[section] = np.vstack(all_embeddings_by_section[section])
            total_embeddings += len(embeddings_by_section[section])
            # Clear chunks to free memory
            all_embeddings_by_section[section] = []
        else:
            embeddings_by_section[section] = None
    
    if total_embeddings == 0:
        raise ValueError("No embeddings were created. Check your input data and processing pipeline.")
    
    # Force cleanup
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return embeddings_by_section, all_metadata



def extract_char_spans_to_token_spans(char_spans: List[Tuple[int, int, str]],
                                     prefix_len: int,
                                     offset_mapping: torch.Tensor,
                                     input_ids: torch.Tensor,
                                     special_token_ids: set,
                                     dedup: bool = False) -> List[Tuple[str, int, int]]:
    """
    Generic char-span -> token-span mapper using tokenizer offset_mapping.

    Each char span is (start_char, end_char, text) relative to the ORIGINAL text used for tokenization.
    If the ORIGINAL text had a prefix that char_spans are not counting, pass prefix_len to shift.

    Returns list of (span_text, start_token_idx, end_token_idx) with end exclusive.
    """
    spans = []
    offset_list = offset_mapping.cpu().tolist()
    input_ids_list = input_ids.cpu().tolist()

    for span_start_char, span_end_char, span_text in char_spans:
        shifted_start = span_start_char + prefix_len
        shifted_end = span_end_char + prefix_len

        token_start = None
        token_end = None

        for idx, (offset_start, offset_end) in enumerate(offset_list):
            # Skip special tokens using actual token ID (robust across tokenizer versions)
            if input_ids_list[idx] in special_token_ids:
                continue
            # Skip padding tokens
            if offset_start == 0 and offset_end == 0:
                continue

            if token_start is None and offset_start <= shifted_start < offset_end:
                token_start = idx

            if token_start is not None and offset_start < shifted_end <= offset_end:
                token_end = idx + 1
                break

        if token_start is not None and token_end is not None and token_start < token_end:
            spans.append((span_text, token_start, token_end))

    if not dedup:
        return spans

    seen = set()
    unique = []
    for span_text, start, end in spans:
        k = (span_text.lower(), start, end)
        if k in seen:
            continue
        seen.add(k)
        unique.append((span_text, start, end))
    return unique


def canonicalize_span_text(span_text: str) -> str:
    """
    Canonicalize span text for deduplication and statistics:
    - Convert to lowercase
    - Preserve technical symbols: - / . + (important for patents like H.264, Wi-Fi, C++, 3.5GHz)
    - Remove other punctuation
    - Normalize whitespace
    - Strip leading/trailing stopwords
    
    Used for df/idf computation and deduplication, not for embedding pooling
    (which still uses original token boundaries).
    """
    s = span_text.lower().strip()
    
    # Preserve technical symbols (- / . +), remove other punctuation
    # This keeps: phase-locked, H.264, Wi-Fi, C++, 3.5GHz intact
    s = re.sub(r'[^a-z0-9\s\-/\.\+]', ' ', s)
    
    # Normalize multiple consecutive symbols/spaces (but keep single technical symbols)
    # e.g., "foo -- bar" -> "foo - bar", "foo  bar" -> "foo bar"
    s = re.sub(r'[\-/\.\+]{2,}', lambda m: m.group(0)[0], s)  # Keep first of repeated symbols
    s = re.sub(r"\s+", " ", s).strip()
    
    # Strip leading/trailing stopwords
    stop = {"a", "an", "the", "this", "that", "these", "those", "any"}
    toks = s.split()
    while toks and toks[0] in stop:
        toks = toks[1:]
    while toks and toks[-1] in stop:
        toks = toks[:-1]
    
    return " ".join(toks)


def filter_span_quality(span_text: str) -> bool:
    """
    Filter out low-quality spans (patent templates, stopwords, etc.).
    Returns True if span should be kept, False if should be filtered out.
    """
    s = span_text.strip()
    # 0) strip weird whitespace
    if not s:
        return False

    span_lower = span_text.lower().strip()
    # 1) Too short or too long
    if len(span_text) < 3 or len(span_text) > 100:
        return False
    
    # 2) reject pure function/connector words (patent discourse)
    hard_stop = {
        "which","wherein","thereof","therein","thereby","herein","hereby",
        "said","such","other","another","any","each","may","can","would","could",
        "including","include","includes","according","respectively"
    }
    if span_lower in hard_stop:
        return False

    # 3) reject 1-word spans unless they look like technical anchors
    words = re.findall(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*", s)
    if len(words) == 1:
        w = words[0]
        # allow acronyms / chemical-like / alnum anchors
        if not (re.match(r"^[A-Z]{2,}$", w) or re.search(r"\d", w) or "-" in w or "/" in w):
            return False

    # 4) reject generic template heads (very common in patents)
    generic_heads = {
        "method","methods","system","systems","apparatus","device","devices",
        "technique","techniques","approach","approaches","solution","solutions",
        "embodiment","embodiments","invention","disclosure"
    }
    # if span is short and ends with generic head -> drop (e.g., "a method", "the technique")
    toks = [t for t in span_lower.split() if t]
    if len(toks) <= 4 and toks[-1] in generic_heads:
        return False

    # 5) Only digits, punctuation, or units
    if re.match(r'^[\d\s\-.,;:()%°/]+$', span_text):
        return False
    
    # 6) Patent template phrases (common boilerplate)
    patent_templates = [
        'at least one', 'plurality of', 'the present invention', 
        'embodiment of the invention', 'one embodiment', 'another embodiment',
        'the invention', 'said invention', 'the method', 'the apparatus',
        'the system', 'the device', 'the present disclosure',
        'according to the invention', 'in accordance with', 'as described',
        'as shown', 'as illustrated', 'such as', 'and the like'
    ]
    
    for template in patent_templates:
        if template in span_lower:
            return False
    
    # 7) High stopword ratio
    stopwords = {'a', 'an', 'the', 'of', 'for', 'to', 'in', 'on', 'at', 'by', 
                'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'or', 'and', 'but', 'if', 'as', 'it', 'this', 'that', 'these', 'those'}
    
    words = span_lower.split()
    if len(words) > 0:
        stopword_ratio = sum(1 for w in words if w in stopwords) / len(words)
        if stopword_ratio > 0.6:
            return False
    
    return True




def chunk_query_text(
    full_text: str,
    tokenizer,
    max_length: int = 512,
    title_prefix_max: int = 64,
    stride_ratio: float = 1.0,
    format_scheme: Optional[str] = None,
    sep: Optional[str] = None,
) -> List[Tuple[str, int]]:
    """
    Split a long query (title + sep + abstract) into chunks that fit within max_length tokens.
    format_scheme and sep (from get_encoder_sep_for_model) must match format_abstract_for_encoder.
    """
    if not full_text or not full_text.strip():
        return [("", 0)]

    if format_scheme is None:
        format_scheme = ENCODER_FORMAT_SECTION_TOKENS
    if sep is None:
        sep = DEFAULT_SEP
    sep_marker = get_chunk_sep_marker(format_scheme, sep)
    if sep_marker not in full_text:
        # No [abstract] format: treat as single chunk (caller may truncate elsewhere)
        return [(full_text.strip(), 0)]

    parts = full_text.split(sep_marker, 1)
    title_part = (parts[0] or "").strip()
    abstract_part = (parts[1] or "").strip()
    prefix_text = title_part + sep_marker

    # Tokenize without special tokens to get content token counts
    prefix_enc = tokenizer(
        prefix_text,
        add_special_tokens=False,
        return_tensors=None,
        return_offsets_mapping=False,
    )
    abstract_enc = tokenizer(
        abstract_part,
        add_special_tokens=False,
        return_tensors=None,
        return_offsets_mapping=False,
    )
    prefix_ids = prefix_enc if isinstance(prefix_enc, list) else prefix_enc["input_ids"]
    abstract_ids = abstract_enc if isinstance(abstract_enc, list) else abstract_enc["input_ids"]
    if not isinstance(prefix_ids, list):
        prefix_ids = prefix_ids.tolist()
    if not isinstance(abstract_ids, list):
        abstract_ids = abstract_ids.tolist()
    # Handle batch-of-one from tokenizer
    if prefix_ids and isinstance(prefix_ids[0], list):
        prefix_ids = prefix_ids[0]
    if abstract_ids and isinstance(abstract_ids[0], list):
        abstract_ids = abstract_ids[0]

    # Truncate prefix to leave room for abstract in each chunk
    content_max = max_length - 2  # [CLS] and [SEP] added by encoder
    prefix_ids = prefix_ids[:title_prefix_max]
    chunk_content_size = content_max - len(prefix_ids)
    if chunk_content_size <= 0:
        chunk_content_size = 1

    if len(abstract_ids) <= chunk_content_size:
        # Single chunk
        chunk_ids = prefix_ids + abstract_ids
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=False)
        return [(chunk_text, 0)]

    stride = max(1, int(chunk_content_size * stride_ratio))
    out = []
    start = 0
    while start < len(abstract_ids):
        end = min(start + chunk_content_size, len(abstract_ids))
        segment_ids = abstract_ids[start:end]
        chunk_ids = prefix_ids + segment_ids
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=False)
        global_offset_base = len(prefix_ids) + start
        out.append((chunk_text, global_offset_base))
        if end >= len(abstract_ids):
            break
        start += stride
    return out


def _visible_char_end_from_offset_mapping(offset_map_1d: torch.Tensor,
                                         input_ids_1d: torch.Tensor,
                                         special_token_ids: set) -> int:
    """
    Given one sequence's offset_mapping [seq_len, 2], return the max visible character end
    among non-special, non-padding tokens. This lets us truncate raw text so spaCy only
    processes what the encoder actually saw (after max_length truncation).
    """
    offset_list = offset_map_1d.cpu().tolist()
    input_ids_list = input_ids_1d.cpu().tolist()

    max_end = 0
    for (start, end), tid in zip(offset_list, input_ids_list):
        # skip special tokens
        if tid in special_token_ids:
            continue
        # skip padding tokens (offset (0,0))
        if start == 0 and end == 0:
            continue
        if end > max_end:
            max_end = end
    return int(max_end)


def process_doc_batch(doc_texts: List[str],
                      doc_ids: List[str],
                      sections: List[str],
                      unit: str,
                      model, tokenizer, device, max_length: int = 512, keep_cls: bool = True, layer: str = "last", span_pooling: str = "mean",
                      keep_doc_mean: bool = False) -> List[Tuple[str, str, str, str, str, np.ndarray]]:
    """
    Process a batch of doc-section texts (one encoder pass per doc-section):
    1) Encode doc_texts in batch (max_length parameter, default 512)
    2) Derive visible char range from offset_mapping; truncate text to what encoder saw
    3) Run spaCy on truncated text ONLY; extract semantic units (token/sentence/noun_chunk)
    4) Map unit char spans -> token spans via offset_mapping
    5) Pool token embeddings for each unit (mean or max)
    6) If keep_doc_mean: append one span per doc = mean of all content token embeddings

    Args:
        span_pooling: 'mean' or 'max' for aggregating token embeddings within a span.
        keep_doc_mean: If True, keep one span per doc-section = mean over all content tokens (symmetric with keep_cls).

    Returns: List of (doc_id, section, doc_text, span_text_raw, span_text_canonical, span_embedding)
    """
    # Encode doc texts
    encoding = tokenizer(
        doc_texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        add_special_tokens=True,
        return_tensors='pt',
        return_offsets_mapping=True
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    offset_mapping = encoding['offset_mapping']  # [batch, seq_len, 2]

    special_token_ids = set(tokenizer.all_special_ids)
    # Check if [CLS] token exists
    cls_token_id = tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None else None
    # Create filtered special token set based on keep_cls parameter
    filtered_special_token_ids = special_token_ids.copy()
    if keep_cls and cls_token_id is not None:
        # If keep_cls=True, exclude CLS from filtered set (so it's kept in output)
        filtered_special_token_ids.discard(cls_token_id)
    # If keep_cls=False, CLS will remain in filtered_special_token_ids (so it's filtered out)

    # Truncation diagnostics (fast tokenizer only)
    truncated_count = 0
    if hasattr(encoding, 'encodings') and encoding.encodings is not None:
        for enc in encoding.encodings:
            if hasattr(enc, 'overflowing') and len(enc.overflowing) > 0:
                truncated_count += 1
    if truncated_count > 0:
        print(f"\n⚠️  {truncated_count}/{len(doc_texts)} docs truncated at {max_length} tokens")

    # Encode
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Select layer: "last" for last layer, "second_last" for second-to-last layer
        if layer == "last":
            batch_embeddings = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        elif layer == "second_last":
            if outputs.hidden_states is None or len(outputs.hidden_states) < 2:
                raise ValueError("Model does not return hidden_states or has less than 2 layers")
            batch_embeddings = outputs.hidden_states[-2]  # [batch, seq_len, hidden_dim]
        else:
            raise ValueError(f"Invalid layer parameter: {layer}. Must be 'last' or 'second_last'")

    # Compute visible text substrings for spaCy (exactly what encoder saw)
    visible_texts = []
    for i in range(len(doc_texts)):
        char_end = _visible_char_end_from_offset_mapping(offset_mapping[i], input_ids[i], special_token_ids)
        visible_texts.append(doc_texts[i][:char_end] if char_end > 0 else "")

    # Run spaCy only if needed (encoder_token bypasses spaCy entirely)
    docs_spacy = None
    if unit != "encoder_token":
        docs_spacy = list(NLP.pipe(visible_texts, batch_size=SPACY_PIPE_BATCH_SIZE))

    all_span_embeddings = []
    if unit == "encoder_token":
        for i, (doc_id, section, doc_text, vis_text) in enumerate(
            zip(doc_ids, sections, doc_texts, visible_texts)
        ):
            st = (vis_text or "").strip()
            if not st or not re.search(r"[a-zA-Z]", st):
                continue

            token_embeddings = batch_embeddings[i]   # [seq_len, hidden_dim]
            seq_input_ids = input_ids[i]             # [seq_len]
            seq_attention = attention_mask[i]        # [seq_len]

            # Convert all ids -> token strings in one shot
            token_strs = tokenizer.convert_ids_to_tokens(seq_input_ids.detach().cpu().tolist())

            for tok_idx, (tok_id, tok_str, attn) in enumerate(zip(seq_input_ids, token_strs, seq_attention)):
                if int(attn.item()) == 0:
                    continue
                # Filter special tokens but keep [CLS] if it exists
                if int(tok_id.item()) in filtered_special_token_ids:
                    continue

                tok_emb = token_embeddings[tok_idx]
                tok_emb = tok_emb.cpu().numpy()
                tok_emb = tok_emb / (np.linalg.norm(tok_emb) + 1e-12)

                # Canonicalize token string (may become empty for pure punctuation)
                tok_canonical = canonicalize_span_text(tok_str)

                all_span_embeddings.append((doc_id, section, doc_text, tok_str, tok_canonical, tok_emb))

        if keep_doc_mean:
            for i in range(len(doc_ids)):
                doc_id, section, doc_text = doc_ids[i], sections[i], doc_texts[i]
                token_embeddings = batch_embeddings[i]
                seq_input_ids = input_ids[i]
                seq_attention = attention_mask[i]
                embs = []
                for tok_idx in range(seq_attention.shape[0]):
                    if int(seq_attention[tok_idx].item()) == 0:
                        continue
                    if int(seq_input_ids[tok_idx].item()) in filtered_special_token_ids:
                        continue
                    embs.append(token_embeddings[tok_idx].cpu().numpy())
                if embs:
                    mean_emb = np.mean(embs, axis=0).astype(np.float32)
                    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
                    all_span_embeddings.append((doc_id, section, doc_text, "[DOC_MEAN]", "[DOC_MEAN]", mean_emb))
        return all_span_embeddings

    for i, (doc_id, section, doc_text, vis_text, doc_spacy) in enumerate(
        zip(doc_ids, sections, doc_texts, visible_texts, docs_spacy)
    ):
        st = (vis_text or "").strip()
        if not st or not re.search(r"[a-zA-Z]", st):
            continue

        token_embeddings = batch_embeddings[i]  # [seq_len, hidden_dim]
        offset_map = offset_mapping[i]          # [seq_len, 2]
        seq_input_ids = input_ids[i]            # [seq_len]

        # Build semantic unit char spans based on spaCy; offsets are relative to vis_text.
        char_spans = []
        if unit == "spacy_token":
            # Strategy: noun_chunks are merged (one embedding per chunk),
            # other tokens remain individual (one embedding per token).
            # Step 1: Collect all noun_chunks (spaCy yields non-overlapping spans)
            noun_chunk_spans = []
            for chunk in doc_spacy.noun_chunks:
                chunk_start = chunk.start_char
                chunk_end = chunk.end_char
                chunk_text = chunk.text.strip()
                if chunk_text:
                    noun_chunk_spans.append((chunk_start, chunk_end, chunk_text))
            
            # Step 2: Add noun_chunks first (they will be merged into one embedding each)
            char_spans.extend(noun_chunk_spans)
            
            # Step 3: Add individual tokens that are NOT completely inside any noun_chunk
            # A token is "inside" a chunk if its char range is fully contained within the chunk's range
            for tok in doc_spacy:
                if tok.is_space:
                    continue
                tok_start = tok.idx
                tok_end = tok.idx + len(tok.text)
                
                # Check if this token is completely contained within any noun_chunk
                token_in_chunk = False
                for chunk_start, chunk_end, _ in noun_chunk_spans:
                    if chunk_start <= tok_start and tok_end <= chunk_end:
                        token_in_chunk = True
                        break
                
                # Only add tokens that are NOT part of any noun_chunk
                if not token_in_chunk:
                    char_spans.append((tok_start, tok_end, tok.text))
        elif unit == "spacy_sentence":
            # For abstract section, split by model's separator first (e.g. [SEP], </s>), then process title and abstract separately
            sep_str = get_encoder_sep_for_model("", tokenizer)
            if section == "abstract" and sep_str.strip() and sep_str in vis_text:
                sep_pos = vis_text.find(sep_str)
                if sep_pos >= 0:
                    # Extract title part (before separator)
                    title_part_raw = vis_text[:sep_pos]
                    title_part = title_part_raw.strip()
                    
                    # Extract abstract part (after separator, keep [abstract] token if present)
                    after_sep = vis_text[sep_pos + len(sep_str):]
                    abstract_part = after_sep.strip()
                    abstract_content_start = sep_pos + len(sep_str)
                    # Skip leading whitespace
                    leading_ws_len = len(after_sep) - len(after_sep.lstrip())
                    abstract_content_start += leading_ws_len
                    
                    sentences_found = False
                    
                    # Process title part
                    if title_part:
                        title_doc = NLP(title_part)
                        # Calculate offset: find where title content actually starts in vis_text
                        title_text_start_in_vis = len(title_part_raw) - len(title_part.lstrip())
                        
                        for sent in title_doc.sents:
                            t = sent.text.strip()
                            if not t or len(t) < 3:
                                continue
                            # Skip special token patterns
                            if re.match(r'^\[.*\]\s*$', t) or re.match(r'^\[.*\]\s*\[.*\]\s*$', t):
                                continue
                            # Adjust offset: sent.start_char is relative to title_part, add title_text_start_in_vis
                            sent_start = title_text_start_in_vis + sent.start_char
                            sent_end = title_text_start_in_vis + sent.end_char
                            char_spans.append((sent_start, sent_end, t))
                            sentences_found = True
                    
                    # Process abstract part (includes [abstract] token if present)
                    if abstract_part:
                        abstract_doc = NLP(abstract_part)
                        
                        for sent in abstract_doc.sents:
                            t = sent.text.strip()
                            if not t:
                                continue
                            # For [abstract] token itself, keep it if it's a standalone sentence
                            # Otherwise, skip very short sentences (but allow [abstract] token)
                            if len(t) < 3 and t != "[abstract]":
                                continue
                            # Skip other special token patterns (but keep [abstract])
                            if t != "[abstract]" and (re.match(r'^\[.*\]\s*$', t) or re.match(r'^\[.*\]\s*\[.*\]\s*$', t)):
                                continue
                            # Adjust offset: sent.start_char is relative to abstract_part, add abstract_content_start
                            sent_start = abstract_content_start + sent.start_char
                            sent_end = abstract_content_start + sent.end_char
                            char_spans.append((sent_start, sent_end, t))
                            sentences_found = True
                else:
                    # Fallback: if separator not found, process whole doc as one
                    sentences_found = False
                    for sent in doc_spacy.sents:
                        t = sent.text.strip()
                        if not t:
                            continue
                        if len(t) < 3:
                            continue
                        if re.match(r'^\[.*\]\s*$', t) or re.match(r'^\[.*\]\s*\[.*\]\s*$', t):
                            continue
                        char_spans.append((sent.start_char, sent.end_char, t))
                        sentences_found = True
            else:
                # For non-abstract sections or abstract without separator, process normally
                sentences_found = False
                for sent in doc_spacy.sents:
                    t = sent.text.strip()
                    if not t:
                        continue
                    # Filter out sentences that are only special tokens or very short
                    if len(t) < 3:
                        continue
                    # Additional check: skip sentences that are only special token patterns
                    # (e.g., "[abstract]", "[claim]", "[invention]", "[SEP]")
                    if re.match(r'^\[.*\]\s*$', t) or re.match(r'^\[.*\]\s*\[.*\]\s*$', t):
                        continue
                    char_spans.append((sent.start_char, sent.end_char, t))
                    sentences_found = True
            
            # Debug: if no sentences found but vis_text exists, it might be too short or only special tokens
            # This is expected for some documents where truncation leaves only special tokens
            if not sentences_found and vis_text and len(vis_text.strip()) > 0:
                # vis_text exists but no sentences found - this is okay, we'll just have CLS token
                pass

        elif unit == "noun_chunk":
            for chunk in doc_spacy.noun_chunks:
                t = chunk.text.strip()
                if not t:
                    continue
                char_spans.append((chunk.start_char, chunk.end_char, t))
        else:
            raise ValueError(f"Unknown unit: {unit}")

        spans = extract_char_spans_to_token_spans(
            char_spans=char_spans,
            prefix_len=0,
            offset_mapping=offset_map,
            input_ids=seq_input_ids,
            special_token_ids=filtered_special_token_ids,  # Use filtered set (excludes CLS)
            dedup=False
        )
        
        # For non-encoder_token modes, also add [CLS] token if keep_cls=True and it exists
        if keep_cls and cls_token_id is not None and unit != "encoder_token":
            # Find [CLS] token position (usually at index 0)
            for tok_idx, tok_id in enumerate(seq_input_ids):
                if int(tok_id.item()) == cls_token_id:
                    cls_emb = token_embeddings[tok_idx]
                    cls_emb = cls_emb.cpu().numpy()
                    cls_emb = cls_emb / (np.linalg.norm(cls_emb) + 1e-12)
                    cls_text = tokenizer.convert_ids_to_tokens([cls_token_id])[0]
                    cls_canonical = canonicalize_span_text(cls_text)
                    # Add [CLS] embedding directly (it's a single token, no pooling needed)
                    all_span_embeddings.append((doc_id, section, doc_text, cls_text, cls_canonical, cls_emb))
                    break

        for span_text, token_start, token_end in spans:
            if token_start >= token_end or token_end > len(token_embeddings):
                continue

            # Quality filtering for non-encoder_token modes (encoder_token outputs all tokens as-is)
            # For spacy_sentence, we're more lenient - only filter if span is clearly invalid
            # because sentences should generally be kept even if they're short or contain common words
            if unit != "encoder_token":
                if unit == "spacy_sentence":
                    # For sentences, only filter if they're extremely short or clearly invalid
                    # Don't apply the full filter_span_quality which is designed for tokens/noun_chunks
                    if len(span_text.strip()) < 3:
                        continue
                else:
                    # For other units (spacy_token, noun_chunk), use full quality filter
                    if not filter_span_quality(span_text):
                        continue

            # IMPORTANT: do not drop tokens; pool over all tokens in the unit span
            # Pool multiple token embeddings into a single span embedding
            if span_pooling == "max":
                span_emb = token_embeddings[token_start:token_end].max(dim=0)[0]  # max returns (values, indices), take values
            elif span_pooling == "mean":
                span_emb = token_embeddings[token_start:token_end].mean(dim=0)
            else:
                raise ValueError(f"Unknown span_pooling method: {span_pooling}. Must be 'mean' or 'max'")

            span_emb = span_emb.cpu().numpy()
            span_emb = span_emb / (np.linalg.norm(span_emb) + 1e-12)

            span_text_canonical = canonicalize_span_text(span_text)
            # Skip if canonical version is empty (all stopwords/punctuation)
            if not span_text_canonical:
                continue

            all_span_embeddings.append((doc_id, section, doc_text, span_text, span_text_canonical, span_emb))

    if keep_doc_mean:
        for i in range(len(doc_ids)):
            doc_id, section, doc_text = doc_ids[i], sections[i], doc_texts[i]
            token_embeddings = batch_embeddings[i]
            seq_input_ids = input_ids[i]
            seq_attention = attention_mask[i]
            embs = []
            for tok_idx in range(seq_attention.shape[0]):
                if int(seq_attention[tok_idx].item()) == 0:
                    continue
                if int(seq_input_ids[tok_idx].item()) in filtered_special_token_ids:
                    continue
                embs.append(token_embeddings[tok_idx].cpu().numpy())
            if embs:
                mean_emb = np.mean(embs, axis=0).astype(np.float32)
                mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
                all_span_embeddings.append((doc_id, section, doc_text, "[DOC_MEAN]", "[DOC_MEAN]", mean_emb))

    # Clean up intermediate tensors and variables to free memory
    del batch_embeddings
    del input_ids
    del attention_mask
    del offset_mapping
    del visible_texts
    if docs_spacy is not None:
        del docs_spacy
    del encoding  # Also clean up the encoding dict
    
    return all_span_embeddings


# ============================================================================
# Embedding file and directory parsing utilities
# ============================================================================

def parse_embedding_filename(filename: str) -> dict:
    """
    Parse embedding filename to extract task, model, tokenization info.
    
    Expected format: patent_contextual_spans_{mode}_{model_name}_{unit}_{cls_suffix}.{ext}
    
    Example: patent_contextual_spans_abstract2abstract_PatentMap-V0-SecPair-Claim_spacy_token_cls.npy
    
    Returns dict with keys: mode, model_name, unit, cls_suffix, or None if parsing fails.
    """
    import os
    basename = os.path.basename(filename)
    # Remove extension
    name_without_ext = os.path.splitext(basename)[0]
    # Handle .npz files (remove .npz extension)
    if name_without_ext.endswith('.npz'):
        name_without_ext = name_without_ext[:-4]
    
    # Pattern: patent_contextual_spans_{mode}_{model_name}_{unit}_{cls_suffix}
    pattern = r'patent_contextual_spans_(.+?)_(.+?)_(.+?)_(cls|nocls)$'
    match = re.match(pattern, name_without_ext)
    
    if match:
        return {
            'mode': match.group(1),  # abstract2abstract or claim2all
            'model_name': match.group(2),  # e.g., PatentMap-V0-SecPair-Claim
            'unit': match.group(3),  # e.g., spacy_token, encoder_token
            'cls_suffix': match.group(4)  # cls or nocls
        }
    else:
        # Try alternative pattern (without cls suffix, for backward compatibility)
        pattern_alt = r'patent_contextual_spans_(.+?)_(.+?)_(.+?)$'
        match_alt = re.match(pattern_alt, name_without_ext)
        if match_alt:
            return {
                'mode': match_alt.group(1),
                'model_name': match_alt.group(2),
                'unit': match_alt.group(3),
                'cls_suffix': 'unknown'
            }
        return None


def parse_embeddings_dir(dirname: str) -> dict:
    """
    Parse embeddings directory name from 1create_N_embeddings.py output.
    
    Expected format: embeddings_{model_name}_{unit}_{cls}_{layer}
    
    Example: embeddings_bert-for-patents_spacy_token_cls_last
    
    Returns dict with keys: model_name, unit, cls_suffix, layer, or None if parsing fails.
    """
    import os
    basename = os.path.basename(dirname)
    
    # Pattern: embeddings_{model_name}_{unit}_{cls}_{layer}
    pattern = r'embeddings_(.+?)_(.+?)_(cls|nocls)_(last|second_last)$'
    match = re.match(pattern, basename)
    
    if match:
        return {
            'model_name': match.group(1),  # e.g., bert-for-patents
            'unit': match.group(2),  # e.g., spacy_token, spacy_sentence
            'cls_suffix': match.group(3),  # cls or nocls
            'layer': match.group(4)  # last or second_last
        }
    return None


def find_embedding_files(embeddings_dir: str, mode: str, unit: str = None) -> list:
    """
    Find embedding files in directory from 1create_N_embeddings.py output based on task mode.
    
    For abstract2abstract: returns [abstract_{unit}.npy/npz]
    For claim2all: returns [abstract_{unit}.npy/npz, claim_{unit}.npy/npz, invention_{unit}.npy/npz]
    
    If unit is not provided, tries to infer from directory name or scans for available files.
    
    Returns list of file paths, or empty list if not found.
    """
    import os
    if not os.path.isdir(embeddings_dir):
        return []
    
    # Determine which sections are needed based on mode
    if mode == "abstract2abstract":
        required_sections = ['abstract']
    elif mode == "claim2all":
        required_sections = ['abstract', 'claim', 'invention']
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes: abstract2abstract, claim2all")
    
    # If unit not provided, try to infer from directory name
    if unit is None:
        dir_info = parse_embeddings_dir(embeddings_dir)
        if dir_info:
            unit = dir_info['unit']
        else:
            # Scan for available files to infer unit
            for f in os.listdir(embeddings_dir):
                for section in required_sections:
                    if f.startswith(f"{section}_") and (f.endswith('.npy') or f.endswith('.npz')):
                        # Extract unit from filename: {section}_{unit}.{ext}
                        unit = f.replace(f"{section}_", "").replace('.npy', '').replace('.npz', '')
                        break
                if unit:
                    break
    
    if unit is None:
        return []
    
    # Find files for each required section
    found_files = []
    for section in required_sections:
        # Try .npy first, then .npz
        for ext in ['.npy', '.npz']:
            filepath = os.path.join(embeddings_dir, f"{section}_{unit}{ext}")
            if os.path.exists(filepath):
                found_files.append(filepath)
                break
    
    # Return only if all required sections are found
    if len(found_files) == len(required_sections):
        return found_files
    else:
        return []


# ============================================================================
# Vector normalization utilities
# ============================================================================

def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize vectors (returns new array, non-destructive).
    
    Args:
        X: Array of shape [N, d] to normalize
        eps: Small epsilon to prevent division by zero
    
    Returns:
        Normalized array of same shape
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, eps, None)


def l2_normalize_inplace(X: np.ndarray, eps: float = 1e-12):
    """
    L2-normalize vectors in-place (modifies input array).
    
    Args:
        X: Array of shape [N, d] to normalize (will be modified)
        eps: Small epsilon to prevent division by zero
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X /= np.clip(norms, eps, None)


# ================== Evaluation / formatting helpers (used by baselines.py) ==================

def log_embeddings_shape(embeddings_dict, context=""):
    """Log embedding shapes consistently."""
    if context:
        print(f"{context}:")
    for name, embeddings in embeddings_dict.items():
        print(f"  {name}: {embeddings.shape}")


def print_subsection_header(title, width=60):
    """Print a formatted subsection header."""
    print(f"\n{'-' * width}")
    print(f" {title}")
    print(f"{'-' * width}")


def print_metric_table(results_dict, task_name, precision=4):
    """
    Print results in a clean table format.
    results_dict: metric names -> values; task_name: evaluation task name.
    """
    print(f"\n📊 {task_name} Results:")
    print("-" * 50)
    if not results_dict:
        print("   No results available")
        return

    def metric_sort_key(metric_name):
        match = re.match(r'([^@]+)@?(\d+)?', metric_name)
        if match:
            base_name = match.group(1)
            number = int(match.group(2)) if match.group(2) else 0
            return (base_name, number)
        return (metric_name, 0)

    sorted_keys = sorted(results_dict.keys(), key=metric_sort_key)
    for key in sorted_keys:
        value = results_dict[key]
        if isinstance(value, float):
            formatted_value = f"{value:.6f}" if abs(value) < 0.001 else f"{value:.{precision}f}"
        elif isinstance(value, dict):
            formatted_value = str(value)
        else:
            formatted_value = str(value)
        print(f"   📋 {key:<25}: {formatted_value}")


def mean_pooling(token_embeddings, attention_mask):
    """Mean pooling on token embeddings. Returns (batch_size, hidden_dim)."""
    input_mask_expanded = attention_mask.unsqueeze(-1).to(token_embeddings.device)
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cls_pooling(model_output, attention_mask):
    """CLS token as sentence representation."""
    return model_output.last_hidden_state[:, 0]


def compute_rankings(top_indices):
    """Convert top_k indices to rank matrix (1-based)."""
    rankings = np.empty_like(top_indices)
    for query_idx, doc_order in enumerate(top_indices):
        rankings[query_idx, doc_order] = np.arange(1, len(doc_order) + 1)
    return rankings


def find_centers(dense_model: str, tokenization_unit: str, include_cls: bool, search_dir: str = ".",
                 mode: str = "abstract2abstract", layer: str = "last", centers_suffix: str = "") -> tuple:
    """
    Find centers directory and .npy file for sparse_coverage eval.
    Returns (centers_path, centers_dir).
    """
    import glob
    model_name = dense_model.strip("/").split("/")[-1].replace("/", "_").replace("\\", "_")
    cls_suffix = "cls" if include_cls else "nocls"
    expected_dir_pattern = f"centers_greedy_{mode}_{model_name}_{tokenization_unit}_{cls_suffix}_{layer}{centers_suffix}"
    search_pattern = os.path.join(search_dir, expected_dir_pattern)
    matching_dirs = glob.glob(search_pattern)
    if not matching_dirs:
        matching_dirs = glob.glob(os.path.join(search_dir, "**", expected_dir_pattern), recursive=True)
    if not matching_dirs:
        raise FileNotFoundError(
            f"Could not find centers directory matching: {expected_dir_pattern}\n"
            f"Searched in: {os.path.abspath(search_dir)}\n"
            f"  dense_model={dense_model}\n  tokenization_unit={tokenization_unit}\n"
            f"  include_cls={include_cls}\n  mode={mode}\n  layer={layer}"
        )
    centers_dir = matching_dirs[0]
    if len(matching_dirs) > 1:
        matching_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        centers_dir = matching_dirs[0]
        print(f"⚠️  Found {len(matching_dirs)} matching directories, using: {centers_dir}")
    centers_pattern = os.path.join(centers_dir, "centers_greedy_r*.npy")
    centers_files = glob.glob(centers_pattern)
    if not centers_files:
        raise FileNotFoundError(f"Could not find centers file in: {centers_dir}\nExpected pattern: centers_greedy_r*.npy")
    centers_path = centers_files[0]
    if len(centers_files) > 1:
        centers_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        centers_path = centers_files[0]
        print(f"⚠️  Found {len(centers_files)} centers files, using: {centers_path}")
    return centers_path, centers_dir