#!/usr/bin/env python
"""
evaluate.py

This script evaluates baseline models (without training) on our patent evaluation tasks.
It loads a pretrained model and computes tokenization and embeddings on-the-fly using the model's tokenizer.
If precomputed embeddings are present in the expected temp directories, the script will load them to
speed up repeated runs instead of recomputing embeddings.


Usage example:
    python evaluate.py --model_name <path_or_model_id> --temp_dir ./temp
"""
import os
import sys
import re
import gc
import json
import time
import hashlib
import argparse
import logging
import contextlib
try:
    import fcntl  # POSIX only; used for cross-process flock on the centers_search cache
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None
from collections import defaultdict
from typing import Optional

from tqdm import trange, tqdm
import numpy as np

import faiss
import torch

from transformers import set_seed,  AutoTokenizer, AutoModel
from scipy.sparse import isspmatrix

import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead."
)

# ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# add patenteval to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
patent_eval_path = os.path.join(current_dir, 'patentmap_eval')
sys.path.append(patent_eval_path)

# Try to import patenteval.utils with better error handling
try:
    from patenteval.utils import (
        mean_recall_at_k,
        mean_ndcg_at_k,
        mean_average_precision,
        mean_pres_at_k,
        mean_mrr_at_k,
    )
    print("Successfully imported patenteval.utils")
except ImportError as e:
    print(f"Warning: Could not import patenteval.utils: {e}")
    print(f"patentmap_eval path: {patent_eval_path}")
    print(f"patentmap_eval exists: {os.path.exists(patent_eval_path)}")
    print("Available paths in sys.path:")
    for p in sys.path[-3:]:  # Show last 3 paths
        print(f"  {p}")
    print("Please ensure patentmap_eval is present and contains an __init__.py file.")
    # You might want to exit here or provide fallback implementations
    sys.exit(1)


# Evaluation/formatting and sparse_coverage helpers (from utils)
from utils import (
    print_subsection_header,
    print_metric_table,
    mean_pooling,
    cls_pooling,
    get_encoder_format_scheme,
    format_abstract_for_encoder,
    format_claim_for_encoder,
    format_invention_for_encoder,
    find_centers,
    hash_query_texts as _hash_query_texts,
    sparse_to_csr as _sparse_to_csr,
    is_st_checkpoint as _is_st_checkpoint,
    auto_batch_size as _auto_batch_size,
)


def load_checkpoint_model(checkpoint_path, max_length=512, hf_model_name=None):
    """
    Load a sentence transformer checkpoint without dense layers, or load a HuggingFace model.
    If hf_model_name is provided, loads from HuggingFace; otherwise loads from checkpoint.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import models

    if hf_model_name:
        print(f"Loading HuggingFace model: {hf_model_name}")
        model = SentenceTransformer(hf_model_name)
        print("Model loaded from HuggingFace Hub!")
        return model
    print(f"Loading checkpoint from: {checkpoint_path}")
    # Load the transformer model
    transformer = models.Transformer(checkpoint_path, max_seq_length=max_length)
    # Load pooling config
    pooling_config_path = os.path.join(checkpoint_path, "1_Pooling", "config.json")
    if os.path.exists(pooling_config_path):
        with open(pooling_config_path, 'r') as f:
            pooling_config = json.load(f)
        pooling = models.Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode_cls_token=pooling_config.get('pooling_mode_cls_token', False),
            pooling_mode_mean_tokens=pooling_config.get('pooling_mode_mean_tokens', True),
            pooling_mode_max_tokens=pooling_config.get('pooling_mode_max_tokens', False)
        )
    else:
        # Default to mean pooling
        pooling = models.Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
    # Create model with only transformer and pooling (no dense layers)
    model = SentenceTransformer(modules=[transformer, pooling])
    print("Model loaded successfully (transformer + pooling only)")
    return model


def _clefip_two_stage_rerank(
    passage_ids: list,
    passage_scores_list: list,
    topk_docs: int = 100,
) -> tuple:
    """
    Two-stage CLEF-IP passage retrieval:
      Stage 1: Rank passages by score, derive document ranking (first occurrence dedup).
               Keep only the top-K documents.
      Stage 2: Among ALL passages in the corpus belonging to those top-K documents,
               re-rank by the original passage scores.  Passages from documents not in
               top-K are excluded — this eliminates noise from spurious high-score passages
               in irrelevant documents.

    Args:
        passage_ids:            full corpus passage_id list (needed to know ALL passages
                                belonging to each document, including unscored ones).
        passage_scores_list:    per-query list of dicts {passage_id: score} covering all
                                scored passages.
        topk_docs:              number of top documents to keep after Stage 1 (default 100).

    Returns:
        (reranked_list, doc_ranking_list, full_doc_ranking_list):
          reranked_list:         per-query list of passage_ids (all passages from top-K docs, sorted by score).
          doc_ranking_list:      per-query list of doc_ids from Stage 1 dedup, truncated to topk_docs.
                                 Used for @k document metrics (PRES@100, recall@100, NDCG@10).
          full_doc_ranking_list: per-query list of doc_ids from Stage 1 dedup, NOT truncated
                                 (all docs touched by any scored passage). Used for untruncated map_doc.
    """
    # Pre-build passage_id -> doc_id mapping and doc_id -> set(passage_ids)
    pid_to_doc = {}
    doc_to_pids: dict[str, set] = {}
    for pid in passage_ids:
        doc_id = _clefip_passage_id_to_doc_id(pid)
        pid_to_doc[pid] = doc_id
        doc_to_pids.setdefault(doc_id, set()).add(pid)

    reranked_list = []
    doc_ranking_list = []
    full_doc_ranking_list = []
    for q_idx in range(len(passage_scores_list)):
        scores = passage_scores_list[q_idx]

        # Stage 1: rank passages by score desc, derive full document ranking (first-occurrence dedup)
        ranked_pids = sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)
        seen_docs = set()
        full_docs = []
        for pid in ranked_pids:
            doc_id = pid_to_doc.get(pid, pid.split("::", 1)[0] if "::" in pid else pid)
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                full_docs.append(doc_id)
        full_doc_ranking_list.append(full_docs)
        top_docs = full_docs[:topk_docs]
        doc_ranking_list.append(top_docs)

        # Stage 2: collect ALL passages from those top-K documents, re-rank by score
        candidate_pids = set()
        for doc_id in top_docs:
            candidate_pids.update(doc_to_pids.get(doc_id, set()))

        scored_candidates = [(pid, scores.get(pid, -1e9)) for pid in candidate_pids]
        scored_candidates.sort(key=lambda x: -x[1])
        reranked_list.append([pid for pid, _ in scored_candidates])

    return reranked_list, doc_ranking_list, full_doc_ranking_list


def clefip_passage_evaluation(
    query_ids,
    passage_ids,
    query_embeddings,
    passage_embeddings,
    qrels_passage_ids,
    k=100,
    model_label="Dense",
    topk_docs=100,
):
    """
    Evaluate CLEF-IP claims-to-passages: rank passages per query and compute metrics.
    qrels_passage_ids: dict topic_id -> list of relevant passage_ids (subset of passage_ids).
    Official CLEF-IP metrics: document-level PRES@100 (pres_doc@100), passage-level MAgP (magp),
    plus passage-level recall_passage@1000 and document-level recall@100 / NDCG@10 / MAP.

    Two-stage retrieval is always applied:
      Stage 1: passage rank → document dedup → top-K documents.
      Stage 2: re-rank ALL passages from those top-K docs by cosine similarity.
    """
    assert len(query_ids) == len(query_embeddings)
    assert len(passage_ids) == len(passage_embeddings)
    set_seed(42)
    Q = np.array(query_embeddings, dtype=np.float32, copy=True)
    D = np.array(passage_embeddings, dtype=np.float32, copy=True)
    assert Q.shape[1] == D.shape[1]
    faiss.normalize_L2(Q)
    faiss.normalize_L2(D)
    _report_dense_flops(Q, D, "CLEF-IP passage", model_label=model_label)
    _n_passages_oom = D.shape[0]
    _oom_threshold = 500_000
    if _n_passages_oom > _oom_threshold:
        import warnings as _w
        _w.warn(
            f"clefip_passage_evaluation: computing full Q@D.T similarity matrix with "
            f"{_n_passages_oom:,} passages × {Q.shape[0]} queries × {Q.shape[1]}d "
            f"≈ {Q.shape[0]*_n_passages_oom*Q.shape[1]*4/1e9:.1f} GB. "
            f"Consider using FAISS IVF for corpora larger than {_oom_threshold:,} passages.",
            ResourceWarning, stacklevel=2,
        )
    sim = Q @ D.T
    passage_scores_list = [
        {passage_ids[j]: float(sim[q_idx, j]) for j in range(sim.shape[1])}
        for q_idx in range(len(query_ids))
    ]
    predicted_labels_list, doc_ranking_list, full_doc_ranking_list = _clefip_two_stage_rerank(
        passage_ids, passage_scores_list, topk_docs=topk_docs,
    )
    print(f"  🔄 Two-stage retrieval: top-{topk_docs} docs → re-ranked passages per query")
    results = _evaluate_and_print_clefip(
        qrels_passage_ids, query_ids, predicted_labels_list,
        "Passage retrieval",
        two_stage=True, topk_docs=topk_docs,
        doc_ranking_list=doc_ranking_list,
        full_doc_ranking_list=full_doc_ranking_list,
    )
    return results


def _clefip_passage_section(passage_id: str) -> str:
    """Derive section label from passage_id (doc_id::xpath). Returns 'abstract', 'description', or 'claim'. Maps to prior-art style: description -> invention for models that use [invention]."""
    if "::" not in passage_id:
        return "abstract"
    xpath = passage_id.split("::", 1)[1]
    if "abstract" in xpath:
        return "abstract"
    if "claims" in xpath:
        return "claim"
    return "description"


def _clefip_passage_id_to_doc_id(passage_id: str) -> str:
    """Extract doc_id from passage_id (format doc_id::xpath)."""
    return passage_id.split("::", 1)[0] if "::" in passage_id else passage_id



def _clefip_mean_agp_passage(true_labels_list: list, predicted_labels_list: list, k: int = 100) -> float:
    """
    Official CLEF-IP passage-level MAP(D) — Mean Average Precision at Document level.

    Per Piroi et al. (CLEF-IP 2012), passage-level evaluation measures system
    performance for ranking passages *within each relevant document*:

      1. For each relevant document D_i of topic T, extract the subsequence of
         passages belonging to D_i from the system's global passage ranking
         (preserving relative order).
      2. Compute AP(D_i; T) on that document-local ranking:
           AP(D_i; T) = (1 / n_p(D_i; T)) * sum_r [Precision(r) * rel(r)]
         where r indexes the document-local ranked list, n_p is the number of
         relevant passages in D_i for topic T.
      3. Average across all relevant documents of the topic:
           AP(D; T) = sum_i AP(D_i; T) / n(T)
      4. Average across all topics to get MAP(D).

    If a relevant document has zero passages retrieved, AP(D_i) = 0 (recall penalty).
    """
    if not true_labels_list:
        return 0.0

    topic_scores = []
    for q_true_passages, q_pred_passages in zip(true_labels_list, predicted_labels_list):
        if not q_true_passages:
            # No relevant passages for this topic — skip (undefined)
            continue

        # Group relevant passages by document
        doc_to_rel_passages: dict[str, set] = {}
        for pid in q_true_passages:
            doc_id = _clefip_passage_id_to_doc_id(pid)
            doc_to_rel_passages.setdefault(doc_id, set()).add(pid)

        n_relevant_docs = len(doc_to_rel_passages)
        if n_relevant_docs == 0:
            continue

        # Truncate predicted list to top-k
        pred_truncated = q_pred_passages[:k] if k else q_pred_passages

        doc_ap_sum = 0.0
        for doc_id, rel_pids in doc_to_rel_passages.items():
            # Extract document-local subsequence from predicted ranking
            doc_local_ranking = [
                pid for pid in pred_truncated
                if _clefip_passage_id_to_doc_id(pid) == doc_id
            ]
            # Compute AP within this document-local ranking
            n_rel = len(rel_pids)
            hits = 0
            ap_sum = 0.0
            for rank_idx, pid in enumerate(doc_local_ranking):
                if pid in rel_pids:
                    hits += 1
                    ap_sum += hits / (rank_idx + 1)
            ap_d = ap_sum / n_rel  # 0 if no hits (recall penalty)
            doc_ap_sum += ap_d

        topic_scores.append(doc_ap_sum / n_relevant_docs)

    return float(np.mean(topic_scores)) if topic_scores else 0.0


def _dense_chunk_text(text: str, tokenizer_or_none, max_tokens: int = 512) -> list[str]:
    """Split *text* into sentence-aligned chunks each fitting within *max_tokens* encoder tokens.

    Uses *tokenizer_or_none* for exact token counting when available; otherwise falls back to a
    word-count heuristic (~350 words ≈ 450–500 tokens for patent text).

    Returns a list with one or more non-empty chunks.
    """
    if tokenizer_or_none is not None:
        tok = tokenizer_or_none
        n_tokens = len(tok.encode(text, add_special_tokens=True))
        if n_tokens <= max_tokens:
            return [text]
        # Split on sentence boundaries (period + whitespace); fall back to whitespace.
        import re as _re2
        sents = [s for s in _re2.split(r'(?<=\.)\s+', text) if s.strip()]
        if len(sents) <= 1:
            sents = [w for w in text.split() if w]
        # Pre-tokenize each sentence ONCE (avoids re-encoding every growing prefix).
        sent_lens = [len(tok.encode(s, add_special_tokens=False)) for s in sents]
        # Special tokens (CLS + SEP) cost 2 tokens in most encoders.
        _special = 2
        chunks: list[str] = []
        current_sents: list[str] = []
        current_len = _special
        for sent, slen in zip(sents, sent_lens):
            # +1 for the space separator between sentences
            added = slen + (1 if current_sents else 0)
            if current_len + added <= max_tokens:
                current_sents.append(sent)
                current_len += added
            else:
                if current_sents:
                    chunks.append(" ".join(current_sents))
                current_sents = [sent]
                current_len = _special + slen
        if current_sents:
            chunks.append(" ".join(current_sents))
        return chunks if chunks else [text]
    else:
        # Word-count heuristic: ~350 words per chunk (≈ 450–500 tokens for patent text).
        words = text.split()
        max_words = 350
        if len(words) <= max_words:
            return [text]
        chunks = []
        for start in range(0, len(words), max_words):
            chunks.append(" ".join(words[start:start + max_words]))
        return chunks


def _dense_chunk_maxsim_scores(
    query_texts_fmt: list[str],
    passage_ids: list[str],
    passage_emb: np.ndarray,
    encode_fn,
    tokenizer_or_none,
    query_max_chunks: int = -1,
    batch_size: int = 32,
) -> list[dict]:
    """Dense chunk + max-sim retrieval.

    Each query is split into ≤512-token sentence-aligned chunks. All chunks are encoded as
    independent query vectors. The score for a (query, passage) pair is the **maximum**
    cosine similarity across all chunks from that query.

    Returns passage_scores_list: per-query list of {passage_id: float}.
    """
    # Build flat chunk list
    flat_texts: list[str] = []
    flat_q_indices: list[int] = []
    n_chunked = 0
    for q_idx, text in enumerate(query_texts_fmt):
        chunks = _dense_chunk_text(text, tokenizer_or_none, max_tokens=512)
        if query_max_chunks > 0 and len(chunks) > query_max_chunks:
            chunks = chunks[:query_max_chunks]
        if len(chunks) > 1:
            n_chunked += 1
        for c in chunks:
            flat_texts.append(c)
            flat_q_indices.append(q_idx)

    cap_str = f"cap={query_max_chunks}" if query_max_chunks > 0 else "unlimited"
    print(f"   Chunk+MaxSim: {n_chunked}/{len(query_texts_fmt)} queries split "
          f"→ {len(flat_texts)} total chunks ({cap_str})")

    # Encode all chunks
    chunk_emb = np.array(encode_fn(flat_texts, batch_size=batch_size), dtype=np.float32)
    faiss.normalize_L2(chunk_emb)

    D = np.array(passage_emb, dtype=np.float32, copy=True)
    faiss.normalize_L2(D)

    # sim_chunks[c, p] = cosine(chunk_c, passage_p)
    sim_chunks = chunk_emb @ D.T  # [n_chunks_total, n_passages]

    flat_q_arr = np.array(flat_q_indices, dtype=np.int32)
    n_q = len(query_texts_fmt)
    n_p = D.shape[0]

    passage_scores_list: list[dict] = []
    for q_idx in range(n_q):
        rows = np.where(flat_q_arr == q_idx)[0]
        if rows.size == 1:
            q_scores = sim_chunks[rows[0]]          # [n_passages]
        else:
            q_scores = sim_chunks[rows].max(axis=0) # max over chunks [n_passages]
        passage_scores_list.append({passage_ids[j]: float(q_scores[j]) for j in range(n_p)})

    return passage_scores_list


def _splade_chunk_merge_query(
    query_texts: list[str],
    splade_model,
    encode_bs: int,
    query_max_chunks: int = -1,
) -> "csr_matrix":
    """Encode long SPLADE queries via chunk + term-level max merge.

    Each query is split into sentence-aligned ≤512-token chunks using the model's own
    tokenizer (accessed via ``splade_model.tokenizer``) for exact token counting.
    Each chunk is encoded with ``encode_query``; the final query sparse vector is the
    elementwise maximum over all chunk vectors for that query.

    Returns a CSR matrix of shape ``[n_queries, vocab_size]``.
    """
    # Obtain the tokenizer from the SparseEncoder (SentenceTransformer-based).
    _tokenizer = getattr(splade_model, "tokenizer", None)

    # Build flat chunk list using _dense_chunk_text for exact 512-token splitting.
    flat_chunks: list[str] = []
    flat_q_idx: list[int] = []
    n_chunked = 0
    for q_idx, text in enumerate(query_texts):
        chunks = _dense_chunk_text(text, _tokenizer, max_tokens=512)
        if query_max_chunks > 0 and len(chunks) > query_max_chunks:
            chunks = chunks[:query_max_chunks]
        if len(chunks) > 1:
            n_chunked += 1
        for c in chunks:
            flat_chunks.append(c)
            flat_q_idx.append(q_idx)

    cap_str = f"cap={query_max_chunks}" if query_max_chunks > 0 else "unlimited"
    print(f"   SPLADE Chunk+TermMax: {n_chunked}/{len(query_texts)} queries split "
          f"→ {len(flat_chunks)} total chunks ({cap_str})")

    # Encode all chunks at once
    chunk_sparse = splade_model.encode_query(flat_chunks, batch_size=encode_bs, show_progress_bar=True)
    Q_chunks = _sparse_to_csr(chunk_sparse)   # [n_chunks_total, vocab_size]

    flat_q_arr = np.array(flat_q_idx, dtype=np.int32)
    n_q = len(query_texts)
    vocab_size = Q_chunks.shape[1]

    # Elementwise max per query: for each query, stack its chunk rows and take column-wise max.
    # Using lil_matrix row assignment for efficiency.
    from scipy.sparse import lil_matrix as _lil
    Q_merged = _lil((n_q, vocab_size), dtype=np.float32)
    for q_idx in range(n_q):
        rows = np.where(flat_q_arr == q_idx)[0]
        if rows.size == 1:
            Q_merged[q_idx] = Q_chunks[int(rows[0])]
        else:
            # Stack chunk rows and take column-wise max (dense path; chunk count is small).
            stacked = Q_chunks[rows].toarray()       # [n_chunks, vocab_size]
            merged = stacked.max(axis=0)             # [vocab_size]
            Q_merged[q_idx] = merged

    return Q_merged.tocsr()


def _get_clefip_dense_encoder(args, model_name: str, device):
    """
    Load the dense model for CLEF-IP and return (encode_fn, model_label, tokenizer_or_none).
    encode_fn(texts: list[str], batch_size=32) -> np.ndarray.
    tokenizer_or_none: HuggingFace tokenizer for chunk-splitting, or None for SentenceTransformer models.
    """
    def _batch_encode(texts, tokenizer, model, device, forward_and_pool, hidden_size, batch_size=32):
        """Batch-encode texts with a custom forward+pool function.
        forward_and_pool(inp_dict) -> np.ndarray (cpu, 2D)."""
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inp = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                embs.append(forward_and_pool(model, inp))
        return np.vstack(embs) if embs else np.zeros((0, hidden_size), dtype=np.float32)

    if model_name in ["allenai/specter2_base"]:
        from adapters import AutoAdapterModel
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoAdapterModel.from_pretrained(args.model_name)
        model.load_adapter("allenai/specter2", source="hf",
                           load_as="proximity", set_active=True)
        model.to(device)
        model.eval()

        def _fwd_cls(m, inp):
            return cls_pooling(m(**inp), inp["attention_mask"]).cpu().numpy()

        def _encode(texts, batch_size=32):
            return _batch_encode(texts, tokenizer, model, device, _fwd_cls,
                                 model.config.hidden_size, batch_size)
        return _encode, "Specter2", tokenizer

    if model_name in ["mpi-inno-comp/paecter", "anferico/bert-for-patents"]:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name)
        if "anferico/bert-for-patents" in model_name:
            tokenizer.add_special_tokens({'additional_special_tokens': ['[abstract]', '[claim]', '[invention]']})
            model.resize_token_embeddings(len(tokenizer))
        model.to(device)

        def _fwd_mean(m, inp):
            return mean_pooling(m(**inp).last_hidden_state, inp["attention_mask"]).cpu().numpy()
        def _encode(texts, batch_size=32):
            return _batch_encode(texts, tokenizer, model, device, _fwd_mean, model.config.hidden_size, batch_size)
        return _encode, "PAECTer" if "paecter" in model_name else "bert-for-patents", tokenizer

    if model_name in ["datalyes/patembed-large", "patembed-large"]:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("datalyes/patembed-large")
        model.to(device)
        PATEN_TEB_RETRIEVAL_PROMPT_NAME = "retrieval_MIXED"
        # Resolve prompt prefixes: PatenTEB stores dict prompts {q_text:..., pos_text:...}
        # which some sentence-transformers versions cannot handle via encode_query/encode_document.
        _patembed_prompt = getattr(model, "prompts", {}).get(PATEN_TEB_RETRIEVAL_PROMPT_NAME, {})
        if isinstance(_patembed_prompt, dict):
            _patembed_q_prefix = _patembed_prompt.get("q_text", "")
            _patembed_d_prefix = _patembed_prompt.get("pos_text", "")
        else:
            _patembed_q_prefix = ""
            _patembed_d_prefix = ""
        print(f"🔍 Loading Patembed (bi-encoder): {model_name}")
        print(f"   Using PatenTEB retrieval prompts: prompt_name={PATEN_TEB_RETRIEVAL_PROMPT_NAME}"
              f" (required for best performance)")
        if _patembed_q_prefix or _patembed_d_prefix:
            print(f"   Query prefix:    {_patembed_q_prefix!r}")
            print(f"   Document prefix: {_patembed_d_prefix!r}")
        def _encode(texts, batch_size=256, role="document"):
            prefix = _patembed_q_prefix if role == "query" else _patembed_d_prefix
            if prefix:
                texts = [prefix + t for t in texts]
            return model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        return _encode, "Patembed", None  # SentenceTransformer: use word-count heuristic for chunking

    # PatentMap models: official usage uses CLS token embedding.
    if "patentmap" in model_name.lower():
        import utils
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
        utils.ensure_section_tokens(tokenizer, model)
        model.to(device).eval()

        def _fwd_patentmap(m, inp):
            out = m(**inp, output_hidden_states=True, return_dict=True)
            return out.last_hidden_state[:, 0].cpu().numpy()
        def _encode(texts, batch_size=32):
            return _batch_encode(texts, tokenizer, model, device, _fwd_patentmap, model.config.hidden_size, batch_size)
        return _encode, "PatentMap", tokenizer

    # SentenceTransformer checkpoint (e.g. ./checkpoint-1142): load without dense layers
    if _is_st_checkpoint(args.model_name):
        model = load_checkpoint_model(args.model_name)
        model.to(device)
        _ckpt_label = os.path.basename(os.path.normpath(args.model_name))
        def _encode(texts, batch_size=32):
            return model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        return _encode, _ckpt_label, None  # SentenceTransformer checkpoint: use word-count heuristic

    # Fallback
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)

    def _fwd_mean(m, inp):
        return mean_pooling(m(**inp).last_hidden_state, inp["attention_mask"]).cpu().numpy()
    def _encode(texts, batch_size=32):
        return _batch_encode(texts, tokenizer, model, device, _fwd_mean, model.config.hidden_size, batch_size)
    return _encode, "Dense", tokenizer


def _clefip_format_for_model(query_texts, passage_ids, passage_texts, model_name, passage_titles=None):
    """
    Format CLEF-IP query/passage texts per utils encoder format scheme. Returns (query_list, passage_list).

    Both schemes get the same shape:
      - query (always a claim) → format_claim_for_encoder
      - abstract passage       → format_abstract_for_encoder(title, text)  ← title from XML
      - description passage    → format_invention_for_encoder
      - claim passage          → format_claim_for_encoder

    Differences come from the scheme's per-section formatting:
      - ENCODER_FORMAT_SECTION_TOKENS (PatentMap, bert-for-patents):
          [claim] query / "title [SEP] [abstract] text" / [invention] text / [claim] text
      - ENCODER_FORMAT_NO_SECTION_MARKERS (Specter2, PAECTer, Patembed):
          plain query / "title [SEP] text" (abstract only) / plain text / plain text

    passage_titles: optional list of title strings parallel to passage_ids/passage_texts.
    """
    if not query_texts or not passage_texts:
        return query_texts, passage_texts
    scheme = get_encoder_format_scheme(model_name)
    query_fmt = [format_claim_for_encoder(scheme, t) for t in query_texts]
    passage_sections = [_clefip_passage_section(pid) for pid in passage_ids]
    titles_iter = passage_titles if passage_titles else [""] * len(passage_texts)
    passage_fmt = []
    for s, t, title in zip(passage_sections, passage_texts, titles_iter):
        if s == "claim":
            passage_fmt.append(format_claim_for_encoder(scheme, t))
        elif s == "abstract":
            passage_fmt.append(format_abstract_for_encoder(scheme, title, t))
        else:
            passage_fmt.append(format_invention_for_encoder(scheme, t))
    return query_fmt, passage_fmt


def _report_flops_and_postings_one_line(
    total_postings: int,
    n_non_empty_terms: int,
    n_terms: int,
    label: str,
    total_flops: Optional[int] = None,
    n_queries: int = 0,
    model_label: str = "Sparse",
):
    """Report FLOPs (if available) and one line: total postings + non-empty term count. FLOPs = 2 * sum over queries of (sum of |L_t| for t in supp(q))."""
    print(f"\n📊 Efficiency — {model_label} {label}")
    if total_flops is not None and n_queries > 0:
        print(f"   FLOPs: total={total_flops:,}, mean per query={total_flops // n_queries:,} (2 × sum of posting lengths per query term)")
    else:
        print(f"   FLOPs: not available (query term ids or vocab mapping unavailable)")
    print(f"   Total postings: {total_postings:,}, non-empty terms: {n_non_empty_terms:,} / {n_terms:,}")


def _report_dense_flops(Q_emb, D_emb, label: str, model_label: str = "Dense"):
    """Dense retrieval: FLOPs = 2 × n_queries × n_docs × dim (similarity matrix Q @ D.T)."""
    n_q, dim = Q_emb.shape[0], Q_emb.shape[1]
    n_d = D_emb.shape[0]
    total_flops = 2 * n_q * n_d * dim
    print(f"\n📊 Efficiency — {model_label} {label}")
    print(f"   FLOPs: total={total_flops:,}, mean per query={total_flops // n_q if n_q else 0:,} (2×Q×N×d similarity)")
    print(f"   Documents: {n_d:,}, dimension: {dim:,}")


def _report_bm25_posting_stats(retriever, label: str, query_tokens_list=None):
    """Report total postings + non-empty terms; if query_tokens_list is provided, compute FLOPs.
    bm25s: tokenize() returns a Tokenized object (.ids, .vocab); use retriever.get_tokens_ids() to map query tokens to index term ids for FLOPs."""
    doc_freq = getattr(retriever, "doc_freq", None)
    if doc_freq is None and hasattr(retriever, "scores") and isinstance(getattr(retriever, "scores", None), dict):
        indptr = retriever.scores.get("indptr")
        if indptr is not None:
            doc_freq = np.diff(np.asarray(indptr)).astype(np.float32)
    if doc_freq is None:
        print(f"\n📊 BM25 ({label}): inverted index built (posting lengths not exposed by library).")
        return
    pl_lens = np.asarray(doc_freq).ravel().astype(np.float32)
    n_empty = int(np.sum(pl_lens == 0))
    total_entries = int(np.sum(pl_lens))
    V = len(pl_lens)
    total_flops = None
    n_queries = 0
    if query_tokens_list is not None and hasattr(retriever, "get_tokens_ids"):
        # Normalise to list of list of token strings so we can use retriever.get_tokens_ids()
        queries_as_str_lists = []
        if hasattr(query_tokens_list, "ids") and hasattr(query_tokens_list, "vocab"):
            # bm25s Tokenized: .vocab can be token->id or id->token; .ids is list of list of id
            vocab = getattr(query_tokens_list, "vocab", {})
            if not vocab:
                pass
            else:
                sample_k = next(iter(vocab.keys()))
                if isinstance(sample_k, (int, np.integer)):
                    id2token = vocab
                else:
                    id2token = {v: k for k, v in vocab.items()}
                for q_ids in query_tokens_list.ids:
                    queries_as_str_lists.append([id2token.get(i, "") for i in q_ids if i in id2token])
        elif isinstance(query_tokens_list, (list, tuple)) and len(query_tokens_list) > 0:
            first = query_tokens_list[0]
            if isinstance(first, (list, tuple)):
                if len(first) > 0 and isinstance(first[0], str):
                    queries_as_str_lists = list(query_tokens_list)
                else:
                    # list of list of int: need id->token from retriever to get strings
                    id2token = getattr(retriever, "vocab_dict", None)
                    if id2token is not None and not callable(id2token) and id2token:
                        # vocab_dict can be token->id; if keys are int then it's id->token
                        sample_k = next(iter(id2token.keys()))
                        if isinstance(sample_k, (int, np.integer)):
                            queries_as_str_lists = [[id2token.get(i, "") for i in q] for q in query_tokens_list]
                        else:
                            queries_as_str_lists = []
                    else:
                        queries_as_str_lists = []
            else:
                queries_as_str_lists = []
        if queries_as_str_lists:
            n_queries = len(queries_as_str_lists)
            flops_sum = 0
            for q_tokens in queries_as_str_lists:
                if not q_tokens:
                    continue
                try:
                    ids = retriever.get_tokens_ids(list(q_tokens))
                except Exception:
                    continue
                ids = [i for i in ids if 0 <= i < len(pl_lens)]
                for t in set(ids):
                    flops_sum += int(pl_lens[t])
            if flops_sum > 0:
                total_flops = 2 * flops_sum
    _report_flops_and_postings_one_line(
        int(total_entries), V - n_empty, V, label,
        total_flops=total_flops, n_queries=n_queries, model_label="BM25"
    )


def _splade_build_inverted_index(doc_sparse, vocab_size: int):
    """Build term -> [(doc_idx, weight), ...] from SPLADE doc sparse (torch/sparse). Returns (posting_lists, n_docs)."""
    if hasattr(doc_sparse, "is_sparse") and doc_sparse.is_sparse:
        doc_sparse = doc_sparse.coalesce()
        row = doc_sparse.indices()[0].cpu().numpy()
        col = doc_sparse.indices()[1].cpu().numpy()
        values = doc_sparse.values().cpu().numpy()
    else:
        if isspmatrix(doc_sparse):
            coo = doc_sparse.tocoo()
            row, col, values = coo.row, coo.col, coo.data
        else:
            arr = np.asarray(doc_sparse)
            row, col = np.where(arr != 0)
            values = arr[row, col].ravel()
    V = int(vocab_size)
    posting_lists = [[] for _ in range(V)]
    for i in range(len(row)):
        c = int(col[i])
        if 0 <= c < V:
            posting_lists[c].append((int(row[i]), float(values[i])))
    n_docs = int(np.max(row) + 1) if len(row) > 0 else 0
    return posting_lists, n_docs


def _splade_retrieve_with_index(query_sparse, posting_lists, top_k: int, *, return_scores: bool = False):
    """Term-at-a-time: for each query accumulate doc scores via posting lists.

    Returns list of top_k doc index arrays.  When *return_scores* is True,
    returns (top_indices, score_dicts) where score_dicts is a per-query list of
    {doc_idx: float} covering **all** scored documents (not just top_k).
    """
    if hasattr(query_sparse, "is_sparse") and query_sparse.is_sparse:
        query_sparse = query_sparse.coalesce()
        q_row = query_sparse.indices()[0].cpu().numpy()
        q_col = query_sparse.indices()[1].cpu().numpy()
        q_val = query_sparse.values().cpu().numpy()
    else:
        if isspmatrix(query_sparse):
            coo = query_sparse.tocoo()
            q_row, q_col, q_val = coo.row, coo.col, coo.data
        else:
            arr = np.asarray(query_sparse)
            q_row, q_col = np.where(arr != 0)
            q_val = arr[q_row, q_col].ravel()
    n_queries = int(np.max(q_row) + 1) if len(q_row) > 0 else 0
    from collections import defaultdict
    q_terms = defaultdict(list)
    for i in range(len(q_row)):
        q_terms[int(q_row[i])].append((int(q_col[i]), float(q_val[i])))
    top_indices = []
    all_scores = [] if return_scores else None
    for q_idx in range(n_queries):
        doc_scores = defaultdict(float)
        for term_id, q_w in q_terms.get(q_idx, []):
            if term_id < len(posting_lists):
                for doc_idx, d_w in posting_lists[term_id]:
                    doc_scores[doc_idx] += q_w * d_w
        if not doc_scores:
            top_indices.append(np.array([], dtype=np.int64))
        else:
            doc_idx_arr = np.array(list(doc_scores.keys()))
            score_arr = np.array([doc_scores[d] for d in doc_idx_arr])
            top_indices.append(doc_idx_arr[np.argsort(-score_arr)[:top_k]])
        if return_scores:
            all_scores.append(dict(doc_scores))
    if return_scores:
        return top_indices, all_scores
    return top_indices


def _report_splade_flops_and_postings(posting_lists, query_sparse, label: str):
    """FLOPs = 2 * sum over queries of (sum of |L_t| for t in supp(q)). Plus one line: total postings, non-empty terms."""
    from collections import defaultdict
    if hasattr(query_sparse, "is_sparse") and query_sparse.is_sparse:
        query_sparse = query_sparse.coalesce()
        q_row = query_sparse.indices()[0].cpu().numpy()
        q_col = query_sparse.indices()[1].cpu().numpy()
    else:
        if isspmatrix(query_sparse):
            coo = query_sparse.tocoo()
            q_row, q_col = coo.row, coo.col
        else:
            arr = np.asarray(query_sparse)
            q_row, q_col = np.where(arr != 0)
    q_terms = defaultdict(set)
    for i in range(len(q_row)):
        q_terms[int(q_row[i])].add(int(q_col[i]))
    n_queries = max(q_terms.keys()) + 1 if q_terms else 0
    total_flops = 0
    for q_idx in range(n_queries):
        for t in q_terms.get(q_idx, []):
            if t < len(posting_lists):
                total_flops += 2 * len(posting_lists[t])
    total_postings = sum(len(pl) for pl in posting_lists)
    n_non_empty = sum(1 for pl in posting_lists if len(pl) > 0)
    V = len(posting_lists)
    _report_flops_and_postings_one_line(
        total_postings, n_non_empty, V, label,
        total_flops=total_flops if n_queries > 0 else None, n_queries=n_queries, model_label="SPLADE"
    )


def _make_clefip_official_metrics(
    true_labels_list: list,
    predicted_labels_list: list,
    doc_ranking_list: list,
    full_doc_ranking_list: list = None,
) -> dict:
    """
    CLEF-IP metrics: passage-level + document-level.

        Passage-level (2):
            - magp                 — MAP(D), official CLEF-IP hierarchical per-document AP (Piroi et al. 2012).
            - recall_passage@1000  — passage recall at depth 1000.

    Document-level (5):
      - pres_doc@100   — official CLEF-IP document PRES.
      - recall_doc@100 — document recall.
      - ndcg_doc@10    — top-of-list document ranking quality.
      - map_doc        — untruncated document-level MAP (uses full_doc_ranking_list).
      - mrr_doc        — untruncated document-level MRR (rank of first relevant doc).

    *doc_ranking_list* (truncated to topk_docs) drives the @k document metrics.
    *full_doc_ranking_list* (untruncated Stage-1 dedup ranking) drives untruncated map_doc;
    if not provided, falls back to doc_ranking_list (yielding truncated MAP).
    """
    # Passage-level
    metrics = {
        "recall_passage@1000": mean_recall_at_k(true_labels_list, predicted_labels_list, k=1000),
        "magp": _clefip_mean_agp_passage(true_labels_list, predicted_labels_list, k=None),
    }
    # Document-level
    true_doc_ids_list = [
        list({_clefip_passage_id_to_doc_id(pid) for pid in rel_passages})
        for rel_passages in true_labels_list
    ]
    metrics["pres_doc@100"] = mean_pres_at_k(true_doc_ids_list, doc_ranking_list, k=100, N_max=100)
    metrics["recall_doc@100"] = mean_recall_at_k(true_doc_ids_list, doc_ranking_list, k=100)
    metrics["ndcg_doc@10"] = mean_ndcg_at_k(true_doc_ids_list, doc_ranking_list, k=10)
    _map_ranking = full_doc_ranking_list if full_doc_ranking_list is not None else doc_ranking_list
    metrics["map_doc"] = mean_average_precision(true_doc_ids_list, _map_ranking, k=None)
    _mrr_k = max((len(r) for r in _map_ranking), default=1) or 1
    metrics["mrr_doc"] = mean_mrr_at_k(true_doc_ids_list, _map_ranking, k=_mrr_k)
    return metrics


def _evaluate_and_print_clefip(
    qrels: dict,
    query_ids: list,
    predicted_labels_list: list,
    model_label: str,
    *,
    two_stage: bool = True,
    topk_docs: int = 100,
    header_extra: str = "",
    doc_ranking_list: list,
    full_doc_ranking_list: list = None,
) -> dict:
    """Evaluate CLEF-IP passage retrieval: compute metrics and print results.

    *doc_ranking_list* is the Stage-1 truncated document ranking (used for @k metrics).
    *full_doc_ranking_list* is the untruncated Stage-1 ranking (used for untruncated map_doc).
    """
    true_labels_list = [qrels.get(qid, []) for qid in query_ids]
    results = _make_clefip_official_metrics(
        true_labels_list, predicted_labels_list,
        doc_ranking_list=doc_ranking_list,
        full_doc_ranking_list=full_doc_ranking_list,
    )
    label_suffix = f" (two-stage top-{topk_docs} docs)" if two_stage else ""
    print_subsection_header(f"CLEF-IP 2013 EN claims-to-passages{header_extra}{label_suffix}")
    print_metric_table(results, f"{model_label}{label_suffix}")
    return results


def _score_queries_against_postings(
    query_sparse: list,
    doc_postings: list,
    idf: "np.ndarray",
    idf_exponent: float,
    top_k: int,
    *,
    length_norm: str = "none",
    length_norm_exp: float = 0.5,
    doc_nspans: "Optional[np.ndarray]" = None,
    show_progress: bool = True,
    return_scores: bool = False,
) -> "list | tuple":
    """Score queries against an inverted index and return top-k doc indices per query.

    Scoring: score(q, d) = sum_{t in supp(q) ∩ supp(d)} q_sim_t * d_sim_t * idf_t^alpha

    Parameters
    ----------
    query_sparse : list of (terms, weights) tuples
    doc_postings : list of posting lists; doc_postings[center_id] = [(doc_idx, d_weight), ...]
    idf : ndarray (V,)
    idf_exponent : float
    top_k : int
    length_norm : str  — "sqrt_spans" to enable length normalization.
    length_norm_exp : float
    doc_nspans : ndarray or None  — per-document span counts.
    show_progress : bool
    return_scores : bool
        If True, also return per-query score dicts {doc_idx: float} covering ALL
        scored documents (not just top_k).  Returns (top_indices, score_dicts).

    Returns
    -------
    list[list[int]]  — per-query top-k document indices, descending by score.
    If return_scores is True: (list[list[int]], list[dict[int, float]])
    """
    _use_len_norm = (length_norm == "sqrt_spans" and doc_nspans is not None)

    top_indices: list[list[int]] = []
    all_scores: list[dict[int, float]] = [] if return_scores else None
    iterator = enumerate(query_sparse)
    if show_progress:
        iterator = enumerate(tqdm(query_sparse, desc="Retrieving"))

    for _q_idx, qpack in iterator:
        terms, weights = qpack[0], qpack[1]
        doc_scores: dict[int, float] = {}

        for i, term in enumerate(terms):
            pl = doc_postings[term]
            if not pl:
                continue
            q_idf = float(weights[i]) * (float(idf[term]) ** idf_exponent)
            for doc_idx, d_weight in pl:
                doc_scores[doc_idx] = doc_scores.get(doc_idx, 0.0) + q_idf * float(d_weight)

        # Apply document length normalization
        if _use_len_norm and doc_scores:
            for doc_idx in list(doc_scores.keys()):
                norm_factor = max(doc_nspans[doc_idx] ** length_norm_exp, 1e-6)
                doc_scores[doc_idx] /= norm_factor

        if doc_scores:
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            top_indices.append([doc_idx for doc_idx, _ in sorted_docs[:top_k]])
            if all_scores is not None:
                all_scores.append(dict(doc_scores))
        else:
            top_indices.append([])
            if all_scores is not None:
                all_scores.append({})

    if return_scores:
        return top_indices, all_scores
    return top_indices


# ── ColBERT helpers ──────────────────────────────────────────────────────────

_COLBERT_MODEL_NAMES = {"colbert", "colbertv2", "colbert-ir/colbertv2.0"}


def _clefip_cache_id(args):
    """Build the (model_clean, sample_tag) tuple used to namespace CLEF-IP per-backend caches.

    - ``model_clean``: ``args.model_name`` with slashes replaced by underscores.
    - ``sample_tag``: ``"{sample_size}"`` plus ``"_hn{int(ratio*100)}"`` when both
      ``clefip_sample_size`` is a real subset (not 0/-1/-2) and ``clefip_hard_neg_ratio > 0``.
    """
    model_clean = (args.model_name or "").rstrip("/").replace("/", "_").strip("_")
    sample_sz = getattr(args, "clefip_sample_size", 0) or 0
    hn_ratio = getattr(args, "clefip_hard_neg_ratio", 0.0) or 0.0
    hn_suffix = f"_hn{int(hn_ratio * 100)}" if (sample_sz not in [0, -1, -2] and hn_ratio > 0) else ""
    sample_tag = f"{sample_sz}{hn_suffix}"
    return model_clean, sample_tag, sample_sz


def _bm25_passage_scores(args, query_ids, query_texts, passage_ids, corpus_jsonl_path):
    """Encode + search with BM25. Returns per-query {passage_id: score} dicts."""
    print("Loading passage texts from corpus for BM25...", flush=True)
    passage_texts = []
    with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            passage_texts.append(json.loads(line)["text"])
    assert len(passage_texts) == len(passage_ids)

    import bm25s
    import snowballstemmer
    stemmer = snowballstemmer.stemmer("english")
    passage_tokens = bm25s.tokenize(passage_texts, stopwords="en", stemmer=stemmer)
    query_tokens = bm25s.tokenize(query_texts, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(passage_tokens)
    _report_bm25_posting_stats(retriever, "CLEF-IP passage (full corpus)", query_tokens_list=query_tokens)
    results, scores = retriever.retrieve(query_tokens, k=len(passage_ids))
    return [
        {passage_ids[int(results[q][j])]: float(scores[q][j]) for j in range(len(results[q]))}
        for q in range(len(results))
    ]


def _splade_passage_scores(args, query_ids, query_texts, passage_ids, corpus_jsonl_path, device):
    """Encode + search with SPLADE-v2 (cocondenser-ensembledistil checkpoint). Cached on disk.

    Returns per-query {passage_id: score} dicts.
    """
    from scipy.sparse import save_npz as _sp_save_npz, load_npz as _sp_load_npz

    actual_model_name = "naver/splade-cocondenser-ensembledistil"

    _splade_clean, _sample_tag, _sample_sz = _clefip_cache_id(args)
    _qmc = getattr(args, "query_max_chunks", -1)
    _use_chunk = _qmc != 0
    _qchunk_tag = "" if not _use_chunk else f"_qchunk{'_unlim' if _qmc < 0 else _qmc}"
    cache_dir = os.path.join("temp", "clefip_splade_cache", f"{_splade_clean}_s{_sample_tag}")
    q_path = os.path.join(cache_dir, f"query_sparse{_qchunk_tag}.npz")
    p_path = os.path.join(cache_dir, "passage_sparse.npz")
    meta_path = os.path.join(cache_dir, f"meta{_qchunk_tag}.json")

    Q = D = None
    cache_hit = False
    query_hash = _hash_query_texts(query_texts)

    if os.path.isfile(p_path) and os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as _mf:
                meta = json.load(_mf)
            if meta.get("n_passages") == len(passage_ids) and meta.get("model") == actual_model_name:
                D = _sp_load_npz(p_path)
                q_hash_ok = meta.get("query_text_hash") == query_hash
                if (os.path.isfile(q_path) and meta.get("n_queries") == len(query_ids) and q_hash_ok):
                    Q = _sp_load_npz(q_path)
                    cache_hit = True
                    print(f"✅ Loaded SPLADE cache from {cache_dir}")
                    print(f"   queries: {Q.shape}, passages: {D.shape}")
                else:
                    print("⚠️  SPLADE query cache missing/mismatched — re-encoding queries (passages reused).")
                    print(f"✅ Loaded SPLADE passage cache: {D.shape}")
            else:
                print("⚠️  SPLADE cache metadata mismatch, re-encoding...")
        except Exception as e:
            print(f"⚠️  SPLADE cache load failed ({e}), re-encoding...")
            Q = D = None

    if not cache_hit:
        from sentence_transformers import SparseEncoder
        splade_model = SparseEncoder(actual_model_name)
        encode_bs = _auto_batch_size(device=device, hidden_size=768, vocab_size=30522)

        if Q is None:
            if _use_chunk:
                print(f"  Encoding {len(query_texts)} queries (chunk+term-max, max_chunks={_qmc})...", flush=True)
                Q = _splade_chunk_merge_query(query_texts, splade_model, encode_bs, query_max_chunks=_qmc)
            else:
                print(f"  Encoding {len(query_texts)} queries...", flush=True)
                query_sparse = splade_model.encode_query(query_texts, batch_size=encode_bs, show_progress_bar=True)
                Q = _sparse_to_csr(query_sparse)

        if D is None:
            print("Loading passage texts from corpus for SPLADE...", flush=True)
            passage_texts = []
            with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    passage_texts.append(json.loads(line)["text"])
            assert len(passage_texts) == len(passage_ids)
            print(f"  Encoding {len(passage_texts)} passages...", flush=True)
            passage_sparse = splade_model.encode_document(passage_texts, batch_size=encode_bs, show_progress_bar=True)
            D = _sparse_to_csr(passage_sparse)
            del passage_texts

        os.makedirs(cache_dir, exist_ok=True)
        _sp_save_npz(q_path, Q)
        if not os.path.isfile(p_path):
            _sp_save_npz(p_path, D)
        with open(meta_path, "w") as _mf:
            json.dump({
                "model": actual_model_name,
                "n_queries": len(query_ids),
                "n_passages": len(passage_ids),
                "vocab_size": int(D.shape[1]),
                "q_nnz": int(Q.nnz),
                "p_nnz": int(D.nnz),
                "query_max_chunks": _qmc,
                "query_text_hash": query_hash,
            }, _mf, indent=2)
        _q_mb = os.path.getsize(q_path) / 1024**2
        _p_mb = os.path.getsize(p_path) / 1024**2
        print(f"💾 Saved SPLADE cache to {cache_dir}")
        print(f"   queries: {_q_mb:.1f} MB ({Q.nnz:,} nnz), passages: {_p_mb:.1f} MB ({D.nnz:,} nnz)")
        del splade_model

    posting_lists, _ = _splade_build_inverted_index(D, D.shape[1])
    _report_splade_flops_and_postings(posting_lists, Q, "CLEF-IP passage (SPLADE)")
    _, idx_scores_list = _splade_retrieve_with_index(
        Q, posting_lists, top_k=len(passage_ids), return_scores=True,
    )
    return [{passage_ids[d]: s for d, s in sd.items()} for sd in idx_scores_list]


def _colbert_passage_scores(args, query_ids, query_texts, passage_ids, corpus_jsonl_path):
    """ColBERTv2 + PLAID end-to-end retrieval. Returns per-query {passage_id: score} or None on import error."""
    try:
        from colbert.modeling.checkpoint import Checkpoint  # noqa: F401
        from colbert.infra import ColBERTConfig as _ColBERTConfig
        from colbert.infra import Run, RunConfig
        from colbert import Indexer as _ColBERTIndexer, Searcher as _ColBERTSearcher
    except ImportError:
        print("❌  colbert-ai package not installed (or missing PLAID components).")
        print("   Install with:  pip install colbert-ai[torch,faiss-gpu]")
        return None

    _model_clean, _sample_tag, _sample_sz = _clefip_cache_id(args)
    _kmeans = getattr(args, "colbert_kmeans_niters", 4)
    _ki_str = f"_ki{_kmeans}" if _kmeans != 20 else ""
    _index_name = f"{_model_clean}_s{_sample_tag}_nbits2{_ki_str}"

    plaid_root = os.path.join("temp", "clefip_colbert_plaid")
    experiment = "clefip"

    # Step 1: collection TSV
    collection_tsv = os.path.join(plaid_root, f"collection_s{_sample_tag}.tsv")
    if not os.path.isfile(collection_tsv):
        print("Preparing collection TSV for PLAID indexing...")
        os.makedirs(os.path.dirname(collection_tsv), exist_ok=True)
        with open(corpus_jsonl_path, "r", encoding="utf-8") as f_in,              open(collection_tsv, "w", encoding="utf-8") as f_out:
            for idx, line in enumerate(f_in):
                rec = json.loads(line)
                text = rec["text"].replace("\t", " ").replace("\n", " ")
                f_out.write(f"{idx}\t{text}\n")
        print(f"  ✅ Collection TSV: {collection_tsv} ({len(passage_ids):,} passages)")
    else:
        print(f"  ✅ Collection TSV exists: {collection_tsv}")

    # Step 2: build PLAID index (cached)
    index_root = os.path.join(plaid_root, experiment, "indexes")
    index_path = os.path.join(index_root, _index_name)
    if not os.path.isdir(index_path) or not os.path.isfile(os.path.join(index_path, "metadata.json")):
        print(f"\n🔨 Building PLAID index: {_index_name} ...")
        print(f"   This encodes all {len(passage_ids):,} passages with ColBERTv2 + residual compression (nbits=2).")
        with Run().context(RunConfig(nranks=1, experiment=experiment)):
            q_maxlen = getattr(args, "colbert_query_maxlen", 512)
            kw = dict(nbits=2, doc_maxlen=512, query_maxlen=q_maxlen, root=plaid_root, kmeans_niters=_kmeans)            # Note: ncells in ColBERTConfig is a search-time probe parameter, not a build parameter.
            # The number of k-means centroids is auto-computed by ColBERT from corpus size.
            cfg = _ColBERTConfig(**kw)
            print(f"   PLAID config: kmeans_niters={_kmeans} (centroids auto-computed from corpus size)")
            indexer = _ColBERTIndexer(checkpoint="colbert-ir/colbertv2.0", config=cfg)
            indexer.index(name=_index_name, collection=collection_tsv, overwrite="reuse")
        print(f"  ✅ PLAID index built: {index_path}")
    else:
        print(f"  ✅ PLAID index exists: {index_path}")

    # Step 3: search
    plaid_topk = min(getattr(args, "colbert_plaid_topk", 1000), len(passage_ids))
    cb_qmc = getattr(args, "query_max_chunks", -1)
    use_chunk = cb_qmc != 0
    cap_str = f"cap={cb_qmc}" if cb_qmc > 0 else "unlimited"
    if use_chunk:
        print(f"\n🔍 Searching with ColBERTv2 PLAID engine (chunk+max-sim, {cap_str}, top-{plaid_topk})...")
    else:
        print(f"\n🔍 Searching with ColBERTv2 PLAID engine (top-{plaid_topk} per query)...")

    passage_scores_list = []
    with Run().context(RunConfig(nranks=1, experiment=experiment)):
        _ncells_search = getattr(args, "colbert_ncells_search", 4) or 4
        _search_qmaxlen = getattr(args, "colbert_search_query_maxlen", 192) or 192
        _cs_threshold = getattr(args, "colbert_centroid_score_threshold", 0.5)
        _ndocs = getattr(args, "colbert_ndocs", 1024) or 1024
        cfg = _ColBERTConfig(root=plaid_root, query_maxlen=_search_qmaxlen,
                             ncells=_ncells_search,
                             centroid_score_threshold=_cs_threshold,
                             ndocs=_ndocs)
        print(f"   PLAID search config: query_maxlen={_search_qmaxlen}, ncells={_ncells_search}, "
              f"centroid_score_threshold={_cs_threshold}, ndocs={_ndocs}")
        searcher = _ColBERTSearcher(index=_index_name, config=cfg, collection=collection_tsv)
        for qtext in tqdm(query_texts, desc="ColBERT PLAID search"):
            if use_chunk:
                chunks = _dense_chunk_text(qtext, None, max_tokens=_search_qmaxlen)
                if cb_qmc > 0 and len(chunks) > cb_qmc:
                    chunks = chunks[:cb_qmc]
                merged = {}
                for chunk in chunks:
                    pids_result, _ranks, scores_result = searcher.search(chunk, k=plaid_topk)
                    for pid_int, score in zip(pids_result, scores_result):
                        if 0 <= pid_int < len(passage_ids):
                            pid_str = passage_ids[pid_int]
                            if score > merged.get(pid_str, -1e9):
                                merged[pid_str] = float(score)
                passage_scores_list.append(merged)
            else:
                pids_result, _ranks, scores_result = searcher.search(qtext, k=plaid_topk)
                passage_scores_list.append({
                    passage_ids[pid_int]: float(score)
                    for pid_int, score in zip(pids_result, scores_result)
                    if 0 <= pid_int < len(passage_ids)
                })

    # Report PLAID efficiency: dominant cost = ndocs passages receiving full MaxSim scoring per query.
    # FLOPs = 2 × n_queries × ndocs  (lower bound; each MaxSim candidate counts as one comparison).
    n_q_done = len(passage_scores_list)
    total_flops_est = 2 * n_q_done * _ndocs
    print(f"\n📊 Efficiency — ColBERT CLEF-IP passage (PLAID)")
    print(f"   PLAID config: ncells={_ncells_search}, ndocs={_ndocs} (full MaxSim candidates/query), topk={plaid_topk}")
    print(f"   FLOPs: total={total_flops_est:,}, mean per query={total_flops_est // n_q_done if n_q_done else 0:,} (2 × ndocs MaxSim candidates)")
    print(f"  ✅ PLAID search complete: {len(query_ids)} queries, top-{plaid_topk} per query")
    return passage_scores_list


def _dense_passage_eval(args, query_ids, query_texts, passage_ids, corpus_jsonl_path,
                        qrels_passage_ids, topk_docs, device, model_name):
    """Encode dense embeddings (cached), then evaluate via either chunk+max-sim or FAISS top-K.

    Self-contained: this branch has its own evaluation path because of the chunk-merge mode,
    so it doesn't fit the (return scores → unified rerank+eval) pattern.
    """
    _model_clean, _sample_tag, _sample_sz = _clefip_cache_id(args)
    cache_dir = os.path.join("temp", "clefip_dense_cache", f"{_model_clean}_s{_sample_tag}")
    q_path = os.path.join(cache_dir, "query_embeddings.npy")
    p_path = os.path.join(cache_dir, "passage_embeddings.npy")
    meta_path = os.path.join(cache_dir, "meta.json")

    query_texts_fmt, _ = _clefip_format_for_model(query_texts, passage_ids[:1], [""], args.model_name)
    encode_fn, model_label, dense_tokenizer = _get_clefip_dense_encoder(args, model_name, device)
    qmc = getattr(args, "query_max_chunks", -1)
    use_chunk = qmc != 0  # -1 (default) or >0 = chunking enabled; 0 = legacy truncate

    cache_hit = False
    passage_emb = query_emb = None
    query_hash = _hash_query_texts(query_texts_fmt)
    # Bump when passage formatting changes; "title_in_abstract_all_schemes" applies title to
    # abstract passages for all schemes (previously only section_tokens).
    PASSAGE_FORMAT_VERSION = "title_in_abstract_all_schemes"

    if os.path.isfile(p_path) and os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            meta_ok = (meta.get("model_name") == args.model_name
                       and meta.get("n_passages") == len(passage_ids)
                       and meta.get("passage_format_version") == PASSAGE_FORMAT_VERSION)
            if meta_ok:
                passage_emb = np.load(p_path)
                assert passage_emb.shape[0] == len(passage_ids)
                q_hash_ok = meta.get("query_text_hash") == query_hash
                if (not use_chunk and os.path.isfile(q_path)
                        and meta.get("n_queries") == len(query_ids) and q_hash_ok):
                    query_emb = np.load(q_path)
                    assert query_emb.shape[0] == len(query_ids)
                    print(f"✅ Loaded CLEF-IP dense embeddings from cache: {cache_dir}")
                    print(f"   queries: {query_emb.shape}, passages: {passage_emb.shape}")
                else:
                    if not q_hash_ok and not use_chunk:
                        print("⚠️  Query text hash mismatch — re-encoding queries (passages reused).")
                    print(f"✅ Loaded CLEF-IP passage embeddings from cache: {passage_emb.shape}")
                cache_hit = True
        except Exception as e:
            print(f"⚠️  CLEF-IP dense cache load failed ({e}), re-encoding...")
            passage_emb = None
            cache_hit = False

    role_models = ["datalyes/patembed-large", "patembed-large"]

    def _encode_query_batch(texts):
        return encode_fn(texts, role="query") if model_name in role_models else encode_fn(texts, batch_size=32)

    def _encode_doc_batch(texts):
        return encode_fn(texts, role="document") if model_name in role_models else encode_fn(texts, batch_size=32)

    if not cache_hit:
        if not use_chunk:
            query_emb = _encode_query_batch(query_texts_fmt)
        batch_size = 256
        passage_emb_list = []
        with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
            batch_pids, batch_texts, batch_titles = [], [], []
            for line in tqdm(f, desc="  corpus", leave=False):
                rec = json.loads(line)
                batch_pids.append(rec["pid"])
                batch_texts.append(rec["text"])
                batch_titles.append(rec.get("title", ""))
                if len(batch_pids) >= batch_size:
                    _, batch_fmt = _clefip_format_for_model([""], batch_pids, batch_texts, args.model_name, batch_titles)
                    passage_emb_list.append(_encode_doc_batch(batch_fmt))
                    batch_pids, batch_texts, batch_titles = [], [], []
            if batch_pids:
                _, batch_fmt = _clefip_format_for_model([""], batch_pids, batch_texts, args.model_name, batch_titles)
                passage_emb_list.append(_encode_doc_batch(batch_fmt))
        passage_emb = np.vstack(passage_emb_list) if passage_emb_list else np.zeros((0, 0), dtype=np.float32)
        del passage_emb_list

        os.makedirs(cache_dir, exist_ok=True)
        np.save(p_path, passage_emb)
        if not use_chunk:
            np.save(q_path, query_emb)
        with open(meta_path, "w") as f:
            json.dump({
                "model_name": args.model_name,
                "n_queries": len(query_ids),
                "n_passages": len(passage_ids),
                "dim": int(passage_emb.shape[1]),
                "sample_size": _sample_sz,
                "query_text_hash": query_hash,
                "passage_format_version": PASSAGE_FORMAT_VERSION,
            }, f, indent=2)
        print(f"💾 Saved CLEF-IP dense embeddings to {cache_dir}")
        print(f"   passages: {passage_emb.shape} ({os.path.getsize(p_path) / 1024**2:.0f} MB)")
        if not use_chunk:
            print(f"   queries: {query_emb.shape} ({os.path.getsize(q_path) / 1024**2:.0f} MB)")
    elif not use_chunk and query_emb is None:
        # Cache hit on passages but query_text_hash mismatched: re-encode queries only.
        query_emb = _encode_query_batch(query_texts_fmt)
        np.save(q_path, query_emb)
        with open(meta_path, "w") as f:
            json.dump({
                "model_name": args.model_name,
                "n_queries": len(query_ids),
                "n_passages": len(passage_ids),
                "dim": int(passage_emb.shape[1]),
                "sample_size": _sample_sz,
                "query_text_hash": query_hash,
                "passage_format_version": PASSAGE_FORMAT_VERSION,
            }, f, indent=2)
        print(f"💾 Updated query embeddings cache: {query_emb.shape}")

    if use_chunk:
        print(f"\n🔗 Dense Chunk+MaxSim (query_max_chunks={qmc})")
        if model_name in role_models:
            def _q_encode_fn(texts, batch_size=32):
                return encode_fn(texts, batch_size=batch_size, role="query")
        else:
            _q_encode_fn = encode_fn
        passage_scores_list = _dense_chunk_maxsim_scores(
            query_texts_fmt, passage_ids, passage_emb, _q_encode_fn, dense_tokenizer,
            query_max_chunks=qmc, batch_size=32,
        )
        predicted_labels_list, doc_ranking_list, full_doc_ranking_list = _clefip_two_stage_rerank(
            passage_ids, passage_scores_list, topk_docs=topk_docs,
        )
        print(f"  🔄 Two-stage retrieval: top-{topk_docs} docs → re-ranked passages per query")
        _evaluate_and_print_clefip(
            qrels_passage_ids, query_ids, predicted_labels_list,
            "Dense Chunk+MaxSim passage retrieval",
            two_stage=True, topk_docs=topk_docs,
            doc_ranking_list=doc_ranking_list,
            full_doc_ranking_list=full_doc_ranking_list,
        )
    else:
        clefip_passage_evaluation(
            query_ids, passage_ids, query_emb, passage_emb, qrels_passage_ids,
            k=100, model_label=model_label + " (full 01)", topk_docs=topk_docs,
        )


def _run_clefip_eval_full_corpus(
    args,
    query_ids: list,
    query_texts: list,
    passage_ids: list,
    corpus_jsonl_path: str,
    ids_txt_path: str,
    qrels_passage_ids: dict,
):
    """Run CLEF-IP retrieval over the full 01 passage corpus (streaming where possible).

    Dispatches to one of four backend functions based on ``args.model_name``.
    BM25/SPLADE/ColBERT all return per-query passage scores which are then put through
    a unified two-stage rerank + evaluate. Dense models have a self-contained branch
    (``_dense_passage_eval``) because of the chunk+max-sim alternate path.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = (args.model_name or "").lower() if hasattr(args.model_name, "lower") else str(args.model_name or "").lower()
    topk_docs = getattr(args, "clefip_two_stage_topk_docs", 100)

    if model_name == "sparse_coverage":
        # Handled by clefip_passage mode in run_sparse_coverage().
        print("sparse_coverage CLEF-IP: skipped here — handled by clefip_passage mode.")
        return

    if model_name == "bm25":
        passage_scores_list = _bm25_passage_scores(args, query_ids, query_texts, passage_ids, corpus_jsonl_path)
        label, header_extra = "BM25 passage retrieval", " (full 01 corpus)"
    elif model_name in ("naver/splade-v2", "splade-v2"):
        passage_scores_list = _splade_passage_scores(args, query_ids, query_texts, passage_ids, corpus_jsonl_path, device)
        label, header_extra = "SPLADE passage retrieval", " (full 01 corpus)"
    elif model_name in _COLBERT_MODEL_NAMES:
        passage_scores_list = _colbert_passage_scores(args, query_ids, query_texts, passage_ids, corpus_jsonl_path)
        if passage_scores_list is None:
            return
        label, header_extra = "ColBERT passage retrieval", " (PLAID)"
    else:
        # Dense: special case (chunk+max-sim path doesn't fit the score-and-rerank pattern).
        _dense_passage_eval(args, query_ids, query_texts, passage_ids, corpus_jsonl_path,
                            qrels_passage_ids, topk_docs, device, model_name)
        return

    predicted_labels_list, doc_ranking_list, full_doc_ranking_list = _clefip_two_stage_rerank(
        passage_ids, passage_scores_list, topk_docs=topk_docs,
    )
    print(f"  🔄 Two-stage retrieval: top-{topk_docs} docs → re-ranked passages per query")
    _evaluate_and_print_clefip(
        qrels_passage_ids, query_ids, predicted_labels_list,
        label,
        two_stage=True, topk_docs=topk_docs, header_extra=header_extra,
        doc_ranking_list=doc_ranking_list,
        full_doc_ranking_list=full_doc_ranking_list,
    )


def run_clefip_eval(args):
    """Load CLEF-IP EN data, run the selected model, and evaluate passage retrieval."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_clefip = getattr(args, "clefip_root", None) or ""
    if raw_clefip and os.path.isabs(raw_clefip):
        clefip_root = raw_clefip
    else:
        clefip_root = os.path.normpath(os.path.join(current_dir, raw_clefip or "clefip2013"))
    if not os.path.isdir(clefip_root):
        fallback = os.path.normpath(os.path.join(current_dir, "clefip2013"))
        if fallback != clefip_root and os.path.isdir(fallback):
            print(f"Warning: CLEF-IP root not found: {clefip_root}; using {fallback}")
            clefip_root = fallback
        else:
            raise FileNotFoundError(
                f"CLEF-IP root not found: {clefip_root}. "
                "Create it or pass --clefip_root with an existing path (e.g. ./clefip2013)."
            )
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from clefip2013.load_clefip import (
        load_clefip_en_for_eval_full_corpus,
        load_clefip_en_for_eval_sampled_corpus,
        FULL_CORPUS_DIR_EN,
        CORPUS_JSONL,
        IDS_TXT,
    )
    doc_root = os.path.join(clefip_root, "01_document_collection", "01_extracted")
    if not os.path.isdir(doc_root):
        raise FileNotFoundError(
            f"CLEF-IP document collection not found: {doc_root}. "
            f"Extract 01 collection: bash clefip2013/extract_01_collection.sh"
        )
    rebuild_corpus = getattr(args, "clefip_rebuild_corpus", False)
    sample_size = getattr(args, "clefip_sample_size", 0) or 0
    en_corpus_cache_dir = os.path.join(clefip_root, FULL_CORPUS_DIR_EN)
    cache_exists = os.path.isfile(os.path.join(en_corpus_cache_dir, CORPUS_JSONL)) and os.path.isfile(os.path.join(en_corpus_cache_dir, IDS_TXT))

    if sample_size != 0:
        if sample_size == -1:
            sample_cache_dir = os.path.join(clefip_root, "01_passage_corpus_en_qrels_only")
            _corpus_label = "qrels-only corpus"
        elif sample_size == -2:
            sample_cache_dir = os.path.join(clefip_root, "01_passage_corpus_en_qrels_docs")
            _corpus_label = "qrels-docs corpus (all passages from cited documents)"
        else:
            _hn_ratio = getattr(args, "clefip_hard_neg_ratio", 0.0) or 0.0
            _hn_suffix = f"_hn{int(_hn_ratio * 100)}" if _hn_ratio > 0 else ""
            sample_cache_dir = os.path.join(clefip_root, f"01_passage_corpus_en_sample_{sample_size}docs{_hn_suffix}")
            _hn_label = f", hard_neg={_hn_ratio:.0%}" if _hn_ratio > 0 else ""
            _corpus_label = f"sampled corpus ({sample_size:,} docs{_hn_label})"
        sample_cache_exists = os.path.isfile(os.path.join(sample_cache_dir, CORPUS_JSONL)) and os.path.isfile(os.path.join(sample_cache_dir, IDS_TXT))
        if sample_cache_exists and not rebuild_corpus:
            print(f"Loading CLEF-IP 2013 EN (claims-to-passages, {_corpus_label}, using cache)...")
        else:
            print(f"Loading CLEF-IP 2013 EN (claims-to-passages, {_corpus_label})...")
        query_ids, query_texts, corpus_jsonl_path, ids_txt_path, num_passages, qrels_passage_ids = load_clefip_en_for_eval_sampled_corpus(
            clefip_root, doc_root, sample_size=sample_size, rebuild_corpus=rebuild_corpus,
            hard_neg_ratio=getattr(args, "clefip_hard_neg_ratio", 0.0) or 0.0,
        )
        print(f"  Queries: {len(query_ids)}, Corpus passages: {num_passages:,}")
    elif cache_exists and not rebuild_corpus:
        print("Loading CLEF-IP 2013 EN (claims-to-passages, **full EN collection**, using cache)...")
        query_ids, query_texts, corpus_jsonl_path, ids_txt_path, num_passages, qrels_passage_ids = load_clefip_en_for_eval_full_corpus(
            clefip_root, doc_root, corpus_dir=None, rebuild_corpus=rebuild_corpus
        )
        print(f"  Queries: {len(query_ids)}, Full corpus passages: {num_passages:,}")
    else:
        print("Loading CLEF-IP 2013 EN (claims-to-passages, **full EN collection**)...")
        if not cache_exists:
            print("  Full EN corpus cache not found; building from 01 collection (this may take a while).")
        query_ids, query_texts, corpus_jsonl_path, ids_txt_path, num_passages, qrels_passage_ids = load_clefip_en_for_eval_full_corpus(
            clefip_root, doc_root, corpus_dir=None, rebuild_corpus=rebuild_corpus
        )
        print(f"  Queries: {len(query_ids)}, Full corpus passages: {num_passages:,}")
    # Load passage_ids for index -> passage_id mapping (same order as corpus)
    with open(ids_txt_path, "r", encoding="utf-8") as f:
        passage_ids = [line.strip() for line in f]
    assert len(passage_ids) == num_passages, "ids file length vs num_passages"
    # Full-corpus retrieval branch (BM25 / Dense / sparse_coverage)
    _run_clefip_eval_full_corpus(
        args, query_ids, query_texts, passage_ids, corpus_jsonl_path, ids_txt_path, qrels_passage_ids,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the evaluate script.

    Kept separate from ``main()`` so the parser can be inspected/used by tests
    or downstream scripts without triggering evaluation side effects.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, 
                       help="Path to pretrained model or model ID. Supported models: "
                            "allenai/specter2_base, mpi-inno-comp/paecter, "
                            "anferico/bert-for-patents, datalyes/patembed-large, naver/splade-v2, bm25, "
                            "colbert / colbertv2 / colbert-ir/colbertv2.0, "
                            "sparse_coverage, SentenceTransformer checkpoint dir (e.g. checkpoint-1142), or other checkpoint paths.")
    parser.add_argument("--temp_dir", type=str, default="./temp", help="Temporary directory for embeddings creation and evaluation.")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    
    # Parameters for sparse_coverage model
    parser.add_argument("--dense_model", type=str, default="ZoeYou/PatentMap-V0-SecPair-Claim",
                       help="Dense encoder model used to build embeddings (for sparse_coverage). "
                            "Example: ZoeYou/PatentMap-V0-SecPair-Claim")
    parser.add_argument("--tokenization_unit", type=str, default="spacy_token",
                       choices=["spacy_token", "encoder_token", "noun_chunk"],
                       help="Tokenization unit used to build embeddings (for sparse_coverage). "
                            "Example: spacy_token, encoder_token")
    # Eval assumes centers were built with CLS included; build script controls that. We look for 'cls' suffix only.
    parser.add_argument("--exclude_cls_spans", action="store_true", default=False,
                       help="Exclude spans whose text is literally 'cls' or '[CLS]' from index and query (for sparse_coverage). "
                            "Use to compare best config with vs without CLS spans. Document posting lists and query encoding skip these spans.")

    # Document side:
    parser.add_argument("--document_assignment", type=str, default="soft", choices=["hard", "soft"],
                       help="Document side: hard = each span -> nearest center only (Voronoi); soft = search(K) + per-center r_c filter + topK cap. Default: soft.")
    parser.add_argument("--weight_aggregation", type=str, default="max", choices=["max", "sum"],
                       help="Per (query, center) and (doc, center): max = use max similarity (default); sum = use sum of similarities (TF-style). Default: max.")
    
    # Query side:
    _soft_grp = parser.add_mutually_exclusive_group()
    _soft_grp.add_argument("--use_soft_assignment", action="store_true", default=True,
                       help="Use soft assignment for query spans: search(K) + per-center r_c filter + topK cap (symmetric with doc side). Default.")
    _soft_grp.add_argument("--no_soft_assignment", action="store_true", default=False,
                       help="Force hard assignment for query spans (nearest center only).")
    parser.add_argument("--soft_assignment_max_centers_per_span", type=int, default=5,
                       help="Cap each span (query or document) to at most this many centers (by similarity) during soft assignment. "
                           "If a span falls in >K centers, keep top-K; if <=K, keep all. "
                           "Default: 5. Applies to BOTH query-side and document-side soft assignment.")
    parser.add_argument("--query_first_span_weight", type=float, default=1.0,
                       help="Multiply weight of first span per query by this factor. Default: 1.0.")
    parser.add_argument("--idf_exponent", type=float, default=2.0,
                       help="Power applied to IDF in scoring: contrib uses idf^idf_exponent. "
                            "Default: 2.0. Try 0.5 (flatter), 1.0 or 1.5 (less discriminative), 3.0 (more).")

    parser.add_argument("--length_norm", type=str, default="sqrt_centers",
                       choices=["none", "sqrt_spans", "sqrt_centers"],
                       help="Document length normalization for sparse_coverage. "
                            "none: no normalization. sqrt_spans: divide by doc_span_count^exponent "
                            "(BM25-like, stable across stop-center changes). "
                            "sqrt_centers: legacy alias for sqrt_spans. Default: sqrt_centers.")
    parser.add_argument("--length_norm_exponent", type=float, default=0.5,
                       help="Exponent for length norm: divide by doc_span_count^exponent. "
                            "0.5 => sqrt (default). 0.8 => stronger penalization of long docs.")
    parser.add_argument("--stop_center_fraction", type=float, default=0.01,
                       help="Fraction of top centers by doc-DF to disable for retrieval. "
                           "Default: 0.01 (top 1%%). Computed from posting lists at eval time. "
                           "Set <= 0 to disable stop-center suppression. "
                           "NOTE: stop centers are applied AFTER the base sparse vectors cache, "
                           "so changing this value reuses the cache.")
    parser.add_argument("--query_max_chunks", type=int, default=-1,
                       help="Max number of 512-token chunks per query for sparse_coverage. "
                            "-1 (default) = unlimited: always split long queries into as many 512-token chunks "
                            "as needed and merge all resulting spans (recommended for patent claims). "
                            "0 = disabled (truncate at 512 tokens, legacy behaviour). "
                            ">0 = cap at this many chunks (silently drops later chunks; use only for ablation).")
    parser.add_argument("--centers_suffix", type=str, default="",
                       help="Suffix appended to centers directory name for discovery. Required when centers were "
                            "built with a suffix: greedy (e.g. '_soft', '_percenter'), k-means (e.g. '_kmeans_V50000'), "
                            "k-center (e.g. '_kcenter_V25000'), or quantile (e.g. '_quantile'). Must match build script output.")
    parser.add_argument("--spacy_model", type=str, default="sci_lg",
                       choices=["sm", "md", "lg", "sci_sm", "sci_md", "sci_lg"],
                       help="SpaCy model for span tokenization (sparse_coverage). Default: sci_lg.")
    parser.add_argument("--centers_search_batch", type=int, default=4096,
                       help="Batch size for FAISS search when building the centers_search cache. "
                            "Larger = fewer calls, more VRAM/RAM per batch. Default: 4096.")
    parser.add_argument("--centers_search_cache_K", type=int, default=16,
                       help="K_max for the per-span FAISS-search cache (D, I). The cache is keyed on "
                            "(centers_sig, embeddings_sig, section) and is reused across all sweeps "
                            "that vary assignment K (hard / soft K=1..K_max), per-center thresholds, "
                            "weight aggregation, IDF, normalization, stop-center selection, etc. "
                            "Must be >= the largest assignment K any sweep will request "
                            "(soft_assignment_max_centers_per_span). Default 16 covers all currently "
                            "planned sweeps and keeps disk to ~n_spans * 16 * 8 bytes per section. "
                            "Cache is auto-rebuilt to a larger K_max when a request exceeds the cached one; "
                            "smaller requests reuse the larger cached file.")
    parser.add_argument("--centers_search_cache_disable", action="store_true",
                       help="Disable the per-span FAISS-search cache (always recompute, do not read/write disk).")

    parser.add_argument("--clefip_root", type=str, default="",
                       help="CLEF-IP data root (02_topics, qrels). Default: ./clefip2013. Use e.g. ./clefip2023 if your full download is there.")
    parser.add_argument("--clefip_rebuild_corpus", action="store_true",
                       help="Force rebuild of the full 01 passage corpus (ignores existing cache). Only applies when using full corpus (default).")
    parser.add_argument("--clefip_sample_size", type=int, default=25000,
                       help="Controls the CLEF-IP corpus size (document-level sampling for >0): "
                            "0 = full EN corpus (~100M passages, cache under 01_passage_corpus_en/). "
                            "-1 = qrels-only corpus: only the exact passages referenced in qrels (~1.8k unique, "
                            "fastest; cache under 01_passage_corpus_en_qrels_only/). "
                            "-2 = qrels-docs corpus: ALL passages from any document cited in qrels — includes "
                            "abstracts, claims, descriptions of each cited doc (~90 docs; "
                            "cache under 01_passage_corpus_en_qrels_docs/). "
                            ">0 (default: 10000) = number of DOCUMENTS to sample. All passages from qrels-cited documents are "
                            "always included; remaining slots filled by reservoir-sampled EN documents. "
                            "All passages from each selected document are kept (preserves document structure). "
                            "Example: --clefip_sample_size 10000 (~10k docs → ~400k passages at ~40 passages/doc). "
                            "Cache is built under 01_passage_corpus_en_sample_<N>docs/ and reused on subsequent runs.")
    parser.add_argument("--clefip_hard_neg_ratio", type=float, default=0.75,
                       help="Fraction of sampled negative documents that should be IPC-based hard negatives "
                            "(0.0 to 1.0, default: 0.75 = 75%% hard negatives). Only applies when --clefip_sample_size > 0. "
                            "The hard negative quota is split evenly into three tiers (each ~25%% of total budget): "
                            "(1) subgroup-hard: shares IPC subgroup (e.g. 'H04N5/44') with query/cited patents; "
                            "(2) maingroup-hard: shares IPC main-group (e.g. 'H04N5') but not subgroup; "
                            "(3) subclass-hard: shares IPC subclass (e.g. 'H04N') but not main-group. "
                            "The remaining budget (25%%) is random negatives. "
                            "Example: --clefip_hard_neg_ratio 0.75 means 25%% subgroup + 25%% maingroup + 25%% subclass + 25%% random. "
                            "Requires IPC index v3 (built once from 01_extracted XMLs, cached as doc_id_ipc_cache_v3.json). "
                            "Cache dir includes hard neg ratio to avoid mixing: 01_passage_corpus_en_sample_<N>docs_hn<pct>/.")
    parser.add_argument("--clefip_two_stage_topk_docs", type=int, default=100,
                       help="Number of top documents to keep in Stage 1 of two-stage retrieval (default: 100). "
                            "Two-stage is always enabled: Stage 1 ranks passages → derives doc ranking → keeps top-K docs; "
                            "Stage 2 re-ranks ALL passages from those top-K docs by original scores. "
                            "Higher values include more candidate documents (higher recall, lower precision of pool).")
    parser.add_argument("--clefip_neg_doc_sizes", type=str, default="",
                       help="Comma-separated list of corpus sizes (number of documents) for CLEF-IP robustness test. "
                            "For each size N, keeps all relevant documents (from qrels) and samples (N - n_relevant) "
                            "negative documents from the full pool. Evaluates with each reduced pool. "
                            "Example: '100,500,1000,5000,10000,20000'. "
                            "Requires CLEF-IP passage embeddings to be cached (runs after normal CLEF-IP eval).")

    # ColBERT-specific parameters (only apply when model_name is colbert/colbertv2/colbert-ir/colbertv2.0)
    parser.add_argument("--colbert_plaid_topk", type=int, default=1000,
                       help="Number of passages to retrieve per query with PLAID engine. "
                            "Default: 1000. Higher values give better recall but slower search.")
    parser.add_argument("--colbert_ncells_search", type=int, default=4,
                       help="Number of IVF cells to probe per query token at search time. "
                            "Default: 4. Lower values use less GPU memory; higher values improve recall. "
                            "PLAID's hierarchical clustering means even small values give strong recall.")
    parser.add_argument("--colbert_kmeans_niters", type=int, default=4,
                       help="Number of k-means iterations for PLAID index building. "
                            "ColBERT default is 20 but 4 is usually sufficient and 5x faster.")
    parser.add_argument("--colbert_query_maxlen", type=int, default=512,
                       help="Maximum query token length for ColBERT at INDEX BUILD TIME. Default: 512. "
                            "Original ColBERTv2 default is 32; patent claims are much longer, "
                            "so 512 (BERT max) is recommended. Shorter queries are padded with [MASK].")
    parser.add_argument("--colbert_search_query_maxlen", type=int, default=192,
                       help="Maximum query token length at SEARCH TIME (also used to chunk long queries). "
                            "Default: 192. Lower values cut PLAID GPU memory dramatically (the strided "
                            "tensor in score_pids scales linearly with this). Long queries are split into "
                            "chunks of this size and scores are max-merged across chunks.")
    parser.add_argument("--colbert_centroid_score_threshold", type=float, default=0.45,
                       help="Min centroid score to keep a document during PLAID's first filter pass. "
                            "Higher = aggressive pruning = less GPU memory. Default 0.45 "
                            "(ColBERT's own default at k>100 is 0.4).")
    parser.add_argument("--colbert_ndocs", type=int, default=8192,
                       help="Cap on candidate document count after PLAID's first filter. "
                            "Default 8192. Lower values reduce score_pids memory. "
                            "ColBERT's default at k>100 is max(k*4, 4096), which can OOM on large corpora.")

    return parser



def run_sparse_coverage(args):
    """
    Sparse Coverage Retrieval

    Uses pre-built vocabulary centers for sparse retrieval.
    Documents from the evaluation corpus are encoded at runtime (or loaded from cache).
    Both queries and documents are encoded and assigned to centers.
    """
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n🔍 Sparse Coverage Retrieval")

    # Validate required parameters
    if args.dense_model is None:
        raise ValueError("--dense_model is required for sparse_coverage")
    if args.tokenization_unit is None:
        raise ValueError("--tokenization_unit is required for sparse_coverage")

    print(f"   Dense model: {args.dense_model}")
    print(f"   Tokenization unit: {args.tokenization_unit}")
    _qmc = getattr(args, 'query_max_chunks', -1)
    _qmc_str = 'off (truncate at 512)' if _qmc == 0 else (f'unlimited' if _qmc < 0 else f'cap={_qmc} chunks')
    print(f"   Query chunking: {_qmc_str}")
    print(f"   Length norm: {getattr(args, 'length_norm', 'none')}" + (f" (exponent={getattr(args, 'length_norm_exponent', 0.5)})" if getattr(args, 'length_norm', 'none') == 'sqrt_centers' else ""))

    # Import span encoding functions from utils module
    # (These functions are defined in utils.py, not in 1create_N_embeddings.py)
    try:
        import utils
        process_doc_batch = utils.process_doc_batch
        ensure_section_tokens = utils.ensure_section_tokens
        print("✅ Successfully loaded span encoding functions from utils module")
    except Exception as e:
        raise RuntimeError(f"Failed to import span encoding functions from utils: {e}")

    # Initialize spaCy if needed
    if args.tokenization_unit != "encoder_token":
        import spacy
        if args.spacy_model.startswith("sci_"):
            spacy_name = f"en_core_sci_{args.spacy_model[4:]}"
        else:
            spacy_name = f"en_core_web_{args.spacy_model}"
        disable = ["ner", "textcat", "lemmatizer"]
        utils.NLP = spacy.load(spacy_name, disable=disable)
        utils.NLP.max_length = 1_000_000
        print(f"   SpaCy model: {spacy_name}")
    else:
        utils.NLP = None

    # Load dense encoder model
    print(f"\n📦 Loading dense encoder model: {args.dense_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.dense_model)
    model = AutoModel.from_pretrained(args.dense_model, trust_remote_code=True)

    # Ensure section tokens are in vocabulary
    ensure_section_tokens(tokenizer, model)

    model.to(device)
    model.eval()
    print(f"✅ Dense encoder loaded (hidden_size={model.config.hidden_size})")

    # Common kwargs for all process_doc_batch calls
    import functools as _ft
    _encode_spans = _ft.partial(
        process_doc_batch,
        unit=args.tokenization_unit,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=512,
        span_pooling="mean",
    )

    # -----------------------------
    # Per-mode evaluation
    # -----------------------------

    def _load_centers_info_json(centers_path: str) -> dict:
        centers_json_path = centers_path.replace(".npy", ".json")
        if os.path.exists(centers_json_path):
            try:
                with open(centers_json_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"   ⚠️  Could not read centers JSON: {e}")
        return {}

    def _get_r_and_sim_threshold(centers_info: dict) -> tuple[float, float]:
        # r: cosine distance threshold; sim_threshold = 1 - r
        sim_threshold = None
        if "sim_threshold" in centers_info:
            sim_threshold = float(centers_info["sim_threshold"])
        elif "r" in centers_info:
            sim_threshold = 1.0 - float(centers_info["r"])

        r = centers_info.get("r", None)
        if r is None and sim_threshold is not None:
            r = 1.0 - sim_threshold
        if r is None:
            raise ValueError("Could not determine radius r from centers JSON (missing 'r' and 'sim_threshold').")

        r = float(r)
        sim_threshold = float(sim_threshold) if sim_threshold is not None else (1.0 - r)
        return r, sim_threshold

    def _doc_cache_dir(mode: str) -> str:
        model_clean = args.dense_model.strip("/").split("/")[-1].replace("/", "_").replace("\\", "_")
        return os.path.join(args.temp_dir, f"sparse_doc_{model_clean}_{args.tokenization_unit}_{mode}")

    def _load_doc_cache_meta(cache_dir: str) -> tuple[dict[int, str], set[int]]:
        """Load only the span_to_doc metadata (no embeddings).

        Returns (span_to_doc, exclude_cls_indices).
        Raises FileNotFoundError if meta file is missing.
        """
        span_to_doc: dict[int, str] = {}
        exclude_cls_indices: set[int] = set()
        meta_path = os.path.join(cache_dir, f"span_to_doc_{args.tokenization_unit}.jsonl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(meta_path)
        with open(meta_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                span_to_doc[entry["i"]] = entry["d"]
                if entry.get("cls"):
                    exclude_cls_indices.add(entry["i"])
        return span_to_doc, exclude_cls_indices

    def _get_section_shape(cache_dir: str, section: str) -> int:
        """Read the row count of a cached .npy without loading the full array."""
        p = os.path.join(cache_dir, f"{section}_{args.tokenization_unit}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        arr = np.load(p, mmap_mode='r')
        n = arr.shape[0]
        del arr
        return n

    def _load_section_emb(cache_dir: str, section: str) -> np.ndarray:
        """Load a single section's embeddings from cache using mmap.

        Returns a memory-mapped array (read-only). Callers should
        np.array(..., copy=True) to get a writable copy when needed,
        then del the mmap reference to keep peak RSS low.
        """
        p = os.path.join(cache_dir, f"{section}_{args.tokenization_unit}.npy")
        return np.load(p, mmap_mode='r')

    def _doc_cache_exists(cache_dir: str, doc_sections: list[str]) -> bool:
        """Lightweight check: do all cache files exist? Does NOT load data."""
        for sec in doc_sections:
            if not os.path.exists(os.path.join(cache_dir, f"{sec}_{args.tokenization_unit}.npy")):
                return False
        return os.path.exists(os.path.join(cache_dir, f"span_to_doc_{args.tokenization_unit}.jsonl"))

    def _base_sparse_cache_paths(cache_dir: str, sig: str = "") -> tuple[str, str, str]:
        suffix = f"_{sig}" if sig else ""
        base = os.path.join(cache_dir, f"base_sparse_vectors{suffix}")
        return (
            f"{base}.meta.json",
            f"{base}.doc.npz",
            f"{base}.query.npz",
        )

    def _base_sparse_cache_sig(meta: dict) -> str:
        return hashlib.sha1(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:12]

    def _hash_float_array(arr: np.ndarray) -> str:
        return hashlib.sha1(np.asarray(arr, dtype=np.float32).tobytes()).hexdigest()

    def _load_base_sparse_vectors(cache_dir: str, expected_meta: dict):
        sig = _base_sparse_cache_sig(expected_meta)
        meta_path, doc_npz_path, query_npz_path = _base_sparse_cache_paths(cache_dir, sig)
        if not (os.path.exists(meta_path) and os.path.exists(doc_npz_path) and os.path.exists(query_npz_path)):
            return None
        try:
            with open(meta_path, "r") as f:
                got_meta = json.load(f)
            if got_meta != expected_meta:
                return None

            doc_pack = np.load(doc_npz_path)
            doc_indptr = doc_pack["indptr"].astype(np.int64, copy=False)
            doc_idx = doc_pack["doc_idx"].astype(np.int32, copy=False)
            doc_w = doc_pack["weight"].astype(np.float32, copy=False)
            V_local = len(doc_indptr) - 1
            doc_postings: list[list[tuple[int, float]]] = [[] for _ in range(V_local)]
            for c in range(V_local):
                s = int(doc_indptr[c])
                e = int(doc_indptr[c + 1])
                if e > s:
                    doc_postings[c] = list(zip(doc_idx[s:e].astype(int).tolist(), doc_w[s:e].astype(float).tolist()))

            q_pack = np.load(query_npz_path)
            q_indptr = q_pack["indptr"].astype(np.int64, copy=False)
            q_centers = q_pack["centers"].astype(np.int32, copy=False)
            q_weights = q_pack["weights"].astype(np.float32, copy=False)
            n_q = len(q_indptr) - 1
            query_sparse: list[tuple[np.ndarray, np.ndarray]] = []
            for q in range(n_q):
                s = int(q_indptr[q])
                e = int(q_indptr[q + 1])
                query_sparse.append((q_centers[s:e].copy(), q_weights[s:e].copy()))

            return doc_postings, query_sparse
        except Exception as e:
            print(f"   ⚠️  Failed to load base sparse cache ({e}), recomputing")
            return None

    def _save_base_sparse_vectors(cache_dir: str, meta: dict, doc_postings: list, query_sparse: list) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        sig = _base_sparse_cache_sig(meta)
        meta_path, doc_npz_path, query_npz_path = _base_sparse_cache_paths(cache_dir, sig)
        # Cross-process flock: avoid duplicate work + half-written files when
        # multiple jobs converge on the same (sig). Re-check inside the lock.
        with _build_lock(meta_path + ".lock", label=f"base_sparse {sig}"):
            try:
                if os.path.exists(meta_path) and os.path.exists(doc_npz_path) and os.path.exists(query_npz_path):
                    with open(meta_path) as f:
                        if json.load(f) == meta:
                            return  # peer already saved an identical copy
            except Exception:
                pass

            # Flatten doc postings as CSR-like arrays.
            V_local = len(doc_postings)
            doc_indptr = np.zeros(V_local + 1, dtype=np.int64)
            doc_idx_flat: list[int] = []
            doc_w_flat: list[float] = []
            for c in range(V_local):
                pl = doc_postings[c]
                doc_indptr[c + 1] = doc_indptr[c] + len(pl)
                for didx, w in pl:
                    doc_idx_flat.append(int(didx))
                    doc_w_flat.append(float(w))
            doc_idx_arr = np.array(doc_idx_flat, dtype=np.int32)
            doc_w_arr = np.array(doc_w_flat, dtype=np.float32)

            # Flatten query sparse vectors as CSR-like arrays.
            n_q = len(query_sparse)
            q_indptr = np.zeros(n_q + 1, dtype=np.int64)
            q_centers_flat: list[np.ndarray] = []
            q_weights_flat: list[np.ndarray] = []
            for q in range(n_q):
                c_arr, w_arr = query_sparse[q]
                c_arr = np.asarray(c_arr, dtype=np.int32)
                w_arr = np.asarray(w_arr, dtype=np.float32)
                q_indptr[q + 1] = q_indptr[q] + c_arr.shape[0]
                q_centers_flat.append(c_arr)
                q_weights_flat.append(w_arr)
            q_centers = np.concatenate(q_centers_flat) if q_centers_flat else np.zeros((0,), dtype=np.int32)
            q_weights = np.concatenate(q_weights_flat) if q_weights_flat else np.zeros((0,), dtype=np.float32)

            # Atomic writes: tmp + os.replace.
            pid = os.getpid()
            doc_tmp = doc_npz_path + f".tmp.{pid}"
            q_tmp = query_npz_path + f".tmp.{pid}"
            meta_tmp = meta_path + f".tmp.{pid}"
            with open(doc_tmp, "wb") as _f:
                np.savez_compressed(_f, indptr=doc_indptr, doc_idx=doc_idx_arr, weight=doc_w_arr)
            with open(q_tmp, "wb") as _f:
                np.savez_compressed(_f, indptr=q_indptr, centers=q_centers, weights=q_weights)
            with open(meta_tmp, "w") as f:
                json.dump(meta, f, indent=2)
            os.replace(doc_tmp, doc_npz_path)
            os.replace(q_tmp, query_npz_path)
            os.replace(meta_tmp, meta_path)

    # ---- Per-span FAISS-search cache ((D, I) at K_max) ----
    # Keyed on (centers, section, K_max). Independent of all post-search filtering
    # (assignment K, hard/soft, per-center thresholds, weight aggregation, IDF,
    # normalization, stop centers, CLS exclusion). One cache hits all 4 sweeps.
    def _array_sig(arr: np.ndarray) -> str:
        return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]

    def _section_emb_sig(cache_dir: str, section: str) -> str:
        """Cheap fingerprint of a section's embeddings file (size + mtime).
        Used to invalidate centers_search and base_sparse_vectors caches when
        the underlying embeddings are regenerated."""
        p = os.path.join(cache_dir, f"{section}_{args.tokenization_unit}.npy")
        try:
            st = os.stat(p)
            return f"{st.st_size}-{int(st.st_mtime)}"
        except FileNotFoundError:
            return "missing"

    def _embeddings_sig(cache_dir: str, doc_sections: list[str]) -> str:
        joined = "|".join(f"{s}:{_section_emb_sig(cache_dir, s)}" for s in doc_sections)
        return hashlib.sha1(joined.encode()).hexdigest()[:12]

    @contextlib.contextmanager
    def _build_lock(lock_path: str, label: str = ""):
        """Cross-process exclusive lock for cache builders.

        Avoids duplicate work when multiple SLURM jobs hit a cold cache at the
        same time: the first job acquires the lock and builds; the rest block,
        then re-check the cache and skip the rebuild. Falls back to a no-op
        context manager when fcntl is unavailable (non-POSIX).
        """
        if fcntl is None:
            yield False
            return
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        f = open(lock_path, "a+")
        acquired_immediately = True
        try:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                acquired_immediately = False
                if label:
                    print(f"   ⏳ Waiting for centers_search build lock ({label})...")
                t0 = time.time()
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                if label:
                    print(f"   ✅ Acquired centers_search lock after {time.time() - t0:.1f}s ({label})")
            yield acquired_immediately
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            finally:
                f.close()

    def _section_search_cache_paths(cache_dir: str, sig: str, section: str) -> tuple[str, str, str]:
        base = os.path.join(cache_dir, "centers_search",
                            f"{sig}_{section}_{args.tokenization_unit}")
        return f"{base}.D.npy", f"{base}.I.npy", f"{base}.meta.json"

    def _get_or_compute_section_search(
        cache_dir: str,
        centers_sig: str,
        K_max: int,
        section_name: str,
        center_index: faiss.Index,
        batch_size: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (D, I) arrays of shape (n_spans, >=K_max) for *section_name*.

        Cache layout:
          - canonical filename per (centers_sig, section, embeddings_sig)
          - meta records the cached K_max; reused as long as cached >= requested
          - on size-up (requested > cached) we rebuild at max(requested, cached)
            so the new file is never smaller than the old one.
        """
        disabled = bool(getattr(args, "centers_search_cache_disable", False))
        emb_sig = _section_emb_sig(cache_dir, section_name)
        D_path, I_path, meta_path = _section_search_cache_paths(
            cache_dir, centers_sig, section_name)

        def _try_load(min_K: int) -> tuple[Optional[tuple[np.ndarray, np.ndarray]], int]:
            """Return ((D_mm, I_mm), cached_K) if the cache is usable for >= min_K, else (None, cached_K_or_-1)."""
            if disabled or not (os.path.exists(D_path) and os.path.exists(I_path) and os.path.exists(meta_path)):
                return None, -1
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if (meta.get("centers_sig") != centers_sig
                        or meta.get("section") != section_name
                        or meta.get("embeddings_sig") != emb_sig):
                    return None, -1
                cached_K = int(meta.get("K_max", -1))
                if cached_K < min_K:
                    return None, cached_K
                D_mm = np.load(D_path, mmap_mode='r')
                I_mm = np.load(I_path, mmap_mode='r')
                if D_mm.shape == I_mm.shape and D_mm.shape[1] >= min_K:
                    return (D_mm, I_mm), cached_K
                return None, cached_K
            except Exception as e:
                print(f"   ⚠️  centers_search cache unreadable for {section_name} ({e}); recomputing")
                return None, -1

        cached, cached_K = _try_load(K_max)
        if cached is not None:
            return cached

        # Miss → take cross-process lock, re-check, then build (size-up to max(requested, cached))
        if not disabled:
            os.makedirs(os.path.dirname(D_path), exist_ok=True)
            lock_path = D_path + ".lock"
            with _build_lock(lock_path, label=f"{section_name} K_max={K_max}"):
                cached, cached_K = _try_load(K_max)
                if cached is not None:
                    return cached
                build_K = max(K_max, cached_K) if cached_K > 0 else K_max
                return _build_section_search(
                    cache_dir, centers_sig, build_K, section_name, emb_sig,
                    center_index, batch_size, D_path, I_path, meta_path)
        return _build_section_search(
            cache_dir, centers_sig, K_max, section_name, emb_sig,
            center_index, batch_size, D_path, I_path, meta_path)

    def _build_section_search(
        cache_dir: str,
        centers_sig: str,
        K_max: int,
        section_name: str,
        emb_sig: str,
        center_index: faiss.Index,
        batch_size: int,
        D_path: str,
        I_path: str,
        meta_path: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        disabled = bool(getattr(args, "centers_search_cache_disable", False))
        _raw = _load_section_emb(cache_dir, section_name)
        n_spans = int(_raw.shape[0])
        if n_spans == 0:
            del _raw; gc.collect()
            empty_D = np.zeros((0, K_max), dtype=np.float32)
            empty_I = np.full((0, K_max), -1, dtype=np.int32)
            if not disabled:
                os.makedirs(os.path.dirname(D_path), exist_ok=True)
                np.save(D_path, empty_D)
                np.save(I_path, empty_I)
                with open(meta_path, "w") as f:
                    json.dump({"centers_sig": centers_sig, "K_max": int(K_max),
                               "section": section_name, "n_spans": 0,
                               "embeddings_sig": emb_sig}, f)
            return empty_D, empty_I

        # Process per-batch: copy+normalize only one batch at a time so peak RSS
        # stays ~batch_size*emb_dim*4 bytes instead of n_spans*emb_dim*4 bytes.
        # This is critical for large sections (e.g. invention ~96M spans ≈ 395 GB
        # if loaded in full), which would OOM any node.
        if disabled:
            print(f"   🔄 Computing centers_search for {section_name} ({n_spans:,} spans × K_max={K_max}) [cache DISABLED]...")
            D_full = np.empty((n_spans, K_max), dtype=np.float32)
            I_full = np.empty((n_spans, K_max), dtype=np.int32)
            for b_start in tqdm(range(0, n_spans, batch_size),
                                desc=f"  search {section_name}", leave=False):
                b_end = min(b_start + batch_size, n_spans)
                batch = np.array(_raw[b_start:b_end], dtype=np.float32, copy=True)
                faiss.normalize_L2(batch)
                D_b, I_b = center_index.search(batch, K_max)
                D_full[b_start:b_end] = D_b.astype(np.float32, copy=False)
                I_full[b_start:b_end] = I_b.astype(np.int32, copy=False)
            del _raw, batch; gc.collect()
            return D_full, I_full

        os.makedirs(os.path.dirname(D_path), exist_ok=True)
        D_tmp = D_path + f".tmp.{os.getpid()}"
        I_tmp = I_path + f".tmp.{os.getpid()}"
        D_mm = np.lib.format.open_memmap(D_tmp, mode='w+', dtype=np.float32, shape=(n_spans, K_max))
        I_mm = np.lib.format.open_memmap(I_tmp, mode='w+', dtype=np.int32, shape=(n_spans, K_max))
        print(f"   🔄 Building centers_search cache for {section_name} ({n_spans:,} spans × K_max={K_max})...")
        for b_start in tqdm(range(0, n_spans, batch_size),
                            desc=f"  search {section_name}", leave=False):
            b_end = min(b_start + batch_size, n_spans)
            batch = np.array(_raw[b_start:b_end], dtype=np.float32, copy=True)
            faiss.normalize_L2(batch)
            D_b, I_b = center_index.search(batch, K_max)
            D_mm[b_start:b_end] = D_b.astype(np.float32, copy=False)
            I_mm[b_start:b_end] = I_b.astype(np.int32, copy=False)
        D_mm.flush(); I_mm.flush()
        del D_mm, I_mm, _raw, batch; gc.collect()
        os.replace(D_tmp, D_path)
        os.replace(I_tmp, I_path)
        with open(meta_path, "w") as f:
            json.dump({"centers_sig": centers_sig, "K_max": int(K_max),
                       "section": section_name, "n_spans": int(n_spans),
                       "embeddings_sig": emb_sig}, f)
        return np.load(D_path, mmap_mode='r'), np.load(I_path, mmap_mode='r')

    def _query_search_cache_path(cache_dir: str, sig: str, query_text_hash: str) -> str:
        return os.path.join(cache_dir, "centers_search",
                            f"queries_{sig}_{query_text_hash}_{args.tokenization_unit}.npz")

    def _get_or_compute_query_search(
        cache_dir: str,
        centers_sig: str,
        K_max: int,
        query_text_hash: str,
        query_spans: list[np.ndarray],
        center_index: faiss.Index,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per-query (D, I) at >=K_max columns. Returns list aligned with query_spans.

        Cached on disk by (centers_sig, query_text_hash). The cache is reused
        as long as the cached K_max is >= the requested one.
        """
        disabled = bool(getattr(args, "centers_search_cache_disable", False))
        path = _query_search_cache_path(cache_dir, centers_sig, query_text_hash)
        n_q = len(query_spans)

        def _try_load() -> Optional[list[tuple[np.ndarray, np.ndarray]]]:
            if disabled or not os.path.exists(path):
                return None
            try:
                pack = np.load(path, allow_pickle=False)
                if (str(pack["centers_sig"].item()) == centers_sig
                        and int(pack["K_max"].item()) >= int(K_max)
                        and str(pack["query_text_hash"].item()) == query_text_hash
                        and int(pack["n_q"].item()) == n_q):
                    indptr = pack["indptr"].astype(np.int64, copy=False)
                    D_flat = pack["D"].astype(np.float32, copy=False)
                    I_flat = pack["I"].astype(np.int32, copy=False)
                    out = []
                    for q in range(n_q):
                        s, e = int(indptr[q]), int(indptr[q + 1])
                        out.append((D_flat[s:e], I_flat[s:e]))
                    return out
            except Exception as e:
                print(f"   ⚠️  query centers_search cache unreadable ({e}); recomputing")
            return None

        cached = _try_load()
        if cached is not None:
            return cached

        if not disabled:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with _build_lock(path + ".lock", label=f"queries K_max={K_max}"):
                cached = _try_load()
                if cached is not None:
                    return cached
                return _build_query_search(
                    centers_sig, K_max, query_text_hash, query_spans,
                    center_index, path, n_q)
        return _build_query_search(
            centers_sig, K_max, query_text_hash, query_spans,
            center_index, path, n_q)

    def _build_query_search(
        centers_sig: str,
        K_max: int,
        query_text_hash: str,
        query_spans: list[np.ndarray],
        center_index: faiss.Index,
        path: str,
        n_q: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        disabled = bool(getattr(args, "centers_search_cache_disable", False))
        # Compute (concat + one FAISS call)
        n_per_q = np.array([int(s.shape[0]) for s in query_spans], dtype=np.int64)
        indptr = np.concatenate(([0], np.cumsum(n_per_q))).astype(np.int64)
        total = int(indptr[-1])
        if total == 0:
            empty = [(np.zeros((0, K_max), dtype=np.float32),
                      np.full((0, K_max), -1, dtype=np.int32)) for _ in range(n_q)]
            return empty
        d_dim = int(query_spans[next(i for i, n in enumerate(n_per_q) if n > 0)].shape[1])
        all_spans = np.empty((total, d_dim), dtype=np.float32)
        for q in range(n_q):
            s, e = int(indptr[q]), int(indptr[q + 1])
            if e > s:
                all_spans[s:e] = query_spans[q].astype(np.float32, copy=False)
        faiss.normalize_L2(all_spans)
        D_flat, I_flat = center_index.search(all_spans, K_max)
        D_flat = D_flat.astype(np.float32, copy=False)
        I_flat = I_flat.astype(np.int32, copy=False)
        if not disabled:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = path + f".tmp.{os.getpid()}"
                with open(tmp, "wb") as _f:
                    np.savez(_f,
                             centers_sig=np.array(centers_sig),
                             K_max=np.array(int(K_max)),
                             query_text_hash=np.array(query_text_hash),
                             n_q=np.array(int(n_q)),
                             indptr=indptr, D=D_flat, I=I_flat)
                os.replace(tmp, path)
            except Exception as e:
                print(f"   ⚠️  failed to save query centers_search cache ({e})")
        out = []
        for q in range(n_q):
            s, e = int(indptr[q]), int(indptr[q + 1])
            out.append((D_flat[s:e], I_flat[s:e]))
        return out

    def _chunk_query_text(text: str, max_tokens: int = 512) -> list[str]:
        """Split a long query text into sentence-aligned chunks that each fit
        within *max_tokens* encoder tokens.  Returns a list of text chunks.
        If the text fits in one window, returns [text] unchanged."""
        # Quick check: does the full text fit?
        n_tokens = len(tokenizer.encode(text, add_special_tokens=True))
        if n_tokens <= max_tokens:
            return [text]

        # Split on sentence boundaries (period + space, or newline).
        # Fall back to whitespace if no sentence boundaries are found.
        import re as _re
        # Try splitting on ". " or ".\n" first (sentence-level)
        sents = _re.split(r'(?<=\.)\s+', text)
        if len(sents) <= 1:
            # No sentence boundaries — split on whitespace
            sents = text.split()

        chunks: list[str] = []
        current_chunk = ""
        for sent in sents:
            if not sent.strip():
                continue
            candidate = (current_chunk + " " + sent).strip() if current_chunk else sent
            n = len(tokenizer.encode(candidate, add_special_tokens=True))
            if n <= max_tokens:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # Start new chunk with this sentence
                current_chunk = sent
                # If a single sentence exceeds max_tokens, it will be truncated
                # by the encoder anyway — just keep it as one chunk.
        if current_chunk:
            chunks.append(current_chunk)
        return chunks if chunks else [text]

    def _encode_query_spans(texts: list[str], section: str, d: int, batch_size: int = 32) -> list[np.ndarray]:
        query_max_chunks = int(getattr(args, "query_max_chunks", -1))
        # -1 = unlimited chunking (default); 0 = disabled (truncate); >0 = cap at N chunks
        use_chunking = query_max_chunks != 0
        query_exclude_cls = getattr(args, "exclude_cls_spans", False)

        # ── Chunking pass: split long queries into 512-token windows ──
        if use_chunking:
            # Build flat lists: (original_query_idx, chunk_text)
            flat_texts: list[str] = []
            flat_query_indices: list[int] = []
            n_chunked = 0
            for q_idx, text in enumerate(texts):
                chunks = _chunk_query_text(text, max_tokens=512)
                if query_max_chunks > 0 and len(chunks) > query_max_chunks:
                    chunks = chunks[:query_max_chunks]
                if len(chunks) > 1:
                    n_chunked += 1
                for chunk_text in chunks:
                    flat_texts.append(chunk_text)
                    flat_query_indices.append(q_idx)
            cap_str = f"cap={query_max_chunks}" if query_max_chunks > 0 else "unlimited"
            if n_chunked > 0:
                print(f"   🔗 Query chunking: {n_chunked}/{len(texts)} queries split into "
                      f"{len(flat_texts)} chunks ({cap_str})")
        else:
            flat_texts = texts
            flat_query_indices = list(range(len(texts)))

        # ── Encode all chunks/texts ──
        all_query_spans: list[list[np.ndarray]] = [[] for _ in range(len(texts))]
        doc_ids = [f"query_{flat_query_indices[i]}" for i in range(len(flat_texts))]
        sections_list = [section] * len(flat_texts)

        for batch_start in range(0, len(flat_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(flat_texts))
            batch_results = _encode_spans(
                doc_texts=flat_texts[batch_start:batch_end],
                doc_ids=doc_ids[batch_start:batch_end],
                sections=sections_list[batch_start:batch_end],
            )

            for doc_id, _, _, _span_text_raw, _span_text_canonical, span_emb in batch_results:
                if query_exclude_cls:
                    t = (_span_text_canonical or "").strip().lower()
                    r = (_span_text_raw or "").strip()
                    if t == "cls" or r == "[CLS]":
                        continue
                q_idx = int(doc_id.split("_")[1])
                all_query_spans[q_idx].append(span_emb)

        # ── Stack into per-query arrays ──
        result: list[np.ndarray] = []
        for q_idx in range(len(texts)):
            spans = all_query_spans[q_idx]
            if spans:
                result.append(np.stack(spans))
            else:
                result.append(np.zeros((0, d), dtype=np.float32))

        return result

    def _assign_query_spans_to_centers(
        query_spans: list[np.ndarray],
        center_index: faiss.Index,
        V: int,
        sim_thr_per_center: np.ndarray,
        query_span_weights: Optional[list[np.ndarray]] = None,
        stop_centers: Optional[set] = None,
        precomputed_search: Optional[list[tuple[np.ndarray, np.ndarray]]] = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Assign each query's spans to vocabulary centers.

        Single vectorized implementation covering all four cases
        (soft / hard × precomputed / live FAISS). Per query we obtain a
        (n_spans, K) matrix of (D, I) — either by slicing the precomputed cache
        or by one FAISS call — then apply a vectorized filter
        (per-center radius + stop-centers + sim>0) and aggregate by center.

        Soft-empty fallback: if all candidates of a query are filtered out, we
        re-run with the per-center threshold relaxed (only stop-centers and
        sim>0 enforced). This preserves the original behaviour while keeping
        a single code path.
        """
        use_soft_assignment = (args.use_soft_assignment if hasattr(args, "use_soft_assignment") else True) \
            and not getattr(args, "no_soft_assignment", False)
        weight_agg = getattr(args, "weight_aggregation", "max")
        query_first_span_weight = float(getattr(args, "query_first_span_weight", 1.0))
        max_centers_per_span = int(getattr(args, "soft_assignment_max_centers_per_span", 10) or 0)

        # Stop-centers as boolean mask (vectorized membership test).
        stop_mask = np.zeros(V, dtype=bool)
        if stop_centers:
            stop_idx = np.fromiter(stop_centers, dtype=np.int64)
            stop_idx = stop_idx[(stop_idx >= 0) & (stop_idx < V)]
            stop_mask[stop_idx] = True

        _have_precomp = precomputed_search is not None and len(precomputed_search) == len(query_spans)

        def _per_span_extra(qi: int, n_spans: int) -> np.ndarray:
            extra = np.ones(n_spans, dtype=np.float32)
            if query_span_weights is not None and qi < len(query_span_weights):
                w = np.asarray(query_span_weights[qi], dtype=np.float32)
                m = min(n_spans, w.shape[0])
                extra[:m] = w[:m]
            if n_spans > 0 and query_first_span_weight != 1.0:
                extra[0] *= query_first_span_weight
            return extra

        def _aggregate(centers_v: np.ndarray, sims_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Group (centers_v, sims_v) by center and reduce by max or sum."""
            if centers_v.size == 0:
                return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
            order = np.argsort(centers_v, kind="stable")
            cs = centers_v[order]
            ws = sims_v[order]
            edges = np.concatenate(([True], cs[1:] != cs[:-1]))
            starts = np.where(edges)[0]
            if weight_agg == "sum":
                agg_w = np.add.reduceat(ws, starts)
            else:
                agg_w = np.maximum.reduceat(ws, starts)
            return cs[starts].astype(np.int32, copy=False), agg_w.astype(np.float32, copy=False)

        query_sparse: list[tuple[np.ndarray, np.ndarray]] = []
        n_fallback = 0
        for q_idx, spans in enumerate(query_spans):
            n_spans_q = int(spans.shape[0])
            if n_spans_q == 0:
                query_sparse.append((np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)))
                continue

            # 1. Obtain (D_q, I_q) of shape (n_spans_q, K_eff).
            if use_soft_assignment:
                K_target = max(1, max_centers_per_span)
            else:
                K_target = 1
            if _have_precomp:
                D_full, I_full = precomputed_search[q_idx]
                K_eff = min(K_target, int(D_full.shape[1]) if D_full.ndim == 2 else 0)
                D_q = np.asarray(D_full[:, :K_eff], dtype=np.float32)
                I_q = np.asarray(I_full[:, :K_eff], dtype=np.int32)
            else:
                spans_norm = spans.astype(np.float32, copy=True)
                faiss.normalize_L2(spans_norm)
                D_q, I_q = center_index.search(spans_norm, K_target)
                D_q = D_q.astype(np.float32, copy=False)
                I_q = I_q.astype(np.int32, copy=False)
                K_eff = K_target

            if K_eff == 0:
                query_sparse.append((np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)))
                continue

            extra = _per_span_extra(q_idx, n_spans_q)  # (n_spans_q,)

            # 2. Vectorized filter.
            I_clamped = np.clip(I_q, 0, V - 1)
            invalid = (I_q < 0) | (D_q <= 0) | stop_mask[I_clamped]
            invalid_with_thr = invalid | (D_q < sim_thr_per_center[I_clamped])

            # TopK cap (soft only); for hard K_eff==1 it's a no-op.
            if use_soft_assignment and max_centers_per_span > 0:
                valid_m = ~invalid_with_thr
                cum = np.cumsum(valid_m, axis=1)
                invalid_with_thr |= (cum > max_centers_per_span) & valid_m

            valid_mask = ~invalid_with_thr
            if not valid_mask.any() and use_soft_assignment:
                # Fallback: relax per-center radius (keep stop + sim>0). This
                # mirrors the original "if not center_weights: search k=1" path
                # but without the second FAISS call.
                n_fallback += 1
                valid_mask = ~invalid
                # Still cap at max_centers_per_span if requested.
                if max_centers_per_span > 0 and valid_mask.any():
                    cum = np.cumsum(valid_mask, axis=1)
                    valid_mask &= ~((cum > max_centers_per_span) & valid_mask)

            if not valid_mask.any():
                query_sparse.append((np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)))
                continue

            rows, cols = np.where(valid_mask)
            centers_v = I_q[rows, cols].astype(np.int32, copy=False)
            sims_v = D_q[rows, cols].astype(np.float32, copy=False) * extra[rows]
            # Drop any non-positive weights resulting from extra<=0.
            keep = sims_v > 0
            if not keep.all():
                centers_v = centers_v[keep]
                sims_v = sims_v[keep]

            centers_arr, weights_arr = _aggregate(centers_v, sims_v)
            query_sparse.append((centers_arr, weights_arr))

        if n_fallback > 0:
            print(f"   ⚠️  soft-assignment fallback (radius relaxed): {n_fallback}/{len(query_spans)} queries")
        return query_sparse



    # ---- Load CLEF-IP data for clefip_passage mode ----
    _clefip_data = None
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        raw_clefip = getattr(args, "clefip_root", None) or ""
        if raw_clefip and os.path.isabs(raw_clefip):
            _clefip_root = raw_clefip
        else:
            _clefip_root = os.path.normpath(os.path.join(current_dir, raw_clefip or "clefip2013"))
        if not os.path.isdir(_clefip_root):
            _fallback = os.path.normpath(os.path.join(current_dir, "clefip2013"))
            if _fallback != _clefip_root and os.path.isdir(_fallback):
                _clefip_root = _fallback
            else:
                raise FileNotFoundError(f"CLEF-IP root not found: {_clefip_root}")
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from clefip2013.load_clefip import (
            load_clefip_en_for_eval_full_corpus as _load_clefip_full,
            load_clefip_en_for_eval_sampled_corpus as _load_clefip_sampled,
        )
        _doc_root = os.path.join(_clefip_root, "01_document_collection", "01_extracted")
        if not os.path.isdir(_doc_root):
            raise FileNotFoundError(f"CLEF-IP document collection not found: {_doc_root}")
        _sample_size = getattr(args, "clefip_sample_size", 0) or 0
        _rebuild = getattr(args, "clefip_rebuild_corpus", False)
        if _sample_size != 0:
            _cq_ids, _cq_texts, _c_jsonl, _c_ids_txt, _c_npsg, _c_qrels = _load_clefip_sampled(
                _clefip_root, _doc_root, sample_size=_sample_size, rebuild_corpus=_rebuild,
                hard_neg_ratio=getattr(args, "clefip_hard_neg_ratio", 0.0) or 0.0,
            )
        else:
            _cq_ids, _cq_texts, _c_jsonl, _c_ids_txt, _c_npsg, _c_qrels = _load_clefip_full(
                _clefip_root, _doc_root, corpus_dir=None, rebuild_corpus=_rebuild
            )
        with open(_c_ids_txt, "r", encoding="utf-8") as f:
            _c_passage_ids = [line.strip() for line in f]
        assert len(_c_passage_ids) == _c_npsg
        # Load passage texts
        print(f"Loading CLEF-IP passage texts for sparse_coverage ({_c_npsg:,} passages)...")
        _c_passage_texts = []
        _c_passage_titles = []
        with open(_c_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                _c_passage_texts.append(obj["text"])
                _c_passage_titles.append(obj.get("title", ""))
        assert len(_c_passage_texts) == _c_npsg
        _clefip_data = {
            "query_ids": _cq_ids,
            "query_texts": _cq_texts,
            "passage_ids": _c_passage_ids,
            "passage_texts": _c_passage_texts,
            "passage_titles": _c_passage_titles,
            "qrels_passage_ids": _c_qrels,
        }
        print(f"   ✅ CLEF-IP data loaded: {len(_cq_ids)} queries, {_c_npsg:,} passages")
    except FileNotFoundError as e:
        print(f"   ⚠️ CLEF-IP data not available ({e}); skipping clefip_passage mode")
        _clefip_data = None
    except Exception as e:
        print(f"   ⚠️ Failed to load CLEF-IP data: {e}; skipping clefip_passage mode")
        _clefip_data = None

    def _encode_clefip_spans(
        passage_ids_list: list[str],
        passage_texts_list: list[str],
        doc_sections: list[str],
        cache_dir: str,
        batch_size: int = 32,
        passage_titles_list: Optional[list[str]] = None,
    ) -> int:
        """Encode CLEF-IP passages and stream directly to disk (memory-efficient).

        Each passage is treated as a single-section document.
        Embeddings are streamed to a raw binary file per section, then
        converted to .npy via memmap so that peak RAM is bounded by one
        batch rather than the entire section.

        Saves .npy files and span_to_doc .jsonl into *cache_dir*.
        Returns total number of spans saved.
        """
        os.makedirs(cache_dir, exist_ok=True)
        hidden_size = model.config.hidden_size
        format_scheme = get_encoder_format_scheme(args.dense_model)
        section_map = {"abstract": "abstract", "description": "invention", "claim": "claim"}

        titles_iter = passage_titles_list if passage_titles_list else [""] * len(passage_ids_list)

        # Group passages by section
        items_by_section: dict[str, list[tuple[str, str, str]]] = {}
        for pid, text, title in zip(passage_ids_list, passage_texts_list, titles_iter):
            sec_raw = _clefip_passage_section(pid)
            sec = section_map.get(sec_raw, sec_raw)
            if sec not in doc_sections:
                continue
            if sec == "claim":
                fmt_text = format_claim_for_encoder(format_scheme, text)
            elif sec == "invention":
                fmt_text = format_invention_for_encoder(format_scheme, text)
            else:
                # Use real title when available; otherwise keep [SEP] prefix so
                # PatentMap's contextual representation is consistent with indexed docs.
                # section_tokens + title → "title [SEP] [abstract] text"
                # section_tokens, no title → "[SEP] [abstract] text"
                fmt_text = format_abstract_for_encoder(format_scheme, title, text)
            items_by_section.setdefault(sec, []).append((pid, sec, fmt_text))

        span_to_doc: dict[int, str] = {}
        exclude_cls_indices: set[int] = set()
        exclude_cls = getattr(args, "exclude_cls_spans", False)
        current_idx = 0
        total_spans = 0

        for section in doc_sections:
            items = items_by_section.get(section, [])
            npy_path = os.path.join(cache_dir, f"{section}_{args.tokenization_unit}.npy")
            if not items:
                np.save(npy_path, np.zeros((0, hidden_size), dtype=np.float32))
                continue

            sec_ids = [it[0] for it in items]
            sec_sections = [it[1] for it in items]
            sec_texts = [it[2] for it in items]

            # Stream embeddings to a temporary raw binary file to avoid OOM.
            # Peak RAM = one batch of embeddings + metadata lists (< 2 GB).
            # Use a process-unique temp file to avoid collisions when multiple
            # jobs build the same cache directory concurrently.
            raw_path = os.path.join(
                cache_dir,
                f"_tmp_{section}_{os.getpid()}_{time.time_ns()}.raw",
            )
            span_count = 0
            section_doc_ids: list[str] = []
            section_is_cls: list[bool] = []

            with open(raw_path, "wb") as raw_f:
                for b_start in tqdm(range(0, len(sec_texts), batch_size),
                                    desc=f"Encoding CLEF-IP {section}", leave=False):
                    b_end = min(b_start + batch_size, len(sec_texts))
                    results = _encode_spans(
                        doc_texts=sec_texts[b_start:b_end],
                        doc_ids=sec_ids[b_start:b_end],
                        sections=sec_sections[b_start:b_end],
                    )
                    for doc_id, _, _, span_raw, span_canon, emb in results:
                        raw_f.write(emb.astype(np.float32).tobytes())
                        span_count += 1
                        is_cls = (span_canon or "").strip().lower() == "cls" or (span_raw or "").strip() == "[CLS]"
                        section_doc_ids.append(doc_id)
                        section_is_cls.append(is_cls)

            # Convert raw binary → proper .npy via memmap (constant RAM)
            if span_count > 0:
                raw_mm = np.memmap(raw_path, dtype=np.float32, mode="r",
                                   shape=(span_count, hidden_size))
                out_mm = np.lib.format.open_memmap(
                    npy_path, mode="w+", dtype=np.float32,
                    shape=(span_count, hidden_size),
                )
                CHUNK = 100_000  # ~400 MB per chunk at 1024-dim float32
                for ci in range(0, span_count, CHUNK):
                    ce = min(ci + CHUNK, span_count)
                    out_mm[ci:ce] = raw_mm[ci:ce]
                out_mm.flush()
                del out_mm, raw_mm
            else:
                np.save(npy_path, np.zeros((0, hidden_size), dtype=np.float32))
            try:
                os.remove(raw_path)
            except FileNotFoundError:
                # Another concurrent process may have already cleaned it up.
                pass

            # Update global metadata
            for did, is_c in zip(section_doc_ids, section_is_cls):
                span_to_doc[current_idx] = did
                if exclude_cls and is_c:
                    exclude_cls_indices.add(current_idx)
                current_idx += 1

            total_spans += span_count
            print(f"   {section}: {span_count:,} spans from {len(items):,} passages")
            del section_doc_ids, section_is_cls

        # Save span_to_doc metadata
        meta_path = os.path.join(cache_dir, f"span_to_doc_{args.tokenization_unit}.jsonl")
        with open(meta_path, "w") as f:
            for idx in sorted(span_to_doc.keys()):
                entry = {"i": idx, "d": span_to_doc[idx]}
                if idx in exclude_cls_indices:
                    entry["cls"] = True
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Sentinel: marks cache as built with title-aware abstract format.
        # Old caches lacking this file will be rebuilt automatically.
        with open(os.path.join(cache_dir, "format_title_in_abstract.txt"), "w") as f:
            f.write("title_in_abstract\n")

        return total_spans

    # Pre-cache CLEF-IP passage embeddings if data is available
    if _clefip_data is not None:
        _clefip_cache = _doc_cache_dir("clefip_passage")
        _clefip_sections = ["abstract", "claim", "invention"]
        _clefip_sentinel = os.path.join(_clefip_cache, "format_title_in_abstract.txt")
        if _doc_cache_exists(_clefip_cache, _clefip_sections) and os.path.exists(_clefip_sentinel):
            print(f"   ✅ CLEF-IP passage cache already exists")
        else:
            if _doc_cache_exists(_clefip_cache, _clefip_sections):
                print(f"   ⚠️  CLEF-IP cache predates title-aware format — rebuilding")
            print(f"   📦 Pre-caching CLEF-IP passage embeddings...")
            _total_clefip_spans = _encode_clefip_spans(
                _clefip_data["passage_ids"], _clefip_data["passage_texts"],
                _clefip_sections, cache_dir=_clefip_cache, batch_size=32,
                passage_titles_list=_clefip_data.get("passage_titles"),
            )
            print(f"   ✅ Cached {_total_clefip_spans:,} spans for clefip_passage")

    # ---- Find centers (shared across all tasks) ----
    print(f"\n🔍 Searching for centers...")
    print(f"   Search directory: {os.path.abspath('.')}")

    try:
        centers_path, _ = find_centers(
            dense_model=args.dense_model,
            tokenization_unit=args.tokenization_unit,
            search_dir=".",
            centers_suffix=getattr(args, 'centers_suffix', ''),
        )
        print(f"✅ Found centers: {centers_path}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Doc embeddings have been cached, but could not find centers:\n"
            f"  dense_model={args.dense_model}\n"
            f"  tokenization_unit={args.tokenization_unit}\n"
            f"Searched in: {os.path.abspath('.')} (recursive)\n"
            f"Expected pattern: centers_greedy_{{model}}_{{unit}}\n"
            f"Please ensure centers were built with matching parameters."
        )

    available_modes = ["clefip_passage"] if _clefip_data is not None else []
    print(f"\n📋 Will process {len(available_modes)} task(s): {', '.join(available_modes)}")

    # ---- Load centers & build FAISS index ONCE (shared across all modes) ----
    print(f"\n📦 Loading centers...")
    centers = np.load(centers_path).astype(np.float32)
    V_original, d = centers.shape
    print(f"   Original vocabulary size: {V_original:,} centers")
    print(f"   Embedding dimension: {d}")

    centers_info = _load_centers_info_json(centers_path)
    V = int(centers.shape[0])
    stop_centers: set = set()  # computed from doc-DF after posting lists are built

    r, sim_threshold = _get_r_and_sim_threshold(centers_info)

    # Per-center similarity thresholds (from k-center Voronoi radii)
    _rpc = centers_info.get("r_per_center", None)
    if _rpc is not None and len(_rpc) >= V:
        sim_thr_per_center = 1.0 - np.array(_rpc[:V], dtype=np.float32)
        print(f"   Per-center r_c loaded: sim thresholds in [{sim_thr_per_center.min():.4f}, {sim_thr_per_center.max():.4f}]")
    else:
        sim_thr_per_center = np.full(V, float(sim_threshold), dtype=np.float32)
        print(f"   No per-center r_c found; using global sim_threshold={sim_threshold:.4f} for all centers")

    if model.config.hidden_size != d:
        raise ValueError(f"Dimension mismatch: model hidden_size={model.config.hidden_size} but centers dimension={d}")

    print(f"\n🔨 Building FAISS index on centers...")
    centers_norm = centers.copy()
    faiss.normalize_L2(centers_norm)
    center_index = faiss.IndexFlatIP(d)
    center_index.add(centers_norm.astype(np.float32))
    # Move centers index to GPU if available — speeds up the per-section
    # centers_search build dramatically (often 10-30x for large span counts).
    # Memory footprint is tiny (V × d × 4 bytes, plus FAISS workspace ~1-2 GB).
    if torch.cuda.is_available():
        try:
            _faiss_gpu_res = faiss.StandardGpuResources()
            center_index = faiss.index_cpu_to_gpu(_faiss_gpu_res, 0, center_index)
            print(f"✅ Center index built (GPU: {torch.cuda.get_device_name(0)})")
        except Exception as _e:
            print(f"⚠️  GPU FAISS index unavailable ({_e}); falling back to CPU index")
            print(f"✅ Center index built (CPU)")
    else:
        print(f"✅ Center index built (CPU)")

    # Signature for the per-span FAISS-search cache (depends only on centers content).
    centers_sig = _array_sig(centers_norm)
    K_max_cache = int(getattr(args, "centers_search_cache_K", 64))
    if K_max_cache > V:
        K_max_cache = V
    print(f"   centers_sig={centers_sig}  K_max_cache={K_max_cache}"
          + ("  [centers_search cache DISABLED]" if getattr(args, "centers_search_cache_disable", False) else ""))

    for mode in available_modes:
        print(f"\n{'='*80}")
        print(f"Processing task: {mode}")
        print(f"{'='*80}")

        # Decide which sections to use for document indexing
        doc_sections = ["abstract", "claim", "invention"]
        query_section = "claim"

        # ---- Obtain doc-side embeddings + span_to_doc mapping ----
        # Cache-first path: load metadata + shapes, and mmap each section on demand.
        exclude_cls_spans = getattr(args, "exclude_cls_spans", False)
        span_to_doc: dict[int, str] = {}
        exclude_cls_span_indices: set[int] = set()
        total_loaded = 0

        cache_dir = _doc_cache_dir(mode)
        _runtime_sentinel = os.path.join(cache_dir, "format_title_in_abstract.txt")
        try:
            if mode == "clefip_passage" and not os.path.exists(_runtime_sentinel):
                raise FileNotFoundError("clefip cache predates title-aware format")
            print(f"\n📦 Trying cached doc embeddings from: {cache_dir}")
            # Lazy path: load only metadata + shapes (no giant arrays)
            span_to_doc, exclude_cls_span_indices = _load_doc_cache_meta(cache_dir)
            section_shapes: dict[str, int] = {}
            for sec in doc_sections:
                n = _get_section_shape(cache_dir, sec)
                section_shapes[sec] = n
                total_loaded += n
            print(f"   Loaded metadata ({len(span_to_doc):,} spans) from cache (lazy mode — arrays loaded per-section)")
            for sec in doc_sections:
                print(f"   {sec}: {section_shapes[sec]:,} spans")
        except FileNotFoundError:
            print(f"\n📦 Encoding CLEF-IP passages at runtime (first run; will cache for reuse)")
            total_loaded = _encode_clefip_spans(
                _clefip_data["passage_ids"], _clefip_data["passage_texts"],
                doc_sections, cache_dir=cache_dir, batch_size=32,
                passage_titles_list=_clefip_data.get("passage_titles"),
            )
            span_to_doc, exclude_cls_span_indices = _load_doc_cache_meta(cache_dir)

        if exclude_cls_spans:
            print(f"   Excluding {len(exclude_cls_span_indices):,} CLS spans")
        print(f"   Total: {len(span_to_doc):,} span-to-doc mappings, {total_loaded:,} embedding rows")

        # Build per-doc span count for length normalization (stable, pre-filtering)
        _clefip_pid_list = _clefip_data["passage_ids"]
        doc_id_to_idx = {pid: idx for idx, pid in enumerate(_clefip_pid_list)}
        N_docs = len(_clefip_pid_list)
        doc_span_count: dict[str, int] = {}
        for span_idx, doc_id in span_to_doc.items():
            if exclude_cls_spans and span_idx in exclude_cls_span_indices:
                continue
            doc_span_count[doc_id] = doc_span_count.get(doc_id, 0) + 1
        doc_nspans = np.ones(N_docs, dtype=np.float32)
        for doc_id, cnt in doc_span_count.items():
            didx = doc_id_to_idx.get(doc_id)
            if didx is not None:
                doc_nspans[didx] = float(cnt)

        # Query text formatting must match doc-side formatting for sparse assignment.
        _fmt_scheme = get_encoder_format_scheme(args.dense_model)
        query_texts = [format_claim_for_encoder(_fmt_scheme, qt) for qt in _clefip_data["query_texts"]]

        use_soft_assignment = (args.use_soft_assignment if hasattr(args, "use_soft_assignment") else True) \
            and not getattr(args, "no_soft_assignment", False)
        base_sparse_meta = {
            "mode": mode,
            "dense_model": args.dense_model,
            "tokenization_unit": args.tokenization_unit,
            "centers_path": os.path.abspath(centers_path),
            "centers_shape": [int(V), int(d)],
            "centers_sig": centers_sig,
            "embeddings_sig": _embeddings_sig(cache_dir, doc_sections),
            "sim_thr_sha1": _hash_float_array(sim_thr_per_center),
            "document_assignment": getattr(args, "document_assignment", "soft"),
            "weight_aggregation": getattr(args, "weight_aggregation", "max"),
            "use_soft_assignment": bool(use_soft_assignment),
            "soft_assignment_max_centers_per_span": int(getattr(args, "soft_assignment_max_centers_per_span", 10) or 0),
            "query_first_span_weight": float(getattr(args, "query_first_span_weight", 1.0)),
            "exclude_cls_spans": bool(exclude_cls_spans),
            "query_max_chunks": int(getattr(args, "query_max_chunks", -1)),
            "n_docs": int(len(_clefip_data["passage_ids"])),
            "query_text_hash": _hash_query_texts(query_texts),
        }

        cached_sparse = _load_base_sparse_vectors(cache_dir, base_sparse_meta)
        doc_postings: list[list[tuple[int, float]]]
        query_sparse: list[tuple[np.ndarray, np.ndarray]]

        if cached_sparse is not None:
            doc_postings, query_sparse = cached_sparse
            pl_lens = np.array([len(pl) for pl in doc_postings], dtype=np.float64)
            n_empty = int(np.sum(pl_lens == 0))
            total_entries = int(np.sum(pl_lens))
            doc_non_empty = pl_lens[pl_lens > 0]
            print(f"\n💾 Loaded base sparse vectors from cache: {cache_dir}")
            if len(doc_non_empty) > 0:
                print(f"📊 Doc-level posting lists (cached): total entries={total_entries:,}, "
                      f"mean/non-empty center={float(doc_non_empty.mean()):.1f}, "
                      f"max={int(pl_lens.max()):,}")
        else:
            # ---- Build posting lists (flat CSR; no list[list[tuple]] ever) ----
            document_assignment = getattr(args, "document_assignment", "soft")
            print(f"\n🔨 Computing posting lists for {V:,} centers...")
            print(f"   Document assignment: {document_assignment}")

            # Flat accumulators across sections: (center, span_global, sim).
            flat_centers_chunks: list[np.ndarray] = []
            flat_globals_chunks: list[np.ndarray] = []
            flat_sims_chunks: list[np.ndarray] = []

            max_centers_per_span = int(getattr(args, "soft_assignment_max_centers_per_span", 10) or 0)
            if document_assignment == "hard":
                K_search = 1
            else:
                # Use the full cached width; no 4×/64 heuristic needed because
                # we slice the cached top-K_max_cache and cap with topK at the end.
                K_search = min(K_max_cache, V)
                if K_search < max_centers_per_span:
                    print(f"   ⚠️  K_max_cache={K_max_cache} < max_centers_per_span={max_centers_per_span}; "
                          f"capping survivors at {K_search}.")
            K_max_eff = max(K_max_cache, K_search)
            min_sim_thr = float(sim_thr_per_center.min())
            if document_assignment == "hard":
                print(f"   Building posting lists: each span -> nearest center (k=1) [via centers_search cache]")
            else:
                print(f"   Soft assignment: top-K={K_search} from cache (K_max={K_max_eff}) "
                      f"+ per-center r_c filter + topK cap={max_centers_per_span}")
            _bs_search = max(1, int(getattr(args, "centers_search_batch", 4096)))

            # CLS exclusion as int64 array (fast np.isin)
            cls_excl_arr: Optional[np.ndarray] = None
            if exclude_cls_spans and exclude_cls_span_indices:
                if isinstance(exclude_cls_span_indices, np.ndarray):
                    cls_excl_arr = exclude_cls_span_indices.astype(np.int64, copy=False)
                else:
                    cls_excl_arr = np.fromiter(exclude_cls_span_indices, dtype=np.int64)

            span_offset = 0
            for section_name in doc_sections:
                D_mm, I_mm = _get_or_compute_section_search(
                    cache_dir, centers_sig, K_max_eff, section_name,
                    center_index, batch_size=_bs_search)
                n_spans_sec = int(D_mm.shape[0])
                if n_spans_sec == 0:
                    continue

                # One-shot whole-section read of the K_search columns we need.
                # mmap → contiguous in-memory copy (n_spans_sec, K_search) is the only
                # working set; no per-batch FAISS calls, no per-batch tqdm noise.
                D_sec = np.array(D_mm[:, :K_search], dtype=np.float32, copy=True)
                I_sec = np.array(I_mm[:, :K_search], dtype=np.int32, copy=True)
                del D_mm, I_mm; gc.collect()

                # --- Vectorized filtering (whole section) ---
                I_clamped = np.clip(I_sec, 0, V - 1)
                invalid = (I_sec < 0) | (D_sec <= 0) | (D_sec < min_sim_thr) \
                          | (D_sec < sim_thr_per_center[I_clamped])

                if cls_excl_arr is not None:
                    global_indices = np.arange(n_spans_sec, dtype=np.int64) + span_offset
                    row_exclude = np.isin(global_indices, cls_excl_arr)
                    invalid[row_exclude] = True

                # TopK cap per row
                if document_assignment != "hard" and max_centers_per_span > 0:
                    valid_mat = ~invalid
                    cumvalid = np.cumsum(valid_mat, axis=1)
                    invalid |= (cumvalid > max_centers_per_span) & valid_mat

                valid_mask = ~invalid
                if valid_mask.any():
                    rows, cols = np.where(valid_mask)
                    centers_valid = I_sec[rows, cols].astype(np.int32, copy=False)
                    sims_valid = D_sec[rows, cols].astype(np.float32, copy=False)
                    globals_valid = rows.astype(np.int64, copy=False) + span_offset
                    flat_centers_chunks.append(centers_valid)
                    flat_globals_chunks.append(globals_valid)
                    flat_sims_chunks.append(sims_valid)

                span_offset += n_spans_sec
                print(f"     {section_name}: {n_spans_sec:,} spans assigned")
                del D_sec, I_sec, invalid, valid_mask; gc.collect()
            print(f"   Total spans: {total_loaded:,}")

            # Alignment sanity check
            if total_loaded != len(span_to_doc):
                raise ValueError(
                    f"Embeddings count ({total_loaded:,}) != span_to_doc count ({len(span_to_doc):,}). "
                    "Posting lists require exact 1:1 alignment."
                )

            # Concatenate flat span-level posting list (centers, globals, sims).
            if flat_centers_chunks:
                flat_centers = np.concatenate(flat_centers_chunks)
                flat_globals = np.concatenate(flat_globals_chunks)
                flat_sims = np.concatenate(flat_sims_chunks)
            else:
                flat_centers = np.zeros((0,), dtype=np.int32)
                flat_globals = np.zeros((0,), dtype=np.int64)
                flat_sims = np.zeros((0,), dtype=np.float32)
            del flat_centers_chunks, flat_globals_chunks, flat_sims_chunks; gc.collect()

            # Span-level posting list stats (numpy bincount, no per-center loop)
            if flat_centers.size > 0:
                span_pl_lens = np.bincount(flat_centers, minlength=V).astype(np.int64)
                span_non_empty = span_pl_lens[span_pl_lens > 0]
                print(f"\n📊 Span-level posting lists (build cost/memory): "
                      f"total entries={int(span_pl_lens.sum()):,}, "
                      f"mean/non-empty center={float(span_non_empty.mean()):.1f}, "
                      f"max={int(span_pl_lens.max()):,}")

            # Build document-level inverted index: convert span->doc, aggregate by (center, doc).
            print(f"\n🔨 Building document-level inverted index from posting lists...")
            weight_agg = getattr(args, "weight_aggregation", "max")

            # Vectorized span_global -> doc_idx via lookup array.
            n_spans_total = int(span_offset)
            span_to_didx = np.full(n_spans_total, -1, dtype=np.int64)
            for span_idx, doc_id in span_to_doc.items():
                didx = doc_id_to_idx.get(doc_id)
                if didx is not None and 0 <= span_idx < n_spans_total:
                    span_to_didx[int(span_idx)] = int(didx)

            if flat_centers.size > 0:
                flat_didx = span_to_didx[flat_globals]
                # Drop entries with no doc mapping or sim<=0
                keep = (flat_didx >= 0) & (flat_sims > 0)
                if not keep.all():
                    flat_centers = flat_centers[keep]
                    flat_didx = flat_didx[keep]
                    flat_sims = flat_sims[keep]
                else:
                    flat_didx = flat_didx
            else:
                flat_didx = np.zeros((0,), dtype=np.int64)

            # Aggregate (center, doc_idx) pairs -> max or sum of sims.
            if flat_centers.size > 0:
                # Pack key = center * N_docs + doc_idx (both fit in int64).
                _N = int(N_docs)
                key = flat_centers.astype(np.int64) * _N + flat_didx
                order = np.argsort(key, kind="stable")
                key_s = key[order]
                sim_s = flat_sims[order]
                # Group boundaries
                edges = np.concatenate(([True], key_s[1:] != key_s[:-1]))
                group_start = np.where(edges)[0]
                group_end = np.append(group_start[1:], key_s.size)
                if weight_agg == "sum":
                    agg_w = np.add.reduceat(sim_s, group_start)
                else:
                    agg_w = np.maximum.reduceat(sim_s, group_start)
                agg_key = key_s[group_start]
                agg_centers = (agg_key // _N).astype(np.int32, copy=False)
                agg_didx = (agg_key % _N).astype(np.int32, copy=False)
                agg_w = agg_w.astype(np.float32, copy=False)
                # Build doc_postings (list-of-list-of-tuple) for downstream code.
                # This is the only place we materialize the legacy structure.
                doc_postings = [[] for _ in range(V)]
                # Group by center via np.split on contiguous runs.
                c_order = np.argsort(agg_centers, kind="stable")
                ac = agg_centers[c_order]
                ad = agg_didx[c_order]
                aw = agg_w[c_order]
                if ac.size > 0:
                    c_edges = np.concatenate(([True], ac[1:] != ac[:-1]))
                    c_starts = np.where(c_edges)[0]
                    c_ends = np.append(c_starts[1:], ac.size)
                    for i in range(c_starts.size):
                        s, e = int(c_starts[i]), int(c_ends[i])
                        cid = int(ac[s])
                        doc_postings[cid] = list(zip(ad[s:e].tolist(), aw[s:e].tolist()))
            else:
                doc_postings = [[] for _ in range(V)]
            del flat_centers, flat_globals, flat_sims, span_to_didx; gc.collect()

            # Doc-level posting list stats (retrieval cost / FLOPs)
            pl_lens = np.array([len(pl) for pl in doc_postings], dtype=np.float64)
            doc_non_empty = pl_lens[pl_lens > 0]
            n_empty = int(np.sum(pl_lens == 0))
            total_entries = int(np.sum(pl_lens))
            if len(doc_non_empty) > 0:
                print(f"📊 Doc-level posting lists (retrieval cost/FLOPs): "
                      f"total entries={total_entries:,}, "
                      f"mean/non-empty center={float(doc_non_empty.mean()):.1f}, "
                      f"max={int(pl_lens.max()):,}")

            # Encode + assign queries once (without stop-center post-processing)
            query_spans = _encode_query_spans(query_texts, section=query_section, d=d)
            query_span_weights = None
            # Pre-compute (and cache) per-query (D, I) at K_max_cache so subsequent
            # sweeps that vary assignment K (1..K_max_cache) hit the cache.
            _q_K = min(K_max_cache, V)
            _query_text_hash = base_sparse_meta.get("query_text_hash", "")
            query_precomp = _get_or_compute_query_search(
                cache_dir, centers_sig, _q_K, _query_text_hash,
                query_spans, center_index)
            query_sparse = _assign_query_spans_to_centers(
                query_spans,
                center_index=center_index,
                V=V,
                sim_thr_per_center=sim_thr_per_center,
                query_span_weights=query_span_weights,
                stop_centers=None,
                precomputed_search=query_precomp,
            )

            _save_base_sparse_vectors(cache_dir, base_sparse_meta, doc_postings, query_sparse)
            print(f"   💾 Saved base sparse vectors cache: {cache_dir}")

        # --- Compute stop_centers from doc-DF (top fraction by posting-list length) ---
        if getattr(args, "stop_center_fraction", 0.0) > 0:
            _df_for_stop = np.array([len(pl) for pl in doc_postings], dtype=np.float64)
            n_stop = max(1, round(V * args.stop_center_fraction))
            _stop_ranked = np.argsort(-_df_for_stop)
            stop_centers = set(int(c) for c in _stop_ranked[:n_stop])
            _stop_thr = int(_df_for_stop[_stop_ranked[n_stop - 1]])
            print(f"   stop_centers: top {n_stop} ({args.stop_center_fraction:.1%}) by doc-DF, "
                  f"min_df={_stop_thr:,} → {len(stop_centers):,} centers cleared")
            for _sc in stop_centers:
                doc_postings[_sc] = []
        else:
            print(f"   stop-center suppression disabled (stop_center_fraction <= 0)")

        N_docs = len(_clefip_data["passage_ids"])
        df = np.array([len(pl) for pl in doc_postings], dtype=np.float32)  # doc_idx unique per center (aggregated above)
        idf = (np.log((N_docs + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
        idf_exponent = float(getattr(args, "idf_exponent", 1.0))
        if idf_exponent != 1.0:
            print(f"   IDF exponent: {idf_exponent} (score term uses idf^{idf_exponent})")

        # Query sparse vectors are precomputed/cached without stop-center post-processing.
        print(f"\n📝 Evaluating: CLEF-IP Claims -> Passages")
        # Per-query retrieval cost diagnostics
        postings_per_query = np.array(
            [sum(len(doc_postings[t]) for t in qpack[0]) for qpack in query_sparse],
            dtype=np.float64,
        )
        centers_per_query = np.array(
            [len(qpack[0]) for qpack in query_sparse], dtype=np.float64,
        )
        total_flops = int(2 * postings_per_query.sum())
        n_queries_flops = len(query_sparse)
        _report_flops_and_postings_one_line(
            int(total_entries), V - n_empty, V, mode,
            total_flops=total_flops, n_queries=n_queries_flops, model_label="sparse_coverage"
        )
        if n_queries_flops > 0:
            pq = postings_per_query
            cq = centers_per_query
            print(f"   Postings scanned per query  — "
                  f"mean={pq.mean():.0f}, p50={np.percentile(pq,50):.0f}, "
                  f"p90={np.percentile(pq,90):.0f}, p99={np.percentile(pq,99):.0f}, "
                  f"max={pq.max():.0f}")
            print(f"   Active centers per query    — "
                  f"mean={cq.mean():.1f}, p50={np.percentile(cq,50):.0f}, "
                  f"p90={np.percentile(cq,90):.0f}, p99={np.percentile(cq,99):.0f}, "
                  f"max={cq.max():.0f}")

        # Length normalization setup
        length_norm = getattr(args, "length_norm", "none")
        if length_norm == "sqrt_centers":
            length_norm = "sqrt_spans"
        length_norm_exp = getattr(args, "length_norm_exponent", 0.5)
        if length_norm == "sqrt_spans":
            print(f"   Length norm: sqrt_spans (exponent={length_norm_exp})")
        q_opts = []
        if getattr(args, "query_first_span_weight", 1.0) != 1.0:
            q_opts.append(f"E: first_span_weight={args.query_first_span_weight}")
        if q_opts:
            print(f"   Query opts: {', '.join(q_opts)}")
        print(f"🔍 Retrieving documents...")
        _retrieval_top_k = len(_clefip_data["passage_ids"])
        _result = _score_queries_against_postings(
            query_sparse, doc_postings, idf, idf_exponent, _retrieval_top_k,
            length_norm=length_norm,
            length_norm_exp=length_norm_exp,
            doc_nspans=doc_nspans,
            return_scores=True,
        )
        _, _sparse_score_dicts = _result

        if mode == "clefip_passage":
            # CLEF-IP evaluation: retrieve passage_ids, use CLEF-IP official metrics
            _clefip_pid_list = _clefip_data["passage_ids"]
            _clefip_qids = _clefip_data["query_ids"]
            _clefip_qrels = _clefip_data["qrels_passage_ids"]

            topk_docs = getattr(args, "clefip_two_stage_topk_docs", 100)
            # Build per-query score dicts for two-stage reranking
            all_passage_scores_list: list[dict] = []
            for q_idx in range(len(_clefip_qids)):
                if _sparse_score_dicts is not None and q_idx < len(_sparse_score_dicts):
                    # Use actual sparse retrieval scores (index-keyed → passage_id-keyed)
                    pscores = {_clefip_pid_list[pi]: s
                               for pi, s in _sparse_score_dicts[q_idx].items()}
                else:
                    pscores = {}
                all_passage_scores_list.append(pscores)
            predicted_labels_list, doc_ranking_list, full_doc_ranking_list = _clefip_two_stage_rerank(
                _clefip_pid_list, all_passage_scores_list,
                topk_docs=topk_docs,
            )
            print(f"  🔄 Two-stage retrieval: top-{topk_docs} docs → re-ranked passages per query")

            _evaluate_and_print_clefip(
                _clefip_qrels, _clefip_qids, predicted_labels_list,
                "Sparse Coverage (CLEF-IP)",
                two_stage=True, topk_docs=topk_docs,
                doc_ranking_list=doc_ranking_list,
                full_doc_ranking_list=full_doc_ranking_list,
            )

            # ---- CLEF-IP Robustness Test: varying negative pool sizes ----
            _neg_doc_sizes_str = getattr(args, "clefip_neg_doc_sizes", "")
            if _neg_doc_sizes_str:
                import random as _random_mod

                # passage_id -> doc_id mapping
                _pid_to_docid = {pid: _clefip_passage_id_to_doc_id(pid) for pid in _clefip_pid_list}
                _all_doc_ids_sorted = sorted(set(_pid_to_docid.values()))

                # Relevant doc_ids (must always be included)
                _rel_doc_ids = set()
                for _rel_pids in _clefip_qrels.values():
                    for _rpid in _rel_pids:
                        _rel_doc_ids.add(_clefip_passage_id_to_doc_id(_rpid))
                _neg_doc_ids_sorted = sorted([d for d in _all_doc_ids_sorted if d not in _rel_doc_ids])
                _n_rel = len(_rel_doc_ids)

                _target_sizes = sorted([int(s.strip()) for s in _neg_doc_sizes_str.split(",") if s.strip()])
                print(f"\n{'='*80}")
                print(f"📊 CLEF-IP Robustness Test: varying negative pool sizes")
                print(f"   Total docs: {len(_all_doc_ids_sorted):,}, Relevant docs: {_n_rel}, "
                      f"Negative docs: {len(_neg_doc_ids_sorted):,}")
                print(f"   Pool sizes to test: {_target_sizes}")
                print(f"{'='*80}")

                _robustness_results = []
                for _target_n in _target_sizes:
                    if _target_n >= len(_all_doc_ids_sorted):
                        print(f"\n   ⏭️  {_target_n} docs >= total {len(_all_doc_ids_sorted)}, already evaluated above")
                        continue
                    if _target_n < _n_rel:
                        print(f"\n   ⏭️  {_target_n} docs < {_n_rel} relevant docs, skipping")
                        continue

                    # Sample negatives (deterministic seed per pool size)
                    _n_neg_keep = min(_target_n - _n_rel, len(_neg_doc_ids_sorted))
                    _rng_local = _random_mod.Random(42 + _target_n)
                    _sampled_negs = set(_rng_local.sample(_neg_doc_ids_sorted, _n_neg_keep))
                    _allowed_docs = _rel_doc_ids | _sampled_negs

                    # Build allowed passage index set
                    _allowed_pidx = set()
                    for _pi, _pid in enumerate(_clefip_pid_list):
                        if _pid_to_docid[_pid] in _allowed_docs:
                            _allowed_pidx.add(_pi)
                    _n_passages_f = len(_allowed_pidx)
                    _n_docs_f = len(_allowed_docs)

                    # Filter doc_postings
                    _dp_f = [
                        [(di, w) for di, w in pl if di in _allowed_pidx]
                        for pl in doc_postings
                    ]

                    # Recompute IDF with filtered pool
                    _df_f = np.array(
                        [len(pl) for pl in _dp_f],
                        dtype=np.float32,
                    )
                    _idf_f = (np.log((_n_passages_f + 1.0) / (_df_f + 1.0)) + 1.0).astype(np.float32)

                    # Retrieve with filtered inverted index (full retrieval for two-stage)
                    _retrieval_top_k_f = _n_passages_f
                    _result_f = _score_queries_against_postings(
                        query_sparse, _dp_f, _idf_f, idf_exponent, _retrieval_top_k_f,
                        length_norm=length_norm,
                        length_norm_exp=length_norm_exp,
                        doc_nspans=doc_nspans,
                        show_progress=False,
                        return_scores=True,
                    )
                    _, _sparse_score_dicts_f = _result_f

                    # Build per-query passage score dicts for two-stage reranking
                    _allowed_pid_list = [pid for pi, pid in enumerate(_clefip_pid_list) if pi in _allowed_pidx]
                    _passage_scores_f: list[dict] = []
                    for _qi_e in range(len(_clefip_qids)):
                        if _sparse_score_dicts_f is not None and _qi_e < len(_sparse_score_dicts_f):
                            _pscores_f = {_clefip_pid_list[pi]: s
                                          for pi, s in _sparse_score_dicts_f[_qi_e].items()}
                        else:
                            _pscores_f = {}
                        _passage_scores_f.append(_pscores_f)
                    _pred_f, _doc_ranking_f, _full_doc_ranking_f = _clefip_two_stage_rerank(
                        _allowed_pid_list, _passage_scores_f, topk_docs=topk_docs,
                    )

                    _true_f = [_clefip_qrels.get(qid, []) for qid in _clefip_qids]
                    _res_f = _make_clefip_official_metrics(_true_f, _pred_f,
                                                          doc_ranking_list=_doc_ranking_f,
                                                          full_doc_ranking_list=_full_doc_ranking_f)
                    _robustness_results.append((_n_docs_f, _n_passages_f, _res_f))
                    print(f"   {_n_docs_f:>6} docs ({_n_passages_f:>8,} passages): "
                          f"recall_passage@1000={_res_f.get('recall_passage@1000', 0):.4f}  "
                          f"map_doc={_res_f.get('map_doc', 0):.4f}  "
                          f"pres_doc@100={_res_f.get('pres_doc@100', 0):.4f}")

                # Summary table
                if _robustness_results:
                    print(f"\n{'='*90}")
                    print(f"CLEF-IP Robustness Summary (relevant docs always included, negatives subsampled)")
                    print(f"{'='*90}")
                    print(f"{'Docs':>8} {'Passages':>10} {'recall_passage@1000':>20} "
                          f"{'map_doc':>10} {'pres_doc@100':>14} {'magp':>8}")
                    print(f"{'-'*8} {'-'*10} {'-'*20} {'-'*10} {'-'*14} {'-'*8}")
                    for _nd, _np, _r in _robustness_results:
                        print(f"{_nd:>8} {_np:>10,} "
                              f"{_r.get('recall_passage@1000', 0):>20.4f} "
                              f"{_r.get('map_doc', 0):>10.4f} "
                              f"{_r.get('pres_doc@100', 0):>14.4f} "
                              f"{_r.get('magp', 0):>8.4f}")
                    print(f"{'='*92}")



        print(f"\n✅ Task {mode} evaluation completed")

        # Free large objects before next mode to avoid peak memory overlap
        del span_to_doc, exclude_cls_span_indices
        if "posting_lists" in locals():
            del posting_lists
        del doc_postings, query_sparse
        gc.collect()

    print(f"\n✅ Sparse Coverage evaluation completed for all available tasks")

############################################## CLEF-IP 2013 EN (claims-to-passages) ##################################################

def main():
    parser = build_parser()
    args = parser.parse_args()
    print(f"Running evaluation for model: {args.model_name}")
    print("=============================================>>>>>>>>>")

    # Handle the case where model_name is None
    if args.model_name is None:
        print("Error: --model_name is required")
        return

    # Print evaluation header
    print(f"📋 Model: {args.model_name}")
    print(f"📁 Output Directory: {args.temp_dir}")

    if args.model_name == "sparse_coverage":
        run_sparse_coverage(args)
    else:
        # Non-sparse_coverage models: load deferred to run_clefip_eval to avoid
        # loading the model twice (once here, once in the evaluation path).
        print("\n" + "=" * 60)
        print("Running CLEF-IP 2013 EN (claims-to-passages)")
        print("=" * 60)
        run_clefip_eval(args)


########################################################################################################################################################
########################################################################################################################################################

def cleanup_resources():
    """Clean up GPU memory and other resources to prevent segfaults"""
    import gc
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # Force synchronization
            torch.cuda.synchronize()
    except Exception as e:
        print(f"Warning: Error during GPU cleanup: {e}")
    
    # Force garbage collection
    gc.collect()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always cleanup resources
        cleanup_resources()
        print("Resource cleanup completed.")