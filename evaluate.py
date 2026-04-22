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
import argparse
import logging
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
    get_encoder_sep_for_model,
    ENCODER_FORMAT_SECTION_TOKENS,
    format_abstract_for_encoder,
    format_claim_for_encoder,
    format_invention_for_encoder,
    collect_doc_texts,
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
        (reranked_list, doc_ranking_list):
          reranked_list:    per-query list of passage_ids (all passages from top-K docs, sorted by score).
          doc_ranking_list: per-query list of doc_ids from Stage 1 dedup (up to topk_docs unique docs,
                            in order of first passage occurrence). Used for document-level metrics.
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
    for q_idx in range(len(passage_scores_list)):
        scores = passage_scores_list[q_idx]

        # Stage 1: rank passages by score desc, derive document ranking (first-occurrence dedup)
        ranked_pids = sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)
        seen_docs = set()
        top_docs = []
        for pid in ranked_pids:
            doc_id = pid_to_doc.get(pid, pid.split("::", 1)[0] if "::" in pid else pid)
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                top_docs.append(doc_id)
                if len(top_docs) >= topk_docs:
                    break
        doc_ranking_list.append(top_docs)

        # Stage 2: collect ALL passages from those top-K documents, re-rank by score
        candidate_pids = set()
        for doc_id in top_docs:
            candidate_pids.update(doc_to_pids.get(doc_id, set()))

        scored_candidates = [(pid, scores.get(pid, -1e9)) for pid in candidate_pids]
        scored_candidates.sort(key=lambda x: -x[1])
        reranked_list.append([pid for pid, _ in scored_candidates])

    return reranked_list, doc_ranking_list


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
    plus recall@k, NDCG@k, MRR, MAP, pres@100 (passage-level).

    Two-stage retrieval is always applied:
      Stage 1: passage rank → document dedup → top-K documents.
      Stage 2: re-rank ALL passages from those top-K docs by cosine similarity.
    """
    assert len(query_ids) == len(query_embeddings)
    assert len(passage_ids) == len(passage_embeddings)
    Q = np.array(query_embeddings, dtype=np.float32, copy=True)
    D = np.array(passage_embeddings, dtype=np.float32, copy=True)
    assert Q.shape[1] == D.shape[1]
    faiss.normalize_L2(Q)
    faiss.normalize_L2(D)
    _report_dense_flops(Q, D, "CLEF-IP passage", model_label=model_label)
    sim = Q @ D.T
    passage_scores_list = [
        {passage_ids[j]: float(sim[q_idx, j]) for j in range(sim.shape[1])}
        for q_idx in range(len(query_ids))
    ]
    predicted_labels_list, doc_ranking_list = _clefip_two_stage_rerank(
        passage_ids, passage_scores_list, topk_docs=topk_docs,
    )
    print(f"  🔄 Two-stage retrieval: top-{topk_docs} docs → re-ranked passages per query")
    results = _evaluate_and_print_clefip(
        qrels_passage_ids, query_ids, predicted_labels_list,
        "Passage retrieval",
        two_stage=True, topk_docs=topk_docs,
        doc_ranking_list=doc_ranking_list,
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
        sents = _re2.split(r'(?<=\.)\s+', text)
        if len(sents) <= 1:
            sents = text.split()
        chunks: list[str] = []
        current = ""
        for sent in sents:
            if not sent.strip():
                continue
            candidate = (current + " " + sent).strip() if current else sent
            if len(tok.encode(candidate, add_special_tokens=True)) <= max_tokens:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sent  # single sent may still exceed max; encoder will truncate
        if current:
            chunks.append(current)
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

    # PatentMap models: pooler_output / CLS pooling + section tokens
    if "patentmap" in model_name.lower():
        import utils
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
        utils.ensure_section_tokens(tokenizer, model)
        model.to(device).eval()

        def _fwd_patentmap(m, inp):
            try:
                out = m(**inp, output_hidden_states=True, return_dict=True, sent_emb=True)
                return out.pooler_output.cpu().numpy()
            except TypeError:
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


def _clefip_format_for_model(query_texts, passage_ids, passage_texts, model_name):
    """
    Format CLEF-IP query/passage texts per utils encoder format scheme. Returns (query_list, passage_list).
    - ENCODER_FORMAT_TITLE_SEP_ONLY (Specter2, PAECTer, Patembed): raw text, no section tokens.
    - ENCODER_FORMAT_SECTION_TOKENS (e.g. bert-for-patents): [claim] query, [abstract]/[invention]/[claim] passage.
    """
    if not query_texts or not passage_texts:
        return query_texts, passage_texts
    scheme = get_encoder_format_scheme(model_name)
    if scheme != ENCODER_FORMAT_SECTION_TOKENS:
        return query_texts, passage_texts
    query_fmt = [format_claim_for_encoder(scheme, t) for t in query_texts]
    passage_sections = [_clefip_passage_section(pid) for pid in passage_ids]
    passage_fmt = []
    for s, t in zip(passage_sections, passage_texts):
        if s == "claim":
            passage_fmt.append(format_claim_for_encoder(scheme, t))
        elif s == "abstract":
            passage_fmt.append(f"[abstract] {t}".strip())
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
            doc_freq = np.diff(np.asarray(indptr)).astype(np.float64)
    if doc_freq is None:
        print(f"\n📊 BM25 ({label}): inverted index built (posting lengths not exposed by library).")
        return
    pl_lens = np.asarray(doc_freq).ravel().astype(np.float64)
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
) -> dict:
    """
    CLEF-IP metrics: passage-level + document-level.

    Passage-level (3):
      - magp        — MAP(D), official CLEF-IP hierarchical per-document AP (Piroi et al. 2012).
      - recall@100  — standard passage recall.
      - ndcg@10     — top-of-list ranking quality.

    Document-level (4):
      - pres_doc@100   — official CLEF-IP document PRES.
      - recall_doc@100 — document recall.
      - ndcg_doc@10    — top-of-list document ranking quality.
      - map_doc        — document-level MAP.

    *doc_ranking_list* is the Stage-1 document ranking from two-stage retrieval and
    is required: all callers compute it via _clefip_two_stage_rerank.
    """
    # Passage-level
    _passage_k = max((len(p) for p in predicted_labels_list), default=100)
    metrics = {
        "recall@100": mean_recall_at_k(true_labels_list, predicted_labels_list, k=100),
        "ndcg@10": mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=10),
        "magp": _clefip_mean_agp_passage(true_labels_list, predicted_labels_list, k=_passage_k),
    }
    # Document-level
    true_doc_ids_list = [
        list({_clefip_passage_id_to_doc_id(pid) for pid in rel_passages})
        for rel_passages in true_labels_list
    ]
    metrics["pres_doc@100"] = mean_pres_at_k(true_doc_ids_list, doc_ranking_list, k=100, N_max=100)
    metrics["recall_doc@100"] = mean_recall_at_k(true_doc_ids_list, doc_ranking_list, k=100)
    metrics["ndcg_doc@10"] = mean_ndcg_at_k(true_doc_ids_list, doc_ranking_list, k=10)
    metrics["map_doc"] = mean_average_precision(true_doc_ids_list, doc_ranking_list, k=100)
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
) -> dict:
    """Evaluate CLEF-IP passage retrieval: compute metrics and print results.

    *doc_ranking_list* is the Stage-1 document ranking from two-stage retrieval,
    used directly for document-level metrics.
    """
    true_labels_list = [qrels.get(qid, []) for qid in query_ids]
    results = _make_clefip_official_metrics(
        true_labels_list, predicted_labels_list,
        doc_ranking_list=doc_ranking_list,
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


def _colbert_encode(checkpoint, texts: list, is_query: bool = False, batch_size: int = 32) -> list:
    """Encode *texts* with a ColBERT checkpoint into per-token embeddings.

    Returns a list of numpy arrays, each shaped ``[n_tokens, dim]``.
    For queries the checkpoint pads with [MASK] to ``query_maxlen`` and returns
    a fixed-length matrix; for documents length varies with actual tokens.
    """
    import torch as _th
    all_embs: list[np.ndarray] = []
    _label = "queries" if is_query else "documents"
    for i in trange(0, len(texts), batch_size, desc=f"ColBERT encode {_label}"):
        batch = texts[i : i + batch_size]
        with _th.no_grad():
            if is_query:
                embs = checkpoint.queryFromText(batch)           # Tensor [B, query_maxlen, dim]
            else:
                embs = checkpoint.docFromText(batch, bsize=batch_size)  # tuple (Tensor,) or (Tensor, mask)

        # docFromText returns a tuple (D,) or (D, mask); queryFromText returns a Tensor.
        # Extract the embedding tensor and optional mask from tuples.
        mask_tensor = None
        if isinstance(embs, (list, tuple)):
            emb_tensor = embs[0]
            if len(embs) > 1 and isinstance(embs[1], _th.Tensor):
                mask_tensor = embs[1]
        else:
            emb_tensor = embs

        if isinstance(emb_tensor, _th.Tensor):
            embs_np = emb_tensor.cpu().float().numpy()
            if embs_np.ndim == 3:
                # [B, max_tok, dim] — split into individual per-text embeddings
                mask_np = (mask_tensor.cpu().bool().numpy()
                           if mask_tensor is not None and isinstance(mask_tensor, _th.Tensor) and mask_tensor.dim() == 2
                           else None)
                for j in range(embs_np.shape[0]):
                    doc_emb = embs_np[j]  # [max_tok, dim]
                    if not is_query:
                        # For documents, remove padding tokens (zero-norm rows)
                        # to get variable-length per-document embeddings.
                        if mask_np is not None:
                            doc_emb = doc_emb[mask_np[j]]
                        else:
                            norms = np.linalg.norm(doc_emb, axis=1)
                            valid_mask = norms > 1e-9
                            if valid_mask.any():
                                doc_emb = doc_emb[valid_mask]
                    all_embs.append(doc_emb)
            else:
                all_embs.append(embs_np)
        else:
            all_embs.append(np.asarray(emb_tensor, dtype=np.float32))
    return all_embs


def _colbert_maxsim_matrix(query_embs: list, doc_embs: list, batch_doc: int = 2048) -> np.ndarray:
    """Compute the MaxSim similarity matrix ``[n_queries, n_docs]``.

    For each query token, take the max cosine similarity with any document
    token, then sum over query tokens:

        score(q, d) = sum_i max_j (q_i · d_j)

    All embeddings are assumed **already** L2-normalised (ColBERT checkpoint
    normalises internally).

    Documents are padded and batched (size *batch_doc*) for vectorised torch
    computation on GPU/CPU.  Docs are padded ONCE per batch and reused for all
    queries.  Uses float16 on GPU for ~2x speedup via Tensor Cores.
    """
    import torch as _th
    _device = _th.device("cuda" if _th.cuda.is_available() else "cpu")
    _dtype = _th.float16 if _device.type == "cuda" else _th.float32
    n_q = len(query_embs)
    n_d = len(doc_embs)
    dim = query_embs[0].shape[1]
    sim = np.zeros((n_q, n_d), dtype=np.float32)

    # Pre-convert all queries to GPU once
    Q_tensors = [_th.from_numpy(query_embs[i]).to(_device, dtype=_dtype) for i in range(n_q)]

    # Outer loop: doc batches (pad once, reuse for all queries)
    for d_start in tqdm(range(0, n_d, batch_doc), desc="MaxSim scoring"):
        d_end = min(d_start + batch_doc, n_d)
        batch = [doc_embs[j] for j in range(d_start, d_end)]
        lengths = [d.shape[0] for d in batch]
        max_d_tok = max(lengths)
        B = len(batch)
        # Pad documents to same length → [B, max_d_tok, dim]
        D = _th.zeros(B, max_d_tok, dim, device=_device, dtype=_dtype)
        mask = _th.zeros(B, max_d_tok, device=_device, dtype=_th.bool)
        for i, d_np in enumerate(batch):
            L = d_np.shape[0]
            D[i, :L] = _th.from_numpy(d_np).to(_dtype)
            mask[i, :L] = True
        mask_expanded = ~mask.unsqueeze(1)  # [B, 1, max_d_tok] — precompute once
        # Inner loop: all queries against this doc batch
        for q_idx, q in enumerate(Q_tensors):
            # Batched MaxSim: q [q_tok, dim] × D [B, d_tok, dim]^T → [B, q_tok, d_tok]
            scores = _th.einsum("qd,bkd->bqk", q, D)
            # Mask padding positions to -inf so they never win the max
            scores.masked_fill_(mask_expanded, float("-inf"))
            # max over d_tok → [B, q_tok], sum over q_tok → [B]
            sim[q_idx, d_start:d_end] = scores.max(dim=2).values.sum(dim=1).cpu().float().numpy()
    return sim


def _colbert_maxsim_rankings_and_scores(
    query_embs: list, doc_embs: list, doc_ids: list
) -> tuple:
    """Compute MaxSim, derive per-query score dicts.

    Returns ``(predicted_labels_list, passage_scores_list)`` where
    ``predicted_labels_list`` is always ``None`` (ranking is derived from
    scores inside ``_clefip_two_stage_rerank``).
    """
    sim = _colbert_maxsim_matrix(query_embs, doc_embs)
    passage_scores_list = [
        {doc_ids[j]: float(sim[q_idx, j]) for j in range(sim.shape[1])}
        for q_idx in range(sim.shape[0])
    ]
    return None, passage_scores_list


# ── ColBERT sharded helpers (OOM-safe for large corpora) ─────────────────────

def _colbert_encode_passages_sharded(
    checkpoint, texts: list, cache_dir: str,
    shard_size: int = 50_000, batch_size: int = 32,
) -> list:
    """Encode passages in shards saved to disk to avoid OOM.

    Instead of accumulating all per-token embeddings in memory (~100 GB for 2M
    passages), this function flushes every *shard_size* passage embeddings to a
    pickle file on disk and clears memory.

    Returns a sorted list of shard file paths.  Each shard is a pickle file
    containing a ``list[np.ndarray]`` of per-passage token embeddings.
    """
    import torch as _th
    import pickle
    os.makedirs(cache_dir, exist_ok=True)
    shard_paths: list[str] = []
    current_shard: list[np.ndarray] = []
    shard_idx = 0

    for i in trange(0, len(texts), batch_size, desc="ColBERT encode passages (sharded)"):
        batch = texts[i : i + batch_size]
        with _th.no_grad():
            embs = checkpoint.docFromText(batch, bsize=batch_size)
        # docFromText returns a tuple (D,) or (D, mask); extract properly
        mask_tensor = None
        if isinstance(embs, (list, tuple)):
            emb_tensor = embs[0]
            if len(embs) > 1 and isinstance(embs[1], _th.Tensor):
                mask_tensor = embs[1]
        else:
            emb_tensor = embs
        if isinstance(emb_tensor, _th.Tensor):
            embs_np = emb_tensor.cpu().float().numpy()
            if embs_np.ndim == 3:
                mask_np = (mask_tensor.cpu().bool().numpy()
                           if mask_tensor is not None and isinstance(mask_tensor, _th.Tensor) and mask_tensor.dim() == 2
                           else None)
                for j in range(embs_np.shape[0]):
                    doc_emb = embs_np[j]
                    if mask_np is not None:
                        doc_emb = doc_emb[mask_np[j]]
                    else:
                        norms = np.linalg.norm(doc_emb, axis=1)
                        valid_mask = norms > 1e-9
                        if valid_mask.any():
                            doc_emb = doc_emb[valid_mask]
                    current_shard.append(doc_emb)
            else:
                current_shard.append(embs_np)
            del embs_np
        else:
            current_shard.append(np.asarray(emb_tensor, dtype=np.float32))
        # Free GPU tensor immediately to prevent memory buildup
        del embs
        if i % (batch_size * 100) == 0:
            if _th.cuda.is_available():
                _th.cuda.empty_cache()
        # Flush shard to disk when full
        if len(current_shard) >= shard_size:
            shard_path = os.path.join(cache_dir, f"passage_shard_{shard_idx:04d}.pkl")
            with open(shard_path, "wb") as f:
                pickle.dump(current_shard, f, protocol=4)
            _mb = os.path.getsize(shard_path) / 1024**2
            print(f"  💾 Shard {shard_idx}: {len(current_shard)} passages ({_mb:.0f} MB) → {shard_path}")
            shard_paths.append(shard_path)
            shard_idx += 1
            current_shard = []
            gc.collect()
            if _th.cuda.is_available():
                _th.cuda.empty_cache()

    # Flush remaining passages
    if current_shard:
        shard_path = os.path.join(cache_dir, f"passage_shard_{shard_idx:04d}.pkl")
        with open(shard_path, "wb") as f:
            pickle.dump(current_shard, f, protocol=4)
        _mb = os.path.getsize(shard_path) / 1024**2
        print(f"  💾 Shard {shard_idx}: {len(current_shard)} passages ({_mb:.0f} MB) → {shard_path}")
        shard_paths.append(shard_path)

    return shard_paths


def _colbert_maxsim_matrix_sharded(
    query_embs: list, shard_paths: list, n_passages: int, batch_doc: int = 2048,
) -> np.ndarray:
    """Compute MaxSim similarity by streaming passage shards from disk.

    Only one shard (~2-3 GB) is loaded at a time.  The similarity matrix
    ``[n_queries, n_passages]`` (e.g. 48 × 2M = 384 MB) is accumulated
    incrementally and stays in memory throughout.

    Optimised: documents are padded once per batch and reused for all queries.
    Uses float16 on GPU for ~2x speedup via Tensor Cores.
    """
    import torch as _th
    import pickle
    _device = _th.device("cuda" if _th.cuda.is_available() else "cpu")
    _dtype = _th.float16 if _device.type == "cuda" else _th.float32
    n_q = len(query_embs)
    dim = query_embs[0].shape[1]
    sim = np.zeros((n_q, n_passages), dtype=np.float32)

    # Pre-convert all queries to GPU once
    Q_tensors = [_th.from_numpy(query_embs[i]).to(_device, dtype=_dtype) for i in range(n_q)]

    col_offset = 0
    for s_idx, shard_path in enumerate(shard_paths):
        print(f"  MaxSim: shard {s_idx + 1}/{len(shard_paths)} — loading {shard_path} ...", flush=True)
        with open(shard_path, "rb") as f:
            shard_embs = pickle.load(f)
        n_shard = len(shard_embs)

        # Outer loop: doc batches (pad once, reuse for all queries)
        _iter = range(0, n_shard, batch_doc)
        if s_idx == 0:
            _iter = tqdm(_iter, desc=f"MaxSim shard {s_idx + 1}/{len(shard_paths)}")
        for d_start in _iter:
            d_end = min(d_start + batch_doc, n_shard)
            batch = shard_embs[d_start:d_end]
            lengths = [d.shape[0] for d in batch]
            max_d_tok = max(lengths)
            B = len(batch)
            D = _th.zeros(B, max_d_tok, dim, device=_device, dtype=_dtype)
            mask = _th.zeros(B, max_d_tok, device=_device, dtype=_th.bool)
            for i, d_np in enumerate(batch):
                L = d_np.shape[0]
                D[i, :L] = _th.from_numpy(d_np).to(_dtype)
                mask[i, :L] = True
            mask_expanded = ~mask.unsqueeze(1)
            # Inner loop: all queries against this doc batch
            for q_idx, q in enumerate(Q_tensors):
                scores = _th.einsum("qd,bkd->bqk", q, D)
                scores.masked_fill_(mask_expanded, float("-inf"))
                sim[q_idx, col_offset + d_start:col_offset + d_end] = (
                    scores.max(dim=2).values.sum(dim=1).cpu().float().numpy()
                )

        col_offset += n_shard
        del shard_embs
        gc.collect()
        if _th.cuda.is_available():
            _th.cuda.empty_cache()

    assert col_offset == n_passages, f"Shard total {col_offset} != n_passages {n_passages}"
    return sim


def _colbert_maxsim_rankings_and_scores_sharded(
    query_embs: list, shard_paths: list, n_passages: int, doc_ids: list,
    batch_doc: int = 512,
) -> tuple:
    """Sharded MaxSim: stream passage shards from disk, derive score dicts.

    Returns ``(predicted_labels_list, passage_scores_list)`` where
    ``predicted_labels_list`` is always ``None`` (ranking is derived from
    scores inside ``_clefip_two_stage_rerank``).
    """
    sim = _colbert_maxsim_matrix_sharded(query_embs, shard_paths, n_passages, batch_doc)
    passage_scores_list = [
        {doc_ids[j]: float(sim[q_idx, j]) for j in range(sim.shape[1])}
        for q_idx in range(sim.shape[0])
    ]
    return None, passage_scores_list


def _run_clefip_eval_full_corpus(
    args,
    query_ids: list,
    query_texts: list,
    passage_ids: list,
    corpus_jsonl_path: str,
    ids_txt_path: str,
    qrels_passage_ids: dict,
):
    """Run CLEF-IP retrieval over the full 01 passage corpus (streaming where possible)."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = (args.model_name or "").lower() if hasattr(args.model_name, "lower") else str(args.model_name or "").lower()

    topk_docs = getattr(args, "clefip_two_stage_topk_docs", 100)

    if model_name == "bm25":
        # Load all passage texts from JSONL (same order as passage_ids); may be heavy for huge corpora
        print("Loading passage texts from corpus for BM25...", flush=True)
        passage_texts = []
        with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                passage_texts.append(rec["text"])
        assert len(passage_texts) == len(passage_ids)
        import bm25s
        import snowballstemmer
        stemmer = snowballstemmer.stemmer("english")
        passage_tokens = bm25s.tokenize(passage_texts, stopwords="en", stemmer=stemmer)
        query_tokens = bm25s.tokenize(query_texts, stopwords="en", stemmer=stemmer)
        retriever = bm25s.BM25()
        retriever.index(passage_tokens)
        _report_bm25_posting_stats(retriever, "CLEF-IP passage (full corpus)", query_tokens_list=query_tokens)
        k = 100
        clefip_results, clefip_scores = retriever.retrieve(query_tokens, k=len(passage_ids))
        # Build per-query score dicts from BM25 retrieval results
        passage_scores_list = []
        for q_idx in range(len(clefip_results)):
            scores = {passage_ids[int(clefip_results[q_idx][j])]: float(clefip_scores[q_idx][j])
                      for j in range(len(clefip_results[q_idx]))}
            passage_scores_list.append(scores)
        predicted_labels_list, doc_ranking_list = _clefip_two_stage_rerank(
            passage_ids, passage_scores_list, topk_docs=topk_docs,
        )
        print(f"  🔄 Two-stage retrieval: top-{topk_docs} docs → re-ranked passages per query")
        _evaluate_and_print_clefip(
            qrels_passage_ids, query_ids, predicted_labels_list,
            "BM25 passage retrieval",
            two_stage=True, topk_docs=topk_docs, header_extra=" (full 01 corpus)",
            doc_ranking_list=doc_ranking_list,
        )
        return

    if model_name in ["naver/splade-v2", "splade-v2", "naver/splade_v2_max", "naver/splade_v2_distil"]:
        splade_model_map = {
            "splade-v2": "naver/splade-cocondenser-ensembledistil",
            "naver/splade-v2": "naver/splade-cocondenser-ensembledistil",
            "naver/splade_v2_max": "naver/splade_v2_max",
            "naver/splade_v2_distil": "naver/splade_v2_distil",
        }
        actual_model_name = splade_model_map.get(model_name, model_name)

        # ---- SPLADE sparse embedding cache ----
        from scipy.sparse import save_npz as _sp_save_npz, load_npz as _sp_load_npz
        _splade_clean = (args.model_name or "").rstrip("/").replace("/", "_").strip("_")
        _sample_sz = getattr(args, "clefip_sample_size", 0) or 0
        _splade_qmc = getattr(args, "query_max_chunks", -1)
        _splade_use_chunk = _splade_qmc != 0
        # Query cache key includes chunking setting; passage cache is always the same.
        _qchunk_tag = "" if not _splade_use_chunk else f"_qchunk{'_unlim' if _splade_qmc < 0 else _splade_qmc}"
        _splade_cache_dir = os.path.join("temp", "clefip_splade_cache", f"{_splade_clean}_s{_sample_sz}")
        _splade_q_path = os.path.join(_splade_cache_dir, f"query_sparse{_qchunk_tag}.npz")
        _splade_p_path = os.path.join(_splade_cache_dir, "passage_sparse.npz")
        _splade_meta_path = os.path.join(_splade_cache_dir, f"meta{_qchunk_tag}.json")

        _splade_cache_hit = False
        _splade_query_hash = _hash_query_texts(query_texts)
        Q = None
        D = None
        if os.path.isfile(_splade_p_path) and os.path.isfile(_splade_meta_path):
            try:
                with open(_splade_meta_path, "r") as _mf:
                    _splade_meta = json.load(_mf)
                if (_splade_meta.get("n_passages") == len(passage_ids)
                        and _splade_meta.get("model") == actual_model_name):
                    D = _sp_load_npz(_splade_p_path)
                    _q_hash_ok = _splade_meta.get("query_text_hash") == _splade_query_hash
                    if (os.path.isfile(_splade_q_path)
                            and _splade_meta.get("n_queries") == len(query_ids)
                            and _q_hash_ok):
                        Q = _sp_load_npz(_splade_q_path)
                        _splade_cache_hit = True
                        print(f"✅ Loaded SPLADE cache from {_splade_cache_dir}")
                        print(f"   queries: {Q.shape}, passages: {D.shape}")
                    else:
                        if not _q_hash_ok:
                            print(f"⚠️  SPLADE query text hash mismatch — re-encoding queries (passages reused).")
                        else:
                            print(f"⚠️  SPLADE query cache missing — re-encoding queries (passages reused).")
                        print(f"✅ Loaded SPLADE passage cache: {D.shape}")
                else:
                    print(f"⚠️  SPLADE cache metadata mismatch, re-encoding...")
            except Exception as e:
                print(f"⚠️  SPLADE cache load failed ({e}), re-encoding...")
                Q = None
                D = None

        if not _splade_cache_hit:
            from sentence_transformers import SparseEncoder
            splade_model = SparseEncoder(actual_model_name)
            encode_bs = _auto_batch_size(device, hidden_size=768)

            if Q is None:
                if _splade_use_chunk:
                    print(f"  Encoding {len(query_texts)} queries (chunk+term-max, max_chunks={_splade_qmc})...", flush=True)
                    Q = _splade_chunk_merge_query(query_texts, splade_model, encode_bs, query_max_chunks=_splade_qmc)
                else:
                    print(f"  Encoding {len(query_texts)} queries...", flush=True)
                    query_sparse = splade_model.encode_query(query_texts, batch_size=encode_bs, show_progress_bar=True)
                    Q = _sparse_to_csr(query_sparse)

            if D is None:
                print(f"Loading passage texts from corpus for SPLADE...", flush=True)
                passage_texts = []
                with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        passage_texts.append(rec["text"])
                assert len(passage_texts) == len(passage_ids)
                print(f"  Encoding {len(passage_texts)} passages...", flush=True)
                passage_sparse = splade_model.encode_document(passage_texts, batch_size=encode_bs, show_progress_bar=True)
                D = _sparse_to_csr(passage_sparse)
                del passage_texts

            # Save cache (always rewrite both meta and query; passage is rewritten only if newly encoded)
            os.makedirs(_splade_cache_dir, exist_ok=True)
            _sp_save_npz(_splade_q_path, Q)
            if not os.path.isfile(_splade_p_path):
                _sp_save_npz(_splade_p_path, D)
            with open(_splade_meta_path, "w") as _mf:
                json.dump({
                    "model": actual_model_name,
                    "n_queries": len(query_ids),
                    "n_passages": len(passage_ids),
                    "vocab_size": int(D.shape[1]),
                    "q_nnz": int(Q.nnz),
                    "p_nnz": int(D.nnz),
                    "query_max_chunks": _splade_qmc,
                    "query_text_hash": _splade_query_hash,
                }, _mf, indent=2)
            _q_mb = os.path.getsize(_splade_q_path) / 1024**2
            _p_mb = os.path.getsize(_splade_p_path) / 1024**2
            print(f"💾 Saved SPLADE cache to {_splade_cache_dir}")
            print(f"   queries: {_q_mb:.1f} MB ({Q.nnz:,} nnz), passages: {_p_mb:.1f} MB ({D.nnz:,} nnz)")
            del splade_model
        vocab_size = D.shape[1]
        posting_lists, _ = _splade_build_inverted_index(D, vocab_size)
        _report_splade_flops_and_postings(posting_lists, Q, "CLEF-IP passage (SPLADE)")
        k = min(100, len(passage_ids))
        top_k_list, idx_scores_list = _splade_retrieve_with_index(
            Q, posting_lists, top_k=len(passage_ids), return_scores=True,
        )
        # Convert index-keyed score dicts to passage_id-keyed score dicts
        passage_scores_list = [
            {passage_ids[d]: s for d, s in sd.items()} for sd in idx_scores_list
        ]
        predicted_labels_list, doc_ranking_list = _clefip_two_stage_rerank(
            passage_ids, passage_scores_list, topk_docs=topk_docs,
        )
        print(f"  \U0001f504 Two-stage retrieval: top-{topk_docs} docs \u2192 re-ranked passages per query")
        _evaluate_and_print_clefip(
            qrels_passage_ids, query_ids, predicted_labels_list,
            "SPLADE passage retrieval",
            two_stage=True, topk_docs=topk_docs, header_extra=" (full 01 corpus)",
            doc_ranking_list=doc_ranking_list,
        )
        return

    if model_name in _COLBERT_MODEL_NAMES:
        # ── ColBERT CLEF-IP passage retrieval ──
        # Default: PLAID end-to-end retrieval (ColBERTv2 centroid index + residual compression).
        # Fallback: brute-force sharded encoding (--colbert_mode bruteforce; requires >200GB).
        import pickle
        try:
            from colbert.modeling.checkpoint import Checkpoint  # noqa: F811
            from colbert.infra import ColBERTConfig as _ColBERTConfig  # noqa: F811
        except ImportError:
            print("❌  colbert-ai package not installed. Install with:  pip install colbert-ai")
            return

        _model_clean = (args.model_name or "").rstrip("/").replace("/", "_").strip("_")
        _sample_sz = getattr(args, "clefip_sample_size", 0) or 0
        _colbert_mode = getattr(args, "colbert_mode", "plaid")

        # ── Path A: PLAID end-to-end retrieval (default) ──
        if _colbert_mode == "plaid":
            try:
                from colbert.infra import Run, RunConfig
                from colbert import Indexer as _ColBERTIndexer, Searcher as _ColBERTSearcher
            except ImportError:
                print("❌  colbert-ai PLAID components not available. ")
                print("   Install with:  pip install colbert-ai[torch,faiss-gpu]")
                print("   Or use --colbert_mode bruteforce as fallback.")
                return

            _plaid_root = os.path.join("temp", "clefip_colbert_plaid")
            _experiment = "clefip"
            _ncells = getattr(args, "colbert_ncells", 0) or 0
            _kmeans = getattr(args, "colbert_kmeans_niters", 4)
            _nc_str = f"_nc{_ncells}" if _ncells > 0 else ""
            _ki_str = f"_ki{_kmeans}" if _kmeans != 20 else ""
            _index_name = f"{_model_clean}_s{_sample_sz}_nbits2{_nc_str}{_ki_str}"

            # Step 1: Prepare collection TSV (pid\ttext) if not exists
            _collection_tsv = os.path.join(_plaid_root, f"collection_s{_sample_sz}.tsv")
            if not os.path.isfile(_collection_tsv):
                print("Preparing collection TSV for PLAID indexing...")
                os.makedirs(os.path.dirname(_collection_tsv), exist_ok=True)
                with open(corpus_jsonl_path, "r", encoding="utf-8") as f_in, \
                     open(_collection_tsv, "w", encoding="utf-8") as f_out:
                    for idx, line in enumerate(f_in):
                        rec = json.loads(line)
                        text = rec["text"].replace("\t", " ").replace("\n", " ")
                        f_out.write(f"{idx}\t{text}\n")
                print(f"  ✅ Collection TSV: {_collection_tsv} ({len(passage_ids):,} passages)")
            else:
                print(f"  ✅ Collection TSV exists: {_collection_tsv}")

            # Step 2: Build PLAID index (cached; requires GPU for first build)
            _index_root = os.path.join(_plaid_root, _experiment, "indexes")
            _index_path = os.path.join(_index_root, _index_name)
            if not os.path.isdir(_index_path) or not os.path.isfile(os.path.join(_index_path, "metadata.json")):
                print(f"\n🔨 Building PLAID index: {_index_name} ...")
                print(f"   This encodes all {len(passage_ids):,} passages with ColBERTv2 + residual compression (nbits=2).")
                print(f"   Index will be cached at: {_index_path}")
                with Run().context(RunConfig(nranks=1, experiment=_experiment)):
                    _q_maxlen = getattr(args, "colbert_query_maxlen", 512)
                    _plaid_kwargs = dict(
                        nbits=2,
                        doc_maxlen=512,
                        query_maxlen=_q_maxlen,
                        root=_plaid_root,
                        kmeans_niters=_kmeans,
                    )
                    if _ncells > 0:
                        _plaid_kwargs["ncells"] = _ncells
                    _plaid_config = _ColBERTConfig(**_plaid_kwargs)
                    print(f"   PLAID config: ncells={'auto' if _ncells == 0 else _ncells}, kmeans_niters={_kmeans}")
                    _indexer = _ColBERTIndexer(
                        checkpoint="colbert-ir/colbertv2.0",
                        config=_plaid_config,
                    )
                    _indexer.index(
                        name=_index_name,
                        collection=_collection_tsv,
                        overwrite="reuse",
                    )
                print(f"  ✅ PLAID index built: {_index_path}")
            else:
                print(f"  ✅ PLAID index exists: {_index_path}")

            # Step 3: Search with PLAID engine (with optional chunk+max-sim for long queries)
            _plaid_topk = getattr(args, "colbert_plaid_topk", 1000)
            _plaid_topk = min(_plaid_topk, len(passage_ids))
            _cb_qmc = getattr(args, "query_max_chunks", -1)
            _cb_use_chunk = _cb_qmc != 0
            _cb_cap_str = f"cap={_cb_qmc}" if _cb_qmc > 0 else "unlimited"
            if _cb_use_chunk:
                print(f"\n🔍 Searching with ColBERTv2 PLAID engine (chunk+max-sim, {_cb_cap_str}, top-{_plaid_topk})...")
            else:
                print(f"\n🔍 Searching with ColBERTv2 PLAID engine (top-{_plaid_topk} per query)...")
            with Run().context(RunConfig(nranks=1, experiment=_experiment)):
                _search_config = _ColBERTConfig(
                    root=_plaid_root,
                    query_maxlen=getattr(args, "colbert_query_maxlen", 512),
                )
                _searcher = _ColBERTSearcher(
                    index=_index_name,
                    config=_search_config,
                    collection=_collection_tsv,
                )

                passage_scores_list = []
                for q_idx, (qid, qtext) in enumerate(zip(
                    tqdm(query_ids, desc="ColBERT PLAID search"), query_texts
                )):
                    if _cb_use_chunk:
                        # Split query into ≤512-token chunks; search each; merge by max score.
                        chunks = _dense_chunk_text(qtext, None, max_tokens=512)
                        if _cb_qmc > 0 and len(chunks) > _cb_qmc:
                            chunks = chunks[:_cb_qmc]
                        merged: dict[str, float] = {}
                        for chunk in chunks:
                            pids_result, _ranks, scores_result = _searcher.search(chunk, k=_plaid_topk)
                            for pid_int, score in zip(pids_result, scores_result):
                                if 0 <= pid_int < len(passage_ids):
                                    pid_str = passage_ids[pid_int]
                                    if score > merged.get(pid_str, -1e9):
                                        merged[pid_str] = float(score)
                        passage_scores_list.append(merged)
                    else:
                        pids_result, _ranks, scores_result = _searcher.search(qtext, k=_plaid_topk)
                        score_dict = {}
                        for pid_int, score in zip(pids_result, scores_result):
                            if 0 <= pid_int < len(passage_ids):
                                score_dict[passage_ids[pid_int]] = float(score)
                        passage_scores_list.append(score_dict)

            print(f"  ✅ PLAID search complete: {len(query_ids)} queries, top-{_plaid_topk} per query")

        elif _colbert_mode == "bruteforce":
            # ── Path B: Brute-force sharded encoding ──
            print("ColBERT brute-force mode: encoding ALL passages (sharded to disk).")
            print("   Peak RAM ~8 GB (one shard loaded at a time). Disk usage ~180 GB for 3.7M passages.")
            _colbert_config = _ColBERTConfig(doc_maxlen=512, query_maxlen=getattr(args, "colbert_query_maxlen", 512), nbits=2)
            _cache_dir = os.path.join("temp", "clefip_colbert_cache", f"{_model_clean}_s{_sample_sz}")
            _cb_qmc_bf = getattr(args, "query_max_chunks", -1)
            _cb_use_chunk_bf = _cb_qmc_bf != 0
            _cb_cap_str_bf = f"cap={_cb_qmc_bf}" if _cb_qmc_bf > 0 else "unlimited"
            # Chunk queries: build flat list (chunk_text → original query index)
            if _cb_use_chunk_bf:
                _flat_chunk_texts: list[str] = []
                _flat_chunk_q_idx: list[int] = []
                for _qi, _qt in enumerate(query_texts):
                    _cks = _dense_chunk_text(_qt, None, max_tokens=512)
                    if _cb_qmc_bf > 0 and len(_cks) > _cb_qmc_bf:
                        _cks = _cks[:_cb_qmc_bf]
                    for _ck in _cks:
                        _flat_chunk_texts.append(_ck)
                        _flat_chunk_q_idx.append(_qi)
                _n_chunked_bf = sum(1 for _qi in range(len(query_texts))
                                    if sum(1 for x in _flat_chunk_q_idx if x == _qi) > 1)
                print(f"   ColBERT bruteforce chunk+max-sim: {_n_chunked_bf}/{len(query_texts)} queries split "
                      f"→ {len(_flat_chunk_texts)} chunks ({_cb_cap_str_bf})")
                _cache_q = os.path.join(_cache_dir, "query_embs_chunk.pkl")
            else:
                _flat_chunk_texts = query_texts
                _flat_chunk_q_idx = list(range(len(query_texts)))
                _cache_q = os.path.join(_cache_dir, "query_embs.pkl")
            _cache_meta = os.path.join(_cache_dir, "meta.json")

            _cache_hit = False
            _shard_paths: list[str] = []
            _cb_query_hash = _hash_query_texts(_flat_chunk_texts)
            query_embs = None
            passage_embs = None  # populated only when legacy single-file cache is loaded

            if os.path.isfile(_cache_meta):
                try:
                    with open(_cache_meta, "r") as _mf:
                        _meta = json.load(_mf)
                    _n_q_expected = len(_flat_chunk_texts) if _cb_use_chunk_bf else len(query_ids)
                    _passage_ok = (_meta.get("n_passages") == len(passage_ids)
                                   and _meta.get("model") == "colbert-ir/colbertv2.0")
                    if _passage_ok:
                        import glob as _glob
                        _shard_candidates = sorted(_glob.glob(os.path.join(_cache_dir, "passage_shard_*.pkl")))
                        _passage_hit = False
                        if _shard_candidates and _meta.get("format") == "sharded":
                            _shard_paths = _shard_candidates
                            _passage_hit = True
                        elif os.path.isfile(os.path.join(_cache_dir, "passage_embs.pkl")):
                            print("⚠️  Legacy single-file ColBERT cache found. Loading...")
                            with open(os.path.join(_cache_dir, "passage_embs.pkl"), "rb") as _f:
                                passage_embs = pickle.load(_f)
                            _passage_hit = True
                        if _passage_hit:
                            _q_hash_ok = _meta.get("query_text_hash") == _cb_query_hash
                            if (os.path.isfile(_cache_q)
                                    and _meta.get("n_queries") == _n_q_expected
                                    and _q_hash_ok):
                                with open(_cache_q, "rb") as _f:
                                    query_embs = pickle.load(_f)
                                _cache_hit = True
                                print(f"✅ Loaded ColBERT cache from {_cache_dir}")
                                print(f"   queries: {len(query_embs)}, passage shards: {len(_shard_paths) or 1}")
                            else:
                                if not _q_hash_ok:
                                    print(f"⚠️  ColBERT query text hash mismatch — re-encoding queries (passages reused).")
                                else:
                                    print(f"⚠️  ColBERT query cache missing — re-encoding queries (passages reused).")
                except Exception as e:
                    print(f"⚠️  ColBERT cache load failed ({e}), re-encoding...")

            if not _cache_hit:
                _colbert_ckpt = Checkpoint("colbert-ir/colbertv2.0", colbert_config=_colbert_config)
                if query_embs is None:
                    _encode_label = "query chunks" if _cb_use_chunk_bf else "queries"
                    print(f"  Encoding {len(_flat_chunk_texts)} {_encode_label} ...", flush=True)
                    query_embs = _colbert_encode(_colbert_ckpt, _flat_chunk_texts, is_query=True, batch_size=32)
                if not _shard_paths and passage_embs is None:
                    print(f"  Encoding {len(passage_ids)} passages (sharded)...", flush=True)
                    passage_texts_all = []
                    with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            rec = json.loads(line)
                            passage_texts_all.append(rec["text"])
                    assert len(passage_texts_all) == len(passage_ids)
                    _shard_paths = _colbert_encode_passages_sharded(
                        _colbert_ckpt, passage_texts_all, _cache_dir,
                        shard_size=50_000, batch_size=32,
                    )
                    del passage_texts_all
                del _colbert_ckpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                os.makedirs(_cache_dir, exist_ok=True)
                with open(_cache_q, "wb") as _f:
                    pickle.dump(query_embs, _f, protocol=4)
                with open(_cache_meta, "w") as _mf:
                    json.dump({
                        "model": "colbert-ir/colbertv2.0",
                        "n_queries": len(_flat_chunk_texts),
                        "n_passages": len(passage_ids),
                        "sample_size": _sample_sz,
                        "format": "sharded" if _shard_paths else "legacy",
                        "n_shards": len(_shard_paths),
                        "shard_size": 50_000,
                        "query_text_hash": _cb_query_hash,
                    }, _mf, indent=2)
                print(f"💾 Saved ColBERT cache to {_cache_dir}")

            print("  Computing MaxSim similarity matrix...", flush=True)
            if _shard_paths:
                _, chunk_scores_list = _colbert_maxsim_rankings_and_scores_sharded(
                    query_embs, _shard_paths, len(passage_ids), passage_ids,
                )
            else:
                _, chunk_scores_list = _colbert_maxsim_rankings_and_scores(
                    query_embs, passage_embs, passage_ids,
                )

            if _cb_use_chunk_bf:
                # Aggregate chunk-level scores into per-original-query scores by max.
                _flat_q_arr_bf = np.array(_flat_chunk_q_idx, dtype=np.int32)
                passage_scores_list = []
                for _qi in range(len(query_texts)):
                    _chunk_rows = np.where(_flat_q_arr_bf == _qi)[0]
                    merged: dict[str, float] = {}
                    for _ci in _chunk_rows:
                        for pid_str, sc in chunk_scores_list[int(_ci)].items():
                            if sc > merged.get(pid_str, -1e9):
                                merged[pid_str] = sc
                    passage_scores_list.append(merged)
            else:
                passage_scores_list = chunk_scores_list

        predicted_labels_list, doc_ranking_list = _clefip_two_stage_rerank(
            passage_ids, passage_scores_list, topk_docs=topk_docs,
        )
        print(f"  🔄 Two-stage retrieval: top-{topk_docs} docs → re-ranked passages per query")
        _header_extra = f" (PLAID)" if _colbert_mode == "plaid" else " (full 01 corpus, brute-force)"
        _evaluate_and_print_clefip(
            qrels_passage_ids, query_ids, predicted_labels_list,
            "ColBERT passage retrieval",
            two_stage=True, topk_docs=topk_docs, header_extra=_header_extra,
            doc_ranking_list=doc_ranking_list,
        )
        return

    if model_name == "sparse_coverage":
        # sparse_coverage CLEF-IP is handled by the `clefip_passage` mode in main();
        # this standalone path was removed (it was a stale duplicate without PCA/angle/residual support).
        print("sparse_coverage CLEF-IP: skipped here — handled by clefip_passage mode in main().")
        return

    # Dense model: encode in batches, build FAISS index
    _model_clean = (args.model_name or "").rstrip("/").replace("/", "_").strip("_")
    _sample_sz = getattr(args, "clefip_sample_size", 0) or 0
    _dense_cache_dir = os.path.join("temp", "clefip_dense_cache", f"{_model_clean}_s{_sample_sz}")
    _dense_q_path = os.path.join(_dense_cache_dir, "query_embeddings.npy")
    _dense_p_path = os.path.join(_dense_cache_dir, "passage_embeddings.npy")
    _dense_meta_path = os.path.join(_dense_cache_dir, "meta.json")

    query_texts_fmt, _ = _clefip_format_for_model(query_texts, passage_ids[:1], [""], args.model_name)
    encode_fn, model_label, _dense_tokenizer = _get_clefip_dense_encoder(args, model_name, device)
    _query_max_chunks = getattr(args, "query_max_chunks", -1)
    _use_chunk_merge = _query_max_chunks != 0  # -1 (default) or >0 = chunking enabled; 0 = truncate (legacy)

    # Passage embeddings are cached independently of query chunking strategy.
    # Query embeddings are only cached in the non-chunk path (chunk merge produces variable-size arrays).
    _cache_hit = False
    passage_emb = None
    query_emb = None

    _query_hash = _hash_query_texts(query_texts_fmt)

    if os.path.isfile(_dense_p_path) and os.path.isfile(_dense_meta_path):
        try:
            with open(_dense_meta_path, "r") as f:
                _meta = json.load(f)
            _meta_ok = (_meta.get("model_name") == args.model_name
                        and _meta.get("n_passages") == len(passage_ids))
            if _meta_ok:
                passage_emb = np.load(_dense_p_path)
                assert passage_emb.shape[0] == len(passage_ids)
                _q_hash_ok = _meta.get("query_text_hash") == _query_hash
                if (not _use_chunk_merge
                        and os.path.isfile(_dense_q_path)
                        and _meta.get("n_queries") == len(query_ids)
                        and _q_hash_ok):
                    query_emb = np.load(_dense_q_path)
                    assert query_emb.shape[0] == len(query_ids)
                    print(f"✅ Loaded CLEF-IP dense embeddings from cache: {_dense_cache_dir}")
                    print(f"   queries: {query_emb.shape}, passages: {passage_emb.shape}")
                else:
                    if not _q_hash_ok and not _use_chunk_merge:
                        print(f"⚠️  Query text hash mismatch — re-encoding queries (passages reused).")
                    print(f"✅ Loaded CLEF-IP passage embeddings from cache: {passage_emb.shape}")
                _cache_hit = True
        except Exception as e:
            print(f"⚠️  CLEF-IP dense cache load failed ({e}), re-encoding...")
            passage_emb = None
            _cache_hit = False

    if not _cache_hit:
        _role_models = ["datalyes/patembed-large", "patembed-large"]
        if not _use_chunk_merge:
            query_emb = encode_fn(query_texts_fmt, role="query") if model_name in _role_models else encode_fn(query_texts_fmt, batch_size=32)
        batch_size = 256
        passage_emb_list = []
        with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
            batch_pids, batch_texts = [], []
            for line in tqdm(f, desc="  corpus", leave=False):
                rec = json.loads(line)
                batch_pids.append(rec["pid"])
                batch_texts.append(rec["text"])
                if len(batch_pids) >= batch_size:
                    _, batch_fmt = _clefip_format_for_model([""], batch_pids, batch_texts, args.model_name)
                    if model_name in _role_models:
                        passage_emb_list.append(encode_fn(batch_fmt, role="document"))
                    else:
                        passage_emb_list.append(encode_fn(batch_fmt, batch_size=32))
                    batch_pids, batch_texts = [], []
            if batch_pids:
                _, batch_fmt = _clefip_format_for_model([""], batch_pids, batch_texts, args.model_name)
                if model_name in _role_models:
                    passage_emb_list.append(encode_fn(batch_fmt, role="document"))
                else:
                    passage_emb_list.append(encode_fn(batch_fmt, batch_size=32))
        passage_emb = np.vstack(passage_emb_list) if passage_emb_list else np.zeros((0, 0), dtype=np.float32)
        del passage_emb_list  # free memory before saving

        # Save to cache
        os.makedirs(_dense_cache_dir, exist_ok=True)
        np.save(_dense_p_path, passage_emb)
        if not _use_chunk_merge:
            np.save(_dense_q_path, query_emb)
        with open(_dense_meta_path, "w") as f:
            json.dump({
                "model_name": args.model_name,
                "n_queries": len(query_ids),
                "n_passages": len(passage_ids),
                "dim": int(passage_emb.shape[1]),
                "sample_size": _sample_sz,
                "query_text_hash": _query_hash,
            }, f, indent=2)
        print(f"💾 Saved CLEF-IP dense embeddings to {_dense_cache_dir}")
        print(f"   passages: {passage_emb.shape} ({os.path.getsize(_dense_p_path) / 1024**2:.0f} MB)")
        if not _use_chunk_merge:
            print(f"   queries: {query_emb.shape} ({os.path.getsize(_dense_q_path) / 1024**2:.0f} MB)")
    elif not _use_chunk_merge and query_emb is None:
        # Cache hit on passages but query_text_hash mismatched: re-encode queries only.
        _role_models = ["datalyes/patembed-large", "patembed-large"]
        query_emb = (encode_fn(query_texts_fmt, role="query")
                     if model_name in _role_models
                     else encode_fn(query_texts_fmt, batch_size=32))
        np.save(_dense_q_path, query_emb)
        with open(_dense_meta_path, "w") as f:
            json.dump({
                "model_name": args.model_name,
                "n_queries": len(query_ids),
                "n_passages": len(passage_ids),
                "dim": int(passage_emb.shape[1]),
                "sample_size": _sample_sz,
                "query_text_hash": _query_hash,
            }, f, indent=2)
        print(f"💾 Updated query embeddings cache: {query_emb.shape}")

    if _use_chunk_merge:
        # Chunk + max-sim merge: each query split into ≤512-token chunks, each chunk encoded
        # independently, score = max cosine similarity across all chunks.
        print(f"\n🔗 Dense Chunk+MaxSim (query_max_chunks={_query_max_chunks})")
        _role_models = ["datalyes/patembed-large", "patembed-large"]
        # Wrap encode_fn so chunks are encoded with the correct role (query vs document).
        if model_name in _role_models:
            def _q_encode_fn(texts, batch_size=32):
                return encode_fn(texts, batch_size=batch_size, role="query")
        else:
            _q_encode_fn = encode_fn
        passage_scores_list = _dense_chunk_maxsim_scores(
            query_texts_fmt, passage_ids, passage_emb, _q_encode_fn, _dense_tokenizer,
            query_max_chunks=_query_max_chunks, batch_size=32,
        )
        predicted_labels_list, doc_ranking_list = _clefip_two_stage_rerank(
            passage_ids, passage_scores_list, topk_docs=topk_docs,
        )
        print(f"  🔄 Two-stage retrieval: top-{topk_docs} docs → re-ranked passages per query")
        _evaluate_and_print_clefip(
            qrels_passage_ids, query_ids, predicted_labels_list,
            "Dense Chunk+MaxSim passage retrieval",
            two_stage=True, topk_docs=topk_docs,
            doc_ranking_list=doc_ranking_list,
        )
    else:
        clefip_passage_evaluation(query_ids, passage_ids, query_emb, passage_emb, qrels_passage_ids,
                                  k=100, model_label=model_label + " (full 01)", topk_docs=topk_docs)


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


def main():
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
                       choices=["spacy_token", "encoder_token", "spacy_sentence", "noun_chunk"],
                       help="Tokenization unit used to build embeddings (for sparse_coverage). "
                            "Example: spacy_token, encoder_token")
    # Eval assumes centers were built with CLS included; build script controls that. We look for 'cls' suffix only.
    parser.add_argument("--exclude_cls_spans", action="store_true", default=False,
                       help="Exclude spans whose text is literally 'cls' or '[CLS]' from index and query (for sparse_coverage). "
                            "Use to compare best config with vs without CLS spans. Document posting lists and query encoding skip these spans.")
    parser.add_argument("--layer", type=str, default="last",
                       choices=["last", "second_last"],
                       help="Which layer to use for embeddings (for sparse_coverage). "
                            "Options: 'last' (default) or 'second_last'. "
                            "This must match the layer used when building centers.")

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
    parser.add_argument("--no_stop_centers", action="store_true",
                       help="Disable stop centers: ignore the stop_centers list from centers.json and "
                            "allow all centers (including high-df ones) for document/query assignment. "
                            "For ablation: measures the effect of stop-center filtering.")
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
    parser.add_argument("--posting_list_batch_size", type=int, default=256,
                       help="For doc soft: batch size for range_search when building posting lists. "
                            "Larger = fewer FAISS calls, may use more memory. Default: 256.")

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
    parser.add_argument("--colbert_mode", type=str, default="plaid",
                       choices=["plaid", "bruteforce"],
                       help="ColBERT retrieval mode. "
                            "'plaid' (default): uses ColBERTv2's PLAID engine — builds a compressed centroid-based index "
                            "(~2-4 GB for 2M passages) with nbits=2 residual quantisation and performs efficient end-to-end "
                            "retrieval. No first-stage retriever needed. Index is built once and cached. "
                            "'bruteforce': encodes ALL passages to per-token float32 embeddings (sharded to disk, "
                            ">200 GB for 2M passages) and computes exact MaxSim. Use only for small corpora or "
                            "when PLAID is unavailable.")
    parser.add_argument("--colbert_plaid_topk", type=int, default=1000,
                       help="Number of passages to retrieve per query with PLAID engine. "
                            "Default: 1000. Higher values give better recall but slower search.")
    parser.add_argument("--colbert_ncells", type=int, default=0,
                       help="Number of centroids for PLAID k-means. 0 = auto (ColBERT default, "
                            "often too large for >1M passages). Recommended: 2**14=16384 or "
                            "2**16=65536 for large corpora to avoid multi-hour k-means.")
    parser.add_argument("--colbert_kmeans_niters", type=int, default=4,
                       help="Number of k-means iterations for PLAID index building. "
                            "ColBERT default is 20 but 4 is usually sufficient and 5x faster.")
    parser.add_argument("--colbert_query_maxlen", type=int, default=512,
                       help="Maximum query token length for ColBERT. Default: 512. "
                            "Original ColBERTv2 default is 32; patent claims are much longer, "
                            "so 512 (BERT max) is recommended. Shorter queries are padded with [MASK].")

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


########################################################################################################################################################
########################################################################################################################################################
    # Set seed for reproducibility (even if not training, for deterministic results)
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Non-sparse_coverage models: model loading is deferred to run_clefip_eval → _get_clefip_dense_encoder
    # to avoid loading the model twice (once here, once in the evaluation path).
    if args.model_name == "sparse_coverage":
        """
        Sparse Coverage Retrieval
        
        Uses pre-built vocabulary centers for sparse retrieval.
        Documents from the evaluation corpus are encoded at runtime (or loaded from cache).
        Both queries and documents are encoded and assigned to centers.
        """
        print(f"\n🔍 Sparse Coverage Retrieval")
        
        # Validate required parameters
        if args.dense_model is None:
            raise ValueError("--dense_model is required for sparse_coverage")
        if args.tokenization_unit is None:
            raise ValueError("--tokenization_unit is required for sparse_coverage")
        
        # Eval assumes centers were built with CLS (build script controls that); only evaluation-time options here
        include_cls = True
        print(f"   Dense model: {args.dense_model}")
        print(f"   Tokenization unit: {args.tokenization_unit}")
        print(f"   Exclude CLS spans (index+query): {getattr(args, 'exclude_cls_spans', False)}")
        _qmc = getattr(args, 'query_max_chunks', -1)
        _qmc_str = 'off (truncate at 512)' if _qmc == 0 else (f'unlimited' if _qmc < 0 else f'cap={_qmc} chunks')
        print(f"   Query chunking: {_qmc_str}")
        print(f"   Layer: {getattr(args, 'layer', 'last')}")
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
            nlp = spacy.load(spacy_name, disable=disable)
            nlp.max_length = 1_000_000
            utils.NLP = nlp
            print(f"   SpaCy model: {spacy_name}")
        else:
            nlp = None
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
            keep_cls=include_cls,
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
        
        def _encode_doc_spans(
            documents: dict,
            doc_sections: list[str],
            batch_size: int = 32,
        ) -> tuple[dict[str, np.ndarray], dict[int, str], set[int]]:
            """Encode evaluation-corpus documents at runtime.

            Returns:
                embeddings_by_section: {section_name: np.ndarray of shape (n_spans, d)}
                span_to_doc: {global_span_idx: doc_id}
                exclude_cls_indices: set of global indices for CLS spans
            """
            format_scheme = get_encoder_format_scheme(args.dense_model)
            sep = get_encoder_sep_for_model(args.dense_model, tokenizer)
            doc_items = collect_doc_texts(documents, format_scheme=format_scheme, sep=sep)

            items_by_section: dict[str, list[tuple[str, str, str]]] = {}
            for doc_id, section, text in doc_items:
                if section in doc_sections:
                    items_by_section.setdefault(section, []).append((doc_id, section, text))

            embeddings_by_section: dict[str, np.ndarray] = {}
            span_to_doc: dict[int, str] = {}
            exclude_cls_indices: set[int] = set()
            exclude_cls = getattr(args, "exclude_cls_spans", False)
            current_idx = 0

            for section in doc_sections:
                items = items_by_section.get(section, [])
                if not items:
                    embeddings_by_section[section] = np.zeros((0, model.config.hidden_size), dtype=np.float32)
                    continue

                sec_ids = [it[0] for it in items]
                sec_sections = [it[1] for it in items]
                sec_texts = [it[2] for it in items]
                section_embs: list[np.ndarray] = []
                section_meta: list[dict] = []

                for b_start in tqdm(range(0, len(sec_texts), batch_size),
                                    desc=f"Encoding doc {section}", leave=False):
                    b_end = min(b_start + batch_size, len(sec_texts))
                    results = _encode_spans(
                        doc_texts=sec_texts[b_start:b_end],
                        doc_ids=sec_ids[b_start:b_end],
                        sections=sec_sections[b_start:b_end],
                        keep_doc_mean=False,
                        layer=getattr(args, "layer", "last"),
                    )
                    for doc_id, _sec, _dtxt, span_raw, span_canon, emb in results:
                        section_embs.append(emb)
                        is_cls = (span_canon or "").strip().lower() == "cls" or (span_raw or "").strip() == "[CLS]"
                        section_meta.append({"d": doc_id, "s": section, "r": span_raw or "", "is_cls": is_cls})

                for meta in section_meta:
                    span_to_doc[current_idx] = meta["d"]
                    if exclude_cls and meta["is_cls"]:
                        exclude_cls_indices.add(current_idx)
                    current_idx += 1

                embeddings_by_section[section] = np.stack(section_embs).astype(np.float32) if section_embs else np.zeros((0, model.config.hidden_size), dtype=np.float32)
                print(f"   {section}: {len(section_embs):,} spans from {len(items):,} docs")

            return embeddings_by_section, span_to_doc, exclude_cls_indices
        
        def _doc_cache_dir(mode: str) -> str:
            model_clean = args.dense_model.strip("/").split("/")[-1].replace("/", "_").replace("\\", "_")
            layer = getattr(args, "layer", "last")
            return os.path.join(args.temp_dir, f"sparse_doc_{model_clean}_{args.tokenization_unit}_{layer}_{mode}")
        
        def _save_doc_cache(
            cache_dir: str,
            doc_sections: list[str],
            embeddings_by_section: dict[str, np.ndarray],
            span_to_doc: dict[int, str],
            exclude_cls_indices: set[int],
        ) -> None:
            os.makedirs(cache_dir, exist_ok=True)
            for sec in doc_sections:
                emb = embeddings_by_section.get(sec)
                if emb is not None and emb.shape[0] > 0:
                    np.save(os.path.join(cache_dir, f"{sec}_{args.tokenization_unit}.npy"), emb, allow_pickle=False)
            meta_path = os.path.join(cache_dir, f"span_to_doc_{args.tokenization_unit}.jsonl")
            with open(meta_path, "w") as f:
                for idx in sorted(span_to_doc.keys()):
                    entry = {"i": idx, "d": span_to_doc[idx]}
                    if idx in exclude_cls_indices:
                        entry["cls"] = True
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"   Cached doc embeddings to {cache_dir}")

        def _load_doc_cache(
            cache_dir: str,
            doc_sections: list[str],
        ) -> tuple[dict[str, np.ndarray], dict[int, str], set[int], int]:
            """Load cached doc embeddings.

            Returns (embeddings_by_section, span_to_doc, exclude_cls_indices, total_loaded).
            Raises FileNotFoundError if any section file is missing.
            """
            embeddings_by_section: dict[str, np.ndarray] = {}
            total_loaded = 0
            for sec in doc_sections:
                p = os.path.join(cache_dir, f"{sec}_{args.tokenization_unit}.npy")
                if not os.path.exists(p):
                    raise FileNotFoundError(p)
                arr = np.load(p)
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                embeddings_by_section[sec] = arr
                total_loaded += arr.shape[0]

            span_to_doc, exclude_cls_indices = _load_doc_cache_meta(cache_dir)
            return embeddings_by_section, span_to_doc, exclude_cls_indices, total_loaded

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
            current_tokens = 0
            for sent in sents:
                if not sent.strip():
                    continue
                candidate = (current_chunk + " " + sent).strip() if current_chunk else sent
                n = len(tokenizer.encode(candidate, add_special_tokens=True))
                if n <= max_tokens:
                    current_chunk = candidate
                    current_tokens = n
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    # Start new chunk with this sentence
                    current_chunk = sent
                    current_tokens = len(tokenizer.encode(sent, add_special_tokens=True))
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

                for doc_id, _section, _doc_text, _span_text_raw, _span_text_canonical, span_emb in batch_results:
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
            idf: Optional[np.ndarray] = None,
            query_span_weights: Optional[list[np.ndarray]] = None,
            stop_centers: Optional[set] = None,
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            use_soft_assignment = (args.use_soft_assignment if hasattr(args, "use_soft_assignment") else True) \
                and not getattr(args, "no_soft_assignment", False)
            weight_agg = getattr(args, "weight_aggregation", "max")
            query_first_span_weight = float(getattr(args, "query_first_span_weight", 1.0))
            _stop = stop_centers if stop_centers is not None else set()

            def _to_weight(sim: float, span_idx: int, span_downweight: float = 1.0, span_extra_weight: float = 1.0) -> float:
                w = sim
                if span_idx == 0 and query_first_span_weight != 1.0:
                    w *= query_first_span_weight
                w *= span_downweight
                w *= span_extra_weight
                return w

            def _update_weight(
                weights: dict, key: int, sim: float, span_idx: int = 0, span_downweight: float = 1.0,
                span_extra_weight: float = 1.0,
            ) -> None:
                w = _to_weight(sim, span_idx, span_downweight, span_extra_weight)
                if w <= 0:
                    return
                if weight_agg == "sum":
                    weights[key] = weights.get(key, 0.0) + w
                else:
                    if w > weights.get(key, 0.0):
                        weights[key] = w

            query_sparse: list[tuple[np.ndarray, np.ndarray]] = []
            for q_idx, spans in enumerate(query_spans):
                if spans.shape[0] == 0:
                    query_sparse.append((np.array([], dtype=np.int32), np.array([], dtype=np.float32)))
                    continue

                spans_norm = spans.astype(np.float32).copy()
                faiss.normalize_L2(spans_norm)

                def _span_extra(qi: int, si: int) -> float:
                    if query_span_weights is None or qi >= len(query_span_weights):
                        return 1.0
                    w = query_span_weights[qi]
                    if si >= w.shape[0]:
                        return 1.0
                    return float(w[si])

                if use_soft_assignment:
                    max_centers_per_span = int(getattr(args, "soft_assignment_max_centers_per_span", 10) or 0)
                    K_search = min(max(max_centers_per_span * 4, 64), V)
                    _min_thr = float(sim_thr_per_center.min())
                    D_q, I_q = center_index.search(spans_norm, K_search)
                    center_weights = {}
                    for span_idx in range(spans_norm.shape[0]):
                        extra = _span_extra(q_idx, span_idx)
                        kept = 0
                        for k in range(K_search):
                            c = int(I_q[span_idx, k])
                            if c < 0:
                                break
                            sim = float(D_q[span_idx, k])
                            if sim < _min_thr:
                                break
                            if c in _stop or sim <= 0:
                                continue
                            if sim < sim_thr_per_center[c]:
                                continue
                            _update_weight(center_weights, c, sim, span_idx, span_downweight=1.0, span_extra_weight=extra)
                            kept += 1
                            if max_centers_per_span > 0 and kept >= max_centers_per_span:
                                break
                    if not center_weights:
                        similarities, assigned = center_index.search(spans_norm, k=1)
                        for span_idx in range(similarities.shape[0]):
                            center_id = int(assigned[span_idx, 0])
                            if center_id in _stop:
                                continue
                            sim = float(similarities[span_idx, 0])
                            extra = _span_extra(q_idx, span_idx)
                            _update_weight(center_weights, center_id, sim, span_idx, span_downweight=1.0, span_extra_weight=extra)

                    if center_weights:
                        centers_arr = np.array(list(center_weights.keys()), dtype=np.int32)
                        weights_arr = np.array([center_weights[c] for c in centers_arr], dtype=np.float32)
                        query_sparse.append((centers_arr, weights_arr))
                    else:
                        query_sparse.append((np.array([], dtype=np.int32), np.array([], dtype=np.float32)))
                else:
                    similarities, assigned = center_index.search(spans_norm, k=1)
                    center_weights = {}
                    for span_idx in range(similarities.shape[0]):
                        center_id = int(assigned[span_idx, 0])
                        if center_id in _stop:
                            continue
                        sim = float(similarities[span_idx, 0])
                        extra = _span_extra(q_idx, span_idx)
                        _update_weight(center_weights, center_id, sim, span_idx, span_downweight=1.0, span_extra_weight=extra)

                    centers_arr = np.array(list(center_weights.keys()), dtype=np.int32)
                    weights_arr = np.array([center_weights[c] for c in centers_arr], dtype=np.float32)
                    query_sparse.append((centers_arr, weights_arr))

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
            with open(_c_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    _c_passage_texts.append(json.loads(line)["text"])
            assert len(_c_passage_texts) == _c_npsg
            _clefip_data = {
                "query_ids": _cq_ids,
                "query_texts": _cq_texts,
                "passage_ids": _c_passage_ids,
                "passage_texts": _c_passage_texts,
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

            # Group passages by section
            items_by_section: dict[str, list[tuple[str, str, str]]] = {}
            for pid, text in zip(passage_ids_list, passage_texts_list):
                sec_raw = _clefip_passage_section(pid)
                sec = section_map.get(sec_raw, sec_raw)
                if sec not in doc_sections:
                    continue
                if sec == "claim":
                    fmt_text = format_claim_for_encoder(format_scheme, text)
                elif sec == "invention":
                    fmt_text = format_invention_for_encoder(format_scheme, text)
                else:
                    fmt_text = format_abstract_for_encoder(format_scheme, "", text, sep="")
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
                raw_path = os.path.join(cache_dir, f"_tmp_{section}.raw")
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
                            keep_doc_mean=False,
                            layer=getattr(args, "layer", "last"),
                        )
                        for doc_id, _sec, _dtxt, span_raw, span_canon, emb in results:
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
                os.remove(raw_path)

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

            return total_spans

        # Pre-cache CLEF-IP passage embeddings if data is available
        if _clefip_data is not None:
            _clefip_cache = _doc_cache_dir("clefip_passage")
            _clefip_sections = ["abstract", "claim", "invention"]
            if _doc_cache_exists(_clefip_cache, _clefip_sections):
                print(f"   ✅ CLEF-IP passage cache already exists")
            else:
                print(f"   📦 Pre-caching CLEF-IP passage embeddings...")
                _total_clefip_spans = _encode_clefip_spans(
                    _clefip_data["passage_ids"], _clefip_data["passage_texts"],
                    _clefip_sections, cache_dir=_clefip_cache, batch_size=32,
                )
                print(f"   ✅ Cached {_total_clefip_spans:,} spans for clefip_passage")

        # ---- Find centers (shared across all tasks) ----
        print(f"\n🔍 Searching for centers...")
        print(f"   Search directory: {os.path.abspath('.')}")
        
        try:
            centers_path, _centers_dir = find_centers(
                dense_model=args.dense_model,
                tokenization_unit=args.tokenization_unit,
                include_cls=include_cls,
                search_dir=".",
                layer=getattr(args, 'layer', 'last'),
                centers_suffix=getattr(args, 'centers_suffix', ''),
            )
            print(f"✅ Found centers: {centers_path}")
        except FileNotFoundError:
            layer = getattr(args, 'layer', 'last')
            raise FileNotFoundError(
                f"Doc embeddings have been cached, but could not find centers:\n"
                f"  dense_model={args.dense_model}\n"
                f"  tokenization_unit={args.tokenization_unit}\n"
                f"  (eval assumes centers built with CLS; layer={layer})\n"
                f"Searched in: {os.path.abspath('.')} (recursive)\n"
                f"Expected pattern: centers_greedy_{{model}}_{{unit}}_{{cls}}_{{layer}}\n"
                f"Please ensure centers were built with matching parameters, or use --layer last if you have centers built with 'last' layer."
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
        stop_centers = set(centers_info.get("stop_centers", []))
        if getattr(args, "no_stop_centers", False):
            print(f"   --no_stop_centers: ignoring {len(stop_centers)} stop centers (ablation)")
            stop_centers = set()
        elif stop_centers:
            print(f"   stop_centers: {len(stop_centers)} disabled for activation (df >= threshold)")
        print(f"   Final vocabulary size: {V:,} centers")
        
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
        print(f"✅ Center index built")
        
        for mode in available_modes:
            print(f"\n{'='*80}")
            print(f"Processing task: {mode}")
            print(f"{'='*80}")

            # Decide which sections to use for document indexing
            doc_sections = ["abstract", "claim", "invention"]
            query_section = "claim"
            
            # ---- Obtain doc-side embeddings + span_to_doc mapping ----
            # Priority: 1) cached runtime embeddings, 2) encode at runtime
            # For cached case, we use LAZY loading: metadata + shapes only,
            # actual embeddings loaded one section at a time during posting-list build.
            exclude_cls_spans = getattr(args, "exclude_cls_spans", False)
            embeddings_by_section: dict[str, np.ndarray] = {}
            span_to_doc: dict[int, str] = {}
            exclude_cls_span_indices: set[int] = set()
            total_loaded = 0
            _lazy_cache_dir = None  # type: str | None  # set when using lazy loading path

            cache_dir = _doc_cache_dir(mode)
            try:
                print(f"\n📦 Trying cached doc embeddings from: {cache_dir}")
                # Lazy path: load only metadata + shapes (no giant arrays)
                span_to_doc, exclude_cls_span_indices = _load_doc_cache_meta(cache_dir)
                section_shapes: dict[str, int] = {}
                for sec in doc_sections:
                    n = _get_section_shape(cache_dir, sec)
                    section_shapes[sec] = n
                    total_loaded += n
                _lazy_cache_dir = cache_dir
                print(f"   Loaded metadata ({len(span_to_doc):,} spans) from cache (lazy mode — arrays loaded per-section)")
                for sec in doc_sections:
                    print(f"   {sec}: {section_shapes[sec]:,} spans")
            except FileNotFoundError:
                print(f"\n📦 Encoding CLEF-IP passages at runtime (first run; will cache for reuse)")
                total_loaded = _encode_clefip_spans(
                    _clefip_data["passage_ids"], _clefip_data["passage_texts"],
                    doc_sections, cache_dir=cache_dir, batch_size=32,
                )
                # Use lazy path for the just-written cache
                span_to_doc, exclude_cls_span_indices = _load_doc_cache_meta(cache_dir)
                _lazy_cache_dir = cache_dir

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
            
            # ---- Build posting lists ----
            document_assignment = getattr(args, "document_assignment", "soft")
            print(f"\n🔨 Computing posting lists for {V:,} centers...")
            print(f"   Document assignment: {document_assignment}")
            
            posting_lists: list[list[tuple[int, float]]] = []
            centers_norm_for_pl = centers.copy()
            faiss.normalize_L2(centers_norm_for_pl)
            
            if document_assignment == "hard":
                print(f"   Building posting lists: each span -> nearest center (k=1)")
                # Pre-build stop_centers mask for vectorized filtering
                _stop_mask_arr_hard = np.zeros(V, dtype=bool)
                for _sc in stop_centers:
                    if 0 <= _sc < V:
                        _stop_mask_arr_hard[_sc] = True
                posting_lists = [[] for _ in range(V)]
                span_offset = 0
                for section_name in doc_sections:
                    # Lazy load: load one section at a time to avoid holding all in memory
                    if _lazy_cache_dir is not None:
                        _raw = _load_section_emb(_lazy_cache_dir, section_name)
                    else:
                        _raw = embeddings_by_section[section_name]
                    if _raw.shape[0] == 0:
                        continue
                    if _raw.shape[1] != d:
                        raise ValueError(f"Embedding dimension mismatch for {section_name}: {_raw.shape[1]} != {d}")
                    sec_emb = np.array(_raw, dtype=np.float32, copy=True)  # single contiguous copy
                    del _raw; gc.collect()  # free before normalize
                    faiss.normalize_L2(sec_emb)
                    sims, assigned = center_index.search(sec_emb, 1)
                    sims_1d = sims[:, 0]
                    assigned_1d = assigned[:, 0]
                    # Build valid mask: not CLS-excluded, not stop center, sim > 0
                    valid = (sims_1d > 0) & (assigned_1d >= 0)
                    valid &= ~_stop_mask_arr_hard[np.clip(assigned_1d, 0, V - 1)]
                    if exclude_cls_spans and exclude_cls_span_indices:
                        global_indices = np.arange(sec_emb.shape[0]) + span_offset
                        valid &= ~np.isin(global_indices, list(exclude_cls_span_indices) if not isinstance(exclude_cls_span_indices, np.ndarray) else exclude_cls_span_indices)
                    # Extract valid entries and group by center
                    valid_idx = np.where(valid)[0]
                    if valid_idx.size > 0:
                        v_centers = assigned_1d[valid_idx]
                        v_sims = sims_1d[valid_idx]
                        v_globals = valid_idx.astype(np.int64) + span_offset
                        order = np.argsort(v_centers)
                        sorted_c = v_centers[order]
                        sorted_g = v_globals[order]
                        sorted_s = v_sims[order]
                        unique_c, counts = np.unique(sorted_c, return_counts=True)
                        splits = np.cumsum(counts)[:-1]
                        g_groups = np.split(sorted_g, splits)
                        s_groups = np.split(sorted_s, splits)
                        for ci in range(len(unique_c)):
                            c = int(unique_c[ci])
                            posting_lists[c].extend(
                                zip(g_groups[ci].tolist(), s_groups[ci].tolist())
                            )
                    span_offset += sec_emb.shape[0]
                    print(f"     {section_name}: {sec_emb.shape[0]:,} spans assigned")
                    del sec_emb; gc.collect()
                print(f"   Total spans: {total_loaded:,}")
            else:
                # Doc soft: search(K) + per-center threshold filter + topK cap
                # VECTORIZED: replaces Python for-j-for-k loop with numpy ops
                max_centers_per_span = int(getattr(args, "soft_assignment_max_centers_per_span", 10) or 0)
                K_search = min(max(max_centers_per_span * 4, 64), V)
                min_sim_thr = float(sim_thr_per_center.min())
                print(f"   Soft assignment: search(K={K_search}) + per-center r_c filter + topK={max_centers_per_span}")
                # Pre-build stop_centers mask for vectorized filtering
                _stop_mask_arr = np.zeros(V, dtype=bool)
                for _sc in stop_centers:
                    if 0 <= _sc < V:
                        _stop_mask_arr[_sc] = True
                posting_lists = [[] for _ in range(V)]
                span_offset = 0
                for section_name in doc_sections:
                    # Lazy load: load one section at a time to avoid holding all in memory
                    if _lazy_cache_dir is not None:
                        _raw = _load_section_emb(_lazy_cache_dir, section_name)
                    else:
                        _raw = embeddings_by_section[section_name]
                    if _raw.shape[0] == 0:
                        span_offset += 0
                        continue
                    if _raw.shape[1] != d:
                        raise ValueError(f"Embedding dimension mismatch for {section_name}: {_raw.shape[1]} != {d}")
                    sec_emb_n = np.array(_raw, dtype=np.float32, copy=True)  # single contiguous copy
                    del _raw; gc.collect()  # free original before normalize
                    faiss.normalize_L2(sec_emb_n)
                    batch_size = max(1, int(getattr(args, "posting_list_batch_size", 4096)))
                    for b_start in tqdm(range(0, sec_emb_n.shape[0], batch_size),
                                        desc=f"  {section_name} spans->centers", leave=False):
                        b_end = min(b_start + batch_size, sec_emb_n.shape[0])
                        batch = sec_emb_n[b_start:b_end]
                        D_batch, I_batch = center_index.search(batch, K_search)
                        n_batch = batch.shape[0]

                        # --- Vectorized filtering ---
                        # 1. Clamp negative center IDs to 0 (will be masked out)
                        I_clamped = np.clip(I_batch, 0, V - 1)
                        # 2. Build boolean masks for invalid entries
                        mask_neg_id = I_batch < 0                                    # (n, K)
                        mask_low_global = D_batch < min_sim_thr                      # (n, K)
                        mask_non_pos = D_batch <= 0                                  # (n, K)
                        mask_stop = _stop_mask_arr[I_clamped]                        # (n, K)
                        # Per-center threshold: sim < r_c[center_id]
                        mask_below_rc = D_batch < sim_thr_per_center[I_clamped]      # (n, K)
                        # Combined invalid mask
                        invalid = mask_neg_id | mask_low_global | mask_non_pos | mask_stop | mask_below_rc

                        # 3. CLS exclusion (row-level mask)
                        if exclude_cls_spans and exclude_cls_span_indices:
                            global_indices = np.arange(b_start, b_end) + span_offset
                            # Build row mask: True for rows to exclude
                            row_exclude = np.isin(global_indices, list(exclude_cls_span_indices) if not isinstance(exclude_cls_span_indices, np.ndarray) else exclude_cls_span_indices)
                            invalid[row_exclude] = True

                        # 4. TopK cap: for each row, keep only the first max_centers_per_span valid entries
                        if max_centers_per_span > 0:
                            valid = ~invalid                                          # (n, K)
                            cumvalid = np.cumsum(valid, axis=1)                       # (n, K)
                            invalid |= (cumvalid > max_centers_per_span) & valid

                        # 5. Extract valid (row, col) pairs and populate posting lists
                        valid_mask = ~invalid
                        rows, cols = np.where(valid_mask)
                        if rows.size > 0:
                            centers_valid = I_batch[rows, cols]
                            sims_valid = D_batch[rows, cols]
                            global_rows = rows.astype(np.int64) + (span_offset + b_start)
                            # Group by center using argsort for batch extend
                            order = np.argsort(centers_valid)
                            sorted_c = centers_valid[order]
                            sorted_g = global_rows[order]
                            sorted_s = sims_valid[order]
                            unique_c, counts = np.unique(sorted_c, return_counts=True)
                            splits = np.cumsum(counts)[:-1]
                            g_groups = np.split(sorted_g, splits)
                            s_groups = np.split(sorted_s, splits)
                            for ci in range(len(unique_c)):
                                c = int(unique_c[ci])
                                posting_lists[c].extend(
                                    zip(g_groups[ci].tolist(), s_groups[ci].tolist())
                                )
                    span_offset += sec_emb_n.shape[0]
                    print(f"     {section_name}: {sec_emb_n.shape[0]:,} spans assigned")
                    del sec_emb_n; gc.collect()
                print(f"   Total spans: {total_loaded:,}")
            
            # Alignment sanity check
            if total_loaded != len(span_to_doc):
                raise ValueError(
                    f"Embeddings count ({total_loaded:,}) != span_to_doc count ({len(span_to_doc):,}). "
                    "Posting lists require exact 1:1 alignment."
                )
            
            # Span-level posting list stats (index build cost / memory)
            span_pl_lens = np.array([len(pl) for pl in posting_lists], dtype=np.float64)
            span_non_empty = span_pl_lens[span_pl_lens > 0]
            if len(span_non_empty) > 0:
                print(f"\n📊 Span-level posting lists (build cost/memory): "
                      f"total entries={int(span_pl_lens.sum()):,}, "
                      f"mean/non-empty center={float(span_non_empty.mean()):.1f}, "
                      f"max={int(span_pl_lens.max()):,}")
            
            # Build document-level inverted index (doc_idx, weight)
            print(f"\n🔨 Building document-level inverted index from posting lists...")
            doc_postings: list[list[tuple[int, float]]] = [[] for _ in range(V)]
            
            weight_agg = getattr(args, "weight_aggregation", "max")
            for center_idx in tqdm(range(V), desc="Building inverted index"):
                span_sims = posting_lists[center_idx]
                if not span_sims:
                    continue
                agg: dict[int, float] = {}  # doc_idx -> weight
                for entry in span_sims:
                    span_idx, similarity = entry[0], entry[1]
                    doc_id = span_to_doc.get(span_idx, None)
                    if doc_id is None:
                        continue
                    didx = doc_id_to_idx.get(doc_id)
                    if didx is None:
                        continue
                    sim = float(similarity)
                    if weight_agg == "sum":
                        agg[didx] = agg.get(didx, 0.0) + max(0.0, sim)
                    else:
                        if didx not in agg or max(0.0, sim) > agg[didx]:
                            agg[didx] = max(0.0, sim)
                for didx, w in agg.items():
                    doc_postings[center_idx].append((didx, float(w)))
            
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

            N_docs = len(_clefip_data["passage_ids"])
            df = np.array([len(pl) for pl in doc_postings], dtype=np.float32)  # doc_idx unique per center (aggregated above)
            idf = (np.log((N_docs + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
            idf_exponent = float(getattr(args, "idf_exponent", 1.0))
            if idf_exponent != 1.0:
                print(f"   IDF exponent: {idf_exponent} (score term uses idf^{idf_exponent})")

            # Encode + assign queries (format must match utils.collect_doc_texts for doc side)
            print(f"\n📝 Evaluating: CLEF-IP Claims -> Passages")
            _fmt_scheme = get_encoder_format_scheme(args.dense_model)
            query_texts = [format_claim_for_encoder(_fmt_scheme, qt) for qt in _clefip_data["query_texts"]]
            
            query_spans = _encode_query_spans(query_texts, section=query_section, d=d)
            query_span_weights = None
            query_sparse = _assign_query_spans_to_centers(query_spans, center_index=center_index, V=V, sim_thr_per_center=sim_thr_per_center, idf=idf, query_span_weights=query_span_weights, stop_centers=stop_centers)
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
            top_k = 100
            _retrieval_top_k = len(_clefip_data["passage_ids"])
            _result = _score_queries_against_postings(
                query_sparse, doc_postings, idf, idf_exponent, _retrieval_top_k,
                length_norm=length_norm,
                length_norm_exp=length_norm_exp,
                doc_nspans=doc_nspans,
                return_scores=True,
            )
            top_indices, _sparse_score_dicts = _result
            
            if mode == "clefip_passage":
                # CLEF-IP passage-level evaluation: retrieve passage_ids, use CLEF-IP official metrics
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
                predicted_labels_list, doc_ranking_list = _clefip_two_stage_rerank(
                    _clefip_pid_list, all_passage_scores_list,
                    topk_docs=topk_docs,
                )
                print(f"  🔄 Two-stage retrieval: top-{topk_docs} docs → re-ranked passages per query")

                _evaluate_and_print_clefip(
                    _clefip_qrels, _clefip_qids, predicted_labels_list,
                    "Sparse Coverage (CLEF-IP)",
                    two_stage=True, topk_docs=topk_docs,
                    doc_ranking_list=doc_ranking_list,
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
                    for _qid_r, _rel_pids in _clefip_qrels.items():
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
                        _top_indices_f, _sparse_score_dicts_f = _result_f

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
                        _pred_f, _doc_ranking_f = _clefip_two_stage_rerank(
                            _allowed_pid_list, _passage_scores_f, topk_docs=topk_docs,
                        )

                        _true_f = [_clefip_qrels.get(qid, []) for qid in _clefip_qids]
                        _res_f = _make_clefip_official_metrics(_true_f, _pred_f,
                                                              doc_ranking_list=_doc_ranking_f)
                        _robustness_results.append((_n_docs_f, _n_passages_f, _res_f))
                        print(f"   {_n_docs_f:>6} docs ({_n_passages_f:>8,} passages): "
                              f"recall@100={_res_f.get('recall@100', 0):.4f}  "
                              f"ndcg@10={_res_f.get('ndcg@10', 0):.4f}  "
                              f"map_doc={_res_f.get('map_doc', 0):.4f}  "
                              f"pres_doc@100={_res_f.get('pres_doc@100', 0):.4f}")

                    # Summary table
                    if _robustness_results:
                        print(f"\n{'='*90}")
                        print(f"CLEF-IP Robustness Summary (relevant docs always included, negatives subsampled)")
                        print(f"{'='*90}")
                        print(f"{'Docs':>8} {'Passages':>10} {'recall@100':>12} {'ndcg@10':>10} "
                              f"{'map_doc':>10} {'pres_doc@100':>14} {'magp':>8}")
                        print(f"{'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*14} {'-'*8}")
                        for _nd, _np, _r in _robustness_results:
                            print(f"{_nd:>8} {_np:>10,} "
                                  f"{_r.get('recall@100', 0):>12.4f} "
                                  f"{_r.get('ndcg@10', 0):>10.4f} "
                                  f"{_r.get('map_doc', 0):>10.4f} "
                                  f"{_r.get('pres_doc@100', 0):>14.4f} "
                                  f"{_r.get('magp', 0):>8.4f}")
                        print(f"{'='*92}")


            
            print(f"\n✅ Task {mode} evaluation completed")
            
            # Free large objects before next mode to avoid peak memory overlap
            del embeddings_by_section, span_to_doc, exclude_cls_span_indices
            del posting_lists, doc_postings, top_indices, query_sparse
            gc.collect()
        
        print(f"\n✅ Sparse Coverage evaluation completed for all available tasks")

    ############################################## CLEF-IP 2013 EN (claims-to-passages) ##################################################
    if args.model_name != "sparse_coverage":
        # sparse_coverage handles CLEF-IP in its own mode loop
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