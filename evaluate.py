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

from __future__ import absolute_import, division, unicode_literals

import os
import re
import sys
import json
import argparse
import logging
from typing import Optional

from tqdm import trange, tqdm
import pandas as pd
import numpy as np

import faiss
import torch

from transformers import set_seed,  AutoTokenizer, AutoModel
from scipy.sparse import csr_matrix, isspmatrix

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

# Global constants
QUALITY_MIN_WORDS = 6  # Minimum number of words required for high-quality text


# add patenteval to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
patent_eval_path = os.path.join(current_dir, 'patentmap_eval')
sys.path.append(patent_eval_path)

# Try to import patenteval.utils with better error handling
try:
    from patenteval.utils import (
        load_corpus,
        citation_to_citing_to_cited_dict,
        mean_recall_at_k,
        mean_ndcg_at_k,
        mean_mrr_at_k,
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
)


def _load_or_compute_prior_art_embeddings(cache_query_path, cache_doc_path, compute_fn, pickle_protocol=None):
    """
    Load prior-art query/document embeddings from cache if present, else compute via compute_fn() and save.
    compute_fn() should return (query_embeddings, document_embeddings) as numpy arrays.
    Returns (query_embeddings, document_embeddings).
    """
    if os.path.exists(cache_query_path) and os.path.exists(cache_doc_path):
        print("Embeddings already created!")
        return (
            torch.load(cache_query_path, weights_only=False),
            torch.load(cache_doc_path, weights_only=False),
        )
    query_embeddings, document_embeddings = compute_fn()
    save_kw = {} if pickle_protocol is None else {"pickle_protocol": pickle_protocol}
    torch.save(query_embeddings, cache_query_path, **save_kw)
    torch.save(document_embeddings, cache_doc_path, **save_kw)
    return query_embeddings, document_embeddings


def _save_rankings_paths_from_args(args):
    """If args.save_rankings (dir) is set, return (path_abstract2abstract, path_claim2all); else (None, None). Creates dir if needed."""
    base = getattr(args, "save_rankings", None)
    if not base:
        return None, None
    base = os.path.abspath(base)
    os.makedirs(base, exist_ok=True)
    return (
        os.path.join(base, "rankings_abstract2abstract.json"),
        os.path.join(base, "rankings_claim2all.json"),
    )


def prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types, save_rankings_path=None, save_rankings_claim2all_path=None, model_label="Dense"):
    """
    Evaluate prior art search performance.

    - If save_rankings_path is set, save abstract->abstract rankings.
    - If save_rankings_claim2all_path is set, save claim->all rankings.
    Format: {query_id: [doc_id, ...]}.
    - model_label: used for FLOPs/efficiency reporting (e.g. "Dense", "Specter2").
    """
    assert len(query_ids) == len(query_embeddings), f"query_ids and query_embeddings length mismatch: {len(query_ids)} vs {len(query_embeddings)}"
    assert len(doc_ids) == len(document_embeddings), f"doc_ids and document_embeddings length mismatch: {len(doc_ids)} vs {len(document_embeddings)}"
    assert len(query_types) == len(query_ids), f"query_types and query_ids length mismatch: {len(query_types)} vs {len(query_ids)}"
    assert len(doc_types) == len(doc_ids), f"doc_types and doc_ids length mismatch: {len(doc_types)} vs {len(doc_ids)}"
    assert len(citation_mapping) == len(query_ids), f"citation_mapping and query_ids length mismatch: {len(citation_mapping)} vs {len(query_ids)}"

    results = {}

    ######## Task1: Abstract-to-Abstract evaluation ########
    texttype_q, texttype_d = "abstract", "abstract"

    # Convert to numpy array to ensure compatibility
    query_types = np.array(query_types)
    doc_types = np.array(doc_types)

    query_type_masks = (query_types == texttype_q)
    doc_type_masks = (doc_types == texttype_d)

    Q_emb = query_embeddings[query_type_masks].astype(np.float32)  # shape: [n_queries, emb_dim]
    D_emb = document_embeddings[doc_type_masks].astype(np.float32)    # shape: [n_docs, emb_dim]

    # Validate shape consistency
    assert Q_emb.shape[1] == D_emb.shape[1], f"Embedding dimension mismatch: Q_emb {Q_emb.shape} vs D_emb {D_emb.shape}"
    assert not np.any(np.isnan(Q_emb)) and not np.any(np.isnan(D_emb)), "NaN detected in embeddings before normalization."

    faiss.normalize_L2(Q_emb)  # Normalize before similarity computation
    faiss.normalize_L2(D_emb)
    _report_dense_flops(Q_emb, D_emb, "abstract->abstract", model_label=model_label)
    distances = Q_emb @ D_emb.T  # FAISS optimized cosine similarity

    # For each query row, we get top_k doc indices (sorted ascending by distance)
    top_k_indices = np.argsort(-distances, axis=1)

    # Evaluate retrieval: we build lists of true labels & predicted labels
    true_labels_list, predicted_labels_list = [], []

    # We'll iterate over each query index
    for q_idx, retrieved_docs_indices in enumerate(top_k_indices):
        # 1) The query ID string, e.g. 'Q1'
        q_id_str = query_ids[q_idx]
        # 2) The set of true doc IDs for that query, e.g. ['D3', 'D27']
        #    Make sure your citation_mapping stores them as a set/list
        true_labels = citation_mapping.get(q_id_str, [])

        # 3) Convert doc indices to doc ID strings
        predicted_labels = [doc_ids[d_idx] for d_idx in retrieved_docs_indices]

        true_labels_list.append(true_labels)
        predicted_labels_list.append(predicted_labels)

    # Optionally save abstract->abstract rankings for hybrid fusion (query_ids here are full list; first n are abstract)
    if save_rankings_path:
        n_abstract = len(predicted_labels_list)
        ranking_dict = {query_ids[q_idx]: predicted_labels_list[q_idx] for q_idx in range(n_abstract)}
        with open(save_rankings_path, "w") as f:
            json.dump(ranking_dict, f, indent=0)
        print(f"   Saved abstract->abstract rankings to {save_rankings_path} ({n_abstract} queries)")

    # Compute metrics
    results_key = "abstract->abstract"
    results[results_key] = _make_prior_art_metrics(true_labels_list, predicted_labels_list)


    ######## Task2: Claim-to-All evaluation ########
    retrieved_sections = []   # for noting which section is retrieved at top_k
    
    # Original counts before tripling (abstract, claim, invention)
    original_doc_count = len(doc_ids) // 3
    original_query_count = len(query_ids) // 3

    query_type_masks = (query_types == "claim")
    Q_emb = query_embeddings[query_type_masks].astype(np.float32)
    D_emb = document_embeddings.astype(np.float32)
    D_ids = doc_ids
    faiss.normalize_L2(Q_emb)
    faiss.normalize_L2(D_emb)
    _report_dense_flops(Q_emb, D_emb, "claim->all", model_label=model_label)
    distances = Q_emb @ D_emb.T

    top_k_indices = np.argsort(-distances, axis=1)[:, :300]  # top_k * 3 to ensure we have enough candidates
    true_labels_list, predicted_labels_list = [], []
    for q_idx, retrieved_docs_indices in enumerate(top_k_indices):
        q_id_str = query_ids[original_query_count + q_idx]  # qid for claim queries
        true_labels = citation_mapping.get(q_id_str, [])
        # Map indices to doc IDs (same doc can appear as abstract/claim/invention → dedupe by first occurrence)
        raw_predicted = [D_ids[d_idx] for d_idx in retrieved_docs_indices]
        _, unique_indices = np.unique(raw_predicted, return_index=True)
        unique_indices_sorted = sorted(unique_indices)[:100]
        predicted_labels = [raw_predicted[i] for i in unique_indices_sorted]
        # Section for each of top-100 (after dedupe): 0=abstract, 1=claim, 2=invention
        section_names = ["abstract", "claim", "invention"]
        retrieved_sections.append([
            section_names[retrieved_docs_indices[i] // original_doc_count] for i in unique_indices_sorted
        ])
        true_labels_list.append(true_labels)
        predicted_labels_list.append(predicted_labels)

    # Compute metrics
    results_key = "claim->all"
    results[results_key] = _make_prior_art_metrics(
        true_labels_list, predicted_labels_list,
        retrieved_sections=f"[{len(retrieved_sections)} queries with retrieved sections]",
    )

    # Optionally save claim->all rankings for hybrid fusion
    if save_rankings_claim2all_path:
        n_claim = len(predicted_labels_list)
        claim_query_ids = [query_ids[original_query_count + q_idx] for q_idx in range(n_claim)]
        ranking_dict = {str(qid): predicted_labels_list[q_idx] for q_idx, qid in enumerate(claim_query_ids)}
        with open(save_rankings_claim2all_path, "w") as f:
            json.dump(ranking_dict, f, indent=0)
        print(f"   Saved claim->all rankings to {save_rankings_claim2all_path} ({n_claim} queries)")

    # Format and display results
    print_subsection_header("Prior Art Search Results")
    
    for task_key, task_results in results.items():
        if isinstance(task_results, dict):
            # Create a clean task name
            if '->' in task_key:
                clean_name = f"Query: {task_key.split('->')[0]} → Document: {task_key.split('->')[1]}"
            else:
                clean_name = task_key
                
            print_metric_table(task_results, clean_name)
        
    # Store the full retrieved_sections in results for analysis, but don't print it
    results[results_key]['retrieved_sections_full'] = retrieved_sections
    
    # Run retrieved sections analysis if we have the data
    if retrieved_sections:
        from patentmap_eval.patenteval.utils import analyze_retrieved_sections_integrated
        
        # Analyze retrieved sections distribution
        section_analysis = analyze_retrieved_sections_integrated(
            retrieved_sections, 
            query_section=texttype_q, 
            print_results=True
        )
        results[results_key]['section_analysis'] = section_analysis


def clefip_passage_evaluation(
    query_ids,
    passage_ids,
    query_embeddings,
    passage_embeddings,
    qrels_passage_ids,
    k=100,
    model_label="Dense",
):
    """
    Evaluate CLEF-IP claims-to-passages: rank passages per query and compute metrics.
    qrels_passage_ids: dict topic_id -> list of relevant passage_ids (subset of passage_ids).
    Uses same metrics as prior_art: recall@k, NDCG@k, MRR, MAP, PRES@100.
    """
    assert len(query_ids) == len(query_embeddings)
    assert len(passage_ids) == len(passage_embeddings)
    Q = np.asarray(query_embeddings, dtype=np.float32)
    D = np.asarray(passage_embeddings, dtype=np.float32)
    assert Q.shape[1] == D.shape[1]
    faiss.normalize_L2(Q)
    faiss.normalize_L2(D)
    _report_dense_flops(Q, D, "CLEF-IP passage", model_label=model_label)
    sim = Q @ D.T
    top_k_indices = np.argsort(-sim, axis=1)[:, :k]
    true_labels_list = []
    predicted_labels_list = []
    for q_idx, qid in enumerate(query_ids):
        true_labels_list.append(qrels_passage_ids.get(qid, []))
        predicted_labels_list.append([passage_ids[j] for j in top_k_indices[q_idx]])
    results = _make_prior_art_metrics(true_labels_list, predicted_labels_list)
    print_subsection_header("CLEF-IP 2013 EN claims-to-passages")
    print_metric_table(results, "Passage retrieval")
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


def _clefip_format_for_model(query_texts, passage_ids, passage_texts, model_name):
    """
    Format CLEF-IP query/passage texts per utils encoder format scheme. Returns (query_list, passage_list).
    - ENCODER_FORMAT_TITLE_SEP_ONLY (Specter2, PatentBERT, PAECTer, Patembed): raw text, no section tokens.
    - ENCODER_FORMAT_SECTION_TOKENS (e.g. bert-for-patents): [claim] query, [abstract]/[invention]/[claim] passage.
    """
    if not query_texts or not passage_texts:
        return query_texts, passage_texts
    scheme = get_encoder_format_scheme(model_name)
    if scheme != ENCODER_FORMAT_SECTION_TOKENS:
        return query_texts, passage_texts
    query_fmt = [format_claim_for_encoder(scheme, t) for t in query_texts]
    passage_sections = [_clefip_passage_section(pid) for pid in passage_ids]
    section_map = {"abstract": "abstract", "description": "invention", "claim": "claim"}
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
        raise RuntimeError(
            f"FLOPs evaluation required but not available for {model_label} {label}: "
            "query term ids or vocab mapping unavailable. Ensure query_tokens_list is passed and retriever supports get_tokens_ids()."
        )
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


def _splade_retrieve_with_index(query_sparse, posting_lists, top_k: int):
    """Term-at-a-time: for each query accumulate doc scores via posting lists. Returns list of top_k doc index arrays."""
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


def _to_numpy_if_torch(*arrays):
    """Convert torch tensors to numpy; leave arrays as np.asarray. Returns tuple of numpy arrays."""
    out = []
    for a in arrays:
        if hasattr(a, "cpu"):
            out.append(a.cpu().numpy())
        else:
            out.append(np.asarray(a))
    return tuple(out)


def _make_prior_art_metrics(true_labels_list, predicted_labels_list, k=100, **extra):
    """Build standard prior-art metric dict (recall@k, ndcg@k, mrr@10, map, pres@100). Merge in extra keys."""
    metrics = {
        "recall@10": mean_recall_at_k(true_labels_list, predicted_labels_list, k=10),
        "recall@20": mean_recall_at_k(true_labels_list, predicted_labels_list, k=20),
        "recall@50": mean_recall_at_k(true_labels_list, predicted_labels_list, k=50),
        "recall@100": mean_recall_at_k(true_labels_list, predicted_labels_list, k=100),
        "ndcg@10": mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=10),
        "ndcg@20": mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=20),
        "ndcg@50": mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=50),
        "ndcg@100": mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=100),
        "mrr@10": mean_mrr_at_k(true_labels_list, predicted_labels_list, k=10),
        "map": mean_average_precision(true_labels_list, predicted_labels_list, k=k),
        "pres@100": mean_pres_at_k(true_labels_list, predicted_labels_list, k=100, N_max=100),
    }
    metrics.update(extra)
    return metrics


def _run_clefip_sparse_coverage(args, query_ids, query_texts, passage_ids, passage_texts, qrels_passage_ids):
    """Sparse-coverage retrieval for CLEF-IP: passages are 'documents', queries are claims."""
    import utils
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dense_model is None:
        raise ValueError("--dense_model is required for sparse_coverage on CLEF-IP")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.dense_model)
    model = AutoModel.from_pretrained(args.dense_model, trust_remote_code=True)
    utils.ensure_section_tokens(tokenizer, model)
    model.to(device).eval()
    d = model.config.hidden_size

    # SpaCy
    if args.tokenization_unit != "encoder_token":
        import spacy
        if args.spacy_model.startswith("sci_"):
            spacy_name = f"en_core_sci_{args.spacy_model[4:]}"
        else:
            spacy_name = f"en_core_web_{args.spacy_model}"
        nlp = spacy.load(spacy_name, disable=["ner", "textcat", "lemmatizer"])
        nlp.max_length = 1_000_000
        utils.NLP = nlp

    process_doc_batch = utils.process_doc_batch
    include_cls = True

    # Find centers (claim2all mode — queries are claims, passages are mixed sections)
    try:
        centers_path, _centers_dir = find_centers(
            dense_model=args.dense_model,
            tokenization_unit=args.tokenization_unit,
            include_cls=include_cls,
            search_dir=".",
            mode="claim2all",
            layer=getattr(args, "layer", "last"),
            centers_suffix=getattr(args, "centers_suffix", ""),
        )
    except FileNotFoundError:
        print("CLEF-IP sparse_coverage: no centers found for claim2all; skipping.")
        return

    centers = np.load(centers_path).astype(np.float32)
    V, d_c = centers.shape
    if d_c != d:
        raise ValueError(f"Dimension mismatch: model={d}, centers={d_c}")
    centers_json = centers_path.replace(".npy", ".json")
    centers_info = {}
    if os.path.exists(centers_json):
        with open(centers_json, "r") as f:
            centers_info = json.load(f)
    r = float(centers_info.get("r", 0.1))
    sim_threshold = 1.0 - r
    stop_centers = set(centers_info.get("stop_centers", []))
    if stop_centers:
        print(f"CLEF-IP sparse_coverage: {len(stop_centers)} stop-centers disabled (df >= threshold)")

    centers_norm = centers.copy()
    faiss.normalize_L2(centers_norm)
    center_index = faiss.IndexFlatIP(d)
    center_index.add(centers_norm.astype(np.float32))
    print(f"CLEF-IP sparse_coverage: {V:,} centers, r={r:.4f}")

    # Derive section per passage and format text with section tokens
    format_scheme = get_encoder_format_scheme(args.dense_model)
    passage_sections_raw = [_clefip_passage_section(pid) for pid in passage_ids]
    section_map = {"abstract": "abstract", "description": "invention", "claim": "claim"}

    def _format_passage(text: str, sec_raw: str) -> str:
        sec = section_map.get(sec_raw, sec_raw)
        if sec == "claim":
            return format_claim_for_encoder(format_scheme, text)
        elif sec == "invention":
            return format_invention_for_encoder(format_scheme, text)
        return f"[abstract] {text}".strip() if format_scheme != "title_sep_only" else text

    formatted_passages = [_format_passage(t, s) for t, s in zip(passage_texts, passage_sections_raw)]

    # Encode passages grouped by section
    print("Encoding passages...")
    passage_embs: list[np.ndarray] = [None] * len(passage_ids)
    for sec_raw in ["abstract", "claim", "description"]:
        sec = section_map.get(sec_raw, sec_raw)
        idxs = [i for i, s in enumerate(passage_sections_raw) if s == sec_raw]
        if not idxs:
            continue
        texts = [formatted_passages[i] for i in idxs]
        pids = [passage_ids[i] for i in idxs]
        for b_start in tqdm(range(0, len(texts), 32), desc=f"  passages ({sec})", leave=False):
            b_end = min(b_start + 32, len(texts))
            results = process_doc_batch(
                doc_texts=texts[b_start:b_end],
                doc_ids=pids[b_start:b_end],
                sections=[sec] * (b_end - b_start),
                unit=args.tokenization_unit,
                model=model, tokenizer=tokenizer, device=device,
                max_length=512, keep_cls=include_cls, span_pooling="mean",
                layer=getattr(args, "layer", "last"),
            )
            span_groups: dict[str, list[np.ndarray]] = {}
            for doc_id, _s, _dt, _sr, _sc, emb in results:
                span_groups.setdefault(doc_id, []).append(emb)
            for local_i, pid in enumerate(pids[b_start:b_end]):
                spans = span_groups.get(pid, [])
                if spans:
                    passage_embs[idxs[b_start + local_i]] = np.stack(spans).astype(np.float32)
                else:
                    passage_embs[idxs[b_start + local_i]] = np.zeros((0, d), dtype=np.float32)

    # Build posting lists (passage-level): for each center, list of (passage_idx, weight)
    print("Building posting lists...")
    posting_lists: list[list[tuple[int, float]]] = [[] for _ in range(V)]
    weight_agg = getattr(args, "weight_aggregation", "max")
    for p_idx, p_spans in enumerate(tqdm(passage_embs, desc="  passages -> centers", leave=False)):
        if p_spans is None or p_spans.shape[0] == 0:
            continue
        p_norm = p_spans.astype(np.float32).copy()
        faiss.normalize_L2(p_norm)
        lims, D, I = center_index.range_search(p_norm, sim_threshold)
        center_weights: dict[int, float] = {}
        for span_i in range(p_norm.shape[0]):
            start, end = int(lims[span_i]), int(lims[span_i + 1])
            for j in range(start, end):
                c = int(I[j])
                if c in stop_centers:
                    continue
                sim = float(D[j])
                if sim <= 0:
                    continue
                if weight_agg == "sum":
                    center_weights[c] = center_weights.get(c, 0.0) + sim
                else:
                    if sim > center_weights.get(c, 0.0):
                        center_weights[c] = sim
        for c, w in center_weights.items():
            posting_lists[c].append((p_idx, w))

    N_passages = len(passage_ids)
    df = np.array([len(set(e[0] for e in pl)) if pl else 0 for pl in posting_lists], dtype=np.float32)
    idf = (np.log((N_passages + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
    idf_exponent = float(getattr(args, "idf_exponent", 1.0))

    # Encode queries (claims — format with section tokens)
    formatted_queries = [format_claim_for_encoder(format_scheme, qt) for qt in query_texts]
    print("Encoding queries...")
    query_spans_list: list[np.ndarray] = []
    for b_start in tqdm(range(0, len(formatted_queries), 32), desc="  queries", leave=False):
        b_end = min(b_start + 32, len(formatted_queries))
        results = process_doc_batch(
            doc_texts=formatted_queries[b_start:b_end],
            doc_ids=[f"q_{i}" for i in range(b_start, b_end)],
            sections=["claim"] * (b_end - b_start),
            unit=args.tokenization_unit,
            model=model, tokenizer=tokenizer, device=device,
            max_length=512, keep_cls=include_cls, span_pooling="mean",
            layer=getattr(args, "layer", "last"),
        )
        q_groups: dict[str, list[np.ndarray]] = {}
        for doc_id, _s, _dt, _sr, _sc, emb in results:
            q_groups.setdefault(doc_id, []).append(emb)
        for qi in range(b_start, b_end):
            spans = q_groups.get(f"q_{qi}", [])
            if spans:
                query_spans_list.append(np.stack(spans).astype(np.float32))
            else:
                query_spans_list.append(np.zeros((0, d), dtype=np.float32))

    # Retrieve: score passages per query
    print("Retrieving...")
    top_k = 100
    predicted_labels: list[list[str]] = []
    for q_idx, q_spans in enumerate(query_spans_list):
        if q_spans.shape[0] == 0:
            predicted_labels.append([])
            continue
        q_norm = q_spans.astype(np.float32).copy()
        faiss.normalize_L2(q_norm)
        lims, D, I = center_index.range_search(q_norm, sim_threshold)
        q_center_weights: dict[int, float] = {}
        for si in range(q_norm.shape[0]):
            start, end = int(lims[si]), int(lims[si + 1])
            for j in range(start, end):
                c = int(I[j])
                if c in stop_centers:
                    continue
                sim = float(D[j])
                if sim > q_center_weights.get(c, 0.0):
                    q_center_weights[c] = sim
        passage_scores: dict[int, float] = {}
        for c, q_sim in q_center_weights.items():
            idf_t = float(idf[c]) ** idf_exponent
            for p_idx, p_sim in posting_lists[c]:
                score = q_sim * p_sim * idf_t
                passage_scores[p_idx] = passage_scores.get(p_idx, 0.0) + score
        sorted_p = sorted(passage_scores.items(), key=lambda x: -x[1])[:top_k]
        predicted_labels.append([passage_ids[pi] for pi, _ in sorted_p])

    true_labels = [qrels_passage_ids.get(qid, []) for qid in query_ids]
    results = _make_prior_art_metrics(true_labels, predicted_labels)
    print_subsection_header("CLEF-IP 2013 EN claims-to-passages")
    print_metric_table(results, "Sparse Coverage (CLEF-IP)")


def run_clefip_eval(args):
    """Load CLEF-IP EN data, run the selected model, and evaluate passage retrieval."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    clefip_root = os.path.join(current_dir, "clefip2013")
    if not os.path.isdir(clefip_root):
        raise FileNotFoundError(f"CLEF-IP root not found: {clefip_root}")
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from clefip2013.load_clefip import load_clefip_en_for_eval
    doc_root = getattr(args, "clefip_doc_root", None)
    if not doc_root:
        print("CLEF-IP skipped: --clefip_doc_root not set (required for official task).")
        return
    print("Loading CLEF-IP 2013 EN (claims-to-passages, official 01 collection)...")
    query_ids, query_texts, passage_ids, passage_texts, qrels_passage_ids = load_clefip_en_for_eval(
        clefip_root, doc_collection_root=doc_root
    )
    print(f"  Queries: {len(query_ids)}, Passages: {len(passage_ids)}")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model_name.lower() if hasattr(args.model_name, "lower") else str(args.model_name).lower()

    if model_name == "sparse_coverage":
        _run_clefip_sparse_coverage(args, query_ids, query_texts, passage_ids, passage_texts, qrels_passage_ids)
        return

    if model_name == "bm25":
        # Use same BM25 stack as prior-art: bm25s + stemmer, one logic for both tasks
        import bm25s
        import snowballstemmer
        stemmer = snowballstemmer.stemmer("english")
        passage_tokens = bm25s.tokenize(passage_texts, stopwords="en", stemmer=stemmer)
        query_tokens = bm25s.tokenize(query_texts, stemmer=stemmer)
        retriever = bm25s.BM25()
        retriever.index(passage_tokens)
        _report_bm25_posting_stats(retriever, "CLEF-IP passage", query_tokens_list=query_tokens)
        k = 100
        clefip_results, _ = retriever.retrieve(query_tokens, k=k)
        predicted_labels_list = [[passage_ids[j] for j in result] for result in clefip_results]
        true_labels_list = [qrels_passage_ids.get(qid, []) for qid in query_ids]
        results = _make_prior_art_metrics(true_labels_list, predicted_labels_list)
        print_subsection_header("CLEF-IP 2013 EN claims-to-passages")
        print_metric_table(results, "BM25 passage retrieval")
        return

    # Format query/passage for dense models (section tags like prior-art)
    query_texts, passage_texts = _clefip_format_for_model(query_texts, passage_ids, passage_texts, args.model_name)

    # Dense models: encode with model-appropriate input format
    if model_name in ["allenai/specter2_base", "patentbert"]:
        from adapters import AutoAdapterModel
        if model_name == "patentbert":
            model_path = "./PatentBert/encoder_only_model"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoAdapterModel.from_pretrained(model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = AutoAdapterModel.from_pretrained(args.model_name)
            model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
        model.to(device)
        # Specter2/PatentBERT use title_sep_only (no section tokens); no add_special_tokens needed for CLEF-IP raw text
        def _encode(texts, batch_size=32):
            embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inp = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(**inp)
                embs.append(cls_pooling(out.last_hidden_state, inp["attention_mask"]).cpu().numpy())
            return np.vstack(embs)
        query_emb = _encode(query_texts)
        passage_emb = _encode(passage_texts)
        _cl_label = "Specter2" if "specter" in model_name else "PatentBERT"
        clefip_passage_evaluation(query_ids, passage_ids, query_emb, passage_emb, qrels_passage_ids, k=100, model_label=_cl_label)
        return

    if model_name in ["mpi-inno-comp/paecter", "anferico/bert-for-patents"]:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name)
        if "anferico/bert-for-patents" in model_name:
            tokenizer.add_special_tokens({'additional_special_tokens': ['[abstract]', '[claim]', '[invention]']})
            model.resize_token_embeddings(len(tokenizer))
        model.to(device)
        def _encode(texts, batch_size=32):
            embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inp = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(**inp)
                embs.append(mean_pooling(out.last_hidden_state, inp["attention_mask"]).cpu().numpy())
            return np.vstack(embs)
        query_emb = _encode(query_texts)
        passage_emb = _encode(passage_texts)
        _cl_label = "PAECTer" if "paecter" in model_name else "bert-for-patents"
        clefip_passage_evaluation(query_ids, passage_ids, query_emb, passage_emb, qrels_passage_ids, k=100, model_label=_cl_label)
        return

    if model_name in ["datalyes/patembed-large", "patembed-large"]:
        from sentence_transformers import SentenceTransformer
        actual_model_id = "datalyes/patembed-large"
        model = SentenceTransformer(actual_model_id)
        model.to(device)
        PATEN_TEB_RETRIEVAL_PROMPT_NAME = "retrieval_MIXED"
        try:
            query_emb = model.encode_query(query_texts, prompt_name=PATEN_TEB_RETRIEVAL_PROMPT_NAME, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
            passage_emb = model.encode_document(passage_texts, prompt_name=PATEN_TEB_RETRIEVAL_PROMPT_NAME, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
        except Exception:
            query_emb = model.encode(query_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
            passage_emb = model.encode(passage_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
        clefip_passage_evaluation(query_ids, passage_ids, query_emb, passage_emb, qrels_passage_ids, k=100, model_label="Patembed")
        return

    # Fallback: generic AutoModel + mean pooling (raw formatted text with section tags)
    print(f"CLEF-IP eval: trying generic encoder for {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    def _encode(texts, batch_size=32):
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inp = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inp)
            embs.append(mean_pooling(out.last_hidden_state, inp["attention_mask"]).cpu().numpy())
        return np.vstack(embs)
    query_emb = _encode(query_texts)
    passage_emb = _encode(passage_texts)
    clefip_passage_evaluation(query_ids, passage_ids, query_emb, passage_emb, qrels_passage_ids, k=100, model_label="Dense")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, 
                       help="Path to pretrained model or model ID. Supported models: "
                            "allenai/specter2_base, patentbert, mpi-inno-comp/paecter, "
                            "anferico/bert-for-patents, datalyes/patembed-large, naver/splade-v2, bm25, "
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
                       help="Document side: hard = each span -> nearest center only (Voronoi); soft = each span in all spheres (range_search). Default: soft.")
    parser.add_argument("--weight_aggregation", type=str, default="max", choices=["max", "sum"],
                       help="Per (query, center) and (doc, center): max = use max similarity (default); sum = use sum of similarities (TF-style). Default: max.")
    
    # Query side:
    parser.add_argument("--use_soft_assignment", action="store_true", default=False,
                       help="Use soft assignment for query spans: all centers with sim >= threshold (same as document side, range_search). "
                            "Default: False (hard assignment: only nearest center per span). If True, query assignment is consistent with posting lists.")
    parser.add_argument("--soft_assignment_max_centers_per_span", type=int, default=None,
                       help="When use_soft_assignment: cap each span to at most this many centers (by similarity). "
                            "Cap-only: if span falls in >K centers, keep top-K; if <=K, keep all (no fill). "
                            "Default: None (no cap). Try 5 or 10 to reduce noise while keeping multi-center recall.")
    parser.add_argument("--query_first_span_weight", type=float, default=1.0,
                       help="Multiply weight of first span per query by this factor (e.g. 1.5 for claim2all). Default: 1.0.")
    parser.add_argument("--query_full_chunks", action="store_true", default=False,
                       help="[Query only] Encode full query by chunking (no truncation). Only for abstract2abstract and tokenization_unit=encoder_token. Doc side unchanged.")
    parser.add_argument("--query_chunk_stride_ratio", type=float, default=1.0,
                       help="When query_full_chunks: stride = chunk_window * this (1.0 = no overlap, <1.0 = overlap; dedup keeps first occurrence).")
    parser.add_argument("--query_chunk_weight", type=str, default="uniform", choices=["uniform", "first"],
                       help="When query_full_chunks: uniform = all chunks equal; first = boost first chunk weight by query_first_chunk_weight.")
    parser.add_argument("--query_first_chunk_weight", type=float, default=1.5,
                       help="When query_chunk_weight=first: multiply first-chunk span weights by this. Compare with uniform to see which is better.")
    parser.add_argument("--idf_exponent", type=float, default=1.0,
                       help="Power applied to IDF in scoring: contrib uses idf^idf_exponent. "
                            "Default: 1.0. Try 0.5 (flatter), 1.5 or 2.0 (more discriminative).")

    parser.add_argument("--length_norm", type=str, default="none",
                       choices=["none", "sqrt_centers"],
                       help="Document length normalization for sparse_coverage. "
                            "none: no normalization. sqrt_centers: divide by num_centers_hit^length_norm_exponent. "
                            "Default: none.")
    parser.add_argument("--length_norm_exponent", type=float, default=0.5,
                       help="Exponent for length norm when length_norm=sqrt_centers: divide by n_centers^exponent. "
                            "0.5 => sqrt (default). 0.8 => stronger penalization of long docs.")
    parser.add_argument("--centers_suffix", type=str, default="",
                       help="Suffix appended to centers directory name for discovery. Required when centers were "
                            "built with a suffix: greedy (e.g. '_soft', '_percenter'), k-means (e.g. '_kmeans_V50000'), "
                            "k-center (e.g. '_kcenter_V25000'), or quantile (e.g. '_quantile'). Must match build script output.")
    parser.add_argument("--embeddings_dir", type=str, default=None,
                       help="Path to directory containing pre-computed doc-side embeddings and metadata. "
                            "Must contain {section}_{unit}.npy/.npz and {section}_{unit}_metadata.jsonl. "
                            "If unset, documents from the evaluation corpus (perf200/clefip) are encoded at runtime.")
    parser.add_argument("--spacy_model", type=str, default="sci_lg",
                       choices=["sm", "md", "lg", "sci_sm", "sci_md", "sci_lg"],
                       help="SpaCy model for span tokenization (sparse_coverage). Default: sci_lg.")
    parser.add_argument("--posting_list_batch_size", type=int, default=256,
                       help="For doc soft: batch size for range_search when building posting lists. "
                            "Larger = fewer FAISS calls, may use more memory. Default: 256.")

    parser.add_argument("--pca_proj_alpha", type=float, default=0.0,
                       help="Within-sphere: contrib = (q_sim*d_sim + alpha*q_proj*d_proj) * idf (approx sim(q,d)). "
                            "Requires center_pca_dirs.npy. 0 = off. Try 0.5 or 1.0.")
    parser.add_argument("--angle_sim_beta", type=float, default=0.0,
                       help="Within-sphere (idea 2): add beta*(1 - |q_sim - d_sim|)*idf to favor similar span–center angles. "
                            "0 = off. Try 0.1–0.5.")
    parser.add_argument("--residual_alpha", type=float, default=0.0,
                       help="Within-sphere (idea 1): add alpha*(q_res_proj*d_res_proj)*idf with res_proj=proj-sim*(c·u). "
                            "Requires center_pca_dirs.npy. 0 = off. Try 0.1–0.5.")
                            
    parser.add_argument("--save_rankings", type=str, default=None,
                       help="If set (directory path), save rankings for hybrid fusion: "
                            "rankings_abstract2abstract.json and rankings_claim2all.json. "
                            "Format: {query_id: [doc_id, ...]}. Use with dense or sparse_coverage runs.")
    parser.add_argument("--clefip_doc_root", type=str, default='./clefip2013/01_document_collection/01_extracted',
                       help="Path to extracted CLEF-IP 01 document collection. If set, official CLEF-IP EN evaluation runs after prior-art. If unset, CLEF-IP is skipped. Extract with: bash clefip2013/extract_01_collection.sh")

    args = parser.parse_args()
    save_rankings_abs, save_rankings_claim = _save_rankings_paths_from_args(args)

    print(f"Running evaluation for model: {args.model_name}")
    print("=============================================>>>>>>>>>")

    # Handle the case where model_name is None
    if args.model_name is None:
        print("Error: --model_name is required")
        return

    # Initialize temp directories for all models (sanitize model_name so HF IDs like org/repo do not create nested dirs)
    _temp_suffix = (args.model_name or "").replace("/", "_").strip("_") or "model"
    priorart_temp_dir = os.path.join(args.temp_dir, f'priorart_temp_{_temp_suffix}')
    
    # Create directories if they don't exist (for non-BM25 models)
    if not (args.model_name and "bm25" in args.model_name.lower()):
        for temp_dir in [priorart_temp_dir]:
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
                print(f"Created directory: {temp_dir}")

    # Print evaluation header
    print(f"📋 Model: {args.model_name}")
    print(f"📁 Output Directory: {args.temp_dir}")


    ############################################## create dataset for prior-art search ##################################################
    print("Running Prior-art search task.")
    Prior_art_dataset_dir = './patentmap_eval/data/downstream/perf200'

    queries = load_corpus(f"{Prior_art_dataset_dir}/content/queries.json")
    documents = load_corpus(f"{Prior_art_dataset_dir}/content/documents.json")

    # Convert dict_keys to lists so we can index them safely
    query_ids = list(queries.keys())       # e.g. ['Q1', 'Q2', 'Q3', ...]
    doc_ids = list(documents.keys())       # e.g. ['D1', 'D2', 'D3', ...]

    # convert to dataframe
    queries_df = pd.DataFrame(queries).T
    documents_df = pd.DataFrame(documents).T

    # 2) Load citation mappings (gold standard)
    citation_file = f"{Prior_art_dataset_dir}/mapping/gold.json"
    with open(citation_file) as f:
        raw_citations = json.load(f)

    # format: {query_id: [list_of_cited_doc_ids], ...}
    citation_mapping = citation_to_citing_to_cited_dict(raw_citations)
    
    # Multiply IDs to match concatenated embeddings
    original_query_count = len(query_ids)
    original_doc_count = len(doc_ids)
    
    query_ids = query_ids * 3
    doc_ids = doc_ids * 3
    
    # Create types to match the order of concatenated embeddings
    # Both query and document embeddings: [abstract1, abstract2, ..., claim1, claim2, ..., invention1, invention2, ...]
    query_types = ['abstract'] * original_query_count + ['claim'] * original_query_count + ['invention'] * original_query_count
    doc_types = ['abstract'] * original_doc_count + ['claim'] * original_doc_count + ['invention'] * original_doc_count


########################################################################################################################################################
########################################################################################################################################################
    # Set seed for reproducibility (even if not training, for deterministic results)
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Choose the model class based on model name or path
    if args.model_name.lower() in ["allenai/specter2_base", "patentbert"]:
        from adapters import AutoAdapterModel
        if args.model_name.lower() == "patentbert":
            model_path = "./PatentBert/encoder_only_model"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoAdapterModel.from_pretrained(model_path)
        else:
            # load the model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = AutoAdapterModel.from_pretrained(args.model_name)
            #load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
            model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
        embedding_dim = model.config.hidden_size
        model.to(device)

        def _compute_specter2_embeddings():
            query_embeddings_dict = {}
            doc_embeddings_dict = {}
            for texttype in ["abstract", "claim", "invention"]:
                if texttype == "abstract":
                    query_texts = [queries_df.iloc[i]['title'] + f" {tokenizer.sep_token} " + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                    doc_texts = [documents_df.iloc[i]['title'] + f" {tokenizer.sep_token} " + documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                else:
                    query_texts = [queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                    doc_texts = [documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                query_encodings = tokenizer(query_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
                doc_encodings = tokenizer(doc_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
                batch_size = 256
                query_embs = np.zeros((len(query_encodings['input_ids']), embedding_dim))
                doc_embs = np.zeros((len(doc_encodings['input_ids']), embedding_dim))
                with torch.no_grad():
                    for i in trange(0, len(query_encodings['input_ids']), batch_size, desc=f"Computing {texttype} query embeddings"):
                        batch = {key: val[i:i+batch_size].to(device) for key, val in query_encodings.items()}
                        outputs = model(**batch)
                        query_embs[i:i+batch_size] = outputs['last_hidden_state'][:, 0, :].detach().cpu().numpy()
                    for i in trange(0, len(doc_encodings['input_ids']), batch_size, desc=f"Computing {texttype} document embeddings"):
                        batch = {key: val[i:i+batch_size].to(device) for key, val in doc_encodings.items()}
                        outputs = model(**batch)
                        doc_embs[i:i+batch_size] = outputs['last_hidden_state'][:, 0, :].detach().cpu().numpy()
                query_embeddings_dict[texttype] = query_embs
                doc_embeddings_dict[texttype] = doc_embs
            q = np.concatenate([query_embeddings_dict["abstract"], query_embeddings_dict["claim"], query_embeddings_dict["invention"]], axis=0)
            d = np.concatenate([doc_embeddings_dict["abstract"], doc_embeddings_dict["claim"], doc_embeddings_dict["invention"]], axis=0)
            print(q.shape, d.shape)
            return q, d

        query_embeddings, document_embeddings = _load_or_compute_prior_art_embeddings(
            f'{priorart_temp_dir}/query_embeddings.pt',
            f'{priorart_temp_dir}/document_embeddings.pt',
            _compute_specter2_embeddings,
        )
        query_embeddings = query_embeddings.astype('float32')
        document_embeddings = document_embeddings.astype('float32')
        prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types, save_rankings_path=save_rankings_abs, save_rankings_claim2all_path=save_rankings_claim, model_label="Specter2")


########################################################################################################################################################
########################################################################################################################################################
    elif args.model_name.lower() == "mpi-inno-comp/paecter" or args.model_name.lower() == "anferico/bert-for-patents":
        # load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name)

        if args.model_name.lower() == "anferico/bert-for-patents":
            # add special tokens to the tokenizer
            tokenizer.add_special_tokens({'additional_special_tokens': ['[abstract]', '[claim]', '[invention]']})
            model.resize_token_embeddings(len(tokenizer))

        embedding_dim = model.config.hidden_size
        model.to(device)

        def _compute_paecter_embeddings():
            query_embeddings_dict = {}
            doc_embeddings_dict = {}
            for texttype in ["abstract", "claim", "invention"]:
                if args.model_name.lower() == "mpi-inno-comp/paecter":
                    if texttype == "abstract":
                        query_texts = [queries_df.iloc[i]['title'] + f" {tokenizer.sep_token} " + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                        doc_texts = [documents_df.iloc[i]['title'] + f" {tokenizer.sep_token} " + documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                    else:
                        query_texts = [queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                        doc_texts = [documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                else:
                    if texttype == "abstract":
                        query_texts = [queries_df.iloc[i]['title'] + f" [SEP] [{texttype}] " + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                        doc_texts = [documents_df.iloc[i]['title'] + f" [SEP] [{texttype}] " + documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                    else:
                        query_texts = [f"[{texttype}] " + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                        doc_texts = [f"[{texttype}] " + documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                query_encodings = tokenizer(query_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
                doc_encodings = tokenizer(doc_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
                batch_size = 256
                query_embs = np.zeros((len(query_encodings['input_ids']), embedding_dim))
                doc_embs = np.zeros((len(doc_encodings['input_ids']), embedding_dim))
                with torch.no_grad():
                    for i in trange(0, len(query_encodings['input_ids']), batch_size, desc=f"Computing {texttype} query embeddings"):
                        batch = {key: torch.tensor(val[i:i+batch_size]).to(device) for key, val in query_encodings.items()}
                        outputs = model(**batch)
                        query_embs[i:i+batch_size] = mean_pooling(outputs.last_hidden_state, query_encodings['attention_mask'][i:i+batch_size]).detach().cpu().numpy()
                    for i in trange(0, len(doc_encodings['input_ids']), batch_size, desc=f"Computing {texttype} document embeddings"):
                        batch = {key: torch.tensor(val[i:i+batch_size]).to(device) for key, val in doc_encodings.items()}
                        outputs = model(**batch)
                        doc_embs[i:i+batch_size] = mean_pooling(outputs.last_hidden_state, doc_encodings['attention_mask'][i:i+batch_size]).detach().cpu().numpy()
                query_embeddings_dict[texttype] = query_embs
                doc_embeddings_dict[texttype] = doc_embs
            q = np.concatenate([query_embeddings_dict["abstract"], query_embeddings_dict["claim"], query_embeddings_dict["invention"]], axis=0)
            d = np.concatenate([doc_embeddings_dict["abstract"], doc_embeddings_dict["claim"], doc_embeddings_dict["invention"]], axis=0)
            print(q.shape, d.shape)
            return q, d

        query_embeddings, document_embeddings = _load_or_compute_prior_art_embeddings(
            f'{priorart_temp_dir}/query_embeddings.pt',
            f'{priorart_temp_dir}/document_embeddings.pt',
            _compute_paecter_embeddings,
            pickle_protocol=4,
        )
        query_embeddings = query_embeddings.astype('float32')
        document_embeddings = document_embeddings.astype('float32')
        _paecter_label = "PAECTer" if "paecter" in args.model_name.lower() else "bert-for-patents"
        prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types, save_rankings_path=save_rankings_abs, save_rankings_claim2all_path=save_rankings_claim, model_label=_paecter_label)


########################################################################################################################################################
########################################################################################################################################################
    elif args.model_name.lower() in ["datalyes/patembed-large", "patembed-large"]:
        # Patembed-large: sentence-transformers bi-encoder (PatenTEB, arxiv 2510.22264)
        # Paper Sec 5.2 & Table 11: retrieval evaluation MUST use task-specific prompt prefixes;
        # Table 16 shows DAPFAM NDCG@100 0.377 with prompt vs 0.044 without.
        #
        # Model loads 16 prompts (model.prompts keys):
        #   Retrieval: retrieval_IN, retrieval_OUT, retrieval_MIXED, retrieval_inventor,
        #              title2full, problem2full, effect2full, effect2substance, problem2solution
        #   Paraphrase: para_problem, para_solution
        #   Classification: class_text2ipc3, class_bloom, class_nli_oldnew
        #   Clustering: clusters_ext_full_ipc, clusters_inventor
        # Usage: encode_query(texts, prompt_name="...") / encode_document(texts, prompt_name="...") use task prompts.
        # Prior-art: citations span same/mixed/different domains (unstratified) → use retrieval_MIXED (not IN/OUT).
        from sentence_transformers import SentenceTransformer

        actual_model_id = "datalyes/patembed-large"
        print(f"\n🔍 Loading Patembed (bi-encoder): {actual_model_id}")
        model = SentenceTransformer(actual_model_id)
        embedding_dim = model.get_sentence_embedding_dimension()
        model.to(device)

        # Use model's built-in retrieval_MIXED prompt (prior-art = unstratified, mixed domain)
        PATEN_TEB_RETRIEVAL_PROMPT_NAME = "retrieval_MIXED"
        print(f"   Using PatenTEB retrieval prompts: prompt_name={PATEN_TEB_RETRIEVAL_PROMPT_NAME} (required for best performance)")

        query_cache = os.path.join(priorart_temp_dir, "query_embeddings_prompted.pt")
        doc_cache = os.path.join(priorart_temp_dir, "document_embeddings_prompted.pt")

        def _compute_patembed_embeddings():
            query_embeddings_dict = {}
            doc_embeddings_dict = {}
            sep = getattr(model.tokenizer, 'sep_token', ' [SEP] ')
            for texttype in ["abstract", "claim", "invention"]:
                if texttype == "abstract":
                    raw_query = [queries_df.iloc[i]['title'] + sep + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                    raw_doc = [documents_df.iloc[i]['title'] + sep + documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                else:
                    raw_query = [queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                    raw_doc = [documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                try:
                    query_embs = model.encode_query(raw_query, prompt_name=PATEN_TEB_RETRIEVAL_PROMPT_NAME, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
                    doc_embs = model.encode_document(raw_doc, prompt_name=PATEN_TEB_RETRIEVAL_PROMPT_NAME, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
                except Exception:
                    PROMPT_QUERY = "encode query for mixed document retrieval: "
                    PROMPT_DOC = "encode document for mixed retrieval: "
                    query_embs = model.encode([PROMPT_QUERY + t for t in raw_query], batch_size=256, show_progress_bar=True, convert_to_numpy=True)
                    doc_embs = model.encode([PROMPT_DOC + t for t in raw_doc], batch_size=256, show_progress_bar=True, convert_to_numpy=True)
                query_embeddings_dict[texttype] = query_embs
                doc_embeddings_dict[texttype] = doc_embs
            q = np.concatenate([query_embeddings_dict["abstract"], query_embeddings_dict["claim"], query_embeddings_dict["invention"]], axis=0)
            d = np.concatenate([doc_embeddings_dict["abstract"], doc_embeddings_dict["claim"], doc_embeddings_dict["invention"]], axis=0)
            print(q.shape, d.shape)
            return q, d

        query_embeddings, document_embeddings = _load_or_compute_prior_art_embeddings(
            query_cache, doc_cache, _compute_patembed_embeddings, pickle_protocol=4,
        )
        query_embeddings = query_embeddings.astype('float32')
        document_embeddings = document_embeddings.astype('float32')
        prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types, save_rankings_path=save_rankings_abs, save_rankings_claim2all_path=save_rankings_claim, model_label="Patembed")


########################################################################################################################################################
########################################################################################################################################################
    elif args.model_name and args.model_name.lower() == "bm25":
        import bm25s
        import snowballstemmer

        ############################ BM25 Evaluation ############################
        print("Running BM25 (Standard) Prior-art search evaluation")
        
        stemmer = snowballstemmer.stemmer('english')
        original_doc_ids = list(documents.keys())
        
        # 1) Abstract-to-Abstract evaluation (like other models' abstract->abstract)
        print("\nBM25 Evaluation 1: Abstract-to-Abstract retrieval")
        abstract_train_corpus = documents_df['title'] + ' ' + documents_df['abstract']
        abstract_test_corpus = queries_df['title'] + ' ' + queries_df['abstract']
        
        # Tokenize corpus
        abstract_corpus_tokens = bm25s.tokenize(abstract_train_corpus.tolist(), stopwords="en", stemmer=stemmer)
        
        # Create and index BM25 model
        abstract_retriever = bm25s.BM25()
        abstract_retriever.index(abstract_corpus_tokens)
        # Tokenize queries then report efficiency (postings + FLOPs when query term ids available)
        abstract_queries_tokens = bm25s.tokenize(abstract_test_corpus.tolist(), stemmer=stemmer)
        _report_bm25_posting_stats(abstract_retriever, "Abstract-to-Abstract", query_tokens_list=abstract_queries_tokens)
        abstract_results, _ = abstract_retriever.retrieve(abstract_queries_tokens, k=100)
        
        # Map results back to document IDs (only abstract docs)
        abstract_retrieved_ids = [[original_doc_ids[i] for i in result] for result in abstract_results]
        
        # Calculate metrics for abstract-to-abstract
        query_ids_list = list(queries.keys())
        true_labels_list = [citation_mapping.get(q, []) for q in query_ids_list]
        
        bm25_abstract_results = _make_prior_art_metrics(true_labels_list, abstract_retrieved_ids)
        print_metric_table(bm25_abstract_results, "BM25: Abstract → Abstract")
        if save_rankings_abs:
            ranking_dict = {str(query_ids_list[i]): abstract_retrieved_ids[i] for i in range(len(abstract_retrieved_ids))}
            with open(save_rankings_abs, "w") as f:
                json.dump(ranking_dict, f, indent=0)
            print(f"   Saved abstract->abstract rankings to {save_rankings_abs} ({len(abstract_retrieved_ids)} queries)")
        
        # 2) Claim-to-All evaluation (like other models' claim->all)
        print("\nBM25 Evaluation 2: Claim-to-All retrieval")
        all_train_corpus = (
            (documents_df['title'] + ' ' + documents_df['abstract']).tolist() + 
            documents_df['claim'].tolist() + 
            documents_df['invention'].tolist()
        )
        claim_test_corpus = queries_df['claim'].tolist()
        
        all_corpus_tokens = bm25s.tokenize(all_train_corpus, stopwords="en", stemmer=stemmer)
        all_retriever = bm25s.BM25()
        all_retriever.index(all_corpus_tokens)
        claim_queries_tokens = bm25s.tokenize(claim_test_corpus, stemmer=stemmer)
        _report_bm25_posting_stats(all_retriever, "Claim-to-All", query_tokens_list=claim_queries_tokens)
        claim_results, _ = all_retriever.retrieve(claim_queries_tokens, k=300)
        
        original_doc_count = len(original_doc_ids)
        claim_retrieved_ids = []
        for result in claim_results:
            doc_ids_for_query = []
            for idx in result:
                if idx < original_doc_count:
                    doc_id = original_doc_ids[idx]
                elif idx < 2 * original_doc_count:
                    doc_id = original_doc_ids[idx - original_doc_count]
                else:
                    doc_id = original_doc_ids[idx - 2 * original_doc_count]
                doc_ids_for_query.append(doc_id)
            unique_doc_ids = list(dict.fromkeys(doc_ids_for_query))[:100]
            claim_retrieved_ids.append(unique_doc_ids)
        
        bm25_claim_results = _make_prior_art_metrics(true_labels_list, claim_retrieved_ids)
        print_metric_table(bm25_claim_results, "BM25: Claim → All Sections")
        if save_rankings_claim:
            ranking_dict = {str(query_ids_list[i]): claim_retrieved_ids[i] for i in range(len(claim_retrieved_ids))}
            with open(save_rankings_claim, "w") as f:
                json.dump(ranking_dict, f, indent=0)
            print(f"   Saved claim->all rankings to {save_rankings_claim} ({len(claim_retrieved_ids)} queries)")
        
        print("\n📝 Note: BM25 evaluation completed.")


########################################################################################################################################################
########################################################################################################################################################
    elif args.model_name.lower() in ["naver/splade-v2", "splade-v2", "naver/splade_v2_max", "naver/splade_v2_distil"]:
        """
        SPLADE-v2 Sparse Retrieval Model (arXiv:2109.10086)
        
        SPLADE produces sparse representations designed for "the efficiency of inverted indexes" (paper).
        This script builds a term->(doc_idx, weight) inverted index and performs term-at-a-time retrieval;
        posting list length is reported for Abstract-to-Abstract and Claim-to-All.
        """
        print(f"\n🔍 Loading SPLADE-v2 model: {args.model_name}")
        
        # Map common names to actual HuggingFace model IDs
        splade_model_map = {
            "splade-v2": "naver/splade-cocondenser-ensembledistil",
            "naver/splade-v2": "naver/splade-cocondenser-ensembledistil",
            "naver/splade_v2_max": "naver/splade_v2_max",
            "naver/splade_v2_distil": "naver/splade_v2_distil"
        }
        
        actual_model_name = splade_model_map.get(args.model_name.lower(), args.model_name)
        print(f"   Using model: {actual_model_name}")
        
        # Load SPLADE model using sentence_transformers SparseEncoder API
        # This is the recommended way to use SPLADE models
        from sentence_transformers import SparseEncoder
        
        model = SparseEncoder(actual_model_name)
        print(f"✅ SPLADE model loaded using SparseEncoder API")
        
        # Set batch size for encoding
        encode_batch_size = 32
        
        ############################ Prior-art Search evaluation ############################
        print("\n🔍 SPLADE-v2 Prior-art search evaluation")
        
        if os.path.exists(f'{priorart_temp_dir}/query_embeddings.npz') and os.path.exists(f'{priorart_temp_dir}/document_embeddings.npz'):
            print("📦 Loading precomputed SPLADE sparse embeddings...")
            from scipy.sparse import load_npz
            
            # Load sparse matrices from disk (scipy format)
            query_scipy = load_npz(f'{priorart_temp_dir}/query_embeddings.npz')
            document_scipy = load_npz(f'{priorart_temp_dir}/document_embeddings.npz')
            
            # Convert to PyTorch sparse tensors for use with model.similarity()
            def scipy_to_torch_sparse(scipy_matrix):
                """Convert scipy sparse matrix to PyTorch sparse tensor."""
                coo = scipy_matrix.tocoo()
                indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
                values = torch.from_numpy(coo.data).float()
                shape = coo.shape
                return torch.sparse_coo_tensor(indices, values, shape)
            
            query_embeddings_sparse = scipy_to_torch_sparse(query_scipy)
            document_embeddings_sparse = scipy_to_torch_sparse(document_scipy)
            
            print(f"   Query embeddings: {query_embeddings_sparse.shape}")
            print(f"   Document embeddings: {document_embeddings_sparse.shape}")
        else:
            print("🔄 Computing SPLADE sparse representations...")
            
            # Compute embeddings for each text type separately
            query_sparse_dict = {}
            doc_sparse_dict = {}
            
            for texttype in ["abstract", "claim", "invention"]:
                print(f"\n   Processing {texttype}...")
                
                # Format texts - SPLADE doesn't need special section tokens
                if texttype == "abstract":
                    query_texts = (queries_df['title'] + '. ' + queries_df['abstract']).fillna('').tolist()
                    doc_texts = (documents_df['title'] + '. ' + documents_df['abstract']).fillna('').tolist()
                else:
                    query_texts = queries_df[texttype].fillna('').tolist()
                    doc_texts = documents_df[texttype].fillna('').tolist()
                
                # Compute sparse embeddings
                print(f"      Computing query embeddings ({len(query_texts)} queries)...")
                query_sparse_dict[texttype] = model.encode_query(
                    query_texts, 
                    batch_size=encode_batch_size,
                    show_progress_bar=True
                )
                
                print(f"      Computing document embeddings ({len(doc_texts)} documents)...")
                doc_sparse_dict[texttype] = model.encode_document(
                    doc_texts, 
                    batch_size=encode_batch_size,
                    show_progress_bar=True
                )
                
                print(f"      ✓ {texttype}: Query shape {query_sparse_dict[texttype].shape}, Doc shape {doc_sparse_dict[texttype].shape}")
            
            # Stack PyTorch tensors vertically (concatenate different text types)
            # Keep in PyTorch format for use with model.similarity()
            query_embeddings_sparse = torch.cat([query_sparse_dict["abstract"], 
                                                 query_sparse_dict["claim"], 
                                                 query_sparse_dict["invention"]], dim=0)
            document_embeddings_sparse = torch.cat([doc_sparse_dict["abstract"], 
                                                    doc_sparse_dict["claim"], 
                                                    doc_sparse_dict["invention"]], dim=0)
            
            print(f"\n📊 Final SPLADE embeddings:")
            print(f"   Query embeddings: {query_embeddings_sparse.shape}")
            print(f"   Document embeddings: {document_embeddings_sparse.shape}")
            
            # Convert to scipy sparse format only for saving
            from scipy.sparse import save_npz, csr_matrix
            
            def torch_sparse_to_scipy(tensor):
                """Convert PyTorch sparse tensor to scipy sparse matrix."""
                if tensor.is_sparse:
                    tensor = tensor.coalesce()
                    indices = tensor.indices().cpu().numpy()
                    values = tensor.values().cpu().numpy()
                    shape = tensor.shape
                    from scipy.sparse import coo_matrix
                    return coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr()
                else:
                    # Dense tensor
                    return csr_matrix(tensor.cpu().numpy())
            
            # Save in scipy format for disk storage
            save_npz(f'{priorart_temp_dir}/query_embeddings.npz', torch_sparse_to_scipy(query_embeddings_sparse))
            save_npz(f'{priorart_temp_dir}/document_embeddings.npz', torch_sparse_to_scipy(document_embeddings_sparse))
            print(f"💾 Saved embeddings to {priorart_temp_dir}")

        print("\n🎯 Running Prior-art search evaluation...")
        
        def splade_prior_art_evaluation(query_ids, doc_ids, query_sparse, doc_sparse, 
                                        citation_mapping, query_types, doc_types, model):
            """SPLADE evaluation via inverted index (term-at-a-time retrieval)."""
            results = {}
            
            # Calculate original counts (before 3x multiplication)
            original_query_count = len(query_ids) // 3
            original_doc_count = len(doc_ids) // 3
            
            # Get original IDs (first segment before multiplication)
            original_query_ids = query_ids[:original_query_count]
            original_doc_ids = doc_ids[:original_doc_count]
            
            # Convert sparse tensors to dense for indexing (sparse tensors don't support boolean indexing)
            if query_sparse.is_sparse:
                query_dense = query_sparse.to_dense()
                doc_dense = doc_sparse.to_dense()
            else:
                query_dense = query_sparse
                doc_dense = doc_sparse
            
            # 1) Abstract-to-Abstract: inverted index (term -> doc postings), then term-at-a-time retrieval
            texttype_q = "abstract"
            texttype_d = "abstract"
            query_types_arr = np.array(query_types)
            doc_types_arr = np.array(doc_types)
            query_type_masks = (query_types_arr == texttype_q)
            doc_type_masks = (doc_types_arr == texttype_d)
            D_abs = doc_dense[doc_type_masks]
            Q_abs = query_dense[query_type_masks]
            D_abs, Q_abs = _to_numpy_if_torch(D_abs, Q_abs)
            vocab_size = D_abs.shape[1]
            posting_abs, _ = _splade_build_inverted_index(csr_matrix(D_abs), vocab_size)
            _report_splade_flops_and_postings(posting_abs, csr_matrix(Q_abs), "Abstract-to-Abstract")
            top_k_list_abs = _splade_retrieve_with_index(csr_matrix(Q_abs), posting_abs, top_k=100)

            # Build true/predicted labels using ORIGINAL IDs (abstract: doc index = original doc index)
            true_labels_list, predicted_labels_list = [], []
            for q_idx, retrieved_docs_indices in enumerate(top_k_list_abs):
                q_id_str = original_query_ids[q_idx]  # Use original query ID
                true_labels = citation_mapping.get(q_id_str, [])
                predicted_labels = [original_doc_ids[d_idx] for d_idx in retrieved_docs_indices]  # Use original doc IDs
                true_labels_list.append(true_labels)
                predicted_labels_list.append(predicted_labels)
            
            results_key = "abstract->abstract"
            results[results_key] = _make_prior_art_metrics(true_labels_list, predicted_labels_list)

            # 2) Claim-to-All: inverted index over all sections, term-at-a-time retrieval, then dedupe by doc
            texttype_q = "claim"
            query_type_masks = (query_types_arr == texttype_q)
            Q_claim = query_dense[query_type_masks]
            D_all = doc_dense  # All document sections (abstract, claim, invention)
            D_all, Q_claim = _to_numpy_if_torch(D_all, Q_claim)
            posting_all, _ = _splade_build_inverted_index(csr_matrix(D_all), D_all.shape[1])
            _report_splade_flops_and_postings(posting_all, csr_matrix(Q_claim), "Claim-to-All")
            top_k_list_claim = _splade_retrieve_with_index(csr_matrix(Q_claim), posting_all, top_k=300)
            # Pad to 300 with -1 so zero-result queries don't pollute; skip d_idx < 0 in loop
            top_k_indices = np.full((len(top_k_list_claim), 300), -1, dtype=np.int64)
            for i, t in enumerate(top_k_list_claim):
                n = min(len(t), 300)
                if n > 0:
                    top_k_indices[i, :n] = t[:n]

            retrieved_sections = []
            true_labels_list, predicted_labels_list = [], []

            for q_idx, retrieved_docs_indices in enumerate(top_k_indices):
                q_id_str = original_query_ids[q_idx]
                true_labels = citation_mapping.get(q_id_str, [])

                # Map (d_idx -> orig doc ID); skip padding -1
                doc_entries = []
                for d_idx in retrieved_docs_indices:
                    if d_idx < 0:
                        continue
                    orig_doc_idx = d_idx % original_doc_count
                    doc_entries.append((d_idx, original_doc_ids[orig_doc_idx]))
                # Dedupe by doc ID, preserve order; track d_idx for section
                seen = set()
                unique_predicted = []
                section_d_indices = []
                for d_idx, label in doc_entries:
                    if label not in seen:
                        seen.add(label)
                        unique_predicted.append(label)
                        section_d_indices.append(d_idx)
                predicted_labels = unique_predicted[:100]
                retrieved_sections.append([
                    ["abstract", "claim", "invention"][d_idx // original_doc_count]
                    for d_idx in section_d_indices[:100]
                ])
                
                true_labels_list.append(true_labels)
                predicted_labels_list.append(predicted_labels)
            
            results_key = f"{texttype_q}->all"
            results[results_key] = _make_prior_art_metrics(
                true_labels_list, predicted_labels_list,
                retrieved_sections=f"[{len(retrieved_sections)} queries with retrieved sections]"
            )

            # Format and display results
            print_subsection_header("Prior Art Search Results (SPLADE)")
            for task_key, task_results in results.items():
                if isinstance(task_results, dict):
                    if '->' in task_key:
                        clean_name = f"Query: {task_key.split('->')[0]} → Document: {task_key.split('->')[1]}"
                    else:
                        clean_name = task_key
                    print_metric_table(task_results, clean_name)
            
            # Store and analyze retrieved sections
            results[results_key]['retrieved_sections_full'] = retrieved_sections
            if retrieved_sections:
                from patentmap_eval.patenteval.utils import analyze_retrieved_sections_integrated
                section_analysis = analyze_retrieved_sections_integrated(
                    retrieved_sections, query_section=texttype_q, print_results=True
                )
                results[results_key]['section_analysis'] = section_analysis
        
        # Run SPLADE-specific evaluation
        splade_prior_art_evaluation(query_ids, doc_ids, query_embeddings_sparse, document_embeddings_sparse,
                                    citation_mapping, query_types, doc_types, model)
        
        print("\n✅ SPLADE-v2 evaluation completed!")

########################################################################################################################################################
########################################################################################################################################################
    elif "patentmap" in args.model_name.lower():
        def _is_hf_model_id(name: str) -> bool:
            """True if name looks like a Hugging Face model ID (org/repo) and is not an existing local path."""
            if not name or "/" not in name:
                return False
            if os.path.exists(name) and os.path.isdir(name):
                return False
            return True

        model_name_or_path = args.model_name.strip()
        if not _is_hf_model_id(model_name_or_path):
            raise ValueError(
                f"PatentMap/checkpoint models must be loaded from Hugging Face. "
                f"Use a model ID like ZoeYou/PatentMap-V0-Dropout (not a local path). Got: {model_name_or_path!r}"
            )
        print(f"🔄 Loading from Hugging Face: {model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        embedding_dim = model.config.hidden_size
        print(f"✅ Loaded tokenizer ({len(tokenizer)} tokens) and model (dim={embedding_dim})")

        batch_size = 512
        # device already set at start of main()
        # Setup model for inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.to(device).eval()
        print(f"🚀 Model ready on {device}")

        ############################ Prior-art Search evaluation ############################
        if not os.path.exists(priorart_temp_dir):
            os.makedirs(priorart_temp_dir)
        if os.path.exists(f'{priorart_temp_dir}/query_embeddings.pt') and os.path.exists(f'{priorart_temp_dir}/document_embeddings.pt'):
            print("Embeddings already created!")
            query_embeddings = torch.load(f'{priorart_temp_dir}/query_embeddings.pt', weights_only=False)
            document_embeddings = torch.load(f'{priorart_temp_dir}/document_embeddings.pt', weights_only=False)
        else:
            # Use EXACT same approach as patent.py: compute embeddings by text type separately
            # This ensures complete consistency when evaluating checkpoint models
            query_embeddings_dict = {}
            doc_embeddings_dict = {}
            
            # Process each text type separately, exactly like patent.py
            for texttype in ["abstract", "claim", "invention"]:
                # Format texts exactly like patent.py
                if texttype == "abstract":
                    query_texts = [queries_df.iloc[i]['title'] + f" [SEP] [{texttype}] " + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                    doc_texts = [documents_df.iloc[i]['title'] + f" [SEP] [{texttype}] " + documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                else:
                    query_texts = [f"[{texttype}] " + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                    doc_texts = [f"[{texttype}] " + documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                
                # Tokenize and compute embeddings for this text type
                query_encodings = tokenizer(query_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
                doc_encodings = tokenizer(doc_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
                
                # Compute embeddings
                query_embs = np.zeros((len(query_encodings['input_ids']), embedding_dim))
                doc_embs = np.zeros((len(doc_encodings['input_ids']), embedding_dim))
                
                def _get_embeddings(batch):
                    """Pooler output (BertForCL) or CLS token (standard Bert); supports HF-loaded PatentMap models."""
                    try:
                        out = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                        return out.pooler_output
                    except TypeError:
                        out = model(**batch, output_hidden_states=True, return_dict=True)
                        return out.last_hidden_state[:, 0]

                with torch.no_grad():
                    for i in trange(0, len(query_encodings['input_ids']), batch_size, desc=f"Computing {texttype} query embeddings"):
                        batch = {key: val[i:i+batch_size].to(device) for key, val in query_encodings.items()}
                        query_embs[i:i+batch_size] = _get_embeddings(batch).detach().cpu().numpy()
                    for i in trange(0, len(doc_encodings['input_ids']), batch_size, desc=f"Computing {texttype} document embeddings"):
                        batch = {key: val[i:i+batch_size].to(device) for key, val in doc_encodings.items()}
                        doc_embs[i:i+batch_size] = _get_embeddings(batch).detach().cpu().numpy()
                
                # Store embeddings by text type
                query_embeddings_dict[texttype] = query_embs
                doc_embeddings_dict[texttype] = doc_embs
            
            # For compatibility with existing evaluation code, we'll create the concatenated versions
            # But the evaluation should use the separated versions to match patent.py exactly
            query_embeddings = np.concatenate([query_embeddings_dict["abstract"], query_embeddings_dict["claim"], query_embeddings_dict["invention"]], axis=0)
            document_embeddings = np.concatenate([doc_embeddings_dict["abstract"], doc_embeddings_dict["claim"], doc_embeddings_dict["invention"]], axis=0)

            print(query_embeddings.shape, document_embeddings.shape)

            torch.save(query_embeddings, f'{priorart_temp_dir}/query_embeddings.pt')
            torch.save(document_embeddings, f'{priorart_temp_dir}/document_embeddings.pt')
            np.savez(f'{priorart_temp_dir}/query_embeddings_by_type.npz', **query_embeddings_dict)
            np.savez(f'{priorart_temp_dir}/doc_embeddings_by_type.npz', **doc_embeddings_dict)

        query_embeddings = query_embeddings.astype('float32')
        document_embeddings = document_embeddings.astype('float32')

        # For checkpoint models, use the exact same evaluation method as patent.py to ensure consistency
        print("Using patent.py-compatible evaluation for checkpoint model...")
        print("This ensures exact consistency with training-time evaluation results.")
        
        # Use the standard evaluation for now, but note that minor differences may exist
        # due to different data organization methods between evaluate.py and patent.py
        prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types, save_rankings_path=save_rankings_abs, save_rankings_claim2all_path=save_rankings_claim, model_label="PatentMap")


########################################################################################################################################################
########################################################################################################################################################
    elif args.model_name == "sparse_coverage":
        """
        Sparse Coverage Retrieval
        
        Uses pre-built vocabulary centers for sparse retrieval.
        Documents from the evaluation corpus are encoded at runtime (or loaded from cache/--embeddings_dir).
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
        print(f"   Layer: {getattr(args, 'layer', 'last')}")
        print(f"   Length norm: {getattr(args, 'length_norm', 'none')}" + (f" (exponent={getattr(args, 'length_norm_exponent', 0.5)})" if getattr(args, 'length_norm', 'none') == 'sqrt_centers' else ""))
        
        import glob
        
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
        
        # Find centers and posting lists files
        # Search in current directory (recursive search is handled by find_centers)
        print(f"\n🔍 Searching for centers...")
        print(f"   Search directory: {os.path.abspath('.')}")
        
        # Find all available centers for both modes
        available_modes = []
        centers_info_dict = {}  # mode -> (centers_path, centers_dir)
        
        for test_mode in ["abstract2abstract", "claim2all"]:
            try:
                c_path, c_dir = find_centers(
                    dense_model=args.dense_model,
                    tokenization_unit=args.tokenization_unit,
                    include_cls=include_cls,
                    search_dir=".",
                    mode=test_mode,
                    layer=getattr(args, 'layer', 'last'),
                    centers_suffix=getattr(args, 'centers_suffix', ''),
                )
                centers_info_dict[test_mode] = (c_path, c_dir)
                available_modes.append(test_mode)
                print(f"✅ Found centers for mode: {test_mode}")
                print(f"   Centers: {c_path}")
            except FileNotFoundError as e:
                layer = getattr(args, 'layer', 'last')
                print(f"⚠️  No centers found for mode: {test_mode} (layer: {layer})")
                continue
        
        if not available_modes:
            layer = getattr(args, 'layer', 'last')
            raise FileNotFoundError(
                f"Could not find centers for any mode:\n"
                f"  dense_model={args.dense_model}\n"
                f"  tokenization_unit={args.tokenization_unit}\n"
                f"  (eval assumes centers built with CLS; layer={layer})\n"
                f"Searched in: {os.path.abspath('.')} (recursive)\n"
                f"Expected pattern: centers_greedy_{{mode}}_{{model}}_{{unit}}_{{cls}}_{{layer}}\n"
                f"Please ensure centers were built with matching parameters, or use --layer last if you have centers built with 'last' layer."
            )
        
        print(f"\n📋 Will process {len(available_modes)} task(s): {', '.join(available_modes)}")
        
        # -----------------------------
        # Correct per-mode evaluation
        # -----------------------------
        # Historically, the legacy code path below was accidentally de-indented, which caused
        # only the last mode (usually claim2all) to actually run and print metrics.
        #
        # We implement a clean per-mode loop here and return early. The legacy block is left
        # in place (but unreachable) to minimize churn.
        
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
        
        def _embeddings_dir_name() -> str:
            if getattr(args, "embeddings_dir", None):
                return os.path.abspath(args.embeddings_dir)
            model_name_clean = args.dense_model.strip("/").split("/")[-1].replace("/", "_").replace("\\", "_")
            cls_suffix = "cls" if include_cls else "nocls"
            layer = getattr(args, "layer", "last")
            return f"embeddings_{model_name_clean}_{args.tokenization_unit}_{cls_suffix}_{layer}"
        
        def _metadata_path(section_name: str) -> str:
            return os.path.join(_embeddings_dir_name(), f"{section_name}_{args.tokenization_unit}_metadata.jsonl")
        
        def _embeddings_path(section_name: str) -> str:
            # Prefer .npy, fallback to .npz
            p_npy = os.path.join(_embeddings_dir_name(), f"{section_name}_{args.tokenization_unit}.npy")
            if os.path.exists(p_npy):
                return p_npy
            p_npz = os.path.join(_embeddings_dir_name(), f"{section_name}_{args.tokenization_unit}.npz")
            if os.path.exists(p_npz):
                return p_npz
            raise FileNotFoundError(f"Could not find embeddings for section={section_name} in {_embeddings_dir_name()}")
        
        def _load_embeddings(file_path: str) -> np.ndarray:
            if file_path.endswith(".npz"):
                with np.load(file_path) as data:
                    keys = list(data.keys())
                    if not keys:
                        raise ValueError(f"Empty .npz file: {file_path}")
                    return data[keys[0]].astype(np.float32)
            return np.load(file_path).astype(np.float32)
        
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
                    results = process_doc_batch(
                        doc_texts=sec_texts[b_start:b_end],
                        doc_ids=sec_ids[b_start:b_end],
                        sections=sec_sections[b_start:b_end],
                        unit=args.tokenization_unit,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        max_length=512,
                        keep_cls=include_cls,
                        span_pooling="mean",
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
            return os.path.join(priorart_temp_dir, f"sparse_doc_{model_clean}_{args.tokenization_unit}_{layer}_{mode}")
        
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
                arr = np.load(p).astype(np.float32)
                embeddings_by_section[sec] = arr
                total_loaded += arr.shape[0]

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
            return embeddings_by_section, span_to_doc, exclude_cls_indices, total_loaded
        
        def _encode_query_spans(texts: list[str], section: str, d: int, batch_size: int = 32) -> list[np.ndarray]:
            all_query_spans: list[np.ndarray] = []
            doc_ids = [f"query_{i}" for i in range(len(texts))]
            sections = [section for _ in range(len(texts))]
            
            for batch_start in range(0, len(texts), batch_size):
                batch_end = min(batch_start + batch_size, len(texts))
                batch_texts = texts[batch_start:batch_end]
                batch_sections = sections[batch_start:batch_end]
                batch_doc_ids = doc_ids[batch_start:batch_end]
                
                batch_results = process_doc_batch(
                    doc_texts=batch_texts,
                    doc_ids=batch_doc_ids,
                    sections=batch_sections,
                    unit=args.tokenization_unit,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=512,
                    keep_cls=include_cls,
                    span_pooling="mean",
                )
                
                query_exclude_cls = getattr(args, "exclude_cls_spans", False)
                query_spans_dict: dict[int, list[np.ndarray]] = {}
                for doc_id, _section, _doc_text, _span_text_raw, _span_text_canonical, span_emb in batch_results:
                    if query_exclude_cls:
                        t = (_span_text_canonical or "").strip().lower()
                        r = (_span_text_raw or "").strip()
                        if t == "cls" or r == "[CLS]":
                            continue
                    q_idx = int(doc_id.split("_")[1])
                    query_spans_dict.setdefault(q_idx, []).append(span_emb)
                
                for q_idx in range(batch_start, batch_end):
                    spans = query_spans_dict.get(q_idx, [])
                    if spans:
                        all_query_spans.append(np.stack(spans))
                    else:
                        all_query_spans.append(np.zeros((0, d), dtype=np.float32))
            
            return all_query_spans
        
        def _encode_query_spans_chunked(
            texts: list[str],
            section: str,
            d: int,
            batch_size: int = 32,
        ) -> tuple[list[np.ndarray], Optional[list[np.ndarray]]]:
            """Encode full query by chunking (no truncation). Span dedup: keep first occurrence by global token index. Returns (query_spans, query_span_weights or None)."""
            stride_ratio = float(getattr(args, "query_chunk_stride_ratio", 1.0))
            chunk_weight_mode = getattr(args, "query_chunk_weight", "uniform")
            first_chunk_weight = float(getattr(args, "query_first_chunk_weight", 1.5))
            chunk_meta: dict[tuple[int, int], int] = {}  # (qidx, chunk_idx) -> offset_base
            chunk_list: list[tuple[str, int, int, int]] = []  # (chunk_text, qidx, chunk_idx, offset_base)
            for qidx, full_text in enumerate(texts):
                chunks = utils.chunk_query_text(
                    full_text,
                    tokenizer,
                    max_length=512,
                    title_prefix_max=64,
                    stride_ratio=stride_ratio,
                )
                for cidx, (c_text, offset_base) in enumerate(chunks):
                    chunk_meta[(qidx, cidx)] = offset_base
                    chunk_list.append((c_text, qidx, cidx, offset_base))
            if not chunk_list:
                return [], None
            batch_texts = [c[0] for c in chunk_list]
            batch_doc_ids = [f"query_{c[1]}_{c[2]}" for c in chunk_list]
            batch_sections = [section for _ in chunk_list]
            query_exclude_cls = getattr(args, "exclude_cls_spans", False)
            # Encode in batches
            all_batch_results: list[tuple] = []
            for b_start in range(0, len(batch_texts), batch_size):
                b_end = min(b_start + batch_size, len(batch_texts))
                br = process_doc_batch(
                    doc_texts=batch_texts[b_start:b_end],
                    doc_ids=batch_doc_ids[b_start:b_end],
                    sections=batch_sections[b_start:b_end],
                    unit=args.tokenization_unit,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=512,
                    keep_cls=include_cls,
                    span_pooling="mean",
                )
                all_batch_results.extend(br)
            # Group spans by (doc_id) to get (qidx, chunk_idx) and local span order
            from collections import defaultdict
            doc_id_to_spans: dict[str, list[np.ndarray]] = defaultdict(list)
            for doc_id, _sec, _dt, _raw, _canon, span_emb in all_batch_results:
                if query_exclude_cls:
                    if (_canon or "").strip().lower() == "cls" or (_raw or "").strip() == "[CLS]":
                        continue
                doc_id_to_spans[doc_id].append(span_emb)
            # Build per-query: (global_index, span_emb, from_first_chunk)
            num_queries = len(texts)
            per_query: dict[int, list[tuple[int, np.ndarray, bool]]] = defaultdict(list)
            for doc_id, span_list in doc_id_to_spans.items():
                parts = doc_id.split("_")
                if len(parts) != 3 or parts[0] != "query":
                    continue
                qidx = int(parts[1])
                cidx = int(parts[2])
                offset_base = chunk_meta.get((qidx, cidx), 0)
                for local_idx, span_emb in enumerate(span_list):
                    global_index = offset_base + local_idx
                    per_query[qidx].append((global_index, span_emb, cidx == 0))
            # Dedup: keep first occurrence per (qidx, global_index); then sort by global_index
            all_query_spans = []
            all_query_weights: Optional[list[np.ndarray]] = None if chunk_weight_mode == "uniform" else []
            for qidx in range(num_queries):
                seen: set[int] = set()
                kept: list[tuple[int, np.ndarray, bool]] = []
                for global_index, span_emb, from_first in sorted(per_query.get(qidx, []), key=lambda x: (x[0], -x[2])):  # same global: prefer first chunk (from_first True first)
                    if global_index in seen:
                        continue
                    seen.add(global_index)
                    kept.append((global_index, span_emb, from_first))
                kept.sort(key=lambda x: x[0])
                if not kept:
                    all_query_spans.append(np.zeros((0, d), dtype=np.float32))
                    if all_query_weights is not None:
                        all_query_weights.append(np.array([], dtype=np.float32))
                else:
                    spans_arr = np.stack([x[1] for x in kept], axis=0).astype(np.float32)
                    all_query_spans.append(spans_arr)
                    if chunk_weight_mode == "first" and all_query_weights is not None:
                        w = np.ones(len(kept), dtype=np.float32)
                        for i, (_, _, from_first) in enumerate(kept):
                            if from_first:
                                w[i] = first_chunk_weight
                        all_query_weights.append(w)
            return all_query_spans, all_query_weights
        
        def _assign_query_spans_to_centers(
            query_spans: list[np.ndarray],
            center_index: faiss.Index,
            V: int,
            sim_threshold: float,
            idf: Optional[np.ndarray] = None,
            center_pca_dirs: Optional[np.ndarray] = None,
            query_span_weights: Optional[list[np.ndarray]] = None,
            stop_centers: Optional[set] = None,
        ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
            use_soft_assignment = args.use_soft_assignment if hasattr(args, "use_soft_assignment") else False
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
                center_projs: Optional[dict] = None, center_max_sim: Optional[dict] = None, proj: float = 0.0,
            ) -> None:
                w = _to_weight(sim, span_idx, span_downweight, span_extra_weight)
                if w <= 0:
                    return
                if weight_agg == "sum":
                    weights[key] = weights.get(key, 0.0) + w
                    if center_projs is not None and center_max_sim is not None and sim > center_max_sim.get(key, -1.0):
                        center_max_sim[key] = sim
                        center_projs[key] = proj
                else:
                    if w > weights.get(key, 0.0):
                        weights[key] = w
                        if center_projs is not None:
                            center_projs[key] = proj

            query_sparse: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for q_idx, spans in enumerate(query_spans):
                if spans.shape[0] == 0:
                    query_sparse.append((np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)))
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

                if use_soft_assignment and sim_threshold is not None:
                    max_centers_per_span = getattr(args, "soft_assignment_max_centers_per_span", None)
                    lims, D, I = center_index.range_search(spans_norm, sim_threshold)
                    center_weights = {}
                    center_projs: dict[int, float] = {}
                    center_max_sim: dict[int, float] = {} if weight_agg == "sum" else None
                    for span_idx in range(spans_norm.shape[0]):
                        start, end = int(lims[span_idx]), int(lims[span_idx + 1])
                        pairs = [(int(I[j]), float(D[j])) for j in range(start, end) if float(D[j]) > 0]
                        if max_centers_per_span is not None and max_centers_per_span > 0 and len(pairs) > max_centers_per_span:
                            pairs = sorted(pairs, key=lambda x: -x[1])[:max_centers_per_span]
                        extra = _span_extra(q_idx, span_idx)
                        for center_id, sim in pairs:
                            if center_id in _stop:
                                continue
                            proj = float(spans_norm[span_idx] @ center_pca_dirs[center_id]) if center_pca_dirs is not None else 0.0
                            _update_weight(center_weights, center_id, sim, span_idx, span_downweight=1.0, span_extra_weight=extra, center_projs=center_projs, center_max_sim=center_max_sim, proj=proj)
                    if not center_weights:
                        similarities, assigned = center_index.search(spans_norm, k=1)
                        for span_idx in range(similarities.shape[0]):
                            center_id = int(assigned[span_idx, 0])
                            if center_id in _stop:
                                continue
                            sim = float(similarities[span_idx, 0])
                            proj = float(spans_norm[span_idx] @ center_pca_dirs[center_id]) if center_pca_dirs is not None else 0.0
                            extra = _span_extra(q_idx, span_idx)
                            _update_weight(center_weights, center_id, sim, span_idx, span_downweight=1.0, span_extra_weight=extra, center_projs=center_projs, center_max_sim=center_max_sim, proj=proj)

                    if center_weights:
                        centers_arr = np.array(list(center_weights.keys()), dtype=np.int32)
                        weights_arr = np.array([center_weights[c] for c in centers_arr], dtype=np.float32)
                        projs_arr = np.array([center_projs.get(c, 0.0) for c in centers_arr], dtype=np.float32)
                        query_sparse.append((centers_arr, weights_arr, projs_arr))
                    else:
                        query_sparse.append((np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)))
                else:
                    similarities, assigned = center_index.search(spans_norm, k=1)
                    center_weights = {}
                    center_projs = {}
                    center_max_sim = {} if weight_agg == "sum" else None
                    for span_idx in range(similarities.shape[0]):
                        center_id = int(assigned[span_idx, 0])
                        if center_id in _stop:
                            continue
                        sim = float(similarities[span_idx, 0])
                        proj = float(spans_norm[span_idx] @ center_pca_dirs[center_id]) if center_pca_dirs is not None else 0.0
                        extra = _span_extra(q_idx, span_idx)
                        _update_weight(center_weights, center_id, sim, span_idx, span_downweight=1.0, span_extra_weight=extra, center_projs=center_projs, center_max_sim=center_max_sim, proj=proj)

                    centers_arr = np.array(list(center_weights.keys()), dtype=np.int32)
                    weights_arr = np.array([center_weights[c] for c in centers_arr], dtype=np.float32)
                    projs_arr = np.array([center_projs.get(c, 0.0) for c in centers_arr], dtype=np.float32)
                    query_sparse.append((centers_arr, weights_arr, projs_arr))

            return query_sparse
        
        for mode in available_modes:
            print(f"\n{'='*80}")
            print(f"Processing task: {mode}")
            print(f"{'='*80}")
            
            centers_path, _centers_dir = centers_info_dict[mode]
            print(f"\n📦 Loading centers...")
            centers = np.load(centers_path).astype(np.float32)
            V_original, d = centers.shape
            print(f"   Original vocabulary size: {V_original:,} centers")
            print(f"   Embedding dimension: {d}")
            
            centers_info = _load_centers_info_json(centers_path)
            V = int(centers.shape[0])
            stop_centers = set(centers_info.get("stop_centers", []))
            if stop_centers:
                print(f"   stop_centers: {len(stop_centers)} disabled for activation (df >= threshold)")
            print(f"   Final vocabulary size: {V:,} centers")
            
            pca_proj_alpha = float(getattr(args, "pca_proj_alpha", 0.0))
            residual_alpha = float(getattr(args, "residual_alpha", 0.0))
            center_pca_dirs: Optional[np.ndarray] = None
            if pca_proj_alpha != 0.0 or residual_alpha != 0.0:
                pca_path = os.path.join(os.path.dirname(centers_path), "center_pca_dirs.npy")
                if os.path.exists(pca_path):
                    center_pca_dirs = np.load(pca_path).astype(np.float32)
                    if center_pca_dirs.shape != (V, d):
                        print(f"   ⚠️  center_pca_dirs.npy shape {center_pca_dirs.shape} != (V,d)=({V},{d}), disabling PCA/residual")
                        center_pca_dirs = None
                    else:
                        if pca_proj_alpha != 0.0:
                            print(f"   PCA proj: alpha={pca_proj_alpha} (center_pca_dirs loaded)")
                        if residual_alpha != 0.0:
                            print(f"   Residual term: alpha={residual_alpha} (center_pca_dirs loaded)")
                else:
                    if pca_proj_alpha != 0.0:
                        print(f"   ⚠️  --pca_proj_alpha={pca_proj_alpha} but {pca_path} not found, disabling")
                    if residual_alpha != 0.0:
                        print(f"   ⚠️  --residual_alpha={residual_alpha} but {pca_path} not found, disabling")
                    center_pca_dirs = None
            else:
                center_pca_dirs = None
            
            r, sim_threshold = _get_r_and_sim_threshold(centers_info)
            
            if model.config.hidden_size != d:
                raise ValueError(f"Dimension mismatch: model hidden_size={model.config.hidden_size} but centers dimension={d}")
            
            print(f"\n🔨 Building FAISS index on centers...")
            centers_norm = centers.copy()
            faiss.normalize_L2(centers_norm)
            center_index = faiss.IndexFlatIP(d)
            center_index.add(centers_norm.astype(np.float32))
            print(f"✅ Center index built")
            center_dot_pca: Optional[np.ndarray] = None
            if center_pca_dirs is not None and residual_alpha != 0.0:
                center_dot_pca = np.array([float(centers_norm[c] @ center_pca_dirs[c]) for c in range(V)], dtype=np.float32)

            # Decide which sections to use for document indexing
            if mode == "abstract2abstract":
                doc_sections = ["abstract"]
                query_section = "abstract"
            elif mode == "claim2all":
                doc_sections = ["abstract", "claim", "invention"]
                query_section = "claim"
            else:
                doc_sections = ["abstract"]
                query_section = "abstract"
            
            # ---- Obtain doc-side embeddings + span_to_doc mapping ----
            # Priority: 1) --embeddings_dir (pre-computed), 2) cached runtime embeddings, 3) encode at runtime
            exclude_cls_spans = getattr(args, "exclude_cls_spans", False)
            embeddings_by_section: dict[str, np.ndarray] = {}
            span_to_doc: dict[int, str] = {}
            exclude_cls_span_indices: set[int] = set()
            total_loaded = 0

            use_precomputed = False
            if getattr(args, "embeddings_dir", None):
                try:
                    _emb_dir = os.path.abspath(args.embeddings_dir)
                    for sec in doc_sections:
                        _embeddings_path(sec)  # will raise if missing
                    use_precomputed = True
                except FileNotFoundError:
                    print(f"   --embeddings_dir set but files missing; falling back to runtime encoding")

            if use_precomputed:
                _emb_dir = os.path.abspath(args.embeddings_dir)
                print(f"\n📦 Loading pre-computed doc embeddings from: {_emb_dir}")
                current_span_idx = 0

                def _is_cls_span(meta: dict) -> bool:
                    t = (meta.get("t") or "").strip().lower()
                    r = (meta.get("r") or "").strip()
                    return t == "cls" or r == "[CLS]"

                for section_name in doc_sections:
                    ep = _embeddings_path(section_name)
                    sec_emb = _load_embeddings(ep)
                    embeddings_by_section[section_name] = sec_emb
                    total_loaded += sec_emb.shape[0]
                    mp = _metadata_path(section_name)
                    if os.path.exists(mp):
                        with open(mp, "r") as f:
                            for line in f:
                                if not line.strip():
                                    continue
                                meta = json.loads(line)
                                doc_id = meta.get("d", meta.get("doc_id", ""))
                                span_to_doc[current_span_idx] = doc_id
                                if exclude_cls_spans and _is_cls_span(meta):
                                    exclude_cls_span_indices.add(current_span_idx)
                                current_span_idx += 1
                    else:
                        raise FileNotFoundError(f"Metadata file not found alongside embeddings: {mp}")
                    print(f"   {section_name}: {sec_emb.shape[0]:,} spans")
            else:
                cache_dir = _doc_cache_dir(mode)
                try:
                    print(f"\n📦 Trying cached doc embeddings from: {cache_dir}")
                    embeddings_by_section, span_to_doc, exclude_cls_span_indices, total_loaded = _load_doc_cache(cache_dir, doc_sections)
                    print(f"   Loaded {total_loaded:,} spans from cache")
                    for sec in doc_sections:
                        print(f"   {sec}: {embeddings_by_section[sec].shape[0]:,} spans")
                except FileNotFoundError:
                    print(f"\n📦 Encoding documents at runtime (first run; will cache for reuse)")
                    embeddings_by_section, span_to_doc, exclude_cls_span_indices = _encode_doc_spans(
                        documents, doc_sections, batch_size=32
                    )
                    total_loaded = sum(e.shape[0] for e in embeddings_by_section.values())
                    _save_doc_cache(cache_dir, doc_sections, embeddings_by_section, span_to_doc, exclude_cls_span_indices)

            if exclude_cls_spans:
                print(f"   Excluding {len(exclude_cls_span_indices):,} CLS spans")
            print(f"   Total: {len(span_to_doc):,} span-to-doc mappings, {total_loaded:,} embedding rows")

            # Build doc_span_count for length normalization
            doc_span_count: dict[str, int] = {}
            for span_idx, doc_id in span_to_doc.items():
                if exclude_cls_spans and span_idx in exclude_cls_span_indices:
                    continue
                doc_span_count[doc_id] = doc_span_count.get(doc_id, 0) + 1
            
            # ---- Build posting lists ----
            document_assignment = getattr(args, "document_assignment", "soft")
            print(f"\n🔨 Computing posting lists for {V:,} centers...")
            print(f"   Document assignment: {document_assignment}")
            
            posting_lists: list[list[tuple[int, float]]] = []
            centers_norm_for_pl = centers.copy()
            faiss.normalize_L2(centers_norm_for_pl)
            
            if document_assignment == "hard":
                print(f"   Building posting lists: each span -> nearest center (k=1)")
                posting_lists = [[] for _ in range(V)]
                span_offset = 0
                for section_name in doc_sections:
                    sec_emb = embeddings_by_section[section_name]
                    if sec_emb.shape[0] == 0:
                        continue
                    if sec_emb.shape[1] != d:
                        raise ValueError(f"Embedding dimension mismatch for {section_name}: {sec_emb.shape[1]} != {d}")
                    sec_emb = sec_emb.astype(np.float32).copy()
                    faiss.normalize_L2(sec_emb)
                    sims, assigned = center_index.search(sec_emb, 1)
                    for j in range(sec_emb.shape[0]):
                        global_idx = span_offset + j
                        if exclude_cls_spans and global_idx in exclude_cls_span_indices:
                            continue
                        c = int(assigned[j, 0])
                        if c in stop_centers:
                            continue
                        sim = float(sims[j, 0])
                        if sim > 0:
                            proj = float(sec_emb[j] @ center_pca_dirs[c]) if center_pca_dirs is not None else 0.0
                            posting_lists[c].append((global_idx, sim, proj))
                    span_offset += sec_emb.shape[0]
                    print(f"     {section_name}: {sec_emb.shape[0]:,} spans assigned")
                print(f"   Total spans: {total_loaded:,}")
            else:
                # Doc soft: sphere (range_search) per center
                print(f"   Using radius r={r:.6f} for range search")
                idx_size = 0
                embeddings_index = faiss.IndexFlatIP(d)
                for section_name in doc_sections:
                    sec_emb = embeddings_by_section[section_name]
                    if sec_emb.shape[0] == 0:
                        continue
                    if sec_emb.shape[1] != d:
                        raise ValueError(f"Embedding dimension mismatch for {section_name}: {sec_emb.shape[1]} != {d}")
                    sec_emb_n = sec_emb.astype(np.float32).copy()
                    faiss.normalize_L2(sec_emb_n)
                    embeddings_index.add(sec_emb_n)
                    idx_size += sec_emb_n.shape[0]
                    print(f"     {section_name}: {sec_emb_n.shape[0]:,} (index size: {idx_size:,})")
                    del sec_emb_n
                print(f"   Total in index: {idx_size:,}")
                r_per_center = centers_info.get("r_per_center", None)
                if r_per_center is not None:
                    if len(r_per_center) < V:
                        r_per_center = None
                    else:
                        if len(r_per_center) > V:
                            r_per_center = r_per_center[:V]
                        print(f"   Using per-center radius (r_per_center from centers JSON)")
                sim_thresh_default = 1.0 - r
                batch_size = max(1, int(getattr(args, "posting_list_batch_size", 256)))
                print(f"   Batch size for range_search: {batch_size}")
                for batch_start in tqdm(range(0, V, batch_size), desc="Computing posting lists"):
                    batch_end = min(batch_start + batch_size, V)
                    centers_batch = centers_norm_for_pl[batch_start:batch_end].astype(np.float32)
                    if r_per_center is not None:
                        sim_thresh_batch = float(np.min([1.0 - r_per_center[i] for i in range(batch_start, batch_end)]))
                    else:
                        sim_thresh_batch = sim_thresh_default
                    lims, D, I = embeddings_index.range_search(centers_batch, sim_thresh_batch)
                    n_batch = centers_batch.shape[0]
                    if lims is None or len(lims) != n_batch + 1:
                        for _ in range(n_batch):
                            posting_lists.append([])
                        continue
                    for i in range(n_batch):
                        start, end = int(lims[i]), int(lims[i + 1])
                        cidx = batch_start + i
                        sim_thresh_c = (1.0 - float(r_per_center[cidx])) if r_per_center else sim_thresh_default
                        entries = [(int(I[j]), float(D[j])) for j in range(start, end) if float(D[j]) >= sim_thresh_c]
                        if exclude_cls_spans:
                            entries = [(idx, s) for idx, s in entries if idx not in exclude_cls_span_indices]
                        if cidx in stop_centers:
                            entries = []
                        posting_lists.append(entries)
                
                # PCA projections: second pass over in-memory embeddings
                if center_pca_dirs is not None:
                    span_to_centers: dict[int, list[tuple[int, float]]] = {}
                    for c in range(V):
                        for (span_idx, sim) in posting_lists[c]:
                            span_to_centers.setdefault(span_idx, []).append((c, sim))
                    posting_lists_new: list[list[tuple[int, float, float]]] = [[] for _ in range(V)]
                    span_offset = 0
                    for section_name in doc_sections:
                        arr = embeddings_by_section[section_name].astype(np.float32).copy()
                        faiss.normalize_L2(arr)
                        for j in range(arr.shape[0]):
                            global_idx = span_offset + j
                            for (c, sim) in span_to_centers.get(global_idx, []):
                                proj = float(arr[j] @ center_pca_dirs[c])
                                posting_lists_new[c].append((global_idx, sim, proj))
                        span_offset += arr.shape[0]
                    posting_lists = posting_lists_new
                else:
                    posting_lists = [[(s, sim, 0.0) for s, sim in pl] for pl in posting_lists]
            
            # Alignment sanity check
            if total_loaded != len(span_to_doc):
                raise ValueError(
                    f"Embeddings count ({total_loaded:,}) != span_to_doc count ({len(span_to_doc):,}). "
                    "Posting lists require exact 1:1 alignment."
                )
            
            # Span-level posting list length (before doc aggregation)
            span_pl_lens = np.array([len(pl) for pl in posting_lists], dtype=np.float64)
            span_non_empty = span_pl_lens[span_pl_lens > 0]
            if len(span_non_empty) > 0:
                print(f"\n📊 Posting list (span-level) length: total entries={int(span_pl_lens.sum()):,}, "
                      f"mean per non-empty center={float(span_non_empty.mean()):.1f}, max={int(span_pl_lens.max()):,}")
            
            # Build document-level inverted index (doc_id, weight, proj)
            print(f"\n🔨 Building document-level inverted index from posting lists...")
            doc_postings: list[list[tuple[str, float, float]]] = [[] for _ in range(V)]
            doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(documents_df.index)}
            doc_centers_hit: dict[str, int] = {}
            
            weight_agg = getattr(args, "weight_aggregation", "max")
            for center_idx in tqdm(range(V), desc="Building inverted index"):
                span_sims = posting_lists[center_idx]
                if not span_sims:
                    continue
                doc_weights: dict[str, tuple[float, float, float]] = {}  # (weight, max_sim_seen, proj_of_max)
                for entry in span_sims:
                    if len(entry) == 3:
                        span_idx, similarity, proj = entry
                    else:
                        span_idx, similarity, proj = entry[0], entry[1], 0.0
                    doc_id = span_to_doc.get(span_idx, None)
                    if doc_id is None or doc_id not in doc_id_to_idx:
                        continue
                    sim = float(similarity)
                    p = float(proj)
                    if weight_agg == "sum":
                        if doc_id in doc_weights:
                            ow, max_s, op = doc_weights[doc_id]
                            new_w = ow + max(0.0, sim)
                            doc_weights[doc_id] = (new_w, max(max_s, sim), op if max_s >= sim else p)
                        else:
                            doc_weights[doc_id] = (max(0.0, sim), sim, p)
                    else:
                        if doc_id not in doc_weights or max(0.0, sim) > doc_weights[doc_id][0]:
                            doc_weights[doc_id] = (max(0.0, sim), sim, p)
                for doc_id, wproj in doc_weights.items():
                    weight = wproj[0]
                    proj = wproj[2]
                    doc_postings[center_idx].append((doc_id, float(weight), float(proj)))
                    doc_centers_hit[doc_id] = doc_centers_hit.get(doc_id, 0) + 1
            
            # For efficiency report: total postings and non-empty count (FLOPs computed after query_sparse)
            pl_lens = np.array([len(pl) for pl in doc_postings], dtype=np.float64)
            n_empty = int(np.sum(pl_lens == 0))
            total_entries = int(np.sum(pl_lens))

            N_docs = len(documents_df)
            df = np.array([len(set(e[0] for e in pl)) if pl else 0 for pl in doc_postings], dtype=np.float32)
            idf = (np.log((N_docs + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
            idf_exponent = float(getattr(args, "idf_exponent", 1.0))
            if idf_exponent != 1.0:
                print(f"   IDF exponent: {idf_exponent} (score term uses idf^{idf_exponent})")

            # Encode + assign queries (format must match utils.collect_doc_texts for doc side)
            if mode == "abstract2abstract":
                print(f"\n📝 Evaluating: Abstract -> Abstract")
                _fmt_scheme = get_encoder_format_scheme(args.dense_model)
                _sep = get_encoder_sep_for_model(args.dense_model, tokenizer)
                query_texts = []
                for idx in queries_df.index:
                    title = queries_df.loc[idx, "title"] if "title" in queries_df.columns else ""
                    abstract = queries_df.loc[idx, "abstract"] if "abstract" in queries_df.columns else ""
                    query_texts.append(format_abstract_for_encoder(_fmt_scheme, title, abstract, sep=_sep))
            else:
                print(f"\n📝 Evaluating: Claim -> All")
                # Must match format used when building embeddings (utils.collect_doc_texts): "[claim] {claim}"
                query_texts = [f"[claim] {c}".strip() for c in queries_df["claim"].fillna("").tolist()]
            
            use_full_chunks = getattr(args, "query_full_chunks", False) and mode == "abstract2abstract" and args.tokenization_unit == "encoder_token"
            if use_full_chunks:
                print(f"   Query encoding: full-query chunks (dedup keep first, chunk_weight={getattr(args, 'query_chunk_weight', 'uniform')})")
                query_spans, query_span_weights = _encode_query_spans_chunked(query_texts, section=query_section, d=d)
            else:
                if getattr(args, "query_full_chunks", False) and mode == "abstract2abstract":
                    print(f"   Query full_chunks requested but tokenization_unit != encoder_token; using single-chunk (truncated) encoding.")
                query_spans = _encode_query_spans(query_texts, section=query_section, d=d)
                query_span_weights = None
            query_sparse = _assign_query_spans_to_centers(query_spans, center_index=center_index, V=V, sim_threshold=sim_threshold, idf=idf, center_pca_dirs=center_pca_dirs, query_span_weights=query_span_weights, stop_centers=stop_centers)
            # FLOPs = 2 * sum over queries of (sum of |L_t| for t in query terms)
            total_flops = sum(2 * sum(len(doc_postings[t]) for t in qpack[0]) for qpack in query_sparse)
            n_queries_flops = len(query_sparse)
            _report_flops_and_postings_one_line(
                int(total_entries), V - n_empty, V, mode,
                total_flops=total_flops, n_queries=n_queries_flops, model_label="sparse_coverage"
            )

            # Length normalization setup
            length_norm = getattr(args, "length_norm", "none")
            length_norm_exp = getattr(args, "length_norm_exponent", 0.5)
            avg_span_count = 1.0
            if length_norm == "sqrt_centers":
                print(f"   Length norm: sqrt_centers (exponent={length_norm_exp})")
            q_opts = []
            if getattr(args, "query_full_chunks", False):
                q_opts.append(f"full_chunks stride={getattr(args, 'query_chunk_stride_ratio', 1.0)} weight={getattr(args, 'query_chunk_weight', 'uniform')}")
                if getattr(args, "query_chunk_weight", "uniform") == "first":
                    q_opts.append(f"first_chunk_weight={getattr(args, 'query_first_chunk_weight', 1.5)}")
            if getattr(args, "query_first_span_weight", 1.0) != 1.0:
                q_opts.append(f"E: first_span_weight={args.query_first_span_weight}")
            if pca_proj_alpha != 0.0:
                q_opts.append(f"PCA_proj: alpha={pca_proj_alpha}")
            angle_sim_beta = float(getattr(args, "angle_sim_beta", 0.0))
            if angle_sim_beta != 0.0:
                q_opts.append(f"angle_sim_beta: {angle_sim_beta}")
            if residual_alpha != 0.0:
                q_opts.append(f"residual_alpha: {residual_alpha}")
            if q_opts:
                print(f"   Query opts: {', '.join(q_opts)}")
            print(f"🔍 Retrieving documents...")
            top_k = 100
            top_indices: list[list[int]] = []
            for _q_idx, qpack in enumerate(tqdm(query_sparse, desc="Retrieving")):
                terms, weights, projs = qpack[0], qpack[1], qpack[2]
                doc_scores = {}
                for i, term in enumerate(terms):
                    if len(doc_postings[term]) == 0:
                        continue
                    q_sim = float(weights[i])
                    q_proj = float(projs[i]) if i < len(projs) else 0.0
                    idf_t = float(idf[term]) ** idf_exponent
                    for doc_id, d_weight, d_proj in doc_postings[term]:
                        doc_idx = doc_id_to_idx.get(doc_id, None)
                        if doc_idx is None:
                            continue
                        d_sim = float(d_weight)
                        sim_approx = q_sim * d_sim
                        if pca_proj_alpha != 0.0:
                            sim_approx += pca_proj_alpha * q_proj * float(d_proj)
                        contrib = sim_approx * idf_t
                        if angle_sim_beta != 0.0:
                            contrib += angle_sim_beta * (1.0 - abs(q_sim - d_sim)) * idf_t
                        if residual_alpha != 0.0 and center_dot_pca is not None:
                            cdu = float(center_dot_pca[term])
                            q_res_proj = q_proj - q_sim * cdu
                            d_res_proj = float(d_proj) - d_sim * cdu
                            contrib += residual_alpha * (q_res_proj * d_res_proj) * idf_t
                        doc_scores[doc_idx] = doc_scores.get(doc_idx, 0.0) + contrib
                # Apply document length normalization
                if length_norm != "none" and doc_scores:
                    for doc_idx in list(doc_scores.keys()):
                        doc_id = documents_df.index[doc_idx]
                        if length_norm == "sqrt_centers":
                            nch = doc_centers_hit.get(doc_id, 1)
                            norm_factor = max(float(nch) ** length_norm_exp, 1e-6)
                        else:
                            norm_factor = 1.0
                        doc_scores[doc_idx] /= norm_factor
                if doc_scores:
                    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
                    top_docs = [doc_idx for doc_idx, _ in sorted_docs[:top_k]]
                    top_indices.append(top_docs)
                else:
                    top_indices.append([])
            
            # ---- Helper: compute metrics for a given query subset ----
            def _compute_metrics_for_qids(allowed_qids=None):
                """Compute recall/ndcg/mrr/map for a subset of queries.
                If allowed_qids is None, use all queries."""
                tl, rl = [], []
                for q_idx, q_id in enumerate(queries_df.index):
                    if allowed_qids is not None and str(q_id) not in allowed_qids:
                        continue
                    tl.append(citation_mapping.get(q_id, []))
                    rl.append([documents_df.index[i] for i in top_indices[q_idx]])
                res = {}
                for k in [10, 20, 50, 100]:
                    res[f"recall@{k}"] = mean_recall_at_k(tl, rl, k=k)
                for k in [10, 20, 50, 100]:
                    res[f"ndcg@{k}"] = mean_ndcg_at_k(tl, rl, k=k)
                res["mrr@10"] = mean_mrr_at_k(tl, rl, k=10)
                res["map"] = mean_average_precision(tl, rl, k=100)
                res["pres@100"] = mean_pres_at_k(tl, rl, k=100)
                return res, len(tl), rl

            task_label = "Abstract -> Abstract" if mode == "abstract2abstract" else "Claim -> All"
            results, _, retrieved_ids_list = _compute_metrics_for_qids(None)

            if mode == "abstract2abstract":
                print_metric_table(results, f"Sparse Coverage: {task_label}")
                if save_rankings_abs:
                    ranking_dict = {str(queries_df.index[i]): retrieved_ids_list[i] for i in range(len(retrieved_ids_list))}
                    with open(save_rankings_abs, "w") as f:
                        json.dump(ranking_dict, f, indent=0)
                    print(f"   Saved abstract->abstract rankings to {save_rankings_abs} ({len(retrieved_ids_list)} queries)")
            else:
                print_metric_table(results, f"Sparse Coverage: {task_label}")
                if save_rankings_claim:
                    ranking_dict = {str(queries_df.index[i]): retrieved_ids_list[i] for i in range(len(retrieved_ids_list))}
                    with open(save_rankings_claim, "w") as f:
                        json.dump(ranking_dict, f, indent=0)
                    print(f"   Saved claim->all rankings to {save_rankings_claim} ({len(retrieved_ids_list)} queries)")
            
            print(f"\n✅ Task {mode} evaluation completed")
        
        print(f"\n✅ Sparse Coverage evaluation completed for all available tasks")

    ############################################## CLEF-IP 2013 EN (claims-to-passages) ##################################################
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