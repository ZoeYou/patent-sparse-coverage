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


def _auto_batch_size(device, hidden_size: int = 768, min_bs: int = 4, max_bs: int = 256) -> int:
    """Return a safe inference batch size scaled to GPU total memory.

    Calibrated for transformer models at seq_len=512:
      ~11 GB GPU  (e.g. GTX 1080 Ti / RTX 2080 Ti) → 32
      ~16 GB GPU  (e.g. V100-16 / RTX 3080/4080)    → 64
      ~24 GB GPU  (e.g. RTX 3090/4090)               → 64
      ~40 GB GPU  (e.g. A100-40)                      → 128
      ~80 GB GPU  (e.g. A100-80)                      → 256
    Hidden-size scaling: larger models get proportionally smaller batches (sqrt scaling).
    Falls back to 32 on CPU or if GPU info is unavailable.
    """
    if not torch.cuda.is_available():
        return 32
    try:
        dev_idx = device.index if (hasattr(device, "index") and device.index is not None) else 0
        total_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 3)
        # Reference: hidden=768 works at bs=64 on a 16 GB GPU
        bs_float = 64.0 * (total_gb / 16.0) * (768.0 / max(hidden_size, 1)) ** 0.5
        # Round down to nearest power of 2
        p2 = min_bs
        while p2 * 2 <= int(bs_float):
            p2 *= 2
        result = max(min_bs, min(max_bs, p2))
        print(f"  [auto batch_size] GPU total={total_gb:.1f} GB, hidden={hidden_size} → batch_size={result}", flush=True)
        return result
    except Exception:
        return 32


def _load_or_compute_prior_art_embeddings(cache_query_path, cache_doc_path, compute_fn, pickle_protocol=None):
    """
    Load prior-art query/document embeddings from cache if present, else compute via compute_fn() and save.
    compute_fn() should return (query_embeddings, document_embeddings) as numpy arrays.
    Returns (query_embeddings, document_embeddings) as float32.
    """
    if os.path.exists(cache_query_path) and os.path.exists(cache_doc_path):
        print("Embeddings already created!")
        q = torch.load(cache_query_path, weights_only=False)
        d = torch.load(cache_doc_path, weights_only=False)
    else:
        q, d = compute_fn()
        save_kw = {} if pickle_protocol is None else {"pickle_protocol": pickle_protocol}
        torch.save(q, cache_query_path, **save_kw)
        torch.save(d, cache_doc_path, **save_kw)
    return np.asarray(q, dtype=np.float32), np.asarray(d, dtype=np.float32)


def _save_rankings_paths_from_args(args):
    """If args.save_rankings (dir) is set, return dict of ranking file paths; else empty dict.

    Keys:
      - 'priorart_abs2abs': prior-art abstract → abstract
      - 'priorart_claim2all': prior-art claim → all sections
      - 'clefip_passage': CLEF-IP passage-level ranking
    Creates directory if needed.
    """
    base = getattr(args, "save_rankings", None)
    if not base:
        return {}
    base = os.path.abspath(base)
    os.makedirs(base, exist_ok=True)
    return {
        "priorart_abs2abs": os.path.join(base, "rankings_abstract2abstract.json"),
        "priorart_claim2all": os.path.join(base, "rankings_claim2all.json"),
        "clefip_passage": os.path.join(base, "rankings_clefip_passage.json"),
    }


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
    unique_query_ids = set(query_ids)
    missing = unique_query_ids - set(citation_mapping)
    if missing:
        print(f"⚠️  {len(missing)} query IDs have no gold citations (first 5: {sorted(missing)[:5]})")

    results = {}

    ######## Task1: Abstract-to-Abstract evaluation ########
    texttype_q, texttype_d = "abstract", "abstract"

    # Convert to numpy array to ensure compatibility
    query_types = np.array(query_types)
    doc_types = np.array(doc_types)
    query_ids_arr = np.array(query_ids)
    doc_ids_arr = np.array(doc_ids)

    query_type_masks = (query_types == texttype_q)
    doc_type_masks = (doc_types == texttype_d)

    Q_emb = query_embeddings[query_type_masks].astype(np.float32)  # shape: [n_queries, emb_dim]
    D_emb = document_embeddings[doc_type_masks].astype(np.float32)    # shape: [n_docs, emb_dim]
    qids_abs = query_ids_arr[query_type_masks]
    dids_abs = doc_ids_arr[doc_type_masks]

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

    for q_idx, retrieved_docs_indices in enumerate(top_k_indices):
        q_id_str = qids_abs[q_idx]
        true_labels = citation_mapping.get(q_id_str, [])
        predicted_labels = [dids_abs[d_idx] for d_idx in retrieved_docs_indices]

        true_labels_list.append(true_labels)
        predicted_labels_list.append(predicted_labels)

    _save_rankings(save_rankings_path, qids_abs, predicted_labels_list, "abstract->abstract")

    # Compute metrics
    results_key = "abstract->abstract"
    results[results_key] = _make_prior_art_metrics(true_labels_list, predicted_labels_list)


    ######## Task2: Claim-to-All evaluation ########
    retrieved_sections = []   # for noting which section is retrieved at top_k
    
    original_doc_count = len(doc_ids) // 3

    query_type_masks_claim = (query_types == "claim")
    qids_claim = query_ids_arr[query_type_masks_claim]
    Q_emb = query_embeddings[query_type_masks_claim].astype(np.float32)
    D_emb = document_embeddings.astype(np.float32)
    faiss.normalize_L2(Q_emb)
    faiss.normalize_L2(D_emb)
    _report_dense_flops(Q_emb, D_emb, "claim->all", model_label=model_label)
    distances = Q_emb @ D_emb.T

    top_k_indices = np.argsort(-distances, axis=1)[:, :300]  # top_k * 3 to ensure we have enough candidates
    true_labels_list, predicted_labels_list = [], []
    section_names = ["abstract", "claim", "invention"]
    for q_idx, retrieved_docs_indices in enumerate(top_k_indices):
        q_id_str = qids_claim[q_idx]
        true_labels = citation_mapping.get(q_id_str, [])
        seen = set()
        predicted_labels = []
        q_sections = []
        for d_idx in retrieved_docs_indices:
            docid = doc_ids_arr[d_idx]
            if docid not in seen:
                seen.add(docid)
                predicted_labels.append(docid)
                q_sections.append(section_names[d_idx // original_doc_count])
            if len(predicted_labels) == 100:
                break
        retrieved_sections.append(q_sections)
        true_labels_list.append(true_labels)
        predicted_labels_list.append(predicted_labels)

    # Compute metrics
    results_key = "claim->all"
    results[results_key] = _make_prior_art_metrics(
        true_labels_list, predicted_labels_list,
        retrieved_sections=f"[{len(retrieved_sections)} queries with retrieved sections]",
    )

    _save_rankings(save_rankings_claim2all_path, qids_claim, predicted_labels_list, "claim->all")

    _display_prior_art_results(results, results_key, retrieved_sections, query_section="claim")


def _clefip_two_stage_rerank(
    passage_ids: list,
    predicted_labels_list: list,
    passage_scores_list: list,
    topk_docs: int = 100,
) -> list:
    """
    Two-stage CLEF-IP passage retrieval:
      Stage 1: From initial passage ranking, derive document ranking (first occurrence dedup).
               Keep only the top-K documents.
      Stage 2: Among ALL passages in the corpus belonging to those top-K documents,
               re-rank by the original passage scores.  Passages from documents not in
               top-K are excluded — this eliminates noise from spurious high-score passages
               in irrelevant documents.

    Args:
        passage_ids:            full corpus passage_id list (same order as scoring index).
        predicted_labels_list:  per-query list of ranked passage_ids from Stage 1 (any length).
        passage_scores_list:    per-query list of dicts {passage_id: score} covering all scored passages
                                (or at least those from the top-ranking). If a dict is None or empty,
                                the corresponding query falls back to Stage 1 ranking.
        topk_docs:              number of top documents to keep after Stage 1 (default 100).

    Returns:
        reranked_list:  per-query list of passage_ids (all passages from top-K docs, sorted by score).
    """
    # Pre-build passage_id -> doc_id mapping and doc_id -> set(passage_ids)
    pid_to_doc = {}
    doc_to_pids: dict[str, set] = {}
    for pid in passage_ids:
        doc_id = _clefip_passage_id_to_doc_id(pid)
        pid_to_doc[pid] = doc_id
        doc_to_pids.setdefault(doc_id, set()).add(pid)

    reranked_list = []
    for q_idx in range(len(predicted_labels_list)):
        pred = predicted_labels_list[q_idx]
        scores = passage_scores_list[q_idx] if q_idx < len(passage_scores_list) else {}
        if not scores:
            # Fallback: no scores available, use Stage 1 ranking as-is
            reranked_list.append(pred)
            continue

        # Stage 1: derive document ranking from passage ranking (first-occurrence dedup)
        seen_docs = set()
        top_docs = []
        for pid in pred:
            doc_id = pid_to_doc.get(pid, pid.split("::", 1)[0] if "::" in pid else pid)
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                top_docs.append(doc_id)
                if len(top_docs) >= topk_docs:
                    break

        # Collect ALL passage_ids from those top-K documents
        candidate_pids = set()
        for doc_id in top_docs:
            candidate_pids.update(doc_to_pids.get(doc_id, set()))

        # Stage 2: re-rank candidates by their original passage score (desc)
        scored_candidates = []
        for pid in candidate_pids:
            s = scores.get(pid, None)
            if s is not None:
                scored_candidates.append((pid, s))
            else:
                # Passage from a top-K doc but not scored (e.g. it was beyond the initial
                # top-N cutoff). Assign a very low score so it still appears after scored ones.
                scored_candidates.append((pid, -1e9))
        scored_candidates.sort(key=lambda x: -x[1])
        reranked_list.append([pid for pid, _ in scored_candidates])

    return reranked_list


def clefip_passage_evaluation(
    query_ids,
    passage_ids,
    query_embeddings,
    passage_embeddings,
    qrels_passage_ids,
    k=100,
    model_label="Dense",
    topk_docs=100,
    save_rankings_path=None,
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
    Q = np.asarray(query_embeddings, dtype=np.float32)
    D = np.asarray(passage_embeddings, dtype=np.float32)
    assert Q.shape[1] == D.shape[1]
    faiss.normalize_L2(Q)
    faiss.normalize_L2(D)
    _report_dense_flops(Q, D, "CLEF-IP passage", model_label=model_label)
    sim = Q @ D.T
    # Full ranking for Stage 1 document derivation (dense has all scores, no need to truncate)
    full_ranking_indices = np.argsort(-sim, axis=1)
    predicted_labels_list_s1 = []
    passage_scores_list = []
    for q_idx, qid in enumerate(query_ids):
        ranked_pids = [passage_ids[j] for j in full_ranking_indices[q_idx]]
        predicted_labels_list_s1.append(ranked_pids)
        scores = {passage_ids[j]: float(sim[q_idx, j]) for j in range(sim.shape[1])}
        passage_scores_list.append(scores)
    predicted_labels_list = _clefip_two_stage_rerank(
        passage_ids, predicted_labels_list_s1, passage_scores_list,
        topk_docs=topk_docs,
    )
    print(f"  🔄 Two-stage retrieval: top-{topk_docs} docs → re-ranked passages per query")
    results = _evaluate_and_print_clefip(
        qrels_passage_ids, query_ids, predicted_labels_list,
        "Passage retrieval", save_path=save_rankings_path,
        two_stage=True, topk_docs=topk_docs,
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


def _save_rankings(save_path: str, query_ids, predicted_labels_list: list, label: str = ""):
    """Save ranked list to JSON if save_path is set. Format: {query_id: [id, ...]}."""
    if not save_path:
        return
    ranking_dict = {str(query_ids[i]): predicted_labels_list[i]
                    for i in range(len(predicted_labels_list))}
    with open(save_path, "w") as f:
        json.dump(ranking_dict, f, indent=0)
    print(f"   Saved {label} rankings to {save_path} ({len(predicted_labels_list)} queries)")


def _clefip_derive_doc_level_rankings(
    query_ids: list,
    qrels_passage_ids: dict,
    predicted_passage_labels_list: list,
    k: int = 100,
) -> tuple:
    """
    Convert passage-level qrels and predictions to document-level for official metrics.
    Returns (true_doc_ids_list, predicted_doc_ranking_list).
    - true_doc_ids_list: for each query, list of relevant doc_ids (from qrels passages).
    - predicted_doc_ranking_list: for each query, list of unique doc_ids in order of first appearance in top-k passages.
    """
    true_doc_ids_list = []
    predicted_doc_ranking_list = []
    for q_idx, qid in enumerate(query_ids):
        # Relevant docs = docs that have at least one relevant passage in qrels
        rel_passages = qrels_passage_ids.get(qid, [])
        true_doc_ids_list.append(list({_clefip_passage_id_to_doc_id(pid) for pid in rel_passages}))
        # Predicted doc ranking: unique doc_ids in order of first occurrence in top-k predicted passages
        seen = set()
        doc_ranking = []
        pred_list = predicted_passage_labels_list[q_idx][:k] if q_idx < len(predicted_passage_labels_list) else []
        for pid in pred_list:
            doc_id = _clefip_passage_id_to_doc_id(pid)
            if doc_id not in seen:
                seen.add(doc_id)
                doc_ranking.append(doc_id)
        predicted_doc_ranking_list.append(doc_ranking)
    return true_doc_ids_list, predicted_doc_ranking_list


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


def _get_clefip_dense_encoder(args, model_name: str, device):
    """
    Load the dense model for CLEF-IP and return (encode_fn, model_label).
    encode_fn(texts: list[str], batch_size=32) -> np.ndarray.
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
        model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
        model.to(device)

        def _fwd_cls(m, inp):
            return cls_pooling(m(**inp), inp["attention_mask"]).cpu().numpy()
        def _encode(texts, batch_size=32):
            return _batch_encode(texts, tokenizer, model, device, _fwd_cls, model.config.hidden_size, batch_size)
        return _encode, "Specter2"

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
        return _encode, "PAECTer" if "paecter" in model_name else "bert-for-patents"

    if model_name in ["datalyes/patembed-large", "patembed-large"]:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("datalyes/patembed-large")
        model.to(device)
        PATEN_TEB_RETRIEVAL_PROMPT_NAME = "retrieval_MIXED"
        def _encode(texts, batch_size=256, role="document"):
            try:
                if role == "query":
                    return model.encode_query(texts, prompt_name=PATEN_TEB_RETRIEVAL_PROMPT_NAME, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
                return model.encode_document(texts, prompt_name=PATEN_TEB_RETRIEVAL_PROMPT_NAME, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
            except (TypeError, AttributeError):
                return model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        return _encode, "Patembed"

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
        return _encode, "PatentMap"

    # Fallback
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)

    def _fwd_mean(m, inp):
        return mean_pooling(m(**inp).last_hidden_state, inp["attention_mask"]).cpu().numpy()
    def _encode(texts, batch_size=32):
        return _batch_encode(texts, tokenizer, model, device, _fwd_mean, model.config.hidden_size, batch_size)
        return np.vstack(embs) if embs else np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return _encode, "Dense"


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


def _display_prior_art_results(results: dict, results_key: str, retrieved_sections: list,
                               query_section: str = "claim", header_suffix: str = ""):
    """Print prior-art results table and run retrieved sections analysis if available."""
    print_subsection_header(f"Prior Art Search Results{header_suffix}")
    for task_key, task_results in results.items():
        if isinstance(task_results, dict):
            if '->' in task_key:
                clean_name = f"Query: {task_key.split('->')[0]} → Document: {task_key.split('->')[1]}"
            else:
                clean_name = task_key
            print_metric_table(task_results, clean_name)
    results[results_key]['retrieved_sections_full'] = retrieved_sections
    if retrieved_sections:
        from patentmap_eval.patenteval.utils import analyze_retrieved_sections_integrated
        section_analysis = analyze_retrieved_sections_integrated(
            retrieved_sections, query_section=query_section, print_results=True,
        )
        results[results_key]['section_analysis'] = section_analysis


def _make_clefip_official_metrics(
    true_labels_list: list,
    predicted_labels_list: list,
    k: int = 100,
) -> dict:
    """
    CLEF-IP official metrics: passage-level + document-level.
    - Passage: recall@k, NDCG@k, MRR, MAP (flat), pres@100.
    - Official passage-level: MAP(D) (per-document AP averaged over relevant docs,
      then over topics; Piroi et al. CLEF-IP 2012 Eq. 1-2). Reported as "magp".
    - Document: pres_doc@100 (PRES on doc ranking derived from passage ranking).
    """
    # Passage-level (same as prior-art)
    metrics = _make_prior_art_metrics(true_labels_list, predicted_labels_list, k=k)
    # Official passage-level: MAP(D) — hierarchical per-document AP (Piroi et al. 2012)
    metrics["magp"] = _clefip_mean_agp_passage(true_labels_list, predicted_labels_list, k=k)
    # Document-level PRES@100: derive doc rankings from passage rankings (official CLEF-IP doc-level metric)
    qids_fake = list(range(len(true_labels_list)))
    qrels_by_idx = {i: true_labels_list[i] for i in range(len(true_labels_list))}
    true_doc_ids_list, predicted_doc_ranking_list = _clefip_derive_doc_level_rankings(
        qids_fake, qrels_by_idx, predicted_labels_list, k=k
    )
    metrics["pres_doc@100"] = mean_pres_at_k(true_doc_ids_list, predicted_doc_ranking_list, k=100, N_max=100)
    # Additional doc-level metrics
    metrics["recall_doc@100"] = mean_recall_at_k(true_doc_ids_list, predicted_doc_ranking_list, k=100)
    metrics["ndcg_doc@100"] = mean_ndcg_at_k(true_doc_ids_list, predicted_doc_ranking_list, k=100)
    metrics["map_doc"] = mean_average_precision(true_doc_ids_list, predicted_doc_ranking_list, k=100)
    return metrics


def _evaluate_and_print_clefip(
    qrels: dict,
    query_ids: list,
    predicted_labels_list: list,
    model_label: str,
    *,
    save_path: "Optional[str]" = None,
    two_stage: bool = True,
    topk_docs: int = 100,
    header_extra: str = "",
) -> dict:
    """Evaluate CLEF-IP passage retrieval: compute metrics, print, save rankings."""
    true_labels_list = [qrels.get(qid, []) for qid in query_ids]
    results = _make_clefip_official_metrics(true_labels_list, predicted_labels_list)
    label_suffix = f" (two-stage top-{topk_docs} docs)" if two_stage else ""
    print_subsection_header(f"CLEF-IP 2013 EN claims-to-passages{header_extra}{label_suffix}")
    print_metric_table(results, f"{model_label}{label_suffix}")
    _save_rankings(save_path, query_ids, predicted_labels_list, "CLEF-IP passage")
    return results


def _score_queries_against_postings(
    query_sparse: list,
    doc_postings: list,
    idf: "np.ndarray",
    idf_exponent: float,
    top_k: int,
    *,
    pca_proj_alpha: float = 0.0,
    angle_sim_beta: float = 0.0,
    residual_alpha: float = 0.0,
    center_dot_pca: "Optional[np.ndarray]" = None,
    length_norm: str = "none",
    length_norm_exp: float = 0.5,
    doc_nspans: "Optional[np.ndarray]" = None,
    show_progress: bool = True,
) -> list:
    """Score queries against an inverted index and return top-k doc indices per query.

    Handles both the fast path (pure q_sim * d_sim * idf^alpha) and the full path
    (with PCA projection, angle similarity, and residual correction terms).

    Parameters
    ----------
    query_sparse : list of (terms, weights, projs) tuples
    doc_postings : list of posting lists; doc_postings[center_id] = [(doc_idx, d_weight, d_proj), ...]
    idf : ndarray (V,)
    idf_exponent : float
    top_k : int
    pca_proj_alpha, angle_sim_beta, residual_alpha : float
        Scoring feature weights (0.0 disables the feature).
    center_dot_pca : ndarray or None
        Pre-computed centers_norm[c] @ center_pca_dirs[c] for residual term.
    length_norm : str  — "sqrt_spans" to enable length normalization.
    length_norm_exp : float
    doc_nspans : ndarray or None  — per-document span counts.
    show_progress : bool

    Returns
    -------
    list[list[int]]  — per-query top-k document indices, descending by score.
    """
    _use_proj = (pca_proj_alpha != 0.0
                 or angle_sim_beta != 0.0
                 or (residual_alpha != 0.0 and center_dot_pca is not None))
    _use_len_norm = (length_norm == "sqrt_spans" and doc_nspans is not None)

    top_indices: list[list[int]] = []
    iterator = enumerate(query_sparse)
    if show_progress:
        iterator = enumerate(tqdm(query_sparse, desc="Retrieving"))

    for _q_idx, qpack in iterator:
        terms, weights, projs = qpack[0], qpack[1], qpack[2]
        doc_scores: dict[int, float] = {}

        if not _use_proj:
            # Fast path: score = q_sim * d_sim * idf^alpha
            for i, term in enumerate(terms):
                pl = doc_postings[term]
                if not pl:
                    continue
                q_idf = float(weights[i]) * (float(idf[term]) ** idf_exponent)
                for doc_idx, d_weight, _ in pl:
                    doc_scores[doc_idx] = doc_scores.get(doc_idx, 0.0) + q_idf * float(d_weight)
        else:
            # Full path: PCA projection / angle similarity / residual terms
            for i, term in enumerate(terms):
                pl = doc_postings[term]
                if not pl:
                    continue
                q_sim = float(weights[i])
                q_proj = float(projs[i]) if i < len(projs) else 0.0
                idf_t = float(idf[term]) ** idf_exponent
                for doc_idx, d_weight, d_proj in pl:
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
        if _use_len_norm and doc_scores:
            for doc_idx in list(doc_scores.keys()):
                norm_factor = max(doc_nspans[doc_idx] ** length_norm_exp, 1e-6)
                doc_scores[doc_idx] /= norm_factor

        if doc_scores:
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            top_indices.append([doc_idx for doc_idx, _ in sorted_docs[:top_k]])
        else:
            top_indices.append([])

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
                embs = checkpoint.queryFromText(batch)           # [B, query_maxlen, dim]
            else:
                embs = checkpoint.docFromText(batch, bsize=batch_size)  # [B, *, dim]
        if isinstance(embs, _th.Tensor):
            embs_np = embs.cpu().float().numpy()
            for j in range(embs_np.shape[0]):
                all_embs.append(embs_np[j])  # [n_tok, dim]
        elif isinstance(embs, (list, tuple)):
            for e in embs:
                all_embs.append(e.cpu().float().numpy() if isinstance(e, _th.Tensor) else np.asarray(e, dtype=np.float32))
        else:
            all_embs.append(embs.cpu().float().numpy() if isinstance(embs, _th.Tensor) else np.asarray(embs, dtype=np.float32))
    return all_embs


def _colbert_maxsim_matrix(query_embs: list, doc_embs: list, batch_doc: int = 512) -> np.ndarray:
    """Compute the MaxSim similarity matrix ``[n_queries, n_docs]``.

    For each query token, take the max cosine similarity with any document
    token, then sum over query tokens:

        score(q, d) = sum_i max_j (q_i · d_j)

    All embeddings are assumed **already** L2-normalised (ColBERT checkpoint
    normalises internally).

    Documents are padded and batched (size *batch_doc*) for vectorised torch
    computation on GPU/CPU.  This is orders of magnitude faster than the naive
    Python loop.
    """
    import torch as _th
    _device = _th.device("cuda" if _th.cuda.is_available() else "cpu")
    n_q = len(query_embs)
    n_d = len(doc_embs)
    sim = np.zeros((n_q, n_d), dtype=np.float32)

    for q_idx in tqdm(range(n_q), desc="MaxSim scoring"):
        q = _th.from_numpy(query_embs[q_idx]).to(_device)  # [q_tok, dim]
        for d_start in range(0, n_d, batch_doc):
            d_end = min(d_start + batch_doc, n_d)
            batch = [doc_embs[j] for j in range(d_start, d_end)]
            lengths = [d.shape[0] for d in batch]
            max_d_tok = max(lengths)
            B = len(batch)
            # Pad documents to same length → [B, max_d_tok, dim]
            D = _th.zeros(B, max_d_tok, q.shape[1], device=_device)
            mask = _th.zeros(B, max_d_tok, device=_device, dtype=_th.bool)
            for i, d_np in enumerate(batch):
                L = d_np.shape[0]
                D[i, :L] = _th.from_numpy(d_np)
                mask[i, :L] = True
            # Batched MaxSim: q [q_tok, dim] × D [B, d_tok, dim]^T → [B, q_tok, d_tok]
            scores = _th.einsum("qd,bkd->bqk", q, D)
            # Mask padding positions to -inf so they never win the max
            scores.masked_fill_(~mask.unsqueeze(1), float("-inf"))
            # max over d_tok → [B, q_tok], sum over q_tok → [B]
            sim[q_idx, d_start:d_end] = scores.max(dim=2).values.sum(dim=1).cpu().numpy()
    return sim


def _colbert_maxsim_rankings_and_scores(
    query_embs: list, doc_embs: list, doc_ids: list
) -> tuple:
    """Compute MaxSim, derive ranked ID lists and per-query score dicts.

    Returns ``(predicted_labels_list, passage_scores_list)`` ready for
    ``_clefip_two_stage_rerank``.
    """
    sim = _colbert_maxsim_matrix(query_embs, doc_embs)
    predicted_labels_list = []
    passage_scores_list = []
    ranking_indices = np.argsort(-sim, axis=1)
    for q_idx in range(sim.shape[0]):
        ranked_ids = [doc_ids[j] for j in ranking_indices[q_idx]]
        scores = {doc_ids[j]: float(sim[q_idx, j]) for j in range(sim.shape[1])}
        predicted_labels_list.append(ranked_ids)
        passage_scores_list.append(scores)
    return predicted_labels_list, passage_scores_list


def _run_clefip_eval_full_corpus(
    args,
    query_ids: list,
    query_texts: list,
    passage_ids: list,
    corpus_jsonl_path: str,
    ids_txt_path: str,
    qrels_passage_ids: dict,
    save_rankings_path: str = None,
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
        query_tokens = bm25s.tokenize(query_texts, stemmer=stemmer)
        retriever = bm25s.BM25()
        retriever.index(passage_tokens)
        _report_bm25_posting_stats(retriever, "CLEF-IP passage (full corpus)", query_tokens_list=query_tokens)
        k = 100
        clefip_results, clefip_scores = retriever.retrieve(query_tokens, k=len(passage_ids))
        predicted_labels_list = [[passage_ids[j] for j in result] for result in clefip_results]
        # Build per-query score dicts from BM25 retrieval results
        passage_scores_list = []
        for q_idx in range(len(clefip_results)):
            scores = {passage_ids[int(clefip_results[q_idx][j])]: float(clefip_scores[q_idx][j])
                      for j in range(len(clefip_results[q_idx]))}
            passage_scores_list.append(scores)
        predicted_labels_list = _clefip_two_stage_rerank(
            passage_ids, predicted_labels_list, passage_scores_list,
            topk_docs=topk_docs,
        )
        print(f"  🔄 Two-stage retrieval: top-{topk_docs} docs → re-ranked passages per query")
        _evaluate_and_print_clefip(
            qrels_passage_ids, query_ids, predicted_labels_list,
            "BM25 passage retrieval", save_path=save_rankings_path,
            two_stage=True, topk_docs=topk_docs, header_extra=" (full 01 corpus)",
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
        _splade_cache_dir = os.path.join("temp", "clefip_splade_cache", f"{_splade_clean}_s{_sample_sz}")
        _splade_q_path = os.path.join(_splade_cache_dir, "query_sparse.npz")
        _splade_p_path = os.path.join(_splade_cache_dir, "passage_sparse.npz")
        _splade_meta_path = os.path.join(_splade_cache_dir, "meta.json")

        _splade_cache_hit = False
        if os.path.isfile(_splade_q_path) and os.path.isfile(_splade_p_path) and os.path.isfile(_splade_meta_path):
            try:
                with open(_splade_meta_path, "r") as _mf:
                    _splade_meta = json.load(_mf)
                if (_splade_meta.get("n_queries") == len(query_ids)
                        and _splade_meta.get("n_passages") == len(passage_ids)
                        and _splade_meta.get("model") == actual_model_name):
                    Q = _sp_load_npz(_splade_q_path)
                    D = _sp_load_npz(_splade_p_path)
                    _splade_cache_hit = True
                    print(f"✅ Loaded SPLADE cache from {_splade_cache_dir}")
                    print(f"   queries: {Q.shape}, passages: {D.shape}")
                else:
                    print(f"⚠️  SPLADE cache metadata mismatch, re-encoding...")
            except Exception as e:
                print(f"⚠️  SPLADE cache load failed ({e}), re-encoding...")

        if not _splade_cache_hit:
            print(f"Loading passage texts from corpus for SPLADE...", flush=True)
            passage_texts = []
            with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    passage_texts.append(rec["text"])
            assert len(passage_texts) == len(passage_ids)

            from sentence_transformers import SparseEncoder
            splade_model = SparseEncoder(actual_model_name)
            encode_bs = _auto_batch_size(device, hidden_size=768)
            print(f"  Encoding {len(query_texts)} queries...", flush=True)
            query_sparse = splade_model.encode_query(query_texts, batch_size=encode_bs, show_progress_bar=True)
            print(f"  Encoding {len(passage_texts)} passages...", flush=True)
            passage_sparse = splade_model.encode_document(passage_texts, batch_size=encode_bs, show_progress_bar=True)

            def _sparse_torch_to_csr(t):
                """Convert torch sparse tensor → scipy CSR without dense."""
                import torch as _th
                t = t.cpu().coalesce()
                idx = t.indices().numpy()
                vals = t.values().numpy()
                from scipy.sparse import coo_matrix as _coo
                return _coo((vals, (idx[0], idx[1])), shape=t.shape).tocsr()

            if hasattr(query_sparse, "is_sparse") and query_sparse.is_sparse:
                Q = _sparse_torch_to_csr(query_sparse)
            else:
                q_np = query_sparse.cpu().numpy() if hasattr(query_sparse, "cpu") else np.asarray(query_sparse)
                Q = csr_matrix(q_np)
            if hasattr(passage_sparse, "is_sparse") and passage_sparse.is_sparse:
                D = _sparse_torch_to_csr(passage_sparse)
            else:
                d_np = passage_sparse.cpu().numpy() if hasattr(passage_sparse, "cpu") else np.asarray(passage_sparse)
                D = csr_matrix(d_np)

            # Save cache
            os.makedirs(_splade_cache_dir, exist_ok=True)
            _sp_save_npz(_splade_q_path, Q)
            _sp_save_npz(_splade_p_path, D)
            with open(_splade_meta_path, "w") as _mf:
                json.dump({
                    "model": actual_model_name,
                    "n_queries": len(query_ids),
                    "n_passages": len(passage_ids),
                    "vocab_size": int(D.shape[1]),
                    "q_nnz": int(Q.nnz),
                    "p_nnz": int(D.nnz),
                }, _mf, indent=2)
            _q_mb = os.path.getsize(_splade_q_path) / 1024**2
            _p_mb = os.path.getsize(_splade_p_path) / 1024**2
            print(f"💾 Saved SPLADE cache to {_splade_cache_dir}")
            print(f"   queries: {_q_mb:.1f} MB ({Q.nnz:,} nnz), passages: {_p_mb:.1f} MB ({D.nnz:,} nnz)")
            del splade_model, passage_texts  # free GPU memory
        vocab_size = D.shape[1]
        posting_lists, _ = _splade_build_inverted_index(D, vocab_size)
        _report_splade_flops_and_postings(posting_lists, Q, "CLEF-IP passage (SPLADE)")
        k = min(100, len(passage_ids))
        top_k_list, idx_scores_list = _splade_retrieve_with_index(
            Q, posting_lists, top_k=len(passage_ids), return_scores=True,
        )
        predicted_labels_list = [[passage_ids[j] for j in result] for result in top_k_list]
        # Convert index-keyed score dicts to passage_id-keyed score dicts
        passage_scores_list = [
            {passage_ids[d]: s for d, s in sd.items()} for sd in idx_scores_list
        ]
        predicted_labels_list = _clefip_two_stage_rerank(
            passage_ids, predicted_labels_list, passage_scores_list,
            topk_docs=topk_docs,
        )
        print(f"  \U0001f504 Two-stage retrieval: top-{topk_docs} docs \u2192 re-ranked passages per query")
        _evaluate_and_print_clefip(
            qrels_passage_ids, query_ids, predicted_labels_list,
            "SPLADE passage retrieval", save_path=save_rankings_path,
            two_stage=True, topk_docs=topk_docs, header_extra=" (full 01 corpus)",
        )
        return

    if model_name in _COLBERT_MODEL_NAMES:
        # ── ColBERT CLEF-IP passage retrieval ──
        import pickle
        try:
            from colbert.modeling.checkpoint import Checkpoint
            from colbert.infra import ColBERTConfig as _ColBERTConfig
        except ImportError:
            print("❌  colbert-ai package not installed. Install with:  pip install colbert-ai")
            return

        _colbert_config = _ColBERTConfig(doc_maxlen=512, query_maxlen=64, nbits=2)

        # ── Cache (per-token embeddings are large; save as .pkl) ──
        _model_clean = (args.model_name or "").rstrip("/").replace("/", "_").strip("_")
        _sample_sz = getattr(args, "clefip_sample_size", 0) or 0
        _cache_dir = os.path.join("temp", "clefip_colbert_cache", f"{_model_clean}_s{_sample_sz}")
        _cache_q = os.path.join(_cache_dir, "query_embs.pkl")
        _cache_p = os.path.join(_cache_dir, "passage_embs.pkl")
        _cache_meta = os.path.join(_cache_dir, "meta.json")

        _cache_hit = False
        if os.path.isfile(_cache_q) and os.path.isfile(_cache_p) and os.path.isfile(_cache_meta):
            try:
                with open(_cache_meta, "r") as _mf:
                    _meta = json.load(_mf)
                if (_meta.get("n_queries") == len(query_ids)
                        and _meta.get("n_passages") == len(passage_ids)
                        and _meta.get("model") == "colbert-ir/colbertv2.0"):
                    with open(_cache_q, "rb") as _f:
                        query_embs = pickle.load(_f)
                    with open(_cache_p, "rb") as _f:
                        passage_embs = pickle.load(_f)
                    _cache_hit = True
                    print(f"✅ Loaded ColBERT cache from {_cache_dir}")
                    print(f"   queries: {len(query_embs)}, passages: {len(passage_embs)}")
            except Exception as e:
                print(f"⚠️  ColBERT cache load failed ({e}), re-encoding...")

        if not _cache_hit:
            _colbert_ckpt = Checkpoint("colbert-ir/colbertv2.0", colbert_config=_colbert_config)

            print(f"  Encoding {len(query_texts)} queries with ColBERT...", flush=True)
            query_embs = _colbert_encode(_colbert_ckpt, query_texts, is_query=True, batch_size=32)

            print(f"  Encoding passages from corpus with ColBERT...", flush=True)
            passage_texts_all = []
            with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    passage_texts_all.append(rec["text"])
            assert len(passage_texts_all) == len(passage_ids)
            passage_embs = _colbert_encode(_colbert_ckpt, passage_texts_all, is_query=False, batch_size=32)
            del passage_texts_all, _colbert_ckpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Save cache
            os.makedirs(_cache_dir, exist_ok=True)
            with open(_cache_q, "wb") as _f:
                pickle.dump(query_embs, _f, protocol=4)
            with open(_cache_p, "wb") as _f:
                pickle.dump(passage_embs, _f, protocol=4)
            with open(_cache_meta, "w") as _mf:
                json.dump({
                    "model": "colbert-ir/colbertv2.0",
                    "n_queries": len(query_ids),
                    "n_passages": len(passage_ids),
                    "sample_size": _sample_sz,
                }, _mf, indent=2)
            _q_mb = os.path.getsize(_cache_q) / 1024**2
            _p_mb = os.path.getsize(_cache_p) / 1024**2
            print(f"💾 Saved ColBERT cache to {_cache_dir}")
            print(f"   queries: {_q_mb:.1f} MB, passages: {_p_mb:.1f} MB")

        # MaxSim scoring → rankings + score dicts → two-stage rerank → evaluate
        print("  Computing MaxSim similarity matrix...", flush=True)
        predicted_labels_list, passage_scores_list = _colbert_maxsim_rankings_and_scores(
            query_embs, passage_embs, passage_ids,
        )
        predicted_labels_list = _clefip_two_stage_rerank(
            passage_ids, predicted_labels_list, passage_scores_list,
            topk_docs=topk_docs,
        )
        print(f"  🔄 Two-stage retrieval: top-{topk_docs} docs → re-ranked passages per query")
        _evaluate_and_print_clefip(
            qrels_passage_ids, query_ids, predicted_labels_list,
            "ColBERT passage retrieval", save_path=save_rankings_path,
            two_stage=True, topk_docs=topk_docs, header_extra=" (full 01 corpus)",
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
    encode_fn, model_label = _get_clefip_dense_encoder(args, model_name, device)

    # Try loading from cache
    _cache_hit = False
    if os.path.isfile(_dense_q_path) and os.path.isfile(_dense_p_path) and os.path.isfile(_dense_meta_path):
        try:
            with open(_dense_meta_path, "r") as f:
                _meta = json.load(f)
            if (_meta.get("model_name") == args.model_name
                    and _meta.get("n_queries") == len(query_ids)
                    and _meta.get("n_passages") == len(passage_ids)):
                query_emb = np.load(_dense_q_path)
                passage_emb = np.load(_dense_p_path)
                assert query_emb.shape[0] == len(query_ids), f"query cache shape mismatch: {query_emb.shape[0]} vs {len(query_ids)}"
                assert passage_emb.shape[0] == len(passage_ids), f"passage cache shape mismatch: {passage_emb.shape[0]} vs {len(passage_ids)}"
                _cache_hit = True
                print(f"✅ Loaded CLEF-IP dense embeddings from cache: {_dense_cache_dir}")
                print(f"   queries: {query_emb.shape}, passages: {passage_emb.shape}")
        except Exception as e:
            print(f"⚠️  CLEF-IP dense cache load failed ({e}), re-encoding...")
            _cache_hit = False

    if not _cache_hit:
        query_emb = encode_fn(query_texts_fmt, batch_size=32) if model_name not in ["datalyes/patembed-large", "patembed-large"] else encode_fn(query_texts_fmt, role="query")
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
                    if model_name in ["datalyes/patembed-large", "patembed-large"]:
                        passage_emb_list.append(encode_fn(batch_fmt, role="document"))
                    else:
                        passage_emb_list.append(encode_fn(batch_fmt, batch_size=32))
                    batch_pids, batch_texts = [], []
            if batch_pids:
                _, batch_fmt = _clefip_format_for_model([""], batch_pids, batch_texts, args.model_name)
                if model_name in ["datalyes/patembed-large", "patembed-large"]:
                    passage_emb_list.append(encode_fn(batch_fmt, role="document"))
                else:
                    passage_emb_list.append(encode_fn(batch_fmt, batch_size=32))
        passage_emb = np.vstack(passage_emb_list) if passage_emb_list else np.zeros((0, query_emb.shape[1]), dtype=np.float32)
        del passage_emb_list  # free memory before saving

        # Save to cache
        os.makedirs(_dense_cache_dir, exist_ok=True)
        np.save(_dense_p_path, passage_emb)
        np.save(_dense_q_path, query_emb)
        with open(_dense_meta_path, "w") as f:
            json.dump({
                "model_name": args.model_name,
                "n_queries": len(query_ids),
                "n_passages": len(passage_ids),
                "dim": int(passage_emb.shape[1]),
                "sample_size": _sample_sz,
            }, f, indent=2)
        print(f"💾 Saved CLEF-IP dense embeddings to {_dense_cache_dir}")
        print(f"   queries: {query_emb.shape} ({os.path.getsize(_dense_q_path) / 1024**2:.0f} MB)")
        print(f"   passages: {passage_emb.shape} ({os.path.getsize(_dense_p_path) / 1024**2:.0f} MB)")

    clefip_passage_evaluation(query_ids, passage_ids, query_emb, passage_emb, qrels_passage_ids, k=100, model_label=model_label + " (full 01)",
                              topk_docs=topk_docs, save_rankings_path=save_rankings_path)


def run_clefip_eval(args, save_rankings_path: str = None):
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
            sample_cache_dir = os.path.join(clefip_root, f"01_passage_corpus_en_sample_{sample_size}docs")
            _corpus_label = f"sampled corpus ({sample_size:,} docs)"
        sample_cache_exists = os.path.isfile(os.path.join(sample_cache_dir, CORPUS_JSONL)) and os.path.isfile(os.path.join(sample_cache_dir, IDS_TXT))
        if sample_cache_exists and not rebuild_corpus:
            print(f"Loading CLEF-IP 2013 EN (claims-to-passages, {_corpus_label}, using cache)...")
        else:
            print(f"Loading CLEF-IP 2013 EN (claims-to-passages, {_corpus_label})...")
        query_ids, query_texts, corpus_jsonl_path, ids_txt_path, num_passages, qrels_passage_ids = load_clefip_en_for_eval_sampled_corpus(
            clefip_root, doc_root, sample_size=sample_size, rebuild_corpus=rebuild_corpus
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
        save_rankings_path=save_rankings_path,
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
    parser.add_argument("--soft_assignment_max_centers_per_span", type=int, default=10,
                       help="Cap each span (query or document) to at most this many centers (by similarity) during soft assignment. "
                           "If a span falls in >K centers, keep top-K; if <=K, keep all. "
                           "Default: 10. Applies to BOTH query-side and document-side soft assignment.")
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

    parser.add_argument("--length_norm", type=str, default="sqrt_centers",
                       choices=["none", "sqrt_spans", "sqrt_centers"],
                       help="Document length normalization for sparse_coverage. "
                            "none: no normalization. sqrt_spans: divide by doc_span_count^exponent "
                            "(BM25-like, stable across stop-center changes). "
                            "sqrt_centers: legacy alias for sqrt_spans. Default: sqrt_centers.")
    parser.add_argument("--length_norm_exponent", type=float, default=0.5,
                       help="Exponent for length norm: divide by doc_span_count^exponent. "
                            "0.5 => sqrt (default). 0.8 => stronger penalization of long docs.")
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
                       help="If set (directory path), save ranking files for hybrid fusion: "
                            "rankings_abstract2abstract.json (prior-art abstract→abstract), "
                            "rankings_claim2all.json (prior-art claim→all), and "
                            "rankings_clefip_passage.json (CLEF-IP passage-level). "
                            "Format: {query_id: [doc_or_passage_id, ...]}. Use with dense or sparse_coverage runs.")
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
                            ">0 (default: 25000) = number of DOCUMENTS to sample. All passages from qrels-cited documents are "
                            "always included; remaining slots filled by reservoir-sampled EN documents. "
                            "All passages from each selected document are kept (preserves document structure). "
                            "Example: --clefip_sample_size 10000 (~10k docs → ~800k passages at ~80 passages/doc). "
                            "Cache is built under 01_passage_corpus_en_sample_<N>docs/ and reused on subsequent runs.")
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
    parser.add_argument("--clefip_only", action="store_true", default=False,
                       help="Run only CLEF-IP evaluation, skip prior-art (perf200) tasks. "
                            "Useful for targeted CLEF-IP hyperparameter sweeps.")

    args = parser.parse_args()
    save_rankings_paths = _save_rankings_paths_from_args(args)
    save_rankings_abs = save_rankings_paths.get("priorart_abs2abs")
    save_rankings_claim = save_rankings_paths.get("priorart_claim2all")
    _clefip_data = None  # Set by sparse_coverage block if CLEF-IP data loads successfully

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
    if getattr(args, 'clefip_only', False):
        print("⏭️  --clefip_only: skipping prior-art (perf200) data loading.")
        queries, documents, queries_df, documents_df = None, None, None, None
        query_ids, doc_ids, query_types, doc_types = [], [], [], []
        citation_mapping = {}
        original_query_count = original_doc_count = 0
    else:
        print("Running Prior-art search task.")
    Prior_art_dataset_dir = './patentmap_eval/data/downstream/perf200'

    if not getattr(args, 'clefip_only', False):
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
    if args.model_name.lower() in ["allenai/specter2_base"]:
        from adapters import AutoAdapterModel
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoAdapterModel.from_pretrained(args.model_name)
        model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
        embedding_dim = model.config.hidden_size
        model.to(device)

        if not getattr(args, 'clefip_only', False):
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
                    batch_size = _auto_batch_size(device, hidden_size=embedding_dim)
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

        if not getattr(args, 'clefip_only', False):
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
                    batch_size = _auto_batch_size(device, hidden_size=embedding_dim)
                    query_embs = np.zeros((len(query_encodings['input_ids']), embedding_dim))
                    doc_embs = np.zeros((len(doc_encodings['input_ids']), embedding_dim))
                    with torch.no_grad():
                        for i in trange(0, len(query_encodings['input_ids']), batch_size, desc=f"Computing {texttype} query embeddings"):
                            batch = {key: val[i:i+batch_size].to(device) for key, val in query_encodings.items()}
                            outputs = model(**batch)
                            query_embs[i:i+batch_size] = mean_pooling(outputs.last_hidden_state, batch['attention_mask']).detach().cpu().numpy()
                        for i in trange(0, len(doc_encodings['input_ids']), batch_size, desc=f"Computing {texttype} document embeddings"):
                            batch = {key: val[i:i+batch_size].to(device) for key, val in doc_encodings.items()}
                            outputs = model(**batch)
                            doc_embs[i:i+batch_size] = mean_pooling(outputs.last_hidden_state, batch['attention_mask']).detach().cpu().numpy()
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

        if not getattr(args, 'clefip_only', False):
            def _compute_patembed_embeddings():
                query_embeddings_dict = {}
                doc_embeddings_dict = {}
                sep = getattr(model.tokenizer, 'sep_token', ' [SEP] ')
                _patembed_bs = _auto_batch_size(device, hidden_size=embedding_dim)
                for texttype in ["abstract", "claim", "invention"]:
                    if texttype == "abstract":
                        raw_query = [queries_df.iloc[i]['title'] + sep + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                        raw_doc = [documents_df.iloc[i]['title'] + sep + documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                    else:
                        raw_query = [queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                        raw_doc = [documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                    try:
                        query_embs = model.encode_query(raw_query, prompt_name=PATEN_TEB_RETRIEVAL_PROMPT_NAME, batch_size=_patembed_bs, show_progress_bar=True, convert_to_numpy=True)
                        doc_embs = model.encode_document(raw_doc, prompt_name=PATEN_TEB_RETRIEVAL_PROMPT_NAME, batch_size=_patembed_bs, show_progress_bar=True, convert_to_numpy=True)
                    except Exception:
                        PROMPT_QUERY = "encode query for mixed document retrieval: "
                        PROMPT_DOC = "encode document for mixed retrieval: "
                        query_embs = model.encode([PROMPT_QUERY + t for t in raw_query], batch_size=_patembed_bs, show_progress_bar=True, convert_to_numpy=True)
                        doc_embs = model.encode([PROMPT_DOC + t for t in raw_doc], batch_size=_patembed_bs, show_progress_bar=True, convert_to_numpy=True)
                    query_embeddings_dict[texttype] = query_embs
                    doc_embeddings_dict[texttype] = doc_embs
                q = np.concatenate([query_embeddings_dict["abstract"], query_embeddings_dict["claim"], query_embeddings_dict["invention"]], axis=0)
                d = np.concatenate([doc_embeddings_dict["abstract"], doc_embeddings_dict["claim"], doc_embeddings_dict["invention"]], axis=0)
                print(q.shape, d.shape)
                return q, d

            query_embeddings, document_embeddings = _load_or_compute_prior_art_embeddings(
                query_cache, doc_cache, _compute_patembed_embeddings, pickle_protocol=4,
            )
            prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types, save_rankings_path=save_rankings_abs, save_rankings_claim2all_path=save_rankings_claim, model_label="Patembed")


########################################################################################################################################################
########################################################################################################################################################
    elif args.model_name and args.model_name.lower() == "bm25":
        import bm25s
        import snowballstemmer

        ############################ BM25 Evaluation ############################
        if not getattr(args, 'clefip_only', False):
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
            _save_rankings(save_rankings_abs, query_ids_list, abstract_retrieved_ids, "abstract->abstract")
        
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
            _save_rankings(save_rankings_claim, query_ids_list, claim_retrieved_ids, "claim->all")
        
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
        
        if not getattr(args, 'clefip_only', False):
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
                from scipy.sparse import save_npz
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
                                            citation_mapping, query_types, doc_types,
                                            save_rankings_path=None, save_rankings_claim2all_path=None):
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
                _save_rankings(save_rankings_path, original_query_ids, predicted_labels_list, "abstract->abstract")

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
                _save_rankings(save_rankings_claim2all_path, original_query_ids, predicted_labels_list, f"{texttype_q}->all")

                _display_prior_art_results(results, results_key, retrieved_sections,
                                           query_section=texttype_q, header_suffix=" (SPLADE)")
        
            # Run SPLADE-specific evaluation
            splade_prior_art_evaluation(query_ids, doc_ids, query_embeddings_sparse, document_embeddings_sparse,
                                        citation_mapping, query_types, doc_types,
                                        save_rankings_path=save_rankings_abs,
                                        save_rankings_claim2all_path=save_rankings_claim)
        
            print("\n✅ SPLADE-v2 evaluation completed!")

########################################################################################################################################################
########################################################################################################################################################
    elif args.model_name.lower() in _COLBERT_MODEL_NAMES:
        """
        ColBERTv2 – Contextualised Late Interaction over BERT (arXiv:2112.01488)

        ColBERT encodes each text into a matrix of **token-level** 128-d embeddings.
        Scoring is via MaxSim: for every query token, take the max cosine similarity
        with any document token, then sum over all query tokens.  This is more
        expressive than a single-vector dot product but more expensive.
        """
        print(f"\n🔍 Loading ColBERTv2 model: colbert-ir/colbertv2.0")
        import pickle
        try:
            from colbert.modeling.checkpoint import Checkpoint
            from colbert.infra import ColBERTConfig as _ColBERTConfig
        except ImportError:
            print("❌  colbert-ai package not installed. Install with:  pip install colbert-ai")
            sys.exit(1)

        _colbert_config = _ColBERTConfig(doc_maxlen=512, query_maxlen=64, nbits=2)

        if not getattr(args, 'clefip_only', False):
            # ── Encode prior-art texts by section ──
            # ColBERT does **not** use section tokens; just raw text.
            _colbert_cache_dir = os.path.join(priorart_temp_dir, "colbert_embs")
            os.makedirs(_colbert_cache_dir, exist_ok=True)
            _colbert_cache_path = os.path.join(_colbert_cache_dir, "embs.pkl")

            if os.path.isfile(_colbert_cache_path):
                print("📦 Loading cached ColBERT prior-art token embeddings...")
                with open(_colbert_cache_path, "rb") as _f:
                    _colbert_embs = pickle.load(_f)
                query_embs_dict = _colbert_embs["query"]
                doc_embs_dict = _colbert_embs["doc"]
                print("✅ ColBERT embeddings loaded from cache (checkpoint not needed)")
            else:
                _colbert_ckpt = Checkpoint("colbert-ir/colbertv2.0", colbert_config=_colbert_config)
                print("✅ ColBERTv2 checkpoint loaded")
                query_embs_dict = {}
                doc_embs_dict = {}
                _bs = 32
                for texttype in ["abstract", "claim", "invention"]:
                    print(f"\n   Encoding {texttype}...")
                    if texttype == "abstract":
                        q_texts = (queries_df['title'] + '. ' + queries_df['abstract']).fillna('').tolist()
                        d_texts = (documents_df['title'] + '. ' + documents_df['abstract']).fillna('').tolist()
                    else:
                        q_texts = queries_df[texttype].fillna('').tolist()
                        d_texts = documents_df[texttype].fillna('').tolist()
                    print(f"      queries ({len(q_texts)})...", flush=True)
                    query_embs_dict[texttype] = _colbert_encode(_colbert_ckpt, q_texts, is_query=True, batch_size=_bs)
                    print(f"      documents ({len(d_texts)})...", flush=True)
                    doc_embs_dict[texttype] = _colbert_encode(_colbert_ckpt, d_texts, is_query=False, batch_size=_bs)

                with open(_colbert_cache_path, "wb") as _f:
                    pickle.dump({"query": query_embs_dict, "doc": doc_embs_dict}, _f, protocol=4)
                print(f"💾 Saved ColBERT embeddings to {_colbert_cache_path}")
                # Free GPU memory — checkpoint no longer needed after encoding
                del _colbert_ckpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            original_query_ids = list(queries.keys())
            original_doc_ids = list(documents.keys())

            # ── Abstract→Abstract ──
            print("\n🎯 ColBERT Abstract → Abstract evaluation")
            sim_abs = _colbert_maxsim_matrix(query_embs_dict["abstract"], doc_embs_dict["abstract"])
            top_k_abs = np.argsort(-sim_abs, axis=1)
            true_labels_list, predicted_labels_list = [], []
            for q_idx in range(len(original_query_ids)):
                q_id = original_query_ids[q_idx]
                true_labels_list.append(citation_mapping.get(q_id, []))
                predicted_labels_list.append([original_doc_ids[d_idx] for d_idx in top_k_abs[q_idx]])
            abs_results = _make_prior_art_metrics(true_labels_list, predicted_labels_list)
            print_metric_table(abs_results, "ColBERT: Abstract → Abstract")
            _save_rankings(save_rankings_abs, original_query_ids, predicted_labels_list, "abstract->abstract")

            # ── Claim→All ──
            print("\n🎯 ColBERT Claim → All evaluation")
            # Query: claim embeddings.  Docs: abstract + claim + invention (3× stacked)
            q_claim_embs = query_embs_dict["claim"]
            d_all_embs = doc_embs_dict["abstract"] + doc_embs_dict["claim"] + doc_embs_dict["invention"]
            sim_all = _colbert_maxsim_matrix(q_claim_embs, d_all_embs)
            n_orig = len(original_doc_ids)
            top_k_all = np.argsort(-sim_all, axis=1)[:, :300]
            true_labels_list, predicted_labels_list = [], []
            retrieved_sections = []
            section_names = ["abstract", "claim", "invention"]
            for q_idx in range(len(original_query_ids)):
                q_id = original_query_ids[q_idx]
                true_labels_list.append(citation_mapping.get(q_id, []))
                seen = set()
                preds, secs = [], []
                for d_idx in top_k_all[q_idx]:
                    orig_d = d_idx % n_orig
                    doc_id = original_doc_ids[orig_d]
                    if doc_id not in seen:
                        seen.add(doc_id)
                        preds.append(doc_id)
                        secs.append(section_names[d_idx // n_orig])
                predicted_labels_list.append(preds[:100])
                retrieved_sections.append(secs[:100])
            results = {"claim->all": _make_prior_art_metrics(
                true_labels_list, predicted_labels_list,
                retrieved_sections=f"[{len(retrieved_sections)} queries with retrieved sections]",
            )}
            _save_rankings(save_rankings_claim, original_query_ids, predicted_labels_list, "claim->all")
            _display_prior_art_results(results, "claim->all", retrieved_sections,
                                       query_section="claim", header_suffix=" (ColBERT)")

            print("\n✅ ColBERT prior-art evaluation completed!")

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

        # device already set at start of main()
        # Setup model for inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.to(device).eval()
        print(f"🚀 Model ready on {device}")
        batch_size = _auto_batch_size(device, hidden_size=embedding_dim)

        if not getattr(args, 'clefip_only', False):
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
                    # Free GPU cache between text types to avoid fragmentation OOM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
                # For compatibility with existing evaluation code, we'll create the concatenated versions
                # But the evaluation should use the separated versions to match patent.py exactly
                query_embeddings = np.concatenate([query_embeddings_dict["abstract"], query_embeddings_dict["claim"], query_embeddings_dict["invention"]], axis=0)
                document_embeddings = np.concatenate([doc_embeddings_dict["abstract"], doc_embeddings_dict["claim"], doc_embeddings_dict["invention"]], axis=0)

                print(query_embeddings.shape, document_embeddings.shape)

                torch.save(query_embeddings, f'{priorart_temp_dir}/query_embeddings.pt')
                torch.save(document_embeddings, f'{priorart_temp_dir}/document_embeddings.pt')
                np.savez(f'{priorart_temp_dir}/query_embeddings_by_type.npz', **query_embeddings_dict)
                np.savez(f'{priorart_temp_dir}/doc_embeddings_by_type.npz', **doc_embeddings_dict)

            query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
            document_embeddings = np.asarray(document_embeddings, dtype=np.float32)

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
        
        def _encode_query_spans(texts: list[str], section: str, d: int, batch_size: int = 32) -> list[np.ndarray]:
            all_query_spans: list[np.ndarray] = []
            doc_ids = [f"query_{i}" for i in range(len(texts))]
            sections = [section for _ in range(len(texts))]
            
            for batch_start in range(0, len(texts), batch_size):
                batch_end = min(batch_start + batch_size, len(texts))
                batch_texts = texts[batch_start:batch_end]
                batch_sections = sections[batch_start:batch_end]
                batch_doc_ids = doc_ids[batch_start:batch_end]
                
                batch_results = _encode_spans(
                    doc_texts=batch_texts,
                    doc_ids=batch_doc_ids,
                    sections=batch_sections,
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
                br = _encode_spans(
                    doc_texts=batch_texts[b_start:b_end],
                    doc_ids=batch_doc_ids[b_start:b_end],
                    sections=batch_sections[b_start:b_end],
                )
                all_batch_results.extend(br)
            # Group spans by doc_id (= "query_{qidx}_{cidx}")
            from collections import defaultdict
            doc_id_to_spans: dict[str, list[np.ndarray]] = defaultdict(list)
            for doc_id, _sec, _dt, _raw, _canon, span_emb in all_batch_results:
                if query_exclude_cls:
                    if (_canon or "").strip().lower() == "cls" or (_raw or "").strip() == "[CLS]":
                        continue
                doc_id_to_spans[doc_id].append(span_emb)

            # For encoder_token unit, each output span = one non-special token.
            # Each chunk's encoder input is: [CLS] prefix_tokens... abstract_chunk_tokens... [SEP]
            # After filtering specials (and optionally CLS), the first n_prefix output
            # spans are prefix (title+section marker), the rest are abstract tokens.
            # offset_base for chunk 0 = len(prefix_ids), giving us n_prefix.
            # We keep prefix spans only from chunk 0, and dedup abstract spans by their
            # position in the original (pre-chunked) abstract sequence.
            num_queries = len(texts)
            per_query_prefix: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
            per_query_abstract: dict[int, list[tuple[int, np.ndarray, bool]]] = defaultdict(list)

            for doc_id, span_list in doc_id_to_spans.items():
                parts = doc_id.split("_")
                if len(parts) != 3 or parts[0] != "query":
                    continue
                qidx = int(parts[1])
                cidx = int(parts[2])
                n_prefix = chunk_meta.get((qidx, 0), 0)
                abstract_start = chunk_meta.get((qidx, cidx), 0) - n_prefix
                for local_idx, span_emb in enumerate(span_list):
                    if local_idx < n_prefix:
                        if cidx == 0:
                            per_query_prefix[qidx].append((local_idx, span_emb))
                    else:
                        abs_pos = abstract_start + (local_idx - n_prefix)
                        per_query_abstract[qidx].append((abs_pos, span_emb, cidx == 0))

            all_query_spans = []
            all_query_weights: Optional[list[np.ndarray]] = None if chunk_weight_mode == "uniform" else []
            for qidx in range(num_queries):
                prefix_spans = sorted(per_query_prefix.get(qidx, []), key=lambda x: x[0])
                abs_entries = per_query_abstract.get(qidx, [])
                abs_entries.sort(key=lambda x: (x[0], -x[2]))
                seen: set[int] = set()
                deduped_abs: list[tuple[int, np.ndarray, bool]] = []
                for abs_pos, span_emb, from_first in abs_entries:
                    if abs_pos in seen:
                        continue
                    seen.add(abs_pos)
                    deduped_abs.append((abs_pos, span_emb, from_first))
                deduped_abs.sort(key=lambda x: x[0])

                kept_embs = [e for _, e in prefix_spans] + [e for _, e, _ in deduped_abs]
                if not kept_embs:
                    all_query_spans.append(np.zeros((0, d), dtype=np.float32))
                    if all_query_weights is not None:
                        all_query_weights.append(np.array([], dtype=np.float32))
                else:
                    all_query_spans.append(np.stack(kept_embs, axis=0).astype(np.float32))
                    if chunk_weight_mode == "first" and all_query_weights is not None:
                        w = np.ones(len(kept_embs), dtype=np.float32)
                        for i in range(len(prefix_spans)):
                            w[i] = first_chunk_weight
                        for i, (_, _, from_first) in enumerate(deduped_abs):
                            if from_first:
                                w[len(prefix_spans) + i] = first_chunk_weight
                        all_query_weights.append(w)
            return all_query_spans, all_query_weights
        
        def _assign_query_spans_to_centers(
            query_spans: list[np.ndarray],
            center_index: faiss.Index,
            V: int,
            sim_thr_per_center: np.ndarray,
            idf: Optional[np.ndarray] = None,
            center_pca_dirs: Optional[np.ndarray] = None,
            query_span_weights: Optional[list[np.ndarray]] = None,
            stop_centers: Optional[set] = None,
        ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
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

                if use_soft_assignment:
                    max_centers_per_span = int(getattr(args, "soft_assignment_max_centers_per_span", 10) or 0)
                    K_search = min(max(max_centers_per_span * 4, 64), V)
                    _min_thr = float(sim_thr_per_center.min())
                    D_q, I_q = center_index.search(spans_norm, K_search)
                    center_weights = {}
                    center_projs: dict[int, float] = {}
                    center_max_sim: dict[int, float] = {} if weight_agg == "sum" else None
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
                            proj = float(spans_norm[span_idx] @ center_pca_dirs[c]) if center_pca_dirs is not None else 0.0
                            _update_weight(center_weights, c, sim, span_idx, span_downweight=1.0, span_extra_weight=extra, center_projs=center_projs, center_max_sim=center_max_sim, proj=proj)
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
        
        # ---- Pre-cache doc embeddings for all potential modes ----
        # This runs BEFORE center search so that even if centers are not
        # built yet the (expensive) doc-side embeddings are cached for later.
        _mode_to_sections = {
            "abstract2abstract": ["abstract"],
            "claim2all": ["abstract", "claim", "invention"],
        }
        for _pre_mode, _pre_sections in _mode_to_sections.items():
            _pre_cache = _doc_cache_dir(_pre_mode)
            if _doc_cache_exists(_pre_cache, _pre_sections):
                print(f"   ✅ Doc cache already exists for {_pre_mode}")
            else:
                print(f"   📦 Pre-caching doc embeddings for {_pre_mode}...")
                _pre_emb, _pre_s2d, _pre_cls = _encode_doc_spans(
                    documents, _pre_sections, batch_size=32
                )
                _save_doc_cache(_pre_cache, _pre_sections, _pre_emb, _pre_s2d, _pre_cls)
                print(f"   ✅ Cached {sum(e.shape[0] for e in _pre_emb.values()):,} spans for {_pre_mode}")
        
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
                FULL_CORPUS_DIR_EN as _CLEFIP_FULL_DIR,
                CORPUS_JSONL as _CLEFIP_JSONL,
                IDS_TXT as _CLEFIP_IDS,
            )
            _doc_root = os.path.join(_clefip_root, "01_document_collection", "01_extracted")
            if not os.path.isdir(_doc_root):
                raise FileNotFoundError(f"CLEF-IP document collection not found: {_doc_root}")
            _sample_size = getattr(args, "clefip_sample_size", 0) or 0
            _rebuild = getattr(args, "clefip_rebuild_corpus", False)
            if _sample_size != 0:
                _cq_ids, _cq_texts, _c_jsonl, _c_ids_txt, _c_npsg, _c_qrels = _load_clefip_sampled(
                    _clefip_root, _doc_root, sample_size=_sample_size, rebuild_corpus=_rebuild
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
        
        if getattr(args, 'clefip_only', False):
            available_modes = ["clefip_passage"] if _clefip_data is not None else []
        else:
            available_modes = ["abstract2abstract", "claim2all"]
            if _clefip_data is not None:
                available_modes.append("clefip_passage")
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
        center_dot_pca: Optional[np.ndarray] = None
        if center_pca_dirs is not None and residual_alpha != 0.0:
            center_dot_pca = np.array([float(centers_norm[c] @ center_pca_dirs[c]) for c in range(V)], dtype=np.float32)
        
        for mode in available_modes:
            print(f"\n{'='*80}")
            print(f"Processing task: {mode}")
            print(f"{'='*80}")

            # Decide which sections to use for document indexing
            if mode == "abstract2abstract":
                doc_sections = ["abstract"]
                query_section = "abstract"
            elif mode == "claim2all":
                doc_sections = ["abstract", "claim", "invention"]
                query_section = "claim"
            elif mode == "clefip_passage":
                doc_sections = ["abstract", "claim", "invention"]
                query_section = "claim"
            else:
                doc_sections = ["abstract"]
                query_section = "abstract"
            
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
                if mode == "clefip_passage":
                    print(f"\n📦 Encoding CLEF-IP passages at runtime (first run; will cache for reuse)")
                    total_loaded = _encode_clefip_spans(
                        _clefip_data["passage_ids"], _clefip_data["passage_texts"],
                        doc_sections, cache_dir=cache_dir, batch_size=32,
                    )
                    # Use lazy path for the just-written cache
                    span_to_doc, exclude_cls_span_indices = _load_doc_cache_meta(cache_dir)
                    _lazy_cache_dir = cache_dir
                else:
                    print(f"\n📦 Encoding documents at runtime (first run; will cache for reuse)")
                    embeddings_by_section, span_to_doc, exclude_cls_span_indices = _encode_doc_spans(
                        documents, doc_sections, batch_size=32
                    )
                    total_loaded = sum(e.shape[0] for e in embeddings_by_section.values())
                    _save_doc_cache(cache_dir, doc_sections, embeddings_by_section, span_to_doc, exclude_cls_span_indices)

            if exclude_cls_spans:
                print(f"   Excluding {len(exclude_cls_span_indices):,} CLS spans")
            print(f"   Total: {len(span_to_doc):,} span-to-doc mappings, {total_loaded:,} embedding rows")

            # Build per-doc span count for length normalization (stable, pre-filtering)
            if mode == "clefip_passage":
                _clefip_pid_list = _clefip_data["passage_ids"]
                doc_id_to_idx = {pid: idx for idx, pid in enumerate(_clefip_pid_list)}
                N_docs = len(_clefip_pid_list)
            else:
                doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(documents_df.index)}
                N_docs = len(documents_df)
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
                    del _raw; import gc; gc.collect()  # free before normalize
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
                    del sec_emb; import gc; gc.collect()
                print(f"   Total spans: {total_loaded:,}")
            else:
                # Doc soft: search(K) + per-center threshold filter + topK cap
                max_centers_per_span = int(getattr(args, "soft_assignment_max_centers_per_span", 10) or 0)
                K_search = min(max(max_centers_per_span * 4, 64), V)
                min_sim_thr = float(sim_thr_per_center.min())
                print(f"   Soft assignment: search(K={K_search}) + per-center r_c filter + topK={max_centers_per_span}")
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
                    del _raw; import gc; gc.collect()  # free original before normalize
                    faiss.normalize_L2(sec_emb_n)
                    batch_size = max(1, int(getattr(args, "posting_list_batch_size", 4096)))
                    for b_start in tqdm(range(0, sec_emb_n.shape[0], batch_size),
                                        desc=f"  {section_name} spans->centers", leave=False):
                        b_end = min(b_start + batch_size, sec_emb_n.shape[0])
                        batch = sec_emb_n[b_start:b_end]
                        D_batch, I_batch = center_index.search(batch, K_search)
                        for j in range(batch.shape[0]):
                            global_idx = span_offset + b_start + j
                            if exclude_cls_spans and global_idx in exclude_cls_span_indices:
                                continue
                            kept = 0
                            for k in range(K_search):
                                c = int(I_batch[j, k])
                                if c < 0:
                                    break
                                sim_val = float(D_batch[j, k])
                                if sim_val < min_sim_thr:
                                    break
                                if c in stop_centers or sim_val <= 0:
                                    continue
                                if sim_val < sim_thr_per_center[c]:
                                    continue
                                proj = float(batch[j] @ center_pca_dirs[c]) if center_pca_dirs is not None else 0.0
                                posting_lists[c].append((global_idx, sim_val, proj))
                                kept += 1
                                if max_centers_per_span > 0 and kept >= max_centers_per_span:
                                    break
                    span_offset += sec_emb_n.shape[0]
                    print(f"     {section_name}: {sec_emb_n.shape[0]:,} spans assigned")
                    del sec_emb_n; import gc; gc.collect()
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
            
            # Build document-level inverted index (doc_idx, weight, proj)
            print(f"\n🔨 Building document-level inverted index from posting lists...")
            doc_postings: list[list[tuple[int, float, float]]] = [[] for _ in range(V)]
            
            weight_agg = getattr(args, "weight_aggregation", "max")
            for center_idx in tqdm(range(V), desc="Building inverted index"):
                span_sims = posting_lists[center_idx]
                if not span_sims:
                    continue
                agg: dict[int, tuple[float, float, float]] = {}  # doc_idx -> (weight, max_sim_seen, proj_of_max)
                for entry in span_sims:
                    if len(entry) == 3:
                        span_idx, similarity, proj = entry
                    else:
                        span_idx, similarity, proj = entry[0], entry[1], 0.0
                    doc_id = span_to_doc.get(span_idx, None)
                    if doc_id is None:
                        continue
                    didx = doc_id_to_idx.get(doc_id)
                    if didx is None:
                        continue
                    sim = float(similarity)
                    p = float(proj)
                    if weight_agg == "sum":
                        if didx in agg:
                            ow, max_s, op = agg[didx]
                            new_w = ow + max(0.0, sim)
                            agg[didx] = (new_w, max(max_s, sim), op if max_s >= sim else p)
                        else:
                            agg[didx] = (max(0.0, sim), sim, p)
                    else:
                        if didx not in agg or max(0.0, sim) > agg[didx][0]:
                            agg[didx] = (max(0.0, sim), sim, p)
                for didx, wproj in agg.items():
                    doc_postings[center_idx].append((didx, float(wproj[0]), float(wproj[2])))
            
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

            N_docs = len(_clefip_data["passage_ids"]) if mode == "clefip_passage" else len(documents_df)
            df = np.array([len(pl) for pl in doc_postings], dtype=np.float32)  # doc_idx unique per center (aggregated above)
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
            elif mode == "clefip_passage":
                print(f"\n📝 Evaluating: CLEF-IP Claims -> Passages")
                _fmt_scheme = get_encoder_format_scheme(args.dense_model)
                query_texts = [format_claim_for_encoder(_fmt_scheme, qt) for qt in _clefip_data["query_texts"]]
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
            query_sparse = _assign_query_spans_to_centers(query_spans, center_index=center_index, V=V, sim_thr_per_center=sim_thr_per_center, idf=idf, center_pca_dirs=center_pca_dirs, query_span_weights=query_span_weights, stop_centers=stop_centers)
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
            top_indices = _score_queries_against_postings(
                query_sparse, doc_postings, idf, idf_exponent, top_k,
                pca_proj_alpha=pca_proj_alpha,
                angle_sim_beta=angle_sim_beta,
                residual_alpha=residual_alpha,
                center_dot_pca=center_dot_pca,
                length_norm=length_norm,
                length_norm_exp=length_norm_exp,
                doc_nspans=doc_nspans,
            )
            
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

            if mode == "clefip_passage":
                # CLEF-IP passage-level evaluation: retrieve passage_ids, use CLEF-IP official metrics
                _clefip_pid_list = _clefip_data["passage_ids"]
                _clefip_qids = _clefip_data["query_ids"]
                _clefip_qrels = _clefip_data["qrels_passage_ids"]

                predicted_labels_list: list[list[str]] = []
                for q_idx in range(len(_clefip_qids)):
                    if q_idx < len(top_indices) and top_indices[q_idx]:
                        predicted_labels_list.append([_clefip_pid_list[pi] for pi in top_indices[q_idx]])
                    else:
                        predicted_labels_list.append([])

                topk_docs = getattr(args, "clefip_two_stage_topk_docs", 100)
                # Build per-query score dicts for two-stage reranking
                all_passage_scores_list: list[dict] = []
                for q_idx in range(len(_clefip_qids)):
                    # Reconstruct passage scores from doc_scores in retrieval
                    # We need the full doc_scores — re-derive from top_indices
                    # (scores are not stored, so just use rank-based two-stage)
                    pscores = {_clefip_pid_list[pi]: float(top_k - rank)
                               for rank, pi in enumerate(top_indices[q_idx])} if q_idx < len(top_indices) else {}
                    all_passage_scores_list.append(pscores)
                predicted_labels_list = _clefip_two_stage_rerank(
                    _clefip_pid_list, predicted_labels_list, all_passage_scores_list,
                    topk_docs=topk_docs,
                )
                print(f"  🔄 Two-stage retrieval: top-{topk_docs} docs → re-ranked passages per query")

                _evaluate_and_print_clefip(
                    _clefip_qrels, _clefip_qids, predicted_labels_list,
                    "Sparse Coverage (CLEF-IP)",
                    save_path=save_rankings_paths.get("clefip_passage") if save_rankings_paths else None,
                    two_stage=True, topk_docs=topk_docs,
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
                            [(di, w, p) for di, w, p in pl if di in _allowed_pidx]
                            for pl in doc_postings
                        ]

                        # Recompute IDF with filtered pool
                        _df_f = np.array(
                            [len(pl) for pl in _dp_f],
                            dtype=np.float32,
                        )
                        _idf_f = (np.log((_n_passages_f + 1.0) / (_df_f + 1.0)) + 1.0).astype(np.float32)

                        # Retrieve with filtered inverted index
                        _top_indices_f = _score_queries_against_postings(
                            query_sparse, _dp_f, _idf_f, idf_exponent, top_k,
                            pca_proj_alpha=pca_proj_alpha,
                            angle_sim_beta=angle_sim_beta,
                            residual_alpha=residual_alpha,
                            center_dot_pca=center_dot_pca,
                            length_norm=length_norm,
                            length_norm_exp=length_norm_exp,
                            doc_nspans=doc_nspans,
                            show_progress=False,
                        )

                        # Build predicted labels and evaluate
                        _pred_f: list[list[str]] = []
                        for _qi_e in range(len(_clefip_qids)):
                            if _qi_e < len(_top_indices_f) and _top_indices_f[_qi_e]:
                                _pred_f.append([_clefip_pid_list[pi] for pi in _top_indices_f[_qi_e]])
                            else:
                                _pred_f.append([])

                        _true_f = [_clefip_qrels.get(qid, []) for qid in _clefip_qids]
                        _res_f = _make_clefip_official_metrics(_true_f, _pred_f)
                        _robustness_results.append((_n_docs_f, _n_passages_f, _res_f))
                        print(f"   {_n_docs_f:>6} docs ({_n_passages_f:>8,} passages): "
                              f"recall@100={_res_f.get('recall@100', 0):.4f}  "
                              f"ndcg@10={_res_f.get('ndcg@10', 0):.4f}  "
                              f"map={_res_f.get('map', 0):.4f}  "
                              f"pres_doc@100={_res_f.get('pres_doc@100', 0):.4f}")

                    # Summary table
                    if _robustness_results:
                        print(f"\n{'='*90}")
                        print(f"CLEF-IP Robustness Summary (relevant docs always included, negatives subsampled)")
                        print(f"{'='*90}")
                        print(f"{'Docs':>8} {'Passages':>10} {'recall@100':>12} {'ndcg@10':>10} "
                              f"{'map':>8} {'pres_doc@100':>14} {'magp':>8}")
                        print(f"{'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*8} {'-'*14} {'-'*8}")
                        for _nd, _np, _r in _robustness_results:
                            print(f"{_nd:>8} {_np:>10,} "
                                  f"{_r.get('recall@100', 0):>12.4f} "
                                  f"{_r.get('ndcg@10', 0):>10.4f} "
                                  f"{_r.get('map', 0):>8.4f} "
                                  f"{_r.get('pres_doc@100', 0):>14.4f} "
                                  f"{_r.get('magp', 0):>8.4f}")
                        print(f"{'='*90}")

            else:
                task_label = "Abstract -> Abstract" if mode == "abstract2abstract" else "Claim -> All"
                results, _, retrieved_ids_list = _compute_metrics_for_qids(None)

                print_metric_table(results, f"Sparse Coverage: {task_label}")
                if mode == "abstract2abstract":
                    _save_rankings(save_rankings_abs, list(queries_df.index), retrieved_ids_list, "abstract->abstract")
                else:
                    _save_rankings(save_rankings_claim, list(queries_df.index), retrieved_ids_list, "claim->all")
            
            print(f"\n✅ Task {mode} evaluation completed")
            
            # Free large objects before next mode to avoid peak memory overlap
            del embeddings_by_section, span_to_doc, exclude_cls_span_indices
            del posting_lists, doc_postings, top_indices, query_sparse
            import gc; gc.collect()
        
        print(f"\n✅ Sparse Coverage evaluation completed for all available tasks")

    ############################################## CLEF-IP 2013 EN (claims-to-passages) ##################################################
    if not (args.model_name == "sparse_coverage" and _clefip_data is not None):
        # Only run standalone CLEF-IP eval if sparse_coverage didn't already handle it in the mode loop
        print("\n" + "=" * 60)
        print("Running CLEF-IP 2013 EN (claims-to-passages)")
        print("=" * 60)
        run_clefip_eval(args, save_rankings_path=save_rankings_paths.get("clefip_passage"))


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