#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3build_centers_kcenter.py

Build vocabulary centers via K-Center (farthest-first traversal).
Alternative to greedy covering and quantile sampling with principled guarantees:

  - V is fixed (= semantic unit granularity)
  - Centers are maximally spread out (2-approximation to K-center problem)
  - Per-center r_c is derived from Voronoi assignment (not a hyperparameter)
  - Coverage is guaranteed by construction (100% with max, ~tau% with percentile)
  - Overlap is naturally low (centers are pushed apart)

Algorithm:
  1. Farthest-first traversal: iteratively pick the point farthest from all
     existing centers. Repeat V times. (Gonzalez, 1985)
  2. Voronoi assignment: assign every point to its nearest center.
  3. Per-center r_c = max (or percentile) of cosine distances within each
     Voronoi cell.

Output:
  - centers_greedy_r{nominal_r}.npy  (baselines-compatible)
  - centers_greedy_r{nominal_r}.json with r_per_center, coverage, stats

Section sampling (--section_sampling):
  - equal + max_per_section: cap each section at the same count; can bias centers toward
    small sections and produce odd df/r_c patterns. Prefer proportional when subsampling.
  - proportional + ref_size: stratified by section size (n_j/N); keeps center distribution
    aligned with corpus.
  - To use only retrieval-candidate side (e.g. avoid query vs doc imbalance), pass
    only the candidate-side embedding files in --embeddings_dir.

Centers are task-agnostic: all available sections (abstract, claim, invention)
are loaded and used for center construction. The same vocabulary is shared
across abstract2abstract and claim2all evaluation tasks.

Usage with evaluate.py: --centers_suffix "_kcenter_V{V}"
"""

import os
import json
import time
import math
import argparse
import numpy as np

from utils import (
    parse_embeddings_dir,
    find_embedding_files,
    l2_normalize_inplace,
)

# ---------------------------------------------------------------------------
# GPU helpers (lazy imports; used when CUDA is available)
# ---------------------------------------------------------------------------

def _check_gpu():
    """Return True if CUDA is available via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _load_span_doc_ids(embedding_files, N):
    """
    Load span index -> doc_id from per-section metadata JSONL (same order as embedding files).
    Each line: {"d": doc_id, ...}. Returns list of length N (doc_id per global span index) or None if missing/mismatch.
    """
    doc_ids = []
    for fp in embedding_files:
        meta_path = os.path.splitext(fp)[0] + "_metadata.jsonl"
        if not os.path.isfile(meta_path):
            return None
        try:
            # Batch read entire file at once
            with open(meta_path, "r") as f:
                content = f.read()
            lines = content.split("\n")
            del content  # free memory early
            file_ids = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                file_ids.append(obj.get("d", obj.get("doc_id", "")))
            doc_ids.extend(file_ids)
        except Exception:
            return None
    if len(doc_ids) != N:
        return None
    return doc_ids


def _filter_embeddings_by_span_kind(embedding_files, section_embeddings, exclude_cls=False, exclude_doc_mean=False):
    """
    Filter embedding rows by span_kind using metadata JSONL.

    For each section .npy, reads the companion _metadata.jsonl and keeps only
    rows whose 'k' field is NOT in the excluded set.  Returns filtered arrays
    (loaded into memory, no longer mmap) and the filtered doc_ids list.

    Returns:
        filtered_embeddings: list of np.ndarray (one per section, content-only rows)
        filtered_doc_ids: list of str (doc_id per kept row, across all sections)
    """
    exclude_kinds = set()
    if exclude_cls:
        exclude_kinds.add("cls")
    if exclude_doc_mean:
        exclude_kinds.add("doc_mean")
    if not exclude_kinds:
        return section_embeddings, None  # nothing to filter

    filtered_embeddings = []
    filtered_doc_ids = []
    total_before = 0
    total_after = 0

    for fp, arr in zip(embedding_files, section_embeddings):
        meta_path = os.path.splitext(fp)[0] + "_metadata.jsonl"
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Metadata file required for --exclude_cls/--exclude_doc_mean but not found: {meta_path}\n"
                f"Re-run 1create_N_embeddings.py with --save_metadata 1"
            )

        n_rows = arr.shape[0]
        total_before += n_rows

        # Read span_kind for every row
        keep_mask = np.ones(n_rows, dtype=bool)
        doc_ids_section = []
        with open(meta_path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                kind = obj.get("k", "content")
                doc_id = obj.get("d", obj.get("doc_id", ""))
                if kind in exclude_kinds:
                    keep_mask[i] = False
                doc_ids_section.append(doc_id)

        if len(doc_ids_section) != n_rows:
            raise ValueError(
                f"Metadata row count ({len(doc_ids_section)}) != embedding rows ({n_rows}) for {fp}"
            )

        # Apply mask: copy into memory (no longer mmap)
        kept_arr = arr[keep_mask]
        kept_doc_ids = [d for d, m in zip(doc_ids_section, keep_mask) if m]
        filtered_embeddings.append(kept_arr)
        filtered_doc_ids.extend(kept_doc_ids)
        total_after += kept_arr.shape[0]
        excluded = n_rows - kept_arr.shape[0]
        print(f"  {os.path.basename(fp)}: {n_rows:,} -> {kept_arr.shape[0]:,} (excluded {excluded:,} {exclude_kinds})")

    print(f"[kcenter] Span filtering: {total_before:,} -> {total_after:,} "
          f"(removed {total_before - total_after:,} rows with kind in {exclude_kinds})")
    return filtered_embeddings, filtered_doc_ids


def _eligible_indices_proportional(store, ref_size, rng):
    """
    Stratified proportional sampling: sample ref_size indices so each section j
    contributes proportionally to its size (n_j / N). Preserves corpus distribution
    and avoids bias toward small sections (no equal cap).
    """
    n_sections = len(store.cumsum) - 1
    n_per = np.diff(store.cumsum).astype(np.int64)
    N = int(store.cumsum[-1])
    # target count per section (float), then round and clamp to [0, n_per]
    targets = ref_size * (n_per / N)
    counts = np.round(targets).astype(np.int64)
    np.clip(counts, 0, n_per, out=counts)
    # ensure total <= ref_size; if we rounded down a lot, sum may be < ref_size
    total = int(counts.sum())
    if total > ref_size:
        need_remove = total - ref_size
        # remove from largest sections first
        for j in np.argsort(-counts):
            take = min(counts[j], need_remove)
            counts[j] -= take
            need_remove -= take
            if need_remove <= 0:
                break
    per_section = []
    for j in range(n_sections):
        c = int(counts[j])
        if c <= 0:
            continue
        n = int(n_per[j])
        if c >= n:
            idx_local = np.arange(n)
        else:
            idx_local = rng.choice(n, size=c, replace=False)
        per_section.append(store.cumsum[j] + idx_local)
    if not per_section:
        return np.array([], dtype=np.int64)
    return np.concatenate(per_section)


def _span_doc_ids_to_ints(span_doc_ids):
    """Map a list of opaque doc ID strings to a contiguous int32 array (and return the dict).

    Used by both DF diagnostic implementations to build a compact ``(N,)`` array suitable
    for ``np.unique`` on ``(center_id, doc_int)`` pairs.
    """
    doc_to_int: dict[str, int] = {}
    doc_ints = np.empty(len(span_doc_ids), dtype=np.int32)
    for i, did in enumerate(span_doc_ids):
        if did not in doc_to_int:
            doc_to_int[did] = len(doc_to_int)
        doc_ints[i] = doc_to_int[did]
    return doc_ints, doc_to_int


def _df_quantiles(df_per_center):
    """Standard {p50, p90, p99, max} dict from a per-center DF array."""
    q = np.percentile(df_per_center, [50, 90, 99])
    return {"p50": int(q[0]), "p90": int(q[1]), "p99": int(q[2]), "max": int(np.max(df_per_center))}


def _compute_df_diagnostic(assignments, assign_dists, span_doc_ids, r_per_center, V_actual, top_k=10):
    """
    Compute document-frequency (df) per center: number of distinct documents that activate each center.
    O(N log N) via sort-based unique counting instead of O(V*N) per-center masking.
    Returns (df_per_center, quantiles, top_df_centers).
    """
    doc_ints, _ = _span_doc_ids_to_ints(span_doc_ids)

    # Sort-based df: unique (center, doc) pairs, then bincount on center
    pairs = np.stack([assignments, doc_ints], axis=1)  # (N, 2)
    unique_pairs = np.unique(pairs, axis=0)             # deduplicated (center, doc)
    df_per_center = np.bincount(unique_pairs[:, 0].astype(np.int64), minlength=V_actual).astype(np.int64)

    quantiles = _df_quantiles(df_per_center)

    # Top-k by df: only need per-center argmin(assign_dists) for the top-k centers
    top_indices = np.argsort(-df_per_center)[:top_k]
    # Vectorized: sort by distance, then np.unique on assignments finds first (=closest) per center
    order = np.argsort(assign_dists)
    sorted_assignments = assignments[order]
    unique_centers, first_idx = np.unique(sorted_assignments, return_index=True)
    rep_span_for_center = np.full(V_actual, -1, dtype=np.int64)
    rep_span_for_center[unique_centers] = order[first_idx]

    top_df_centers = []
    for c in top_indices:
        rep_i = int(rep_span_for_center[c])
        if rep_i < 0:
            top_df_centers.append({"center_id": int(c), "df": 0, "r_c": float(r_per_center[c]), "rep_span_idx": None, "rep_span_dist": None})
            continue
        top_df_centers.append({
            "center_id": int(c),
            "df": int(df_per_center[c]),
            "r_c": float(r_per_center[c]),
            "rep_span_idx": rep_i,
            "rep_span_dist": float(assign_dists[rep_i]),
        })
    return df_per_center, quantiles, top_df_centers


def _compute_df_diagnostic_activation(
    store, center_index, N, d, use_gpu,
    span_doc_ids, r_per_center, V_actual,
    max_centers_per_span=10, top_k_report=10,
):
    """
    Activation-based DF diagnostic: matches evaluate.py's soft assignment.

    For each span, search(K) nearest centers, filter by per-center r_c,
    cap at max_centers_per_span.  df_c = |{doc : exists span in doc activating c}|.
    This reflects the actual posting-list lengths in retrieval and is typically
    much larger than the Voronoi df (nearest-1 only).

    Returns (df_per_center, quantiles, top_report_centers, activation_stats).
    """
    import time as _time
    t0 = _time.time()

    doc_ints, _ = _span_doc_ids_to_ints(span_doc_ids)

    # Precompute per-center similarity threshold: sim >= 1 - r_c[c]
    sim_thr_per_center = (1.0 - r_per_center).astype(np.float32)
    min_sim_thr = float(sim_thr_per_center.min())

    # K_search: match evaluate.py logic — search more than max_centers_per_span
    # to allow filtering by r_c before capping
    K_search = min(max(max_centers_per_span * 4, 64), V_actual)

    assign_batch = _adaptive_batch_size(d, use_gpu)

    # Collect (center_id, doc_int) pairs in streaming batches
    all_centers_list = []
    all_docs_list = []
    total_activations = 0

    for start in range(0, N, assign_batch):
        end = min(start + assign_batch, N)
        batch = store.get_chunk(start, end)
        l2_normalize_inplace(batch)
        sims, idxs = center_index.search(batch, K_search)  # (batch_size, K_search)

        batch_size = end - start
        batch_doc = doc_ints[start:end]  # (batch_size,)

        # Vectorized filtering: flatten (batch, K) -> (batch*K,)
        center_ids_flat = idxs.ravel().astype(np.int64)   # (batch*K,)
        sims_flat = sims.ravel()                           # (batch*K,)
        docs_flat = np.repeat(batch_doc, K_search)         # (batch*K,)
        span_indices = np.repeat(np.arange(batch_size), K_search)  # for topK cap

        # Filter 1: valid center id and positive similarity
        valid = (center_ids_flat >= 0) & (sims_flat > 0)
        # Filter 2: global min threshold (early exit equivalent)
        valid &= sims_flat >= min_sim_thr
        # Filter 3: per-center r_c threshold
        valid &= sims_flat >= sim_thr_per_center[center_ids_flat]

        # Apply validity filter
        v_centers = center_ids_flat[valid]
        v_sims = sims_flat[valid]
        v_docs = docs_flat[valid]
        v_spans = span_indices[valid]

        # Filter 4: top-K cap per span (keep at most max_centers_per_span per span)
        # FAISS returns results sorted by descending sim per span; after boolean
        # masking the relative order within each span is preserved.
        # Fully vectorized group-wise cumcount via sort + maximum.accumulate.
        if max_centers_per_span > 0 and len(v_centers) > 0:
            order = np.argsort(v_spans, kind='stable')
            sorted_spans = v_spans[order]
            positions = np.arange(len(sorted_spans), dtype=np.int64)
            # Detect group boundaries (new span id)
            changes = np.empty(len(sorted_spans), dtype=np.bool_)
            changes[0] = True
            np.not_equal(sorted_spans[1:], sorted_spans[:-1], out=changes[1:])
            # Propagate each group's start position forward
            group_start = np.maximum.accumulate(np.where(changes, positions, 0))
            cumcount_sorted = positions - group_start   # 0, 1, 2, ... within each group
            # Unsort back to original order
            cumcount = np.empty_like(cumcount_sorted)
            cumcount[order] = cumcount_sorted
            keep_mask = cumcount < max_centers_per_span
            v_centers = v_centers[keep_mask]
            v_docs = v_docs[keep_mask]

        all_centers_list.append(v_centers)
        all_docs_list.append(v_docs)
        total_activations += len(v_centers)

    # Concatenate all batches
    all_centers = np.concatenate(all_centers_list) if all_centers_list else np.array([], dtype=np.int64)
    all_docs = np.concatenate(all_docs_list) if all_docs_list else np.array([], dtype=np.int32)

    # Unique (center, doc) pairs -> df per center
    if len(all_centers) > 0:
        pairs = np.stack([all_centers, all_docs.astype(np.int64)], axis=1)
        unique_pairs = np.unique(pairs, axis=0)
        df_per_center = np.bincount(unique_pairs[:, 0].astype(np.int64), minlength=V_actual).astype(np.int64)
    else:
        df_per_center = np.zeros(V_actual, dtype=np.int64)

    quantiles = _df_quantiles(df_per_center)

    # Top centers by activation df
    top_indices = np.argsort(-df_per_center)[:top_k_report]
    top_df_centers = []
    for c in top_indices:
        top_df_centers.append({
            "center_id": int(c),
            "df": int(df_per_center[c]),
            "r_c": float(r_per_center[c]),
        })

    mean_activations_per_span = total_activations / N if N > 0 else 0.0
    activation_stats = {
        "total_activations": int(total_activations),
        "mean_activations_per_span": float(mean_activations_per_span),
        "K_search": int(K_search),
        "max_centers_per_span": int(max_centers_per_span),
    }

    elapsed = _time.time() - t0
    print(f"[kcenter] Activation DF diagnostic done in {elapsed:.1f}s "
          f"({total_activations:,} activations, {mean_activations_per_span:.1f} per span)")

    return df_per_center, quantiles, top_df_centers, activation_stats


def _compute_r_c_and_coverage(assignments, assign_dists, V_actual, r_c_percentile, hist_bins):
    """
    Single entry point for r_c and coverage: max (percentile>=100) or histogram percentile.

    Small-cell fallback: for percentile p, a cell needs at least
    ceil(100 / (100-p)) points for the p-th percentile to trim ≥ 1 outlier.
    Below that, percentile ≈ max anyway and histogram interpolation just adds
    bin-quantisation noise.  We replace those with exact max (conservative:
    wider r_c → no lost recall, zero extra cost).

      p=99 → threshold 100    p=95 → 20    p=90 → 10

    Returns (r_per_center, points_per_center, coverage).
    """
    N = len(assignments)
    r_per_center = np.zeros(V_actual, dtype=np.float64)
    points_per_center = np.bincount(assignments, minlength=V_actual).astype(np.int64)
    if r_c_percentile >= 100.0:
        np.maximum.at(r_per_center, assignments, assign_dists.astype(np.float64))
        return r_per_center, points_per_center, 1.0
    r_per_center, points_per_center, n_covered = _r_c_from_histogram(
        assignments, assign_dists, V_actual, r_c_percentile, hist_bins
    )
    # Small-cell fallback: threshold derived from percentile.
    # n_c * (1 - p/100) >= 1  ⟺  n_c >= ceil(100 / (100-p))
    small_cell_thr = int(math.ceil(100.0 / (100.0 - r_c_percentile)))
    small_mask = (points_per_center > 0) & (points_per_center < small_cell_thr)
    if np.any(small_mask):
        r_max = np.zeros(V_actual, dtype=np.float64)
        np.maximum.at(r_max, assignments, assign_dists.astype(np.float64))
        r_per_center = np.where(small_mask, r_max, r_per_center)
        n_small = int(small_mask.sum())
        print(f"[kcenter] Small-cell fallback (p={r_c_percentile}, n<{small_cell_thr}): "
              f"{n_small}/{V_actual} centers -> r_c = max")
    coverage = n_covered / N if N > 0 else 0.0
    return r_per_center, points_per_center, coverage


def _r_c_from_histogram(assignments, assign_dists, V_actual, percentile, B):
    """
    Compute per-center r_c as percentile of Voronoi cell distances using B-bin histograms.
    Cosine distance in [0, 2]; O(V*B) memory, vectorized update.
    Returns (r_per_center, points_per_center, n_covered).
    """
    # [0, 2] -> bin index in [0, B-1]; d=2 goes to last bin
    hist = np.zeros((V_actual, B), dtype=np.int32)
    bin_idx = (assign_dists * (B / 2.0)).astype(np.int32)
    np.clip(bin_idx, 0, B - 1, out=bin_idx)
    np.add.at(hist, (assignments, bin_idx), 1)
    points_per_center = np.bincount(assignments, minlength=V_actual).astype(np.int64)

    # Fully vectorized percentile computation over all centers
    cumhist = np.cumsum(hist, axis=1)                           # (V, B)
    totals = cumhist[:, -1].astype(np.float64)                  # (V,)
    target_counts = totals * (percentile / 100.0)               # (V,)

    # For each center, find the bin where cumhist crosses target_count
    # (cumhist < target_counts[:, None]) is True for bins before the target
    below = cumhist < target_counts[:, None]                    # (V, B) bool
    bin_indices = below.sum(axis=1).astype(np.int64)            # (V,)
    np.clip(bin_indices, 0, B - 1, out=bin_indices)

    # Vectorized linear interpolation within the target bin
    left_edges = (bin_indices / B) * 2.0
    right_edges = ((bin_indices + 1) / B) * 2.0
    c_hi = cumhist[np.arange(V_actual), bin_indices].astype(np.float64)
    c_lo = np.where(bin_indices > 0,
                    cumhist[np.arange(V_actual), bin_indices - 1].astype(np.float64),
                    0.0)
    denom = c_hi - c_lo
    frac = np.where(denom > 0, (target_counts - c_lo) / denom, 0.0)
    r_per_center = left_edges + frac * (right_edges - left_edges)
    r_per_center = np.where(totals > 0, r_per_center, 0.0)

    # Vectorized coverage: count points within each center's r_c bin
    bin_r = np.minimum((r_per_center * (B / 2.0)).astype(np.int64), B - 1)
    # cumhist already has cumulative sums; coverage per center = cumhist[c, bin_r[c]]
    n_covered = int(cumhist[np.arange(V_actual), bin_r].sum())

    return r_per_center, points_per_center, n_covered


def _build_faiss_center_index(centers, use_gpu):
    """Build FAISS IndexFlatIP on centers, optionally move to GPU. Returns the index."""
    import faiss
    d = centers.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(centers.astype(np.float32))
    if use_gpu:
        try:
            gpu_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        except Exception:
            pass
    return index


def _adaptive_batch_size(d, use_gpu, target_mb=512):
    """Batch size that keeps per-batch memory near target_mb (default 512 MB)."""
    bytes_per_row = d * 4  # float32
    target_bytes = target_mb * 1024 * 1024
    batch = target_bytes // bytes_per_row
    lo = 10_000
    hi = 500_000 if use_gpu else 200_000
    return max(lo, min(hi, batch))


def _voronoi_assign(store, center_index, N, d, use_gpu):
    """Assign each of N points to nearest center; return (assignments, assign_dists)."""
    assign_batch = _adaptive_batch_size(d, use_gpu)
    assignments = np.empty(N, dtype=np.int64)
    assign_dists = np.empty(N, dtype=np.float32)
    for start in range(0, N, assign_batch):
        end = min(start + assign_batch, N)
        batch = store.get_chunk(start, end)
        l2_normalize_inplace(batch)
        sims, idxs = center_index.search(batch, 1)
        assignments[start:end] = idxs[:, 0]
        assign_dists[start:end] = 1.0 - sims[:, 0]
    return assignments, assign_dists


class ChunkedEmbeddingStore:
    """
    Stream embeddings from multiple files without loading all into memory.
    Keeps section arrays (e.g. mmap); get_chunk/get_row copy only the requested slice.
    """
    def __init__(self, section_embeddings):
        self.section_embeddings = list(section_embeddings)
        self.d = int(self.section_embeddings[0].shape[1])
        n_per = [arr.shape[0] for arr in self.section_embeddings]
        self.cumsum = np.concatenate([[0], np.cumsum(n_per)])
        self.N = int(self.cumsum[-1])

    def get_row(self, i):
        """Return row i as (d,) float32."""
        j = np.searchsorted(self.cumsum, i, side="right") - 1
        local_i = i - self.cumsum[j]
        return np.asarray(self.section_embeddings[j][local_i], dtype=np.float32).copy()

    def get_chunk(self, start, end):
        """Return rows [start, end) as (end-start, d) float32. Contiguous read for streaming."""
        n = end - start
        out = np.empty((n, self.d), dtype=np.float32)
        row, offset = start, 0
        while offset < n:
            j = np.searchsorted(self.cumsum, row, side="right") - 1
            local_row = row - self.cumsum[j]
            section = self.section_embeddings[j]
            n_in_section = section.shape[0] - local_row
            n_need = n - offset
            n_take = min(n_in_section, n_need)
            out[offset:offset + n_take] = section[local_row:local_row + n_take]
            offset += n_take
            row += n_take
        return out

    def get_rows(self, indices):
        """Return rows at indices as (len(indices), d) float32. Batch by section."""
        indices = np.asarray(indices, dtype=np.int64).ravel()
        out = np.empty((len(indices), self.d), dtype=np.float32)
        # Map each index to its section, then batch-read per section
        sections = np.searchsorted(self.cumsum, indices, side="right") - 1
        for j in np.unique(sections):
            mask = sections == j
            local_idx = indices[mask] - self.cumsum[j]
            out[mask] = self.section_embeddings[j][local_idx]
        return out


def build_output_dir(base_out_dir: str, embeddings_dir: str, suffix: str = "_kcenter") -> str:
    """Build output directory, appending suffix (e.g. _kcenter).

    Embeddings dir layout: {model_name}_{unit}[_fp16]
    Centers dir layout:    centers_greedy_{model_name}_{unit}{suffix}
    """
    if base_out_dir not in ("./centers", "centers"):
        return base_out_dir

    basename = os.path.basename(embeddings_dir.rstrip("/"))
    # Strip optional _fp16 suffix
    if basename.endswith("_fp16"):
        basename = basename[:-len("_fp16")]
    return f"centers_greedy_{basename}{suffix}"



def _make_ff_backend_dense(X_ref, *, use_gpu: bool):
    """Distance backend for in-memory L2-normalized data.

    Returns a callable ``dists(idxs: np.ndarray) -> np.ndarray (N,)`` that, given
    one or more newly-added center indices, returns the per-point min cosine
    distance to those new centers (assumes X_ref is already L2-normalized).

    If ``use_gpu`` the data is uploaded to CUDA once and reused. The returned
    callable exposes a ``.cleanup()`` method to free GPU memory.
    """
    if use_gpu:
        import torch
        device = torch.device("cuda")
        X = torch.from_numpy(X_ref).to(device)

        def _dists(idxs):
            idx_t = torch.as_tensor(np.asarray(idxs), device=device, dtype=torch.long)
            new_centers = X[idx_t]                       # (k, d)
            sims = X @ new_centers.T                     # (N, k)
            dists = 1.0 - sims
            if dists.dim() == 1:
                return dists.cpu().numpy()
            return torch.min(dists, dim=1).values.cpu().numpy()

        def _cleanup():
            nonlocal X
            del X
            torch.cuda.empty_cache()

        _dists.cleanup = _cleanup
        return _dists

    def _dists(idxs):
        idxs = np.asarray(idxs)
        new_centers = X_ref[idxs]                        # (k, d)
        sims = X_ref @ new_centers.T                     # (N, k) or (N,)
        if sims.ndim == 1:
            return (1.0 - sims).astype(np.float32)
        return (1.0 - sims.max(axis=1)).astype(np.float32)

    return _dists


def _make_ff_backend_streaming(store, N: int, *, chunk_size: int, use_gpu: bool):
    """Distance backend that streams a ChunkedEmbeddingStore.

    Reads ``chunk_size`` rows at a time, L2-normalizes (data on disk may not be),
    and computes per-point distance to the newly-added center(s).
    """
    if use_gpu:
        import torch
        device = torch.device("cuda")

        def _dists(idxs):
            idxs = np.asarray(idxs)
            centers = store.get_rows(idxs) if idxs.size > 1 else store.get_row(int(idxs[0]))[None, :]
            l2_normalize_inplace(centers)
            centers_t = torch.from_numpy(centers).to(device)         # (k, d)
            out = np.empty(N, dtype=np.float32)
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                batch = store.get_chunk(start, end)
                l2_normalize_inplace(batch)
                batch_t = torch.from_numpy(batch).to(device)
                sims = (batch_t @ centers_t.T).cpu().numpy()         # (b, k) or (b,)
                if sims.ndim == 1:
                    out[start:end] = 1.0 - sims
                else:
                    out[start:end] = 1.0 - sims.max(axis=1)
            return out

        return _dists

    def _dists(idxs):
        idxs = np.asarray(idxs)
        centers = store.get_rows(idxs) if idxs.size > 1 else store.get_row(int(idxs[0]))[None, :]
        l2_normalize_inplace(centers)
        out = np.empty(N, dtype=np.float32)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch = store.get_chunk(start, end)
            l2_normalize_inplace(batch)
            sims = batch @ centers.T                                 # (b, k) or (b,)
            if sims.ndim == 1:
                out[start:end] = 1.0 - sims
            else:
                out[start:end] = 1.0 - sims.max(axis=1)
        return out

    return _dists


def _farthest_first_traversal_core(
    N: int,
    V: int,
    dists_from_centers,
    *,
    batch_size_ff: int = 1,
    seed: int = 0,
    log_every: int = 500,
    label: str = "kcenter",
) -> np.ndarray:
    """Backend-agnostic farthest-first traversal (Gonzalez 1985).

    Args:
        N: number of candidate points.
        V: number of centers to select.
        dists_from_centers(idxs) -> np.ndarray (N,):
            given indices of newly-added centers, return per-point min cosine
            distance to those new centers.
        batch_size_ff: 1 = exact; >1 = mini-batch approximation (faster).
    """
    if V > N:
        raise ValueError(f"V={V} > N={N}: cannot select more centers than available points.")
    rng = np.random.default_rng(seed)
    min_dist = np.full(N, np.inf, dtype=np.float32)
    center_indices: list = []
    t0 = time.time()

    def _push(new_idxs):
        np.minimum(min_dist, dists_from_centers(new_idxs), out=min_dist)

    first = int(rng.integers(N))
    center_indices.append(first)
    _push(np.array([first]))

    if V == 1:
        return np.array(center_indices, dtype=np.int64)

    n_rounds = math.ceil((V - 1) / batch_size_ff)
    for _ in range(n_rounds):
        n_to_add = min(batch_size_ff, V - len(center_indices))
        if n_to_add <= 0:
            break
        if n_to_add == 1:
            new = int(np.argmax(min_dist))
            center_indices.append(new)
            _push(np.array([new]))
        else:
            top_b = np.argpartition(-min_dist, n_to_add)[:n_to_add]
            center_indices.extend(int(i) for i in top_b)
            _push(top_b)

        n_selected = len(center_indices)
        if n_selected % log_every == 0 or n_selected == V:
            elapsed = time.time() - t0
            max_d = float(np.max(min_dist))
            med_d = float(np.median(min_dist))
            rate = n_selected / elapsed if elapsed > 0 else 0
            eta = (V - n_selected) / rate if rate > 0 else 0
            print(f"[{label}] {n_selected:6d}/{V} centers  "
                  f"max_dist={max_d:.4f}  med_dist={med_d:.4f}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"[{label}] Farthest-first done: {len(center_indices)} centers in {elapsed:.1f}s")
    return np.array(center_indices, dtype=np.int64)


# ── Backward-compatible wrappers ─────────────────────────────────────────────

def farthest_first_traversal(X_ref, V, batch_size_ff=1, seed=0, log_every=500):
    """CPU farthest-first on in-memory L2-normalized data."""
    backend = _make_ff_backend_dense(X_ref, use_gpu=False)
    return _farthest_first_traversal_core(
        len(X_ref), V, backend,
        batch_size_ff=batch_size_ff, seed=seed, log_every=log_every, label="kcenter",
    )


def farthest_first_traversal_gpu(X_ref_np, V, batch_size_ff=1, seed=0, log_every=500):
    """GPU farthest-first on in-memory L2-normalized data (uploaded once)."""
    backend = _make_ff_backend_dense(X_ref_np, use_gpu=True)
    try:
        return _farthest_first_traversal_core(
            len(X_ref_np), V, backend,
            batch_size_ff=batch_size_ff, seed=seed, log_every=log_every, label="kcenter-gpu",
        )
    finally:
        if hasattr(backend, "cleanup"):
            backend.cleanup()


def farthest_first_traversal_streaming(store, N, d, V, chunk_size=100_000, batch_size_ff=1, seed=0, log_every=500):
    """CPU farthest-first that streams chunks from a ChunkedEmbeddingStore."""
    backend = _make_ff_backend_streaming(store, N, chunk_size=chunk_size, use_gpu=False)
    return _farthest_first_traversal_core(
        N, V, backend,
        batch_size_ff=batch_size_ff, seed=seed, log_every=log_every, label="kcenter (streaming)",
    )


def farthest_first_traversal_streaming_gpu(store, N, d, V, chunk_size=100_000, batch_size_ff=1, seed=0, log_every=500):
    """GPU farthest-first that streams chunks from a ChunkedEmbeddingStore."""
    backend = _make_ff_backend_streaming(store, N, chunk_size=chunk_size, use_gpu=True)
    try:
        return _farthest_first_traversal_core(
            N, V, backend,
            batch_size_ff=batch_size_ff, seed=seed, log_every=log_every, label="kcenter-gpu (streaming)",
        )
    finally:
        import torch
        torch.cuda.empty_cache()



def _compute_df_and_stop_centers(
    args, store, span_doc_ids, assignments, assign_dists,
    r_per_center, V_actual, center_index, N, d, use_gpu,
):
    """Compute Voronoi/activation DF diagnostics and select stop-centers (top fraction by DF).

    Returns ``(df_diagnostic_dict_or_None, stop_centers_list, stop_center_threshold_int_or_None)``.
    Returns ``(None, [], None)`` when ``span_doc_ids`` is unavailable.
    """
    if span_doc_ids is None:
        return None, [], None

    # --- Voronoi DF (nearest-1 only; lower bound) ---
    df_per_center_voronoi, df_quantiles_voronoi, top10_df_voronoi = _compute_df_diagnostic(
        assignments, assign_dists, span_doc_ids, r_per_center, V_actual, top_k=10
    )
    print(f"\n[kcenter] ── Voronoi DF (nearest-1 assignment, lower bound) ──")
    print(f"[kcenter] df quantiles: p50={df_quantiles_voronoi['p50']:,}, p90={df_quantiles_voronoi['p90']:,}, "
          f"p99={df_quantiles_voronoi['p99']:,}, max={df_quantiles_voronoi['max']:,}")

    # --- Activation DF (top-K + r_c filter; matches evaluate.py soft assignment) ---
    df_top_k = args.df_top_k
    if df_top_k > 0:
        print(f"\n[kcenter] ── Activation DF (top-{df_top_k} + per-center r_c, matches evaluate.py) ──")
        df_per_center_act, df_quantiles_act, top10_df_act, act_stats = _compute_df_diagnostic_activation(
            store, center_index, N, d, use_gpu,
            span_doc_ids, r_per_center, V_actual,
            max_centers_per_span=df_top_k, top_k_report=10,
        )
        print(f"[kcenter] activation df quantiles: p50={df_quantiles_act['p50']:,}, p90={df_quantiles_act['p90']:,}, "
              f"p99={df_quantiles_act['p99']:,}, max={df_quantiles_act['max']:,}")
        ratio_p50 = df_quantiles_act['p50'] / max(df_quantiles_voronoi['p50'], 1)
        ratio_max = df_quantiles_act['max'] / max(df_quantiles_voronoi['max'], 1)
        print(f"[kcenter] activation/voronoi ratio: p50={ratio_p50:.1f}x, max={ratio_max:.1f}x")
        print(f"[kcenter] mean activations per span: {act_stats['mean_activations_per_span']:.1f} "
              f"(K_search={act_stats['K_search']}, cap={act_stats['max_centers_per_span']})")
        df_for_stop = df_per_center_act
        df_label = "activation"
    else:
        df_for_stop = df_per_center_voronoi
        df_label = "voronoi"
        df_quantiles_act = None
        top10_df_act = None
        act_stats = None

    # Stop-center rule: top fraction of centers by df (using activation df when available).
    n_stop = max(1, int(math.ceil(V_actual * args.stop_center_fraction)))
    ranked = np.argsort(-df_for_stop)
    stop_centers = [int(ranked[i]) for i in range(n_stop)]
    stop_center_threshold = int(df_for_stop[ranked[n_stop - 1]])
    print(f"[kcenter] stop_centers (top {n_stop} by {df_label} df, min_df={stop_center_threshold:,}): "
          f"{len(stop_centers):,} disabled for retrieval")

    df_diagnostic = {
        "voronoi_df_p50": df_quantiles_voronoi["p50"],
        "voronoi_df_p90": df_quantiles_voronoi["p90"],
        "voronoi_df_p99": df_quantiles_voronoi["p99"],
        "voronoi_df_max": df_quantiles_voronoi["max"],
        "voronoi_top10_df_centers": top10_df_voronoi,
    }
    if df_quantiles_act is not None:
        df_diagnostic["activation_df_p50"] = df_quantiles_act["p50"]
        df_diagnostic["activation_df_p90"] = df_quantiles_act["p90"]
        df_diagnostic["activation_df_p99"] = df_quantiles_act["p99"]
        df_diagnostic["activation_df_max"] = df_quantiles_act["max"]
        df_diagnostic["activation_top10_df_centers"] = top10_df_act
        df_diagnostic["activation_stats"] = act_stats
        df_diagnostic["stop_centers_source"] = "activation"
    else:
        df_diagnostic["stop_centers_source"] = "voronoi"

    print(f"\n[kcenter] Top-10 df centers ({df_label}):")
    top10_for_print = top10_df_act if top10_df_act is not None else top10_df_voronoi
    for row in top10_for_print:
        print(f"  center {row['center_id']}: df={row['df']:,}, r_c={row['r_c']:.4f}")
    return df_diagnostic, stop_centers, stop_center_threshold


def _save_outputs(
    *, args, centers, refine_iters, V_actual, nominal_r, min_r, max_r,
    N, d, ff_pool_size, points_per_center, empty_cells, use_gpu, dir_info,
    coverage, coverage_history, r_per_center,
    df_diagnostic, stop_centers, stop_center_threshold,
):
    """Build the output directory, save centers .npy, stop_centers files, and stats JSON."""
    suffix = f"_kcenter_V{args.V}"
    if args.r_c_percentile < 100.0:
        suffix += f"_r{args.r_c_percentile:g}"
    if refine_iters > 0:
        suffix += f"_refine{refine_iters}"
    if args.exclude_cls and args.exclude_doc_mean:
        suffix += "_nocls_nomean"
    elif args.exclude_cls:
        suffix += "_nocls"
    elif args.exclude_doc_mean:
        suffix += "_nomean"
    args.out_dir = build_output_dir(args.out_dir, args.embeddings_dir, suffix=suffix)
    os.makedirs(args.out_dir, exist_ok=True)

    # Include min/max r in hash suffix to avoid filename collisions when r distributions differ.
    r_range_hash = hash((min_r, max_r, nominal_r, int(V_actual))) & 0xFFFFFFFF
    out_name = f"centers_greedy_V{V_actual}_r{nominal_r:.3f}_r{min_r:.3f}-{max_r:.3f}_{r_range_hash:08x}.npy"
    out_path = os.path.join(args.out_dir, out_name)
    np.save(out_path, centers)

    # Standalone stop_centers files (fast loading, no JSON parse).
    if df_diagnostic is not None and stop_centers:
        sc_arr = np.array(stop_centers, dtype=np.int64)
        sc_npy_path = os.path.join(args.out_dir, "stop_centers.npy")
        np.save(sc_npy_path, sc_arr)
        sc_txt_path = os.path.join(args.out_dir, "stop_centers.txt")
        with open(sc_txt_path, "w") as f:
            for c in sorted(stop_centers):
                f.write(f"{c}\n")
        print(f"[kcenter] Saved: {sc_npy_path} ({len(sc_arr)} centers)")
        print(f"[kcenter] Saved: {sc_txt_path}")

    stats = {
        "method": "kcenter",
        "algorithm": "farthest_first_traversal",
        "embeddings_dir": args.embeddings_dir,
        "N": int(N),
        "d": int(d),
        "V": int(V_actual),
        "r": float(nominal_r),
        "r_per_center": [float(x) for x in r_per_center],
        "r_c_percentile": float(args.r_c_percentile),
        "coverage_estimated": float(coverage),
        "coverage_history": [float(c) for c in coverage_history],
        "coverage_history_note": "final coverage only; k-center does not track per-step curve",
        "ref_size": int(ff_pool_size),
        "batch_size_ff": int(args.batch_size_ff),
        "sim_threshold": float(1.0 - nominal_r),
        "points_per_center_min": int(np.min(points_per_center)),
        "points_per_center_median": int(np.median(points_per_center)),
        "points_per_center_max": int(np.max(points_per_center)),
        "empty_cells": int(empty_cells),
        "seed": int(args.seed),
        "use_gpu": bool(use_gpu),
        "refine_iterations": int(refine_iters),
    }
    if dir_info:
        stats["embeddings_dir_info"] = dir_info
    if df_diagnostic is not None:
        stats["df_diagnostic"] = df_diagnostic
        stats["stop_centers"] = stop_centers
        stats["stop_center_threshold"] = stop_center_threshold
    else:
        stats["stop_centers"] = []
    stats_path = os.path.join(args.out_dir, out_name.replace(".npy", ".json"))
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[kcenter] Saved: {out_path} ({centers.shape})")
    print(f"[kcenter] Saved: {stats_path}")
    print(f"[kcenter] Use evaluate.py with: --centers_suffix '{suffix}'")


def main():
    ap = argparse.ArgumentParser(
        description="Build vocabulary centers via K-Center (farthest-first traversal). "
        "Principled alternative to greedy covering and quantile sampling."
    )
    ap.add_argument("--embeddings_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./centers")
    ap.add_argument("--V", type=int, required=True,
                    help="Number of centers (= semantic unit granularity). The only required hyperparameter.")
    ap.add_argument("--r_c_percentile", type=float, default=99.0,
                    help="Percentile for per-center r_c from Voronoi cell distances. "
                         "100=max (coverage 100%%), 99=trim outliers, 95=trade coverage for tighter spheres. Default: 99.")
    ap.add_argument("--batch_size_ff", type=int, default=1,
                    help="Mini-batch size for farthest-first: 1=exact (default), >1=faster but approximate. "
                         "E.g. 10 is ~10x faster with minimal quality loss.")
    ap.add_argument("--ref_size", type=int, default=0,
                    help="Reference subset size for farthest-first (0=full). "
                         "For N=5M, use 500000-1000000 to speed up center selection. "
                         "Voronoi assignment + r_c are always computed on full data.")
    ap.add_argument("--max_per_section", type=int, default=0,
                    help="Cap per-section spans (section_sampling=equal only). Each section contributes at most this many. "
                         "0=no cap. Voronoi assignment still uses all spans.")
    ap.add_argument("--section_sampling", type=str, default="equal", choices=["equal", "proportional"],
                    help="How to subsample sections for center selection. "
                         "equal: uniform cap per section (max_per_section) or uniform random (ref_size). "
                         "proportional: sample ref_size points stratified by section size (ref_size required); "
                         "keeps center distribution aligned with corpus, avoids bias toward small sections.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--refine_iterations", type=int, default=1,
                    help="Number of K-means-style centroid refinement iterations after farthest-first. "
                         "Each iteration: replace each center with mean of its Voronoi cell, re-assign, recompute r_c. "
                         "1-2 often helps for models with broad centers (e.g. PatentMap).")
    ap.add_argument("--r_c_hist_bins", type=int, default=512,
                    help="Number of histogram bins for percentile r_c (cosine distance in [0, 2]). "
                         "Streaming histogram avoids storing all N distances; 512–1024 is typically enough.")
    ap.add_argument("--df_top_k", type=int, default=10,
                    help="max_centers_per_span for activation DF diagnostic (matches evaluate.py soft assignment). "
                         "0 = skip activation DF and use Voronoi DF for stop-centers. Default: 10.")
    ap.add_argument("--stop_center_fraction", type=float, default=0.01,
                    help="Fraction of top centers (by document frequency) to disable for retrieval. "
                         "Default: 0.01 (top 1%%).")
    ap.add_argument("--exclude_cls", action="store_true", default=False,
                    help="Exclude [CLS] embedding rows (span_kind='cls') from center construction. "
                         "Requires metadata JSONL (saved with --save_metadata 1 in 1create_N_embeddings.py).")
    ap.add_argument("--exclude_doc_mean", action="store_true", default=False,
                    help="Exclude doc-mean embedding rows (span_kind='doc_mean') from center construction. "
                         "Requires metadata JSONL (saved with --save_metadata 1 in 1create_N_embeddings.py).")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.V <= 0:
        raise ValueError("--V must be positive")
    if args.batch_size_ff < 1:
        raise ValueError("--batch_size_ff must be >= 1")
    if not (0 < args.r_c_percentile <= 100):
        raise ValueError("--r_c_percentile must be in (0, 100]")
    if args.r_c_hist_bins < 32:
        raise ValueError("--r_c_hist_bins must be >= 32")
    if args.section_sampling == "proportional" and (not args.ref_size or args.ref_size <= 0):
        raise ValueError("--section_sampling proportional requires --ref_size > 0")
    if not os.path.isdir(args.embeddings_dir):
        raise ValueError(f"Embeddings directory not found: {args.embeddings_dir}")

    # Auto-detect CUDA for farthest-first and FAISS
    use_gpu = _check_gpu()
    vram_bytes = 0
    if use_gpu:
        import torch
        props = torch.cuda.get_device_properties(0)
        vram_bytes = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        print(f"[kcenter] GPU mode (auto): {torch.cuda.get_device_name(0)}, "
              f"VRAM={vram_bytes / 1e9:.1f}GB")

    dir_info = parse_embeddings_dir(args.embeddings_dir)
    embedding_files = find_embedding_files(args.embeddings_dir, dir_info["unit"] if dir_info else None)
    if not embedding_files:
        raise ValueError(f"No embedding files in {args.embeddings_dir}")

    print(f"[kcenter] V={args.V}, r_c_percentile={args.r_c_percentile}, "
          f"batch_size_ff={args.batch_size_ff}")
    print(f"[kcenter] Embeddings: {[os.path.basename(f) for f in embedding_files]}")

    # ── Load embeddings ──
    section_embeddings = []
    for fp in embedding_files:
        if fp.endswith(".npz"):
            data = np.load(fp, mmap_mode="r")
            arr = data["embeddings"] if "embeddings" in data else data[list(data.keys())[0]]
        else:
            arr = np.load(fp, mmap_mode="r")
        section_embeddings.append(arr)
        print(f"  Loaded {arr.shape[0]:,} x {arr.shape[1]} from {os.path.basename(fp)}")

    # ── Filter out CLS / doc_mean rows if requested ──
    if args.exclude_cls or args.exclude_doc_mean:
        print(f"[kcenter] Filtering: exclude_cls={args.exclude_cls}, exclude_doc_mean={args.exclude_doc_mean}")
        section_embeddings, filtered_doc_ids = _filter_embeddings_by_span_kind(
            embedding_files, section_embeddings,
            exclude_cls=args.exclude_cls, exclude_doc_mean=args.exclude_doc_mean,
        )
    else:
        filtered_doc_ids = None

    store = ChunkedEmbeddingStore(section_embeddings)
    N, d = store.N, store.d
    print(f"[kcenter] Total: N={N:,}, d={d} (streaming: no full concatenate)")

    if filtered_doc_ids is not None:
        span_doc_ids = filtered_doc_ids
    else:
        span_doc_ids = _load_span_doc_ids(embedding_files, N)
    if span_doc_ids is None:
        print("[kcenter] No metadata (or mismatch): df diagnostic will be skipped. Use {section}_{unit}_metadata.jsonl for df/posting-list stats.")

    # ── Prepare reference set and run farthest-first ──
    eligible_indices = None
    if args.section_sampling == "proportional" and args.ref_size > 0:
        # Stratified proportional: pool reflects section sizes (n_j/N), avoids bias toward small sections
        eligible_indices = _eligible_indices_proportional(store, args.ref_size, rng)
        print(f"[kcenter] Section sampling=proportional, ref_size={args.ref_size:,} -> eligible pool {len(eligible_indices):,}")
    elif args.section_sampling == "equal" and getattr(args, "max_per_section", 0) > 0:
        # Per-section equal cap: each section contributes at most max_per_section (can bias toward small sections)
        n_per = np.diff(store.cumsum).astype(np.int64)
        n_sections = len(n_per)
        per_section = []
        for j in range(n_sections):
            n = int(n_per[j])
            cap = min(n, args.max_per_section)
            if cap < n:
                idx_local = rng.choice(n, size=cap, replace=False)
            else:
                idx_local = np.arange(n)
            per_section.append(store.cumsum[j] + idx_local)
        eligible_indices = np.concatenate(per_section)
        print(f"[kcenter] Section sampling=equal, max_per_section={args.max_per_section:,} -> eligible pool {len(eligible_indices):,}")

    # ── Fit heuristic: load all into memory if RAM (or GPU VRAM) allows ──
    data_bytes = N * d * 4  # float32
    # GPU: need X (N×d) + min_dist (N) + scratch; use 50% VRAM threshold
    gpu_can_fit_all = use_gpu and vram_bytes > 0 and data_bytes < vram_bytes * 0.50
    # CPU: need X (N×d) + min_dist (N) + scratch; use 50% system RAM threshold
    try:
        import psutil
        total_ram = psutil.virtual_memory().total
    except ImportError:
        # Fallback: read /proc/meminfo on Linux
        total_ram = 0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        total_ram = int(line.split()[1]) * 1024  # kB → bytes
                        break
        except Exception:
            pass
    ram_can_fit_all = (not use_gpu) and total_ram > 0 and data_bytes < total_ram * 0.50

    # Adaptive chunk size for streaming fallback
    ff_chunk_size = 100_000
    if use_gpu and vram_bytes > 0:
        # Use up to 20% VRAM per chunk
        max_chunk_rows = int(vram_bytes * 0.20 / (d * 4))
        ff_chunk_size = max(100_000, min(max_chunk_rows, N))
    elif total_ram > 0:
        # CPU: use up to 10% RAM per chunk (leave room for min_dist, scratch, etc.)
        max_chunk_rows = int(total_ram * 0.10 / (d * 4))
        ff_chunk_size = max(100_000, min(max_chunk_rows, N))

    if eligible_indices is not None:
        X_ff = store.get_rows(eligible_indices)
        l2_normalize_inplace(X_ff)
        ff_to_global = eligible_indices
        use_streaming_ff = False
        print(f"[kcenter] Farthest-first on eligible set: {len(X_ff):,}")
    elif args.ref_size and args.ref_size < N:
        ref_idx = rng.choice(N, size=args.ref_size, replace=False)
        X_ff = store.get_rows(ref_idx)
        l2_normalize_inplace(X_ff)
        ff_to_global = ref_idx
        use_streaming_ff = False
        print(f"[kcenter] Farthest-first on ref subset: {args.ref_size:,}")
    elif gpu_can_fit_all or ram_can_fit_all:
        # Data fits in GPU VRAM or system RAM → load all at once for in-memory FFT
        # (orders of magnitude faster than streaming: 1 matmul/round vs N/chunk_size transfers/round)
        if gpu_can_fit_all:
            print(f"[kcenter] Data fits in GPU VRAM ({data_bytes / 1e9:.1f}GB < {vram_bytes * 0.50 / 1e9:.1f}GB threshold)")
        else:
            print(f"[kcenter] Data fits in RAM ({data_bytes / 1e9:.1f}GB < {total_ram * 0.50 / 1e9:.1f}GB threshold)")
        print(f"[kcenter] Loading all {N:,} points into RAM (bulk read, no mmap)...")
        t_load = time.time()
        # Read from store's section_embeddings (already filtered & in-memory after
        # _filter_embeddings_by_span_kind) instead of re-reading raw files, which
        # would include filtered-out rows (cls/doc_mean) and cause an index mismatch.
        parts = []
        for j, se in enumerate(store.section_embeddings):
            print(f"  Reading section {j}...", end=" ", flush=True)
            arr = np.asarray(se, dtype=np.float32).copy()
            parts.append(arr)
            print(f"{arr.shape[0]:,} x {arr.shape[1]}")
        X_ff = np.concatenate(parts, axis=0)
        del parts
        print(f"  Concatenated: {X_ff.shape}, normalizing...", flush=True)
        l2_normalize_inplace(X_ff)
        ff_to_global = np.arange(N, dtype=np.int64)
        use_streaming_ff = False
        print(f"[kcenter] Loaded + normalized in {time.time() - t_load:.1f}s")
    else:
        ff_to_global = None
        use_streaming_ff = True
        if use_gpu:
            print(f"[kcenter] Data too large for GPU VRAM ({data_bytes / 1e9:.1f}GB); using streaming")
        else:
            print(f"[kcenter] Data too large for RAM ({data_bytes / 1e9:.1f}GB); using streaming")
        print(f"[kcenter] Farthest-first on full data (streaming): {N:,} (chunk={ff_chunk_size:,})")

    ff_pool_size = (len(eligible_indices) if eligible_indices is not None else
                    (args.ref_size if (args.ref_size and args.ref_size < N) else N))

    # Auto-tune batch_size_ff for streaming: with default (1), each round requires a full
    # data pass.  For V=30K streaming that's 30K passes — painfully slow.  Batch=64 cuts
    # passes by 64× with minimal quality loss (k-center is approximate anyway).
    if use_streaming_ff and args.batch_size_ff == 1 and args.V > 1000:
        auto_batch = min(128, max(16, args.V // 500))
        print(f"[kcenter] Auto batch_size_ff: 1 → {auto_batch} (streaming + V={args.V:,}; "
              f"override with --batch_size_ff 1 to force exact)")
        args.batch_size_ff = auto_batch

    if use_streaming_ff:
        ff_label = "GPU" if use_gpu else "CPU"
        print(f"\n[kcenter] Running farthest-first (streaming) on {ff_label} (V={args.V}, chunk={ff_chunk_size:,})...")
        ff_stream = farthest_first_traversal_streaming_gpu if use_gpu else farthest_first_traversal_streaming
        center_indices = ff_stream(
            store, N, d, args.V,
            chunk_size=ff_chunk_size,
            batch_size_ff=args.batch_size_ff,
            seed=args.seed,
            log_every=args.log_every,
        )
    else:
        ff_label = "GPU" if use_gpu else "CPU"
        print(f"\n[kcenter] Running farthest-first on {ff_label} (V={args.V}, batch={args.batch_size_ff})...")
        ff_func = farthest_first_traversal_gpu if use_gpu else farthest_first_traversal
        local_indices = ff_func(
            X_ff, args.V,
            batch_size_ff=args.batch_size_ff,
            seed=args.seed,
            log_every=args.log_every,
        )
        center_indices = ff_to_global[local_indices]

    centers = store.get_rows(center_indices)
    l2_normalize_inplace(centers)
    V_actual = len(centers)
    print(f"[kcenter] Selected {V_actual} centers")

    # ── Voronoi assignment + per-center r_c ──
    print(f"\n[kcenter] Voronoi assignment (all {N:,} points, {'GPU' if use_gpu else 'CPU'})...")
    import faiss
    try:
        if hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(min(os.cpu_count() or 1, 16))
    except Exception:
        pass
    center_index = _build_faiss_center_index(centers, use_gpu)
    if use_gpu:
        print(f"[kcenter] FAISS index on GPU")
    t0 = time.time()
    assignments, assign_dists = _voronoi_assign(store, center_index, N, d, use_gpu)
    print(f"[kcenter] Voronoi assignment (streaming) done in {time.time() - t0:.1f}s")

    # Per-center r_c from Voronoi cell distances
    r_per_center, points_per_center, coverage = _compute_r_c_and_coverage(
        assignments, assign_dists, V_actual, args.r_c_percentile, args.r_c_hist_bins
    )
    if args.r_c_percentile >= 100.0:
        print(f"[kcenter] r_c = max(Voronoi cell distances) -> coverage = 100%")
    else:
        print(f"[kcenter] r_c = percentile({args.r_c_percentile}) of Voronoi cell distances (hist B={args.r_c_hist_bins}) -> coverage = {coverage:.4%}")

    # Centroid refinement (K-means-style: replace center with mean of Voronoi cell; stream over data)
    refine_iters = max(0, int(getattr(args, "refine_iterations", 0)))
    assign_batch = _adaptive_batch_size(d, use_gpu)
    for ref_it in range(refine_iters):
        print(f"\n[kcenter] Refinement {ref_it + 1}/{refine_iters}: centroid update (streaming)...")
        sum_c = np.zeros((V_actual, d), dtype=np.float64)
        count_c = np.zeros(V_actual, dtype=np.int64)
        for start in range(0, N, assign_batch):
            end = min(start + assign_batch, N)
            batch = store.get_chunk(start, end)
            l2_normalize_inplace(batch)
            c_batch = assignments[start:end]
            # Sort-based grouped sum: contiguous memory access, much faster than np.add.at
            order = np.argsort(c_batch, kind="mergesort")
            sorted_c = c_batch[order]
            sorted_batch = batch[order].astype(np.float64)
            # Find boundaries between groups
            change = np.empty(len(sorted_c), dtype=np.bool_)
            change[0] = True
            np.not_equal(sorted_c[1:], sorted_c[:-1], out=change[1:])
            group_starts = np.nonzero(change)[0]
            group_ids = sorted_c[group_starts]
            group_ends = np.empty_like(group_starts)
            group_ends[:-1] = group_starts[1:]
            group_ends[-1] = len(sorted_c)
            group_counts = group_ends - group_starts
            # Cumsum trick: compute prefix sums, then diff at boundaries
            cumsum_batch = np.cumsum(sorted_batch, axis=0)
            for gi in range(len(group_starts)):
                s, e = int(group_starts[gi]), int(group_ends[gi])
                gid = int(group_ids[gi])
                group_sum = cumsum_batch[e - 1] - (cumsum_batch[s - 1] if s > 0 else 0)
                sum_c[gid] += group_sum
                count_c[gid] += group_counts[gi]
        centers_new = np.zeros_like(centers, dtype=np.float32)
        # Vectorized division with safe handling of empty cells (count_c=0)
        active = count_c > 0
        centers_new[active] = (sum_c[active] / count_c[active, None]).astype(np.float32)
        centers_new[~active] = centers[~active]
        l2_normalize_inplace(centers_new)
        centers = centers_new
        center_index = _build_faiss_center_index(centers, use_gpu)
        assignments, assign_dists = _voronoi_assign(store, center_index, N, d, use_gpu)
        r_per_center, points_per_center, coverage = _compute_r_c_and_coverage(
            assignments, assign_dists, V_actual, args.r_c_percentile, args.r_c_hist_bins
        )
        print(f"[kcenter] After refine {ref_it + 1}: r_median={np.median(r_per_center):.4f}, cov={coverage:.4%}")

    # Ensure r_c >= small epsilon
    r_per_center = np.maximum(r_per_center, 1e-6)

    nominal_r = float(np.median(r_per_center))
    max_r = float(np.max(r_per_center))
    min_r = float(np.min(r_per_center))
    empty_cells = int(np.sum(points_per_center == 0))

    print(f"[kcenter] Per-center r_c: min={min_r:.4f}, median={nominal_r:.4f}, max={max_r:.4f}")
    print(f"[kcenter] Points per center: min={int(np.min(points_per_center))}, "
          f"median={int(np.median(points_per_center))}, max={int(np.max(points_per_center))}")
    if empty_cells > 0:
        print(f"[kcenter] Warning: {empty_cells} centers have 0 assigned points")
    if empty_cells > 0:
        print(f"[kcenter] Warning: {empty_cells} centers have 0 assigned points")

    # ── DF diagnostic + stop-center selection ──
    df_diagnostic, stop_centers, stop_center_threshold = _compute_df_and_stop_centers(
        args, store, span_doc_ids, assignments, assign_dists,
        r_per_center, V_actual, center_index, N, d, use_gpu,
    )

    # Final coverage only (k-center does not track per-step coverage; no fake curve)
    coverage_history = [float(coverage)]

    # ── Save ──
    _save_outputs(
        args=args, centers=centers, refine_iters=refine_iters,
        V_actual=V_actual, nominal_r=nominal_r, min_r=min_r, max_r=max_r,
        N=N, d=d, ff_pool_size=ff_pool_size, points_per_center=points_per_center,
        empty_cells=empty_cells, use_gpu=use_gpu, dir_info=dir_info,
        coverage=coverage, coverage_history=coverage_history, r_per_center=r_per_center,
        df_diagnostic=df_diagnostic, stop_centers=stop_centers,
        stop_center_threshold=stop_center_threshold,
    )


if __name__ == "__main__":
    main()
