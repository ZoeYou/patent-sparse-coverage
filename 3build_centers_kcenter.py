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
    only the candidate-side embedding files in --embeddings_dir / mode.

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
        with open(meta_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    doc_ids.append(obj.get("d", obj.get("doc_id", "")))
                except Exception:
                    return None
    if len(doc_ids) != N:
        return None
    return doc_ids


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


def _compute_df_diagnostic(assignments, assign_dists, span_doc_ids, r_per_center, V_actual, top_k=10):
    """
    Compute document-frequency (df) per center: number of distinct documents that activate each center.
    O(N log N) via sort-based unique counting instead of O(V*N) per-center masking.
    Returns (df_per_center, quantiles, top_df_centers).
    """
    unique_docs = sorted(set(span_doc_ids))
    doc_to_int = {d: i for i, d in enumerate(unique_docs)}
    doc_ints = np.array([doc_to_int[d] for d in span_doc_ids], dtype=np.int32)

    # Sort-based df: unique (center, doc) pairs, then bincount on center
    pairs = np.stack([assignments, doc_ints], axis=1)  # (N, 2)
    unique_pairs = np.unique(pairs, axis=0)             # deduplicated (center, doc)
    df_per_center = np.bincount(unique_pairs[:, 0].astype(np.int64), minlength=V_actual).astype(np.int64)

    q = np.percentile(df_per_center, [50, 90, 99])
    quantiles = {"p50": int(q[0]), "p90": int(q[1]), "p99": int(q[2]), "max": int(np.max(df_per_center))}

    # Top-k by df: only need per-center argmin(assign_dists) for the top-k centers
    top_indices = np.argsort(-df_per_center)[:top_k]
    # Pre-compute per-span argmin via argsort(assign_dists) grouped by assignment
    order = np.argsort(assign_dists)
    rep_span_for_center = np.full(V_actual, -1, dtype=np.int64)
    seen = np.zeros(V_actual, dtype=np.bool_)
    for i in order:
        c = int(assignments[i])
        if not seen[c]:
            rep_span_for_center[c] = i
            seen[c] = True
        if seen.all():
            break

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


def _compute_r_c_and_coverage(assignments, assign_dists, V_actual, r_c_percentile, hist_bins):
    """
    Single entry point for r_c and coverage: max (percentile>=100) or histogram percentile.
    Returns (r_per_center, points_per_center, coverage).
    """
    N = len(assignments)
    r_per_center = np.zeros(V_actual, dtype=np.float64)
    points_per_center = np.bincount(assignments, minlength=V_actual).astype(np.int64)
    if r_c_percentile >= 100.0:
        for c in range(V_actual):
            mask = assignments == c
            if np.any(mask):
                r_per_center[c] = float(np.max(assign_dists[mask]))
        return r_per_center, points_per_center, 1.0
    r_per_center, points_per_center, n_covered = _r_c_from_histogram(
        assignments, assign_dists, V_actual, r_c_percentile, hist_bins
    )
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

    r_per_center = np.zeros(V_actual, dtype=np.float64)
    n_covered = 0
    for c in range(V_actual):
        total = hist[c].sum()
        if total == 0:
            r_per_center[c] = 0.0
            continue
        target_count = total * percentile / 100.0
        cum = np.cumsum(hist[c])
        bin_idx = np.searchsorted(cum, target_count, side="left")
        if bin_idx >= B:
            bin_idx = B - 1
        left_edge = (bin_idx / B) * 2.0
        right_edge = ((bin_idx + 1) / B) * 2.0
        c_lo = cum[bin_idx - 1] if bin_idx > 0 else 0
        c_hi = cum[bin_idx]
        frac = (target_count - c_lo) / (c_hi - c_lo) if c_hi > c_lo else 0.0
        r_per_center[c] = left_edge + frac * (right_edge - left_edge)
        bin_r = min(int(r_per_center[c] * B / 2.0), B - 1)
        n_covered += hist[c, : bin_r + 1].sum()
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
        """Return rows at indices as (len(indices), d) float32."""
        indices = np.asarray(indices, dtype=np.int64).ravel()
        out = np.empty((len(indices), self.d), dtype=np.float32)
        for i, idx in enumerate(indices):
            out[i] = self.get_row(int(idx))
        return out


def farthest_first_traversal_gpu(X_ref_np, V, batch_size_ff=1, seed=0, log_every=500):
    """
    GPU-accelerated farthest-first traversal using PyTorch.

    Same semantics as the CPU version but runs all matrix ops on GPU.
    Falls back to CPU argmax/argpartition when needed.

    Args:
        X_ref_np: (N_ref, d) float32 numpy array, L2-normalized.
        V: number of centers to select.
        batch_size_ff: mini-batch size (1=exact, >1=approx).
        seed: random seed for first center.
        log_every: print progress every this many centers.

    Returns:
        center_indices: (V,) int64 numpy array of selected indices into X_ref_np.
    """
    import torch

    N_ref, d = X_ref_np.shape
    if V > N_ref:
        raise ValueError(f"V={V} > N_ref={N_ref}: cannot select more centers than available points.")

    device = torch.device("cuda")
    rng = np.random.default_rng(seed)

    # Upload data to GPU
    X = torch.from_numpy(X_ref_np).to(device)           # (N_ref, d) float32
    min_dist = torch.full((N_ref,), float("inf"), device=device, dtype=torch.float32)

    center_indices = []
    t0 = time.time()

    # First center: random
    first = int(rng.integers(N_ref))
    center_indices.append(first)
    sims = X @ X[first]       # (N_ref,)
    dists = 1.0 - sims
    torch.minimum(min_dist, dists, out=min_dist)

    if V == 1:
        return np.array(center_indices, dtype=np.int64)

    n_rounds = math.ceil((V - 1) / batch_size_ff)

    for round_idx in range(n_rounds):
        n_to_add = min(batch_size_ff, V - len(center_indices))
        if n_to_add <= 0:
            break

        if n_to_add == 1:
            new_center = int(torch.argmax(min_dist).item())
            center_indices.append(new_center)
            sims = X @ X[new_center]
            dists = 1.0 - sims
            torch.minimum(min_dist, dists, out=min_dist)
        else:
            # top-B farthest on GPU
            _, top_b = torch.topk(min_dist, n_to_add)
            top_b_list = top_b.cpu().tolist()
            center_indices.extend(top_b_list)
            new_centers_mat = X[top_b]                     # (n_to_add, d)
            sims_batch = X @ new_centers_mat.T             # (N_ref, n_to_add)
            dists_batch = 1.0 - sims_batch
            min_dists_batch, _ = torch.min(dists_batch, dim=1)
            torch.minimum(min_dist, min_dists_batch, out=min_dist)

        n_selected = len(center_indices)
        if n_selected % log_every == 0 or n_selected == V:
            elapsed = time.time() - t0
            max_d = float(torch.max(min_dist).item())
            med_d = float(torch.median(min_dist).item())
            rate = n_selected / elapsed if elapsed > 0 else 0
            eta = (V - n_selected) / rate if rate > 0 else 0
            print(f"[kcenter-gpu] {n_selected:6d}/{V} centers  "
                  f"max_dist={max_d:.4f}  med_dist={med_d:.4f}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"[kcenter-gpu] Farthest-first done: {len(center_indices)} centers in {elapsed:.1f}s")

    # Free GPU memory
    del X, min_dist
    torch.cuda.empty_cache()

    return np.array(center_indices, dtype=np.int64)


def build_output_dir(base_out_dir: str, embeddings_dir: str, mode: str, suffix: str = "_kcenter") -> str:
    """Build output directory, appending suffix (e.g. _kcenter)."""
    if base_out_dir not in ("./centers", "centers"):
        return base_out_dir

    normalized_dir = embeddings_dir.rstrip("/")
    dir_info = parse_embeddings_dir(normalized_dir)

    if dir_info:
        model_name = dir_info["model_name"]
        tokenization_unit = dir_info["unit"]
        cls_suffix = dir_info["cls_suffix"]
        layer = dir_info.get("layer", "last")
        return f"centers_greedy_{mode}_{model_name}_{tokenization_unit}_{cls_suffix}_{layer}{suffix}"

    basename = os.path.basename(normalized_dir)
    if basename.startswith("embeddings_"):
        parts = basename.replace("embeddings_", "").split("_")
        if len(parts) >= 4:
            cls_pos = next((i for i, p in enumerate(parts) if p in ("cls", "nocls")), None)
            layer_pos = next((i for i, p in enumerate(parts) if p in ("last", "second_last")), None)
            if cls_pos is not None and layer_pos is not None and cls_pos < layer_pos:
                model_name = "_".join(parts[:cls_pos])
                tokenization_unit = "_".join(parts[cls_pos + 1 : layer_pos])
                cls_suffix = parts[cls_pos]
                layer = parts[layer_pos]
                return f"centers_greedy_{mode}_{model_name}_{tokenization_unit}_{cls_suffix}_{layer}{suffix}"

    return f"centers_greedy_{basename}_{mode}{suffix}"


def farthest_first_traversal(X_ref, V, batch_size_ff=1, seed=0, log_every=500):
    """
    Farthest-first traversal (Gonzalez 1985) for K-center.

    Args:
        X_ref: (N_ref, d) float32, L2-normalized vectors.
        V: number of centers to select.
        batch_size_ff: how many centers to add per round (1=exact, >1=mini-batch approx).
        seed: random seed for first center.
        log_every: print progress every this many centers.

    Returns:
        center_indices: (V,) int64 array of selected indices into X_ref.
    """
    N_ref, d = X_ref.shape
    if V > N_ref:
        raise ValueError(f"V={V} > N_ref={N_ref}: cannot select more centers than available points. "
                         "Reduce --V or add more embedding data.")
    rng = np.random.default_rng(seed)

    # min_dist[i] = min cosine distance from point i to any selected center
    # Initialize to infinity (no centers yet)
    min_dist = np.full(N_ref, np.inf, dtype=np.float32)

    center_indices = []
    t0 = time.time()

    # First center: random
    first = int(rng.integers(N_ref))
    center_indices.append(first)

    # Update min_dist with first center
    sims = X_ref @ X_ref[first]  # (N_ref,) inner products
    dists = 1.0 - sims
    np.minimum(min_dist, dists, out=min_dist)

    if V == 1:
        return np.array(center_indices, dtype=np.int64)

    n_rounds = math.ceil((V - 1) / batch_size_ff)

    for round_idx in range(n_rounds):
        n_to_add = min(batch_size_ff, V - len(center_indices))
        if n_to_add <= 0:
            break

        if n_to_add == 1:
            # Exact: pick the single farthest point
            new_center = int(np.argmax(min_dist))
            center_indices.append(new_center)
            sims = X_ref @ X_ref[new_center]
            dists = 1.0 - sims
            np.minimum(min_dist, dists, out=min_dist)
        else:
            # Mini-batch: pick top-B farthest points, add them all, then update
            top_b_indices = np.argpartition(-min_dist, n_to_add)[:n_to_add]
            for idx in top_b_indices:
                center_indices.append(int(idx))
            # Batch update: compute distances to all new centers at once
            new_centers_mat = X_ref[top_b_indices]  # (n_to_add, d)
            sims_batch = X_ref @ new_centers_mat.T  # (N_ref, n_to_add)
            dists_batch = 1.0 - sims_batch  # (N_ref, n_to_add)
            min_dists_batch = np.min(dists_batch, axis=1)  # (N_ref,)
            np.minimum(min_dist, min_dists_batch, out=min_dist)

        n_selected = len(center_indices)
        if n_selected % log_every == 0 or n_selected == V:
            elapsed = time.time() - t0
            max_d = float(np.max(min_dist))
            med_d = float(np.median(min_dist))
            rate = n_selected / elapsed if elapsed > 0 else 0
            eta = (V - n_selected) / rate if rate > 0 else 0
            print(f"[kcenter] {n_selected:6d}/{V} centers  "
                  f"max_dist={max_d:.4f}  med_dist={med_d:.4f}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"[kcenter] Farthest-first done: {len(center_indices)} centers in {elapsed:.1f}s")
    return np.array(center_indices, dtype=np.int64)


def farthest_first_traversal_streaming(store, N, d, V, chunk_size=100_000, batch_size_ff=1, seed=0, log_every=500):
    """
    Farthest-first on full data without loading all embeddings. Uses store.get_chunk() in chunks.
    Returns center_indices (global indices 0..N-1).
    """
    rng = np.random.default_rng(seed)
    min_dist = np.full(N, np.inf, dtype=np.float32)
    center_indices = []
    t0 = time.time()
    first = int(rng.integers(N))
    center_indices.append(first)
    center_row = store.get_row(first)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        batch = store.get_chunk(start, end)
        l2_normalize_inplace(batch)
        sims = batch @ center_row
        dists = 1.0 - sims
        np.minimum(min_dist[start:end], dists, out=min_dist[start:end])
    if V == 1:
        return np.array(center_indices, dtype=np.int64)
    n_rounds = math.ceil((V - 1) / batch_size_ff)
    for round_idx in range(n_rounds):
        n_to_add = min(batch_size_ff, V - len(center_indices))
        if n_to_add <= 0:
            break
        if n_to_add == 1:
            new_center = int(np.argmax(min_dist))
            center_indices.append(new_center)
            center_row = store.get_row(new_center)
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                batch = store.get_chunk(start, end)
                l2_normalize_inplace(batch)
                sims = batch @ center_row
                dists = 1.0 - sims
                np.minimum(min_dist[start:end], dists, out=min_dist[start:end])
        else:
            top_b_indices = np.argpartition(-min_dist, n_to_add)[:n_to_add]
            center_indices.extend([int(i) for i in top_b_indices])
            new_centers = store.get_rows(top_b_indices)
            l2_normalize_inplace(new_centers)
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                batch = store.get_chunk(start, end)
                l2_normalize_inplace(batch)
                sims_batch = batch @ new_centers.T
                dists_batch = 1.0 - sims_batch
                min_dists_batch = np.min(dists_batch, axis=1)
                np.minimum(min_dist[start:end], min_dists_batch, out=min_dist[start:end])
        n_selected = len(center_indices)
        if n_selected % log_every == 0 or n_selected == V:
            elapsed = time.time() - t0
            max_d = float(np.max(min_dist))
            med_d = float(np.median(min_dist))
            rate = n_selected / elapsed if elapsed > 0 else 0
            eta = (V - n_selected) / rate if rate > 0 else 0
            print(f"[kcenter] {n_selected:6d}/{V} centers  "
                  f"max_dist={max_d:.4f}  med_dist={med_d:.4f}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")
    elapsed = time.time() - t0
    print(f"[kcenter] Farthest-first (streaming) done: {len(center_indices)} centers in {elapsed:.1f}s")
    return np.array(center_indices, dtype=np.int64)


def farthest_first_traversal_streaming_gpu(store, N, d, V, chunk_size=100_000, batch_size_ff=1, seed=0, log_every=500):
    """
    GPU streaming FFT: stream chunks to GPU, update min_dist on CPU, no full X on GPU.
    Returns center_indices (global indices 0..N-1).
    """
    import torch
    device = torch.device("cuda")
    rng = np.random.default_rng(seed)
    min_dist = np.full(N, np.inf, dtype=np.float32)
    center_indices = []
    t0 = time.time()
    first = int(rng.integers(N))
    center_indices.append(first)
    center_row = store.get_row(first)
    center_t = torch.from_numpy(center_row).to(device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        batch = store.get_chunk(start, end)
        l2_normalize_inplace(batch)
        batch_t = torch.from_numpy(batch).to(device)
        sims = (batch_t @ center_t).cpu().numpy()
        dists = 1.0 - sims
        np.minimum(min_dist[start:end], dists, out=min_dist[start:end])
    if V == 1:
        return np.array(center_indices, dtype=np.int64)
    n_rounds = math.ceil((V - 1) / batch_size_ff)
    for round_idx in range(n_rounds):
        n_to_add = min(batch_size_ff, V - len(center_indices))
        if n_to_add <= 0:
            break
        if n_to_add == 1:
            new_center = int(np.argmax(min_dist))
            center_indices.append(new_center)
            center_row = store.get_row(new_center)
            center_t = torch.from_numpy(center_row).to(device)
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                batch = store.get_chunk(start, end)
                l2_normalize_inplace(batch)
                batch_t = torch.from_numpy(batch).to(device)
                sims = (batch_t @ center_t).cpu().numpy()
                dists = 1.0 - sims
                np.minimum(min_dist[start:end], dists, out=min_dist[start:end])
        else:
            top_b_indices = np.argpartition(-min_dist, n_to_add)[:n_to_add]
            center_indices.extend([int(i) for i in top_b_indices])
            new_centers = store.get_rows(top_b_indices)
            l2_normalize_inplace(new_centers)
            new_centers_t = torch.from_numpy(new_centers).to(device)
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                batch = store.get_chunk(start, end)
                l2_normalize_inplace(batch)
                batch_t = torch.from_numpy(batch).to(device)
                sims_batch = (batch_t @ new_centers_t.T).cpu().numpy()
                dists_batch = 1.0 - sims_batch
                min_dists_batch = np.min(dists_batch, axis=1)
                np.minimum(min_dist[start:end], min_dists_batch, out=min_dist[start:end])
        n_selected = len(center_indices)
        if n_selected % log_every == 0 or n_selected == V:
            elapsed = time.time() - t0
            max_d = float(np.max(min_dist))
            med_d = float(np.median(min_dist))
            rate = n_selected / elapsed if elapsed > 0 else 0
            eta = (V - n_selected) / rate if rate > 0 else 0
            print(f"[kcenter-gpu] {n_selected:6d}/{V} centers  "
                  f"max_dist={max_d:.4f}  med_dist={med_d:.4f}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")
    elapsed = time.time() - t0
    print(f"[kcenter-gpu] Farthest-first (streaming) done: {len(center_indices)} centers in {elapsed:.1f}s")
    torch.cuda.empty_cache()
    return np.array(center_indices, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(
        description="Build vocabulary centers via K-Center (farthest-first traversal). "
        "Principled alternative to greedy covering and quantile sampling."
    )
    ap.add_argument("--embeddings_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./centers")
    ap.add_argument("--mode", type=str, required=True, choices=["abstract2abstract", "claim2all"])
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
    if use_gpu:
        import torch
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        print(f"[kcenter] GPU mode (auto): {torch.cuda.get_device_name(0)}, "
              f"VRAM={vram / 1e9:.1f}GB")

    dir_info = parse_embeddings_dir(args.embeddings_dir)
    embedding_files = find_embedding_files(args.embeddings_dir, args.mode, dir_info["unit"] if dir_info else None)
    if not embedding_files:
        raise ValueError(f"No embedding files for mode '{args.mode}' in {args.embeddings_dir}")

    print(f"[kcenter] Mode: {args.mode}, V={args.V}, r_c_percentile={args.r_c_percentile}, "
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

    store = ChunkedEmbeddingStore(section_embeddings)
    N, d = store.N, store.d
    print(f"[kcenter] Total: N={N:,}, d={d} (streaming: no full concatenate)")

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

    ff_chunk_size = 100_000
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
    else:
        ff_to_global = None
        use_streaming_ff = True
        print(f"[kcenter] Farthest-first on full data (streaming): {N:,}")

    ff_pool_size = (len(eligible_indices) if eligible_indices is not None else
                    (args.ref_size if (args.ref_size and args.ref_size < N) else N))

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
            c_batch = assignments[start:end]
            np.add.at(sum_c, c_batch, batch)
            np.add.at(count_c, c_batch, 1)
        centers_new = np.zeros_like(centers, dtype=np.float32)
        for c in range(V_actual):
            if count_c[c] > 0:
                centers_new[c] = (sum_c[c] / count_c[c]).astype(np.float32)
            else:
                centers_new[c] = centers[c]
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

    # ── DF diagnostic (document frequency per center = posting list length; critical for retrieval) ──
    df_diagnostic = None
    if span_doc_ids is not None:
        df_per_center, df_quantiles, top10_df = _compute_df_diagnostic(
            assignments, assign_dists, span_doc_ids, r_per_center, V_actual, top_k=10
        )
        # Stop-center rule: df >= p99 -> disabled at retrieval (avoids long posting lists + noise)
        df_p99 = df_quantiles["p99"]
        stop_centers = [int(c) for c in range(V_actual) if df_per_center[c] >= df_p99]
        stop_center_threshold = df_p99

        df_diagnostic = {
            "df_p50": df_quantiles["p50"],
            "df_p90": df_quantiles["p90"],
            "df_p99": df_quantiles["p99"],
            "df_max": df_quantiles["max"],
            "top10_df_centers": top10_df,
        }
        print(f"\n[kcenter] ── DF diagnostic (doc frequency per center = posting list length) ──")
        print(f"[kcenter] df quantiles: p50={df_quantiles['p50']:,}, p90={df_quantiles['p90']:,}, p99={df_quantiles['p99']:,}, max={df_quantiles['max']:,}")
        print(f"[kcenter] stop_centers (df >= p99): {len(stop_centers):,} disabled for retrieval (threshold={stop_center_threshold:,})")
        print(f"[kcenter] Top-10 df centers (r_c + representative span):")
        for row in top10_df:
            print(f"  center {row['center_id']}: df={row['df']:,}, r_c={row['r_c']:.4f}, rep_span_idx={row['rep_span_idx']}, rep_span_dist={row['rep_span_dist']:.4f}")

    # Final coverage only (k-center does not track per-step coverage; no fake curve)
    coverage_history = [float(coverage)]

    # ── Save ──
    suffix = f"_kcenter_V{args.V}"
    if args.r_c_percentile < 100.0:
        suffix += f"_r{args.r_c_percentile:g}"
    if refine_iters > 0:
        suffix += f"_refine{refine_iters}"
    args.out_dir = build_output_dir(args.out_dir, args.embeddings_dir, args.mode, suffix=suffix)
    os.makedirs(args.out_dir, exist_ok=True)

    out_name = f"centers_greedy_r{nominal_r:.3f}.npy"
    out_path = os.path.join(args.out_dir, out_name)
    np.save(out_path, centers)

    stats = {
        "method": "kcenter",
        "algorithm": "farthest_first_traversal",
        "embeddings_dir": args.embeddings_dir,
        "task_mode": args.mode,
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


if __name__ == "__main__":
    main()
