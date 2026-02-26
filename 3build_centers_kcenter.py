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
# GPU helpers (lazy imports; only used when --use_gpu is set)
# ---------------------------------------------------------------------------

def _check_gpu():
    """Return True if CUDA is available via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


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
                         "Reduce --V or increase eligible span pool (adjust --max_spans_per_doc).")
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
    ap.add_argument("--r_c_percentile", type=float, default=100.0,
                    help="Percentile for per-center r_c from Voronoi cell distances. "
                         "100=max (coverage 100%%), 99=trim outliers, 95=trade coverage for tighter spheres. Default: 100.")
    ap.add_argument("--batch_size_ff", type=int, default=1,
                    help="Mini-batch size for farthest-first: 1=exact (default), >1=faster but approximate. "
                         "E.g. 10 is ~10x faster with minimal quality loss.")
    ap.add_argument("--ref_size", type=int, default=0,
                    help="Reference subset size for farthest-first (0=full). "
                         "For N=5M, use 500000-1000000 to speed up center selection. "
                         "Voronoi assignment + r_c are always computed on full data.")
    ap.add_argument("--max_spans_per_doc", type=str, default="0",
                    help="Per-doc cap on spans eligible as centers: 0=no limit, positive int=fixed K, "
                         "'auto'=min(p90(spans_per_doc), max(10, ceil(3*V/D))). "
                         "Requires metadata (.jsonl) files.")
    ap.add_argument("--use_gpu", action="store_true", default=False,
                    help="Use GPU (PyTorch CUDA) for farthest-first traversal and FAISS Voronoi. "
                         "Much faster for large N (50-100x). Requires torch with CUDA and faiss-gpu.")
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--refine_iterations", type=int, default=0,
                    help="Number of K-means-style centroid refinement iterations after farthest-first. "
                         "Each iteration: replace each center with mean of its Voronoi cell, re-assign, recompute r_c. "
                         "0=no refinement (default). 1-2 often helps for models with broad centers (e.g. PatentMap).")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.V <= 0:
        raise ValueError("--V must be positive")
    if args.batch_size_ff < 1:
        raise ValueError("--batch_size_ff must be >= 1")
    if not (0 < args.r_c_percentile <= 100):
        raise ValueError("--r_c_percentile must be in (0, 100]")
    if not os.path.isdir(args.embeddings_dir):
        raise ValueError(f"Embeddings directory not found: {args.embeddings_dir}")

    # GPU check
    use_gpu = args.use_gpu
    if use_gpu:
        if not _check_gpu():
            print("[kcenter] WARNING: --use_gpu requested but CUDA not available, falling back to CPU")
            use_gpu = False
        else:
            import torch
            props = torch.cuda.get_device_properties(0)
            vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
            print(f"[kcenter] GPU mode: {torch.cuda.get_device_name(0)}, "
                  f"VRAM={vram / 1e9:.1f}GB")

    # Parse --max_spans_per_doc
    _max_spans_raw = args.max_spans_per_doc.strip().lower()
    if _max_spans_raw == "0":
        max_spans_per_doc_value = 0
        max_spans_per_doc_auto = False
    elif _max_spans_raw == "auto":
        max_spans_per_doc_value = 0
        max_spans_per_doc_auto = True
    elif _max_spans_raw.isdigit() and int(_max_spans_raw) >= 0:
        max_spans_per_doc_value = int(_max_spans_raw)
        max_spans_per_doc_auto = False
    else:
        raise ValueError("--max_spans_per_doc must be 0, 'auto', or a non-negative integer")

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

    # ── Load metadata (span → doc_id) for per-doc cap ──
    required_sections = ["abstract"] if args.mode == "abstract2abstract" else ["abstract", "claim", "invention"]
    unit_for_meta = (dir_info.get("unit") if dir_info else None) or None
    span_to_doc = []
    for fp in embedding_files:
        basename = os.path.basename(fp)
        section = None
        for sec in required_sections:
            if basename.startswith(sec + "_"):
                section = sec
                break
        if section is None:
            continue
        if unit_for_meta is None:
            unit_for_meta = basename[len(section) + 1:].replace(".npy", "").replace(".npz", "")
        meta_path = os.path.join(args.embeddings_dir, f"{section}_{unit_for_meta}_metadata.jsonl")
        if not os.path.isfile(meta_path):
            span_to_doc = []
            break
        with open(meta_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    meta = json.loads(line)
                    span_to_doc.append(meta.get("d", meta.get("doc_id", "")))
                except json.JSONDecodeError:
                    span_to_doc.append("")
    if span_to_doc and len(span_to_doc) != N:
        print(f"[kcenter] Warning: metadata count ({len(span_to_doc):,}) != N ({N:,}); per-doc cap disabled")
        span_to_doc = []
    if not span_to_doc:
        span_to_doc = None

    # ── Per-doc cap: build eligible_indices ──
    eligible_indices = None
    if span_to_doc is not None and (max_spans_per_doc_auto or max_spans_per_doc_value > 0):
        from collections import Counter
        doc_span_counts = list(Counter(span_to_doc).values())
        D_docs = len(doc_span_counts)
        if max_spans_per_doc_auto:
            K_fair = max(10, int(np.ceil(3.0 * args.V / D_docs)))
            p90 = float(np.percentile(doc_span_counts, 90))
            K_cap = max(1, int(np.ceil(p90)))
            K = min(K_cap, K_fair)
            print(f"[kcenter] max_spans_per_doc=auto: D={D_docs:,}, K_fair={K_fair}, p90={p90:.0f} -> K={K}")
        else:
            K = max_spans_per_doc_value
            print(f"[kcenter] max_spans_per_doc={K}")
        doc_to_indices = {}
        for i in range(N):
            doc_to_indices.setdefault(span_to_doc[i], []).append(i)
        eligible_list = []
        for _did, indices in doc_to_indices.items():
            if len(indices) <= K:
                eligible_list.extend(indices)
            else:
                eligible_list.extend(rng.choice(indices, size=K, replace=False).tolist())
        eligible_indices = np.array(eligible_list, dtype=np.int64)
        if len(eligible_indices) < args.V:
            K_min = max(1, int(np.ceil(args.V / D_docs)))
            print(f"[kcenter] Warning: eligible {len(eligible_indices):,} < V={args.V}; increasing K to {K_min}")
            K = max(K, K_min)
            eligible_list = []
            for _did, indices in doc_to_indices.items():
                if len(indices) <= K:
                    eligible_list.extend(indices)
                else:
                    eligible_list.extend(rng.choice(indices, size=K, replace=False).tolist())
            eligible_indices = np.array(eligible_list, dtype=np.int64)
        print(f"[kcenter] Per-doc cap: K={K}, eligible set {len(eligible_indices):,}")

    # ── Prepare reference set and run farthest-first ──
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
    # Assign ALL points (full X) to nearest center
    print(f"\n[kcenter] Voronoi assignment (all {N:,} points, {'GPU' if use_gpu else 'CPU'})...")
    import faiss
    try:
        if hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(min(os.cpu_count() or 1, 16))
    except Exception:
        pass

    center_index = faiss.IndexFlatIP(d)
    center_index.add(centers)

    # Move index to GPU if available
    if use_gpu:
        try:
            gpu_res = faiss.StandardGpuResources()
            center_index = faiss.index_cpu_to_gpu(gpu_res, 0, center_index)
            print(f"[kcenter] FAISS index moved to GPU")
        except Exception as e:
            print(f"[kcenter] WARNING: FAISS GPU failed ({e}), using CPU index")

    assign_batch = 500_000 if use_gpu else 100_000
    assignments = np.empty(N, dtype=np.int64)
    assign_dists = np.empty(N, dtype=np.float32)

    t0 = time.time()
    for start in range(0, N, assign_batch):
        end = min(start + assign_batch, N)
        batch = store.get_chunk(start, end)
        l2_normalize_inplace(batch)
        sims, idxs = center_index.search(batch, 1)
        assignments[start:end] = idxs[:, 0]
        assign_dists[start:end] = 1.0 - sims[:, 0]
    print(f"[kcenter] Voronoi assignment (streaming) done in {time.time() - t0:.1f}s")

    # Per-center r_c from Voronoi cell distances
    r_per_center = np.zeros(V_actual, dtype=np.float64)
    points_per_center = np.zeros(V_actual, dtype=np.int64)

    if args.r_c_percentile >= 100.0:
        # r_c = max distance in Voronoi cell (coverage = 100%)
        for i in range(N):
            c = int(assignments[i])
            d_val = float(assign_dists[i])
            points_per_center[c] += 1
            if d_val > r_per_center[c]:
                r_per_center[c] = d_val
        coverage = 1.0
        print(f"[kcenter] r_c = max(Voronoi cell distances) -> coverage = 100%")
    else:
        # r_c = percentile of distances in Voronoi cell
        cell_dists = [[] for _ in range(V_actual)]
        for i in range(N):
            c = int(assignments[i])
            cell_dists[c].append(float(assign_dists[i]))
            points_per_center[c] += 1
        n_covered = 0
        for c in range(V_actual):
            if cell_dists[c]:
                r_per_center[c] = float(np.percentile(cell_dists[c], args.r_c_percentile))
                n_covered += sum(1 for d_val in cell_dists[c] if d_val <= r_per_center[c])
            else:
                r_per_center[c] = 0.0
        coverage = n_covered / N if N > 0 else 0.0
        print(f"[kcenter] r_c = percentile({args.r_c_percentile}) of Voronoi cell distances -> coverage = {coverage:.4%}")

    # Centroid refinement (K-means-style: replace center with mean of Voronoi cell; stream over data)
    refine_iters = max(0, int(getattr(args, "refine_iterations", 0)))
    for ref_it in range(refine_iters):
        print(f"\n[kcenter] Refinement {ref_it + 1}/{refine_iters}: centroid update (streaming)...")
        sum_c = np.zeros((V_actual, d), dtype=np.float64)
        count_c = np.zeros(V_actual, dtype=np.int64)
        for start in range(0, N, assign_batch):
            end = min(start + assign_batch, N)
            batch = store.get_chunk(start, end)
            for i in range(batch.shape[0]):
                c = int(assignments[start + i])
                sum_c[c] += batch[i]
                count_c[c] += 1
        centers_new = np.zeros_like(centers, dtype=np.float32)
        for c in range(V_actual):
            if count_c[c] > 0:
                centers_new[c] = (sum_c[c] / count_c[c]).astype(np.float32)
            else:
                centers_new[c] = centers[c]
        l2_normalize_inplace(centers_new)
        centers = centers_new
        center_index = faiss.IndexFlatIP(d)
        center_index.add(centers)
        if use_gpu:
            try:
                gpu_res = faiss.StandardGpuResources()
                center_index = faiss.index_cpu_to_gpu(gpu_res, 0, center_index)
            except Exception:
                pass
        for start in range(0, N, assign_batch):
            end = min(start + assign_batch, N)
            batch = store.get_chunk(start, end)
            l2_normalize_inplace(batch)
            sims, idxs = center_index.search(batch, 1)
            assignments[start:end] = idxs[:, 0]
            assign_dists[start:end] = 1.0 - sims[:, 0]
        r_per_center = np.zeros(V_actual, dtype=np.float64)
        points_per_center = np.zeros(V_actual, dtype=np.int64)
        if args.r_c_percentile >= 100.0:
            for i in range(N):
                c = int(assignments[i])
                d_val = float(assign_dists[i])
                points_per_center[c] += 1
                r_per_center[c] = max(r_per_center[c], d_val)
            coverage = 1.0
        else:
            cell_dists = [[] for _ in range(V_actual)]
            for i in range(N):
                c = int(assignments[i])
                cell_dists[c].append(float(assign_dists[i]))
                points_per_center[c] += 1
            n_covered = 0
            for c in range(V_actual):
                if cell_dists[c]:
                    r_per_center[c] = float(np.percentile(cell_dists[c], args.r_c_percentile))
                    n_covered += sum(1 for d_val in cell_dists[c] if d_val <= r_per_center[c])
                else:
                    r_per_center[c] = 0.0
            coverage = n_covered / N if N > 0 else 0.0
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

    # Final coverage only (k-center does not track per-step coverage; no fake curve)
    coverage_history = [float(coverage)]

    # ── Save ──
    suffix = f"_kcenter_V{args.V}"
    if args.r_c_percentile < 100.0:
        suffix += f"_r{args.r_c_percentile:g}"
    if eligible_indices is not None:
        suffix += f"_pd{'auto' if max_spans_per_doc_auto else max_spans_per_doc_value}"
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
        "ref_size": int(len(X_ff)),
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
    if eligible_indices is not None:
        stats["max_spans_per_doc"] = "auto" if max_spans_per_doc_auto else int(max_spans_per_doc_value)
        stats["eligible_set_size"] = int(len(eligible_indices))
    if dir_info:
        stats["embeddings_dir_info"] = dir_info
    stats_path = os.path.join(args.out_dir, out_name.replace(".npy", ".json"))
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[kcenter] Saved: {out_path} ({centers.shape})")
    print(f"[kcenter] Saved: {stats_path}")
    print(f"[kcenter] Use evaluate.py with: --centers_suffix '{suffix}'")


if __name__ == "__main__":
    main()
