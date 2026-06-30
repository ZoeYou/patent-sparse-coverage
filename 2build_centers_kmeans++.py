#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2build_centers_kmeans++.py

Build vocabulary centers via MiniBatch K-means with k-means++ initialization.

Design goal: keep the same downstream contract as k-center script
(centers + per-center r_c + coverage/stats JSON) so comparisons are fair.

Pipeline:
  1) Load embeddings as chunked store (no full concatenate required).
  2) Build a fit pool (full / ref subset / proportional section subset).
  3) Train MiniBatchKMeans(init="k-means++") on L2-normalized vectors.
  4) Voronoi assignment on full data (streaming) using FAISS IP index.
  5) Compute per-center r_c (max or percentile), coverage, quantization metrics.
  6) Save .npy centers and .json stats.

Usage with evaluate.py: --centers_suffix "_kmeanspp_V{V}"
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


def _check_gpu():
    """Return True if CUDA is available via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class ChunkedEmbeddingStore:
    """Stream embeddings from multiple files without loading all into memory."""

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
        """Return rows [start, end) as (end-start, d) float32."""
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
        sections = np.searchsorted(self.cumsum, indices, side="right") - 1
        for j in np.unique(sections):
            mask = sections == j
            local_idx = indices[mask] - self.cumsum[j]
            out[mask] = self.section_embeddings[j][local_idx]
        return out


def build_output_dir(base_out_dir: str, embeddings_dir: str, suffix: str = "_kmeanspp") -> str:
    """Build output directory, appending suffix."""
    if base_out_dir not in ("./centers", "centers"):
        return base_out_dir

    basename = os.path.basename(embeddings_dir.rstrip("/"))
    if basename.endswith("_fp16"):
        basename = basename[:-len("_fp16")]
    return f"centers_{basename}{suffix}"


def _eligible_indices_proportional(store, ref_size, rng):
    """Sample ref_size indices proportionally to section sizes."""
    n_sections = len(store.cumsum) - 1
    n_per = np.diff(store.cumsum).astype(np.int64)
    N = int(store.cumsum[-1])

    targets = ref_size * (n_per / N)
    counts = np.round(targets).astype(np.int64)
    np.clip(counts, 0, n_per, out=counts)

    total = int(counts.sum())
    if total > ref_size:
        need_remove = total - ref_size
        for j in np.argsort(-counts):
            take = min(int(counts[j]), need_remove)
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


def _compute_r_c_and_coverage(assignments, assign_dists, V_actual, r_c_percentile, hist_bins):
    """Compute per-center r_c and estimated coverage."""
    N = len(assignments)
    r_per_center = np.zeros(V_actual, dtype=np.float64)
    points_per_center = np.bincount(assignments, minlength=V_actual).astype(np.int64)

    if r_c_percentile >= 100.0:
        np.maximum.at(r_per_center, assignments, assign_dists.astype(np.float64))
        return r_per_center, points_per_center, 1.0

    r_per_center, points_per_center, n_covered = _r_c_from_histogram(
        assignments, assign_dists, V_actual, r_c_percentile, hist_bins
    )

    # For tiny cells percentile is unstable; fallback to exact max conservatively.
    small_cell_thr = int(math.ceil(100.0 / (100.0 - r_c_percentile)))
    small_mask = (points_per_center > 0) & (points_per_center < small_cell_thr)
    if np.any(small_mask):
        r_max = np.zeros(V_actual, dtype=np.float64)
        np.maximum.at(r_max, assignments, assign_dists.astype(np.float64))
        r_per_center = np.where(small_mask, r_max, r_per_center)
        n_small = int(small_mask.sum())
        print(
            f"[kmeans++] Small-cell fallback (p={r_c_percentile}, n<{small_cell_thr}): "
            f"{n_small}/{V_actual} centers -> r_c = max"
        )

    coverage = n_covered / N if N > 0 else 0.0
    return r_per_center, points_per_center, coverage


def _r_c_from_histogram(assignments, assign_dists, V_actual, percentile, B):
    """Histogram percentile for per-center distances (cos distance in [0, 2])."""
    hist = np.zeros((V_actual, B), dtype=np.int32)
    bin_idx = (assign_dists * (B / 2.0)).astype(np.int32)
    np.clip(bin_idx, 0, B - 1, out=bin_idx)
    np.add.at(hist, (assignments, bin_idx), 1)
    points_per_center = np.bincount(assignments, minlength=V_actual).astype(np.int64)

    cumhist = np.cumsum(hist, axis=1)
    totals = cumhist[:, -1].astype(np.float64)
    target_counts = totals * (percentile / 100.0)

    below = cumhist < target_counts[:, None]
    bin_indices = below.sum(axis=1).astype(np.int64)
    np.clip(bin_indices, 0, B - 1, out=bin_indices)

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

    bin_r = np.minimum((r_per_center * (B / 2.0)).astype(np.int64), B - 1)
    n_covered = int(cumhist[np.arange(V_actual), bin_r].sum())

    return r_per_center, points_per_center, n_covered


def _build_faiss_center_index(centers, use_gpu):
    """Build FAISS IndexFlatIP on centers, optionally move to GPU."""
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
    """Batch size that keeps per-batch memory near target_mb."""
    bytes_per_row = d * 4
    target_bytes = target_mb * 1024 * 1024
    batch = target_bytes // bytes_per_row
    lo = 10_000
    hi = 500_000 if use_gpu else 200_000
    return max(lo, min(hi, batch))


def _voronoi_assign(store, center_index, N, d, use_gpu):
    """Assign each point to nearest center; return assignments and cos distances."""
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


def _quantization_metrics(assign_dists):
    """Quantization diagnostics from per-point cosine distances."""
    d = np.asarray(assign_dists, dtype=np.float64)
    mean_cd = float(d.mean())
    return {
        "mean_cos_distance": mean_cd,
        "quantization_error_sq_l2": float(2.0 * mean_cd),
        "mean_sq_cos_distance": float((d * d).mean()),
        "cos_distance_p50": float(np.percentile(d, 50)),
        "cos_distance_p95": float(np.percentile(d, 95)),
        "cos_distance_p99": float(np.percentile(d, 99)),
        "cos_distance_max": float(d.max()),
    }


def _save_outputs(
    *,
    args,
    centers,
    V_actual,
    nominal_r,
    min_r,
    max_r,
    N,
    d,
    fit_size,
    points_per_center,
    empty_cells,
    use_gpu,
    dir_info,
    coverage,
    coverage_history,
    r_per_center,
    quant_metrics,
    inertia,
):
    """Build output directory, save centers .npy and stats JSON."""
    suffix = f"_kmeanspp_V{args.V}"
    if args.r_c_percentile < 100.0:
        suffix += f"_r{args.r_c_percentile:g}"

    args.out_dir = build_output_dir(args.out_dir, args.embeddings_dir, suffix=suffix)
    os.makedirs(args.out_dir, exist_ok=True)

    r_range_hash = hash((min_r, max_r, nominal_r, int(V_actual))) & 0xFFFFFFFF
    out_name = (
        f"centers_kmeanspp_V{V_actual}_r{nominal_r:.3f}_"
        f"r{min_r:.3f}-{max_r:.3f}_{r_range_hash:08x}.npy"
    )
    out_path = os.path.join(args.out_dir, out_name)
    np.save(out_path, centers)

    stats = {
        "method": "kmeanspp",
        "algorithm": "MiniBatchKMeans(k-means++)",
        "embeddings_dir": args.embeddings_dir,
        "N": int(N),
        "d": int(d),
        "V": int(V_actual),
        "r": float(nominal_r),
        "r_per_center": [float(x) for x in r_per_center],
        "r_c_percentile": float(args.r_c_percentile),
        "coverage_estimated": float(coverage),
        "coverage_history": [float(c) for c in coverage_history],
        "coverage_history_note": "final coverage only; kmeans++ does not track per-step curve",
        "fit_size": int(fit_size),
        "max_iter": int(args.max_iter),
        "batch_size": int(args.batch_size),
        "n_init": int(args.n_init),
        "sim_threshold": float(1.0 - nominal_r),
        "points_per_center_min": int(np.min(points_per_center)),
        "points_per_center_median": int(np.median(points_per_center)),
        "points_per_center_max": int(np.max(points_per_center)),
        "empty_cells": int(empty_cells),
        "seed": int(args.seed),
        "use_gpu": bool(use_gpu),
        "inertia": float(inertia),
        "quantization": {k: float(v) for k, v in quant_metrics.items()},
    }
    if dir_info:
        stats["embeddings_dir_info"] = dir_info

    stats_path = os.path.join(args.out_dir, out_name.replace(".npy", ".json"))
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[kmeans++] Saved: {out_path} ({centers.shape})")
    print(f"[kmeans++] Saved: {stats_path}")
    print(f"[kmeans++] Use evaluate.py with: --centers_suffix '{suffix}'")


def main():
    ap = argparse.ArgumentParser(
        description="Build vocabulary centers via MiniBatch K-means with k-means++ init."
    )
    ap.add_argument("--embeddings_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./centers")
    ap.add_argument("--V", type=int, required=True,
                    help="Number of centers (= semantic unit granularity).")
    ap.add_argument("--r_c_percentile", type=float, default=99.0,
                    help="Percentile for per-center r_c from Voronoi cell distances. "
                         "100=max, 99 trims outliers. Default: 99.")
    ap.add_argument("--max_iter", type=int, default=300,
                    help="Max MiniBatchKMeans iterations.")
    ap.add_argument("--batch_size", type=int, default=10000,
                    help="MiniBatchKMeans batch size.")
    ap.add_argument("--n_init", type=int, default=3,
                    help="Number of K-means restarts (best by inertia).")
    ap.add_argument("--ref_size", type=int, default=0,
                    help="Fit subset size (0=full). Voronoi assignment/r_c always on full data.")
    ap.add_argument("--max_per_section", type=int, default=0,
                    help="Cap per-section spans for fit subset (equal sampling mode only). 0=no cap.")
    ap.add_argument("--section_sampling", type=str, default="equal", choices=["equal", "proportional"],
                    help="How to subsample for fit set. proportional requires --ref_size > 0.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--r_c_hist_bins", type=int, default=512,
                    help="Histogram bins for percentile r_c (cos distance in [0,2]).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.V <= 0:
        raise ValueError("--V must be positive")
    if args.max_iter <= 0:
        raise ValueError("--max_iter must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.n_init <= 0:
        raise ValueError("--n_init must be positive")
    if not (0 < args.r_c_percentile <= 100):
        raise ValueError("--r_c_percentile must be in (0, 100]")
    if args.r_c_hist_bins < 32:
        raise ValueError("--r_c_hist_bins must be >= 32")
    if args.section_sampling == "proportional" and (not args.ref_size or args.ref_size <= 0):
        raise ValueError("--section_sampling proportional requires --ref_size > 0")
    if not os.path.isdir(args.embeddings_dir):
        raise ValueError(f"Embeddings directory not found: {args.embeddings_dir}")

    use_gpu = _check_gpu()
    if use_gpu:
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            vram_bytes = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            print(f"[kmeans++] GPU available: {torch.cuda.get_device_name(0)}, VRAM={vram_bytes / 1e9:.1f}GB")
        except Exception:
            pass

    dir_info = parse_embeddings_dir(args.embeddings_dir)
    embedding_files = find_embedding_files(args.embeddings_dir, dir_info["unit"] if dir_info else None)
    if not embedding_files:
        raise ValueError(f"No embedding files in {args.embeddings_dir}")

    print(f"[kmeans++] V={args.V}, r_c_percentile={args.r_c_percentile}, max_iter={args.max_iter}, "
          f"batch_size={args.batch_size}, n_init={args.n_init}")
    print(f"[kmeans++] Embeddings: {[os.path.basename(f) for f in embedding_files]}")

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
    print(f"[kmeans++] Total: N={N:,}, d={d} (streaming store)")

    if args.V > N:
        raise ValueError(f"V={args.V} > N={N}: cannot train more centers than points")

    eligible_indices = None
    if args.section_sampling == "proportional" and args.ref_size > 0:
        eligible_indices = _eligible_indices_proportional(store, args.ref_size, rng)
        print(f"[kmeans++] Section sampling=proportional, ref_size={args.ref_size:,} -> "
              f"eligible pool {len(eligible_indices):,}")
    elif args.max_per_section > 0:
        per_section = []
        for j in range(len(store.cumsum) - 1):
            s, e = int(store.cumsum[j]), int(store.cumsum[j + 1])
            n_j = e - s
            take = min(args.max_per_section, n_j)
            if take <= 0:
                continue
            if take == n_j:
                idx = np.arange(s, e, dtype=np.int64)
            else:
                idx = rng.choice(np.arange(s, e, dtype=np.int64), size=take, replace=False)
            per_section.append(idx)
        if per_section:
            eligible_indices = np.concatenate(per_section)
            print(f"[kmeans++] Section cap max_per_section={args.max_per_section:,} -> "
                  f"eligible pool {len(eligible_indices):,}")

    if eligible_indices is not None and len(eligible_indices) < args.V:
        raise ValueError(
            f"Eligible fit pool too small for V={args.V}: {len(eligible_indices)}. "
            "Increase --ref_size or --max_per_section."
        )

    if eligible_indices is not None:
        fit_idx = eligible_indices
    elif args.ref_size and args.ref_size < N:
        fit_idx = rng.choice(N, size=args.ref_size, replace=False)
    else:
        fit_idx = None

    if fit_idx is None:
        print("[kmeans++] Building full fit matrix in RAM...")
        t_load = time.time()
        parts = []
        for j, se in enumerate(store.section_embeddings):
            print(f"  Reading section {j}...", end=" ", flush=True)
            arr = np.asarray(se, dtype=np.float32).copy()
            parts.append(arr)
            print(f"{arr.shape[0]:,} x {arr.shape[1]}")
        X_fit = np.concatenate(parts, axis=0)
        print(f"[kmeans++] Full fit matrix ready in {time.time() - t_load:.1f}s")
    else:
        X_fit = store.get_rows(fit_idx)

    l2_normalize_inplace(X_fit)
    fit_size = int(X_fit.shape[0])
    print(f"[kmeans++] Fit set size: {fit_size:,}")

    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for 2build_centers_kmeans++.py. "
            "Please install it in the current environment."
        ) from e

    print("[kmeans++] Training MiniBatchKMeans(init='k-means++')...")
    t0 = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=args.V,
        init="k-means++",
        n_init=args.n_init,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        random_state=args.seed,
    )
    kmeans.fit(X_fit)
    fit_elapsed = time.time() - t0
    print(f"[kmeans++] K-means done in {fit_elapsed:.1f}s, inertia={kmeans.inertia_:.2e}")

    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    l2_normalize_inplace(centers)
    V_actual = int(len(centers))

    # Free fit matrix before full-data Voronoi pass when possible.
    del X_fit

    import faiss
    try:
        if hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(min(os.cpu_count() or 1, 16))
    except Exception:
        pass

    center_index = _build_faiss_center_index(centers, use_gpu)
    if use_gpu:
        print("[kmeans++] FAISS index on GPU")

    print(f"\n[kmeans++] Voronoi assignment on full data ({N:,} points)...")
    t0 = time.time()
    assignments, assign_dists = _voronoi_assign(store, center_index, N, d, use_gpu)
    print(f"[kmeans++] Voronoi assignment done in {time.time() - t0:.1f}s")

    quant_metrics = _quantization_metrics(assign_dists)
    print(f"[kmeans++] Quantization: mean_cos_d={quant_metrics['mean_cos_distance']:.6f}, "
          f"sq_l2={quant_metrics['quantization_error_sq_l2']:.6f}, "
          f"p50={quant_metrics['cos_distance_p50']:.4f}, p95={quant_metrics['cos_distance_p95']:.4f}, "
          f"p99={quant_metrics['cos_distance_p99']:.4f}, max={quant_metrics['cos_distance_max']:.4f}")

    r_per_center, points_per_center, coverage = _compute_r_c_and_coverage(
        assignments, assign_dists, V_actual, args.r_c_percentile, args.r_c_hist_bins
    )
    if args.r_c_percentile >= 100.0:
        print("[kmeans++] r_c = max(Voronoi cell distances) -> coverage = 100%")
    else:
        print(f"[kmeans++] r_c = percentile({args.r_c_percentile}) of Voronoi cell distances "
              f"(hist B={args.r_c_hist_bins}) -> coverage = {coverage:.4%}")

    r_per_center = np.maximum(r_per_center, 1e-6)

    nominal_r = float(np.median(r_per_center))
    max_r = float(np.max(r_per_center))
    min_r = float(np.min(r_per_center))
    empty_cells = int(np.sum(points_per_center == 0))

    print(f"[kmeans++] Per-center r_c: min={min_r:.4f}, median={nominal_r:.4f}, max={max_r:.4f}")
    print(f"[kmeans++] Points per center: min={int(np.min(points_per_center))}, "
          f"median={int(np.median(points_per_center))}, max={int(np.max(points_per_center))}")
    if empty_cells > 0:
        print(f"[kmeans++] Warning: {empty_cells} centers have 0 assigned points")

    coverage_history = [float(coverage)]

    _save_outputs(
        args=args,
        centers=centers,
        V_actual=V_actual,
        nominal_r=nominal_r,
        min_r=min_r,
        max_r=max_r,
        N=N,
        d=d,
        fit_size=fit_size,
        points_per_center=points_per_center,
        empty_cells=empty_cells,
        use_gpu=use_gpu,
        dir_info=dir_info,
        coverage=coverage,
        coverage_history=coverage_history,
        r_per_center=r_per_center,
        quant_metrics=quant_metrics,
        inertia=kmeans.inertia_,
    )


if __name__ == "__main__":
    main()
