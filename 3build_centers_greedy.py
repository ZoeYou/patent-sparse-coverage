#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_centers_greedy.py

Approximate greedy covering to build vocabulary centers.

Cosine distance threshold r:
  cosine_dist(x, c) <= r  <=>  dot(x, c) >= 1 - r   (for L2-normalized vectors)

If --r is not provided, automatically selects radius using kNN distance distribution
(integrated from evaluate_r.py strategy).

Input:
  --embeddings_dir: Directory from 1create_N_embeddings.py output
  --mode: Task mode (abstract2abstract or claim2all)

Output:
  centers_greedy_r{r}.npy, [V_est, d], float32, L2-normalized
  and a .json with stats.

"""

import os
import json
import time
import argparse
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# Import utilities from utils.py
import utils
from utils import (
    parse_embedding_filename,
    parse_embeddings_dir,
    find_embedding_files,
    l2_normalize_inplace,
)

# Import radius evaluation functions from evaluate_r.py
script_dir = os.path.dirname(os.path.abspath(__file__))
evaluate_r_path = os.path.join(script_dir, "evaluate_r.py")

EVALUATE_R_AVAILABLE = False
try:
    import importlib.util
    if os.path.exists(evaluate_r_path):
        spec = importlib.util.spec_from_file_location("evaluate_r", evaluate_r_path)
        evaluate_r_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluate_r_module)
        
        # Import necessary functions and classes
        NNIndex = evaluate_r_module.NNIndex
        build_nn_index = evaluate_r_module.build_nn_index
        compute_knn_distance_distribution = evaluate_r_module.compute_knn_distance_distribution
        choose_radii_from_quantiles = evaluate_r_module.choose_radii_from_quantiles
        evaluate_multiple_radii = evaluate_r_module.evaluate_multiple_radii
        recommend_radius = evaluate_r_module.recommend_radius
        
        EVALUATE_R_AVAILABLE = True
        print("[auto_r] Successfully loaded evaluation functions from evaluate_r.py")
except Exception as e:
    # Fallback: if import fails, we'll use simplified version
    EVALUATE_R_AVAILABLE = False
    print(f"[WARNING] Could not import from evaluate_r.py: {e}")
    print("[WARNING] Will use simplified radius selection (single quantile).")



def build_output_dir(base_out_dir: str, embeddings_dir: str, mode: str) -> str:
    """
    Build output directory name that includes embedding source information.
    
    Format matches baselines.py expectations: centers_greedy_{mode}_{model_name}_{tokenization_unit}_{cls_suffix}_{layer}
    
    If base_out_dir is provided and doesn't look like a default, use it as-is.
    Otherwise, extract info from embeddings_dir and build a descriptive directory name.
    """
    # If user provided a custom out_dir (not default), use it as-is
    if base_out_dir != "./centers" and base_out_dir != "centers":
        return base_out_dir
    
    # Extract info from embeddings_dir
    # Normalize path first (remove trailing slash)
    normalized_dir = embeddings_dir.rstrip('/')
    dir_info = parse_embeddings_dir(normalized_dir)
    
    if dir_info:
        # Format: centers_greedy_{mode}_{model_name}_{unit}_{cls_suffix}_{layer}
        # This matches baselines.py expected format: centers_greedy_{mode}_{model_name}_{tokenization_unit}_{cls_suffix}_{layer}
        model_name = dir_info['model_name']
        tokenization_unit = dir_info['unit']  # e.g., spacy_token, spacy_sentence
        cls_suffix = dir_info['cls_suffix']  # cls or nocls
        layer = dir_info.get('layer', 'last')  # last or second_last, default to last if not found
        
        out_dir = f"centers_greedy_{mode}_{model_name}_{tokenization_unit}_{cls_suffix}_{layer}"
        return out_dir
    
    # Fallback: try to extract from directory name manually
    # Handle case where parse_embeddings_dir fails (e.g., due to path format)
    basename = os.path.basename(normalized_dir)
    if basename.startswith('embeddings_'):
        # Try to extract components manually
        parts = basename.replace('embeddings_', '').split('_')
        if len(parts) >= 4:
            # Assume format: {model}_{unit}_{cls}_{layer}
            # Model name might contain underscores, so we need to be careful
            # Try to find cls/nocls and last/second_last positions
            cls_pos = None
            layer_pos = None
            for i, part in enumerate(parts):
                if part in ['cls', 'nocls']:
                    cls_pos = i
                elif part in ['last', 'second_last']:
                    layer_pos = i
            
            if cls_pos is not None and layer_pos is not None and cls_pos < layer_pos:
                model_name = '_'.join(parts[:cls_pos])
                tokenization_unit = '_'.join(parts[cls_pos+1:layer_pos])
                cls_suffix = parts[cls_pos]
                layer = parts[layer_pos]  # last or second_last
                out_dir = f"centers_greedy_{mode}_{model_name}_{tokenization_unit}_{cls_suffix}_{layer}"
                return out_dir
    
    # Final fallback: use directory name as-is
    return f"centers_greedy_{basename}_{mode}"



def compute_knn_distance_distribution_simple(
    X: np.ndarray,
    k: int = 20,
    n_query: int = 20000,
    seed: int = 0,
    faiss_index=None,
) -> np.ndarray:
    """
    Simplified version: Compute kNN distance distribution using FAISS index.
    
    This computes the distribution of distances to the k-th nearest neighbor,
    which is used to automatically select an appropriate radius for greedy covering.
    
    Parameters:
    -----------
    X : np.ndarray
        Normalized embedding matrix (same as the one used to build faiss_index)
    k : int
        k for kNN (default 20)
    n_query : int
        Number of query points to sample (default 20000)
    seed : int
        Random seed
    faiss_index : faiss.IndexFlatIP
        Prebuilt FAISS index on X
    
    Returns:
    --------
    np.ndarray: Distances to the k-th nearest neighbor for each query point
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    
    # Sample query set
    n_q = min(n_query, N)
    query_idx = rng.choice(N, size=n_q, replace=False)
    X_query = X[query_idx].astype(np.float32)
    
    # Query kNN (k+1 to handle potential self-matches when query point is in index)
    # IndexFlatIP returns similarities (inner products), not distances
    sims, idxs = faiss_index.search(X_query, k + 1)
    
    # Convert from similarity to cosine distance
    # For L2-normalized vectors: cosine_distance = 1 - cosine_similarity = 1 - inner_product
    cosine_dists = 1.0 - sims
    
    kth_dists = []
    for i in range(n_q):
        d = cosine_dists[i]
        # Skip near-zero distances (self-matches: when query point is in index, first neighbor is itself)
        valid = d[d > 1e-8]
        if len(valid) >= k:
            # Have at least k valid neighbors, take the k-th
            kth_dists.append(valid[k - 1])
        elif len(valid) > 0:
            # Fewer than k valid neighbors, take the last one
            kth_dists.append(valid[-1])
        else:
            # No valid neighbors (shouldn't happen), take the last distance anyway
            kth_dists.append(d[-1])
    
    return np.array(kth_dists)


def auto_select_radius_simple(
    X: np.ndarray,
    faiss_index,
    quantile: float = 0.8,
    k: int = 20,
    n_query: int = 20000,
    seed: int = 0,
) -> float:
    """
    Simplified radius selection (fallback if evaluate_r.py is not available).
    
    Uses a single quantile of the kNN distance distribution.
    """
    print(f"[auto_r] Computing kNN distance distribution (k={k}, n_query={n_query})...")
    D = compute_knn_distance_distribution_simple(X, k=k, n_query=n_query, seed=seed, faiss_index=faiss_index)
    
    r = float(np.quantile(D, quantile))
    print(f"[auto_r] Selected radius r={r:.6f} (quantile={quantile:.0%} of kNN distances)")
    print(f"[auto_r]   kNN distance stats: min={D.min():.4f}, median={np.median(D):.4f}, max={D.max():.4f}")
    
    return r


def auto_select_radius_full(
    X: np.ndarray,
    X_ref: np.ndarray,
    faiss_index,
    quantiles: list,
    k: int = 20,
    n_query: int = 20000,
    n_sample_centers: int = 2000,
    min_coverage: float = 0.95,
    assignment_mode: str = "multi",
    fast_mode: bool = False,
    seed: int = 0,
) -> float:
    """
    Full radius selection using evaluation strategy from evaluate_r.py.
    
    Evaluates multiple candidate radii (based on quantiles) and selects the best one
    based on coverage constraint.
    
    Parameters:
    -----------
    X : np.ndarray
        Full embedding matrix (for center sampling)
    X_ref : np.ndarray
        Reference set (normalized, same as faiss_index)
    faiss_index : faiss.IndexFlatIP
        Prebuilt FAISS index on X_ref
    quantiles : list
        List of quantiles to evaluate)
    k : int
        k for kNN (default 20)
    n_query : int
        Number of query points (default 20000)
    n_sample_centers : int
        Number of centers to sample for sphere neighbor stats
    min_coverage : float
        Minimum required coverage fraction
    assignment_mode : str
        "multi" or "hard"
    fast_mode : bool
        If True, skip covering evaluation
    seed : int
        Random seed
    
    Returns:
    --------
    float: Selected radius (cosine distance)
    """
    print(f"[auto_r] Full evaluation mode:")
    print(f"  Evaluating {len(quantiles)} candidate radii: {quantiles}")
    print(f"  Constraint: coverage >= {min_coverage:.0%}")
    if n_sample_centers == 0:
        print(f"  Mode: coverage-only (skipping sphere neighbor stats for faster evaluation)")
        print(f"  Note: Coverage calculation does not depend on assignment_mode")
    else:
        print(f"  Assignment mode: {assignment_mode} (for sphere stats only; coverage and posting lists use sphere membership)")
    
    # Step 1: Compute kNN distance distribution
    print(f"\n[auto_r] Step 1: Computing kNN distance distribution (k={k}, n_query={n_query})...")
    D = compute_knn_distance_distribution_simple(X_ref, k=k, n_query=n_query, seed=seed, faiss_index=faiss_index)
    
    # Step 2: Choose candidate radii from quantiles
    radii = choose_radii_from_quantiles(D, quantiles=tuple(quantiles))
    
    # Sort radii by value (from largest to smallest) for efficient evaluation
    sorted_radii = sorted(radii.items(), key=lambda x: x[1], reverse=True)
    print(f"[auto_r] Candidate radii (sorted from largest to smallest): {[(q, f'{r:.6f}') for q, r in sorted_radii]}")
    
    # Step 3: Build NNIndex wrapper for evaluation functions
    nn_index = NNIndex(
        index=faiss_index,
        X_ref=X_ref,
        metric="cosine",
        use_faiss=True,
        faiss_index_type="IP"
    )
    
    # Step 4: Evaluate radii from largest to smallest, stop when we find one that meets coverage
    print(f"\n[auto_r] Step 2: Evaluating radii from largest to smallest...")
    print(f"  Strategy: Evaluate each radius, continue to smaller ones if coverage >= {min_coverage:.0%}")
    
    radius_stats = {}
    selected_q = None
    selected_r = None
    is_first_radius = True  # Track if this is the first (largest) radius
    
    for q, r in sorted_radii:
        print(f"\n[auto_r] Evaluating radius r={r:.6f} (q={q:.0%})...")
        
        # Evaluate this single radius
        single_radii = {q: r}
        # Skip sphere stats if n_sample_centers=0 (coverage-only mode, faster)
        skip_sphere_stats = (n_sample_centers == 0)
        
        stats_dict = evaluate_multiple_radii(
            X=X,  # Full dataset for center sampling
            radii=single_radii,
            nn_index=nn_index,
            n_sample=0 if fast_mode else 50000,  # Skip covering if fast mode
            n_sample_centers=n_sample_centers,
            seed=seed,
            min_coverage=min_coverage,
            assignment_mode=assignment_mode,
            skip_covering=fast_mode,
            skip_sphere_stats=skip_sphere_stats,  # Skip sphere stats if n_sample_centers=0
        )
        
        stats = stats_dict[q]
        radius_stats[q] = stats
        
        # Check if this radius meets coverage requirement
        if stats.meets_coverage_threshold:
            selected_q = q
            selected_r = r
            print(f"[auto_r]   ✓ Coverage {stats.covered_frac_mean:.1%} >= {min_coverage:.0%} - meets requirement")
            print(f"[auto_r]   Continuing to evaluate smaller radii to find the smallest that meets coverage...")
            is_first_radius = False
        else:
            print(f"[auto_r]   ✗ Coverage {stats.covered_frac_mean:.1%} < {min_coverage:.0%} - does not meet requirement")
            # If we already found a valid radius, stop here (we want the smallest valid one)
            if selected_q is not None:
                print(f"[auto_r]   Stopping evaluation - found smallest valid radius: q={selected_q:.0%}, r={selected_r:.6f}")
                break
            # If this is the largest radius and it doesn't meet coverage, stop immediately
            # because smaller radii will have even lower coverage
            if is_first_radius:
                print(f"[auto_r]   WARNING: Largest radius does not meet coverage requirement.")
                print(f"[auto_r]   Smaller radii will have even lower coverage, stopping evaluation.")
                break
            is_first_radius = False
    
    # Step 5: Select best radius
    print(f"\n[auto_r] Step 3: Selecting best radius...")
    
    if selected_q is None:
        # Fallback: use largest radius (first one we evaluated)
        if sorted_radii:
            selected_q, selected_r = sorted_radii[0]
            print(f"[auto_r] WARNING: No radius meets coverage constraint >= {min_coverage:.0%}.")
            print(f"[auto_r] Fallback to largest radius: q={selected_q:.0%}, r={selected_r:.6f}")
        else:
            raise ValueError("No candidate radii to evaluate")
    else:
        print(f"[auto_r] >>> Selected radius: r={selected_r:.6f} (quantile={selected_q:.0%})")
        print(f"[auto_r]   This is the smallest radius that meets coverage >= {min_coverage:.0%}")
    
    # Print summary of selected radius
    if selected_q in radius_stats:
        stats = radius_stats[selected_q]
        status = "✓ RECOMMENDED" if stats.is_recommended else "⚠ FALLBACK"
        print(f"[auto_r]   Status: {status}")
        print(f"[auto_r]   Coverage: {stats.covered_frac_mean:.1%} (threshold: {min_coverage:.0%})")
        print(f"[auto_r]   Avg degree (scaled): {stats.mean_degree_mean:.1f}")
        print(f"[auto_r]   Vocabulary size estimate: {stats.n_centers_mean:.0f}")
    
    return selected_r

def main():
    ap = argparse.ArgumentParser(
        description="Build vocabulary centers using greedy covering algorithm. "
                    "Uses embeddings directory from 1create_N_embeddings.py output. "
                    "Task modes: abstract2abstract (uses abstract section only), "
                    "claim2all (uses all sections: abstract, claim, invention)."
    )
    
    ap.add_argument("--embeddings_dir", type=str, required=True,
                    help="Directory from 1create_N_embeddings.py output. "
                         "Format: embeddings_{model}_{unit}_{cls}_{layer}.")
    ap.add_argument("--mode", type=str, required=True, choices=['abstract2abstract', 'claim2all'],
                    help="Task mode. "
                         "abstract2abstract: uses only abstract section. "
                         "claim2all: uses all sections (abstract, claim, invention).")
    
    ap.add_argument("--out_dir", type=str, default="./centers")
    ap.add_argument("--seed", type=int, default=666)

    ap.add_argument("--r", type=float, default=None,
                    help="Cosine distance radius (0~2). If not provided, will auto-select based on kNN distribution.")
    ap.add_argument("--ref_size", type=int, default=0,
                    help="Build FAISS index on a reference subset of this size. Default 0=use full dataset (consistent with kmeans). Set to >0 to use subset (faster but less accurate).")
    ap.add_argument("--max_centers", type=int, default=50000, 
                    help="Maximum number of vocabulary centers to select (default: 50000).")
    ap.add_argument("--log_every", type=int, default=5)

    ap.add_argument("--n_candidates_per_iter", type=int, default=2000,
                    help="Number of candidate centers to evaluate per iteration (default: 2000). "
                         "Larger values = more thorough search but slower. "
                         "Consider reducing if step3 is too slow.")
    ap.add_argument("--n_centers_per_iter", type=int, default=10,
                    help="Number of centers to select per iteration (default: 10). "
                         "Selecting multiple centers per iteration can reduce total iterations needed. "
                         "Centers are selected by gain (number of newly covered points), with density as tie-breaker. "
                         "When selecting multiple centers, uses incremental gain to minimize overlap.")
    ap.add_argument("--no_minimize_overlap", dest="minimize_overlap", action="store_false", default=True,
                    help="Disable overlap minimization when selecting multiple centers per iteration. "
                         "Default: minimize_overlap=True (enabled). When enabled, each subsequent center "
                         "only counts points not covered by previously selected centers in the same iteration.")
    # Note: minimize_overlap defaults to True. Use --no_minimize_overlap to disable.

    ap.add_argument("--no_use_density_sampling", dest="use_density_sampling", action="store_false", default=True,
                    help="Disable density-aware candidate sampling. Default: use_density_sampling=True (enabled). "
                         "When disabled, uses random sampling instead. When enabled, samples a larger pool, "
                         "computes density (neighbor count), then selects top candidates by density. "
                         "Requires EVALUATE_R_AVAILABLE=True.")
    # Note: use_density_sampling defaults to True. Use --no_use_density_sampling to disable.
    ap.add_argument("--density_pool_size", type=int, default=5000,
                    help="Size of candidate pool for density evaluation (default: 5000). "
                         "Only used if --use_density_sampling is enabled. "
                         "Should be >= n_candidates_per_iter. Larger pool = better density estimates but slower.")
    ap.add_argument("--early_stop_gain_ratio", type=float, default=0.5,
                    help="Early stopping threshold for candidate evaluation (default: 0.5). "
                         "In each iteration, we evaluate n_candidates_per_iter candidates. "
                         "If we find a candidate with gain >= this_ratio * n_uncovered, we stop "
                         "evaluating remaining candidates (they're unlikely to be much better). "
                         "Example: If 1000 points are uncovered and ratio=0.5, we stop if we find "
                         "a candidate covering >= 500 points. Set to 1.0 to disable (evaluate all candidates).")
    ap.add_argument("--n_threads", type=int, default=None,
                    help="Number of threads for parallel candidate evaluation (default: None=auto, uses CPU count). "
                         "Set to 1 to disable parallelization. Only used for gain calculation, not FAISS operations.")
    
    # Auto radius selection parameters
    ap.add_argument("--auto_r_quantiles", type=float, nargs="+", default=[0.75, 0.8, 0.85, 0.9, 0.95],
                    help="Quantiles of kNN distance distribution to evaluate (default: 0.75 0.8 0.85 0.9 0.95). "
                         "Multiple quantiles will be evaluated and the best one selected based on constraints. "
                         "Only used if --r is not provided. Fewer quantiles = faster but less thorough search.")
    ap.add_argument("--auto_r_k", type=int, default=20,
                    help="k for kNN distance distribution (default: 20). Only used if --r is not provided.")
    ap.add_argument("--auto_r_n_query", type=float, default=0.01,
                    help="Fraction of embeddings to use as query points for kNN distribution (default: 0.01, i.e., 1%%). "
                         "The actual number will be min(n_query_ratio * N, N, 5000) where N is the total number of embeddings. "
                         "Maximum is capped at 5k to avoid excessive computation. Only used if --r is not provided.")
    ap.add_argument("--auto_r_n_sample_centers", type=int, default=0,
                    help="Number of centers to sample for sphere neighbor stats (default: 0, skip for faster coverage-only evaluation). "
                         "Set to >0 (e.g., 2000) to compute posting length statistics. "
                         "Note: Coverage calculation does not depend on this parameter. Only used if --r is not provided.")
    ap.add_argument("--auto_r_min_coverage", type=float, default=0.9,
                    help="Minimum required coverage fraction (default: 0.9). Only used if --r is not provided for radius auto-selection.")

    ap.add_argument("--auto_r_assignment_mode", type=str, default="multi", choices=["multi", "hard"],
                    help="Assignment mode for radius evaluation (default: multi). "
                         "'multi' for sphere membership (matches greedy algorithm's actual behavior), "
                         "'hard' for Voronoi cell size. "
                         "Note: This only affects radius selection evaluation. "
                         "The actual posting lists saved are always based on sphere membership (multi-assignment). "
                         "Only used if --r is not provided.")
    ap.add_argument("--auto_r_fast", action="store_true",
                    help="Fast mode: skip covering evaluation, only compute sphere neighbor stats. "
                         "Only used if --r is not provided.")
    
    args = ap.parse_args()

    # Input validation
    if args.r is not None and (args.r < 0 or args.r > 2):
        raise ValueError(f"--r (cosine distance radius) must be in [0, 2], got {args.r}")
    if args.r is None:
        # Validate auto selection parameters
        for q in args.auto_r_quantiles:
            if q < 0 or q > 1:
                raise ValueError(f"All quantiles must be in [0, 1], got {q}")
        if args.auto_r_min_coverage < 0 or args.auto_r_min_coverage > 1:
            raise ValueError(f"--auto_r_min_coverage must be in [0, 1], got {args.auto_r_min_coverage}")
        if args.auto_r_n_query < 0 or args.auto_r_n_query > 1:
            raise ValueError(f"--auto_r_n_query must be in [0, 1] (as a fraction), got {args.auto_r_n_query}")
    if args.max_centers <= 0:
        raise ValueError(f"--max_centers must be positive, got {args.max_centers}")
    if args.n_candidates_per_iter <= 0:
        raise ValueError(f"--n_candidates_per_iter must be positive, got {args.n_candidates_per_iter}")
    if args.n_centers_per_iter <= 0:
        raise ValueError(f"--n_centers_per_iter must be positive, got {args.n_centers_per_iter}")
    if args.n_centers_per_iter > args.n_candidates_per_iter:
        raise ValueError(f"--n_centers_per_iter ({args.n_centers_per_iter}) cannot exceed "
                         f"--n_candidates_per_iter ({args.n_candidates_per_iter})")
    if args.use_density_sampling and args.density_pool_size < args.n_candidates_per_iter:
        raise ValueError(f"--density_pool_size ({args.density_pool_size}) should be >= "
                         f"--n_candidates_per_iter ({args.n_candidates_per_iter}) when using density sampling")
    if args.ref_size < 0:
        raise ValueError(f"--ref_size must be non-negative, got {args.ref_size}")
    if args.early_stop_gain_ratio < 0 or args.early_stop_gain_ratio > 1:
        raise ValueError(f"--early_stop_gain_ratio must be in [0, 1], got {args.early_stop_gain_ratio}")
    
    # Validate inputs
    if not os.path.isdir(args.embeddings_dir):
        raise ValueError(f"Embeddings directory does not exist: {args.embeddings_dir}")
    
    # Parse directory name to get info
    embeddings_dir_info = parse_embeddings_dir(args.embeddings_dir)
    if embeddings_dir_info is None:
        print(f"[WARNING] Could not parse embeddings directory name: {args.embeddings_dir}")
        print(f"[WARNING] Expected format: embeddings_{{model}}_{{unit}}_{{cls}}_{{layer}}")
    
    # Find embedding files based on mode
    embedding_files = find_embedding_files(args.embeddings_dir, args.mode,
                                           embeddings_dir_info['unit'] if embeddings_dir_info else None)
    
    if len(embedding_files) == 0:
        raise ValueError(f"Could not find embedding files for mode '{args.mode}' in directory: {args.embeddings_dir}")
    
    print(f"[greedy] Directory: {args.embeddings_dir}")
    print(f"[greedy] Mode: {args.mode}")
    print(f"[greedy] Found {len(embedding_files)} file(s): {[os.path.basename(f) for f in embedding_files]}")
    if embeddings_dir_info:
        print(f"[greedy] Parsed info: model={embeddings_dir_info['model_name']}, "
              f"unit={embeddings_dir_info['unit']}, cls={embeddings_dir_info['cls_suffix']}, "
              f"layer={embeddings_dir_info['layer']}")

    # Build output directory with embedding source information
    # Format: centers_greedy_{mode}_{model_name}_{tokenization_unit}_{cls_suffix}_{layer}
    # This matches baselines.py expected format for automatic discovery
    args.out_dir = build_output_dir(args.out_dir, args.embeddings_dir, args.mode)
    print(f"[greedy] Output directory: {args.out_dir}")
    if embeddings_dir_info:
        layer = embeddings_dir_info.get('layer', 'last')
        print(f"[greedy] Output dir format matches baselines.py expectations:")
        print(f"[greedy]   Format: centers_greedy_{args.mode}_{embeddings_dir_info['model_name']}_{embeddings_dir_info['unit']}_{embeddings_dir_info['cls_suffix']}_{layer}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Load embeddings: load and concatenate multiple section files if needed
    section_embeddings = []
    for filepath in embedding_files:
        print(f"[greedy] Loading: {os.path.basename(filepath)}")
        if filepath.endswith('.npz'):
            # For .npz files, load the 'embeddings' key
            npz_data = np.load(filepath, mmap_mode="r")
            if 'embeddings' in npz_data:
                section_emb = npz_data['embeddings']
            else:
                # If no 'embeddings' key, try to use the first array
                keys = list(npz_data.keys())
                if len(keys) > 0:
                    section_emb = npz_data[keys[0]]
                else:
                    raise ValueError(f"No arrays found in .npz file: {filepath}")
        else:
            section_emb = np.load(filepath, mmap_mode="r")
        
        section_embeddings.append(section_emb)
        print(f"[greedy]   Loaded {section_emb.shape[0]:,} embeddings (dim={section_emb.shape[1]})")
    
    # Concatenate all sections
    if len(section_embeddings) > 1:
        print(f"[greedy] Concatenating {len(section_embeddings)} sections...")
        X_mm = np.concatenate(section_embeddings, axis=0)
        print(f"[greedy]   Total: {X_mm.shape[0]:,} embeddings")
    else:
        X_mm = section_embeddings[0]
    
    N, d = X_mm.shape
    print(f"[greedy] Loaded embeddings: N={N:,}, d={d}")
    
    # Validate input data
    if N == 0:
        raise ValueError("Empty embedding matrix")
    if d == 0:
        raise ValueError("Zero-dimensional embeddings")

    # Choose reference set for indexing (for neighbor queries)
    if args.ref_size and args.ref_size < N:
        ref_idx = rng.choice(N, size=args.ref_size, replace=False)
        X_ref = np.asarray(X_mm[ref_idx], dtype=np.float32)
        print(f"[greedy] Using reference subset for index: N_ref={len(ref_idx):,}")
    else:
        ref_idx = None
        # Use .copy() to ensure writable array (memmap is read-only)
        X_ref = np.array(X_mm, dtype=np.float32, copy=True)
        print(f"[greedy] Using FULL data for index: N_ref={X_ref.shape[0]:,} (RAM heavy)")

    # Normalize reference vectors for cosine/IP
    l2_normalize_inplace(X_ref)
    print("[greedy] L2-normalized reference vectors.")

    # Build FAISS index
    import faiss
    index = faiss.IndexFlatIP(d)
    index.add(X_ref)
    
    # Enable FAISS threading if available (can speed up range_search)
    try:
        if hasattr(faiss, 'omp_set_num_threads'):
            # Set number of threads for FAISS (default is usually CPU count)
            faiss.omp_set_num_threads(os.cpu_count())
            print(f"[greedy] FAISS threading enabled: {os.cpu_count()} threads")
    except:
        pass  # FAISS threading not available or already set
    
    print("[greedy] FAISS IndexFlatIP built.")

    # Coverage bookkeeping is over the reference set (which equals full dataset by default)
    N_ref = X_ref.shape[0]

    # Auto-select radius if not provided
    if args.r is None:
        # Calculate actual number of query points based on fraction and reference set size
        n_query_actual = max(1, int(args.auto_r_n_query * N_ref))
        n_query_actual = min(n_query_actual, N_ref, 5000)  # Cap at 5k to avoid excessive computation
        print(f"[auto_r] Using {args.auto_r_n_query:.1%} of reference set ({N_ref:,} points) = {n_query_actual:,} query points for kNN distribution (capped at 5k)")
        
        if EVALUATE_R_AVAILABLE:
            # Use full evaluation strategy from evaluate_r.py
            # For center sampling, use full dataset if available, otherwise use reference set
            if ref_idx is None:
                X_for_sampling = np.asarray(X_mm, dtype=np.float32)
            else:
                # If using subset, we can still sample from full dataset (just need to load it)
                # But for efficiency, we'll use the reference set
                X_for_sampling = X_ref
                print("[auto_r] NOTE: Using reference subset for center sampling (ref_size < N)")
            
            args.r = auto_select_radius_full(
                X=X_for_sampling,
                X_ref=X_ref,
                faiss_index=index,
                quantiles=args.auto_r_quantiles,
                k=args.auto_r_k,
                n_query=n_query_actual,
                n_sample_centers=args.auto_r_n_sample_centers,
                min_coverage=args.auto_r_min_coverage,
                assignment_mode=args.auto_r_assignment_mode,
                fast_mode=args.auto_r_fast,
                seed=args.seed
            )
        else:
            # Fallback to simple quantile selection
            if len(args.auto_r_quantiles) > 1:
                print(f"[auto_r] WARNING: Multiple quantiles specified but evaluate_r.py not available.")
                print(f"[auto_r] Using first quantile: {args.auto_r_quantiles[0]}")
            args.r = auto_select_radius_simple(
                X_ref, index,
                quantile=args.auto_r_quantiles[0],
                k=args.auto_r_k,
                n_query=n_query_actual,
                seed=args.seed
            )
        print(f"[greedy] Using auto-selected radius: r={args.r:.6f}")

    covered = np.zeros(N_ref, dtype=bool)
    # Use a set to track uncovered indices for faster sampling (O(1) removal vs O(N) scan)
    # This optimization significantly speeds up the greedy loop when many points are covered
    uncovered_set = set(range(N_ref))

    sim_th = 1.0 - float(args.r)  # cosine_dist <= r  <=> sim >= 1-r
    print(f"[greedy] radius r={args.r:.6f} => sim_threshold={sim_th:.6f}")
    if args.n_centers_per_iter > 1:
        if args.minimize_overlap:
            print(f"[greedy] Multi-center mode: selecting {args.n_centers_per_iter} centers per iteration "
                  f"with incremental gain (minimizing overlap)")
        else:
            print(f"[greedy] Multi-center mode: selecting {args.n_centers_per_iter} centers per iteration "
                  f"(sorted by gain, allowing overlap)")
    
    # Validate density sampling parameters
    density_sampling_enabled = args.use_density_sampling
    if density_sampling_enabled:
        if not EVALUATE_R_AVAILABLE:
            print("[greedy] WARNING: --use_density_sampling requires evaluate_r.py but it's not available.")
            print("[greedy] Falling back to random sampling.")
            density_sampling_enabled = False
        else:
            print(f"[greedy] Density-aware sampling enabled: pool_size={args.density_pool_size}, "
                  f"selecting top {args.n_candidates_per_iter} candidates by density")

    centers_ref_ids = []
    gains = []
    coverage_history = []  # Store coverage after each center is added (for post-processing)
    t0 = time.time()
    t_last_iter = t0  # Track time of last iteration for time estimation
    iter_times = []  # Track recent iteration times for averaging (keep last 5 iterations)
    recent_coverages = []  # Track coverage at each logged iteration for rate estimation
    
    # Pre-allocate uncovered_array for reuse (updated each iteration)
    uncovered_array = np.zeros(N_ref, dtype=bool)
    
    # Pre-compute n_threads once
    n_threads = args.n_threads if args.n_threads is not None else os.cpu_count()

    # Optimized greedy covering loop
    for it in range(args.max_centers):
        # Check if we should log this iteration (compute once per iteration)
        should_log = (it + 1) % args.log_every == 0 or it == 0
        
        # Fast path: use set for uncovered tracking
        n_uncovered = len(uncovered_set)
        if n_uncovered == 0:
            print("[greedy] All reference points covered. Stop.")
            break

        covered_frac = 1.0 - (n_uncovered / N_ref)

        # Adaptive optimization: reduce candidate pool size and increase centers per iteration as coverage increases
        # When coverage is high, we can use fewer candidates since remaining points are sparse
        # Also increase centers per iteration to reduce total iterations needed
        if covered_frac < 0.3:
            # Early stage: use full candidate pool and default centers per iter
            adaptive_n_cand = args.n_candidates_per_iter
            adaptive_pool_size = args.density_pool_size
            adaptive_n_centers_per_iter = args.n_centers_per_iter
        elif covered_frac < 0.6:
            # Mid stage: reduce candidate pool by 25%, keep default centers per iter
            adaptive_n_cand = max(100, int(args.n_candidates_per_iter * 0.75))
            adaptive_pool_size = max(adaptive_n_cand, int(args.density_pool_size * 0.75))
            adaptive_n_centers_per_iter = args.n_centers_per_iter
        elif covered_frac < 0.9:
            # Late stage: reduce candidate pool by 50%, increase centers per iter by 2x
            adaptive_n_cand = max(100, int(args.n_candidates_per_iter * 0.5))
            adaptive_pool_size = max(adaptive_n_cand, int(args.density_pool_size * 0.5))
            adaptive_n_centers_per_iter = args.n_centers_per_iter * 2
        else:
            # Very late stage (coverage > 90%): reduce candidate pool further, increase centers per iter by 3x
            # This significantly reduces iterations when gain is very small
            adaptive_n_cand = max(100, int(args.n_candidates_per_iter * 0.5))
            adaptive_pool_size = max(adaptive_n_cand, int(args.density_pool_size * 0.5))
            adaptive_n_centers_per_iter = args.n_centers_per_iter * 3
        
        n_cand = min(adaptive_n_cand, n_uncovered)
        # Ensure adaptive_n_centers_per_iter doesn't exceed available candidates
        adaptive_n_centers_per_iter = min(adaptive_n_centers_per_iter, n_cand)
        
        # Auto-disable density sampling when we have many uncovered points (more efficient)
        use_density_this_iter = (density_sampling_enabled and 
                                n_uncovered > n_cand)
        
        # Convert uncovered_set to array once for reuse (used in both density and direct sampling)
        uncovered_array_indices = np.fromiter(uncovered_set, dtype=np.int64, count=n_uncovered)
        
        # Merge density sampling and candidate evaluation into single range_search
        if use_density_this_iter:
            # Density-aware sampling: sample larger pool, compute density, select top candidates
            # This is more efficient than random sampling when we have many uncovered points
            pool_size = min(adaptive_pool_size, n_uncovered)
            pool_candidates = rng.choice(uncovered_array_indices, size=pool_size, replace=False)
            
            # Do range_search once and reuse results for both density and gain calculation
            Xq_pool = X_ref[pool_candidates]
            try:
                lims_pool, D_pool, I_pool = index.range_search(Xq_pool, sim_th)
            except Exception as e:
                raise RuntimeError(f"FAISS range_search failed in density sampling: {e}")
            
            # Count neighbors for each candidate (density) - vectorized for speed
            # np.diff(lims_pool) gives [lims[1]-lims[0], lims[2]-lims[1], ...] = neighbor counts
            densities = np.diff(lims_pool).astype(np.int32)
            
            # Select top n_cand candidates by density
            top_density_idx = np.argsort(-densities)[:n_cand]
            cand = pool_candidates[top_density_idx]
            
            # Reuse range_search results - extract lims and I for selected candidates
            lims = np.zeros(n_cand + 1, dtype=np.int64)
            lims[0] = 0
            I_list = []
            for i, pool_idx in enumerate(top_density_idx):
                start, end = int(lims_pool[pool_idx]), int(lims_pool[pool_idx + 1])
                I_list.append(I_pool[start:end])
                lims[i + 1] = lims[i] + (end - start)
            I = np.concatenate(I_list) if I_list else np.array([], dtype=np.int64)
            
            if should_log:
                adaptive_info = ""
                if adaptive_n_cand < args.n_candidates_per_iter or adaptive_pool_size < args.density_pool_size or adaptive_n_centers_per_iter != args.n_centers_per_iter:
                    adaptive_info = f" [adaptive: n_cand={adaptive_n_cand}, pool={adaptive_pool_size}, centers/iter={adaptive_n_centers_per_iter}]"
                print(f"[greedy] Density sampling: pool={pool_size}, selected top {n_cand} by density "
                        f"(max={densities[top_density_idx[0]]}, min={densities[top_density_idx[-1]]}){adaptive_info}")
        else:
            # Direct sampling: use all uncovered points if few, otherwise random sample
            # This is more efficient when coverage is high (few uncovered points)
            if n_uncovered <= n_cand:
                # If we need all uncovered points, use them directly
                cand = uncovered_array_indices
            else:
                # Sample randomly from uncovered set
                cand = rng.choice(uncovered_array_indices, size=n_cand, replace=False)
            
            # range_search on candidates (batch) - only needed if not using density sampling
            Xq = X_ref[cand]  # already normalized
            try:
                lims, D, I = index.range_search(Xq, sim_th)
            except Exception as e:
                raise RuntimeError(f"FAISS range_search failed: {e}")
        
        # Validate range_search output
        if len(lims) != n_cand + 1:
            raise RuntimeError(f"range_search returned unexpected lims shape: {len(lims)} != {n_cand + 1}")

        # Evaluate gain for all candidates (to support multi-center selection)
        # Use numpy array operations instead of set operations for better performance
        # Update uncovered_array for fast boolean indexing (reuse pre-allocated array)
        uncovered_array.fill(False)
        uncovered_array[uncovered_array_indices] = True
        
        # Early stopping logic:
        # - max_possible_gain = n_uncovered (theoretical maximum: one center could cover all uncovered points)
        # - early_stop_threshold = early_stop_gain_ratio * max_possible_gain
        # - If we find a candidate with gain >= early_stop_threshold, we stop evaluating remaining candidates
        #   because it's already "good enough" (e.g., 95% of maximum possible)
        # Example: If n_uncovered=1000, early_stop_gain_ratio=0.95, then:
        #   - max_possible_gain = 1000
        #   - early_stop_threshold = 950
        #   - If we find a candidate covering 950+ points, we stop (even if there might be one covering 980)
        max_possible_gain = n_uncovered  # Theoretical maximum: cover all uncovered points
        early_stop_threshold = int(args.early_stop_gain_ratio * max_possible_gain) if args.early_stop_gain_ratio < 1.0 else max_possible_gain + 1
        
        # Parallelize candidate gain evaluation
        use_parallel = (n_threads > 1 and n_cand > 50)  # Only parallelize if many candidates
        
        def compute_gain(i, lims, I, uncovered_array, N_ref):
            """Compute gain for a single candidate."""
            start, end = int(lims[i]), int(lims[i + 1])
            # Validate indices
            if start < 0 or end > len(I) or start > end:
                return None
            neigh = I[start:end]
            if neigh.size == 0:
                return None
            # Validate neighbor indices are within bounds
            if np.any(neigh < 0) or np.any(neigh >= N_ref):
                return None
            
            # Use boolean indexing instead of set intersection
            # Only count neighbors that are both in neigh AND in uncovered_array
            new_covered_mask = uncovered_array[neigh]
            new_covered_array = neigh[new_covered_mask]
            g = len(new_covered_array)
            
            if g > 0:
                return (i, g, new_covered_array)
            return None
        
        candidate_gains = []  # List of (cand_pos, gain, new_covered_array)
        
        if use_parallel:
            # Parallel evaluation of candidates
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                futures = {executor.submit(compute_gain, i, lims, I, uncovered_array, N_ref): i 
                          for i in range(n_cand)}
                
                # Early stopping only works in sequential mode, so check results as they complete
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        candidate_gains.append(result)
                        
                        # Early stopping check (only for adaptive_n_centers_per_iter == 1)
                        # Note: We check args.n_centers_per_iter here because early stopping only makes sense
                        # when we're selecting 1 center per iteration (not adaptive)
                        if args.n_centers_per_iter == 1 and adaptive_n_centers_per_iter == 1 and result[1] >= early_stop_threshold:
                            if should_log:
                                print(f"[greedy] Early stop: found candidate with gain={result[1]} >= {early_stop_threshold} "
                                        f"({args.early_stop_gain_ratio:.0%} of max possible)")
                            # Cancel remaining futures (they'll complete but we'll ignore results)
                            break
        else:
            # Sequential evaluation (with early stopping support)
            for i in range(n_cand):
                result = compute_gain(i, lims, I, uncovered_array, N_ref)
                if result is not None:
                    candidate_gains.append(result)
                    
                    # Early stopping: if we found enough candidates with high gain, stop evaluating
                    # (only if adaptive_n_centers_per_iter == 1, otherwise we need to evaluate all to select top-k)
                    # Note: We check args.n_centers_per_iter here because early stopping only makes sense
                    # when we're selecting 1 center per iteration (not adaptive)
                    if args.n_centers_per_iter == 1 and adaptive_n_centers_per_iter == 1 and result[1] >= early_stop_threshold:
                        if should_log:
                            print(f"[greedy] Early stop: found candidate with gain={result[1]} >= {early_stop_threshold} "
                                    f"({args.early_stop_gain_ratio:.0%} of max possible), skipping remaining candidates")
                        break
       

        # Select top adaptive_n_centers_per_iter candidates by gain
        if len(candidate_gains) == 0:
            print(f"[greedy] No candidates with gain > 0. Stop.")
            break
        
        # Select centers: either simple top-k or incremental gain (to minimize overlap)
        if adaptive_n_centers_per_iter == 1 or not args.minimize_overlap:
            # Simple selection: just pick top-k by gain
            candidate_gains.sort(key=lambda x: x[1], reverse=True)
            n_select = min(adaptive_n_centers_per_iter, len(candidate_gains))
            selected_candidates = candidate_gains[:n_select]
        else:
            # Incremental gain selection: minimize overlap by recalculating gain after each selection
            # This ensures each new center covers mostly different points from previously selected ones
            # Use boolean array for faster incremental gain calculation (instead of set operations)
            selected_candidates = []
            remaining_candidates = candidate_gains.copy()
            # Use boolean array for tracking coverage within this iteration (much faster for large sets)
            covered_in_iter = np.zeros(N_ref, dtype=bool)
            
            for _ in range(min(adaptive_n_centers_per_iter, len(candidate_gains))):
                if len(remaining_candidates) == 0:
                    break
                
                best_incremental_gain = -1
                best_candidate = None
                best_candidate_idx = -1
                
                # Recalculate incremental gain for each remaining candidate
                # Parallelize incremental gain calculation if many candidates remain
                n_remaining = len(remaining_candidates)
                use_parallel_incremental = (args.n_threads is None or args.n_threads > 1) and n_remaining > 50
                
                if use_parallel_incremental:
                    # Parallel evaluation of incremental gains
                    def compute_incremental_gain(idx, cand_pos, original_gain, new_covered_array, covered_in_iter):
                        """Compute incremental gain for a single candidate."""
                        incremental_mask = ~covered_in_iter[new_covered_array]
                        incremental_covered_array = new_covered_array[incremental_mask]
                        incremental_gain = len(incremental_covered_array)
                        return (idx, cand_pos, incremental_gain, incremental_covered_array)
                    
                    with ThreadPoolExecutor(max_workers=n_threads) as executor:
                        futures = {executor.submit(compute_incremental_gain, idx, cand_pos, original_gain, 
                                                   new_covered_array, covered_in_iter): idx
                                  for idx, (cand_pos, original_gain, new_covered_array) in enumerate(remaining_candidates)}
                        
                        for future in as_completed(futures):
                            idx, cand_pos, incremental_gain, incremental_covered_array = future.result()
                            if incremental_gain > best_incremental_gain:
                                best_incremental_gain = incremental_gain
                                best_candidate = (cand_pos, incremental_gain, incremental_covered_array)
                                best_candidate_idx = idx
                else:
                    # Sequential evaluation
                    for idx, (cand_pos, original_gain, new_covered_array) in enumerate(remaining_candidates):
                        # Calculate incremental gain: only count points not covered by already selected centers
                        # Use boolean indexing: much faster than set difference for large sets
                        # Only count points that are in new_covered_array AND not yet covered_in_iter
                        incremental_mask = ~covered_in_iter[new_covered_array]
                        incremental_covered_array = new_covered_array[incremental_mask]
                        incremental_gain = len(incremental_covered_array)
                        
                        if incremental_gain > best_incremental_gain:
                            best_incremental_gain = incremental_gain
                            # Store incremental gain and incremental covered array (keep as array, no conversion)
                            best_candidate = (cand_pos, incremental_gain, incremental_covered_array)
                            best_candidate_idx = idx
                
                # Check if best incremental gain meets minimum threshold
                if best_incremental_gain <= 0:
                    # No more candidates with sufficient incremental gain
                    break
                
                # Add best candidate to selected list
                selected_candidates.append(best_candidate)
                
                # Update covered_in_iter using boolean array (faster than set union)
                # best_candidate[2] is already a numpy array (no conversion needed)
                covered_in_iter[best_candidate[2]] = True
                
                # Remove selected candidate from remaining list
                remaining_candidates.pop(best_candidate_idx)
            
            n_select = len(selected_candidates)
        
        # Calculate total gain for logging
        # For incremental gain mode, total gain is the union of all covered points
        # (in incremental mode, each candidate's new_covered is already incremental, so union = total)
        if adaptive_n_centers_per_iter > 1 and args.minimize_overlap:
            # In incremental mode, selected_candidates already contain incremental covered arrays
            # Total gain is the union of all incremental covered points (use numpy unique)
            if len(selected_candidates) > 0:
                total_covered_array = np.unique(np.concatenate([inc for _, _, inc in selected_candidates]))
                total_gain_this_iter = len(total_covered_array)
            else:
                total_gain_this_iter = 0
        else:
            # In non-incremental mode, sum individual gains (may have overlap)
            total_gain_this_iter = sum(g for _, g, _ in selected_candidates)
        best_gain_this_iter = selected_candidates[0][1] if selected_candidates else 0
        
        # Process each selected center
        # Use list of arrays instead of set, concatenate at end (faster)
        all_new_covered_arrays = []
        
        for cand_pos, gain, new_covered_array in selected_candidates:
            center_ref = int(cand[cand_pos])
            centers_ref_ids.append(center_ref)
            # Store the gain (incremental gain if minimize_overlap, original gain otherwise)
            gains.append(gain)
            
            # Note: Posting lists are no longer stored here - they will be computed in baselines.py
            # after target_coverage truncation. This saves memory and ensures posting lists are computed
            # for all documents (not just ref_size subset) and only for centers that will be used.
            
            # Collect newly covered arrays (will concatenate and update at end)
            all_new_covered_arrays.append(new_covered_array)
        
        # Mark all newly covered points (batch update for efficiency)
        # Concatenate all arrays at once (faster than iteratively updating set)
        if len(all_new_covered_arrays) > 0:
            # Concatenate all arrays and remove duplicates using numpy
            all_new_covered_array = np.unique(np.concatenate(all_new_covered_arrays))
            covered[all_new_covered_array] = True
            # Remove from uncovered set (O(1) per element, much faster than np.where)
            # Directly remove from set using array (no need to convert to list first)
            uncovered_set.difference_update(all_new_covered_array)
        
        # Record coverage after adding centers in this iteration
        # Calculate coverage from uncovered_set (more efficient than covered.mean())
        current_coverage = 1.0 - (len(uncovered_set) / N_ref)
        # Record coverage for each center added in this iteration (use extend for efficiency)
        # Note: All centers added in the same iteration have the same coverage (after the iteration)
        coverage_history.extend([current_coverage] * len(selected_candidates))

        # Track iteration time for remaining time estimation
        t_current = time.time()
        iter_time = t_current - t_last_iter
        iter_times.append(iter_time)
        # Keep only last 5 iterations for averaging (to account for adaptive optimization)
        if len(iter_times) > 5:
            iter_times.pop(0)
        t_last_iter = t_current
        
        if should_log:
            # Calculate coverage from uncovered_set (more efficient than covered.mean())
            covered_frac = 1.0 - (len(uncovered_set) / N_ref)
            elapsed = time.time() - t0
            
            # Track coverage for rate estimation
            recent_coverages.append(covered_frac)
            # Keep only last 3 logged iterations for rate calculation
            if len(recent_coverages) > 3:
                recent_coverages.pop(0)
            
            # Estimate remaining time (based on max_centers limit)
            remaining_time_str = ""
            if len(iter_times) >= 2:  # Need at least 2 iterations to estimate
                avg_iter_time = np.mean(iter_times)
                remaining_iters = args.max_centers - (it + 1)
                
                if remaining_iters > 0:
                    estimated_remaining_time = avg_iter_time * remaining_iters
                    remaining_time_str = f"  ETA={estimated_remaining_time/60:.1f}m"
            
            # Show adaptive adjustment info if different from default
            adaptive_info = ""
            if adaptive_n_centers_per_iter != args.n_centers_per_iter:
                adaptive_info = f" [adaptive: {adaptive_n_centers_per_iter} centers/iter]"
            
            if adaptive_n_centers_per_iter > 1:
                # Calculate overlap ratio for logging (only if minimize_overlap is enabled)
                if args.minimize_overlap and n_select > 1:
                    # Overlap = (sum of individual gains - total union gain) / sum of individual gains
                    sum_individual_gains = sum(g for _, g, _ in selected_candidates)
                    overlap_ratio = (sum_individual_gains - total_gain_this_iter) / sum_individual_gains if sum_individual_gains > 0 else 0.0
                    print(f"[greedy] iter={it+1:6d}  centers={len(centers_ref_ids):6d}  "
                          f"covered={covered_frac:.4f}  selected={n_select} centers  "
                          f"best_gain={best_gain_this_iter:6d}  total_gain={total_gain_this_iter:6d}  "
                          f"overlap={overlap_ratio:.1%}  elapsed={elapsed/60:.1f}m{remaining_time_str}{adaptive_info}")
                else:
                    print(f"[greedy] iter={it+1:6d}  centers={len(centers_ref_ids):6d}  "
                          f"covered={covered_frac:.4f}  selected={n_select} centers  "
                          f"best_gain={best_gain_this_iter:6d}  total_gain={total_gain_this_iter:6d}  elapsed={elapsed/60:.1f}m{remaining_time_str}{adaptive_info}")
            else:
                print(f"[greedy] iter={it+1:6d}  centers={len(centers_ref_ids):6d}  "
                      f"covered={covered_frac:.4f}  best_gain={best_gain_this_iter:6d}  elapsed={elapsed/60:.1f}m{remaining_time_str}{adaptive_info}")

    # Build final centers matrix in original space
    if len(centers_ref_ids) == 0:
        raise RuntimeError("No centers were selected. Try increasing --max_centers or decreasing --r.")
    
    centers_ref_ids = np.array(centers_ref_ids, dtype=np.int64)
    centers = X_ref[centers_ref_ids].astype(np.float32)  # already normalized
    
    # Validate centers
    if np.any(np.isnan(centers)) or np.any(np.isinf(centers)):
        raise RuntimeError("Centers contain NaN or Inf values")
    if centers.shape[0] != len(centers_ref_ids):
        raise RuntimeError(f"Center shape mismatch: {centers.shape[0]} != {len(centers_ref_ids)}")
    
    out_name = f"centers_greedy_r{args.r:.3f}.npy"
    out_path = os.path.join(args.out_dir, out_name)
    np.save(out_path, centers)
    
    # Note: Posting lists are no longer saved here.
    # They will be computed in baselines.py after target_coverage truncation,
    # ensuring we only compute posting lists for centers that will be used.
    # This also allows posting lists to be computed for all documents,
    # even if ref_size < N was used during center building.

    # Note: embeddings_path is not saved in stats because:
    # 1. 3build_centers_greedy.py loads section-separated files (abstract_{unit}.npy, claim_{unit}.npy, etc.)
    # 2. baselines.py expects a single concatenated file (patent_contextual_spans_{mode}_{model}_{unit}_{cls}.npy)
    # 3. baselines.py will find the embeddings file using pattern matching
    # If a concatenated file exists, it will be used; otherwise baselines.py will need to load and concatenate sections
    embeddings_path_for_stats = None  # Not saved - baselines.py will find embeddings via pattern matching
    
    stats = {
        "embeddings_dir": args.embeddings_dir,
        "task_mode": args.mode,
        "embedding_files": [os.path.basename(f) for f in embedding_files],
        "embeddings_path": embeddings_path_for_stats,  # Path to embeddings file (for baselines.py)
        "N": int(N),
        "d": int(d),
        "r": float(args.r),
        "r_auto_selected": args.r is None,  # Track if radius was auto-selected
        "auto_r_quantiles": [float(q) for q in args.auto_r_quantiles] if args.r is None else None,
        "auto_r_evaluation_mode": "full" if (EVALUATE_R_AVAILABLE and args.r is None) else "simple",
        "sim_threshold": float(sim_th),
        "ref_size": int(N_ref),
        "use_full_index": bool(ref_idx is None),
        "n_centers": int(centers.shape[0]),
        "achieved_coverage": float(1.0 - len(uncovered_set) / N_ref),  # Coverage on reference set (full dataset by default)
        "coverage_history": [float(c) for c in coverage_history],  # Coverage after each center is added (for post-processing)
        "avg_gain": float(np.mean(gains) if gains else 0.0),
        "median_gain": float(np.median(gains) if gains else 0.0),
        "p90_gain": float(np.percentile(gains, 90) if gains else 0.0),
        "max_gain": int(np.max(gains) if gains else 0),
        "seed": int(args.seed),
        "n_candidates_per_iter": int(args.n_candidates_per_iter),
        "max_centers": int(args.max_centers),
        # Note: posting_lists_path removed - posting lists are computed in baselines.py
        # Note: total_assignments and avg_posting_length removed - computed in baselines.py
    }
    
    # Add parsed directory info if available
    if embeddings_dir_info:
        stats["embeddings_dir_info"] = embeddings_dir_info
    stats_path = os.path.join(args.out_dir, out_name.replace(".npy", ".json"))
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[greedy] Saved centers: {out_path} shape={centers.shape} dtype={centers.dtype}")
    print(f"[greedy] Saved stats:   {stats_path}")

if __name__ == "__main__":
    main()
