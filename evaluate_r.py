import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import argparse

import umap

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

# Optional FAISS support for better performance
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Using sklearn (slower for large datasets).")


@dataclass
class NNIndex:
    """
    Wrapper for nearest neighbor index (sklearn or FAISS).
    
    For cosine similarity with FAISS, we use IndexFlatIP (inner product) on L2-normalized
    vectors, which is more intuitive than IndexFlatL2:
    - cosine_similarity = dot(x, y) for unit vectors
    - cosine_distance = 1 - cosine_similarity
    - range_search threshold: similarity >= (1 - radius) for cosine_distance <= radius
    """
    index: Any
    X_ref: np.ndarray
    metric: str
    use_faiss: bool
    faiss_index_type: str = "IP"  # "IP" (inner product) or "L2"
    
    def query(self, X_query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Query k nearest neighbors. Returns (distances, indices)."""
        if self.use_faiss:
            sims_or_dists, idxs = self.index.search(X_query.astype(np.float32), k)
            if self.metric == "cosine":
                if self.faiss_index_type == "IP":
                    # IndexFlatIP returns similarities (higher = closer)
                    # Convert to cosine distance: dist = 1 - similarity
                    dists = 1.0 - sims_or_dists
                else:
                    # IndexFlatL2 returns squared L2 distances
                    # For unit vectors: L2^2 = 2(1 - cos_sim), so cos_dist = L2^2 / 2
                    dists = sims_or_dists / 2.0
            else:
                dists = sims_or_dists
            return dists, idxs
        else:
            dists, idxs = self.index.kneighbors(X_query, n_neighbors=k, return_distance=True)
            return dists, idxs
    
    def radius_neighbors(self, X_query: np.ndarray, radius: float, max_neighbors: int = 5000) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Find all neighbors within radius. Returns list of (distances, indices) per query.
        
        For FAISS with IndexFlatIP (recommended for cosine):
        - cosine_distance <= radius  <=>  dot(x, y) >= 1 - radius
        - range_search uses similarity threshold directly
        
        For FAISS with IndexFlatL2:
        - cosine_distance <= radius  <=>  L2² <= 2 * radius
        - range_search uses squared L2 threshold
        """
        if self.use_faiss:
            X_q = X_query.astype(np.float32)
            
            if self.faiss_index_type == "IP":
                # IndexFlatIP: range_search finds points with similarity >= threshold
                # cosine_distance <= radius  <=>  similarity >= (1 - radius)
                sim_threshold = 1.0 - radius
                
                # FAISS IP range_search returns points with score >= threshold
                lims, D, I = self.index.range_search(X_q, sim_threshold)
                results = []
                for i in range(X_q.shape[0]):
                    start, end = int(lims[i]), int(lims[i + 1])
                    sims_i = D[start:end].copy()
                    idxs_i = I[start:end].copy()
                    # Convert similarity to cosine distance
                    dists_i = 1.0 - sims_i
                    results.append((dists_i, idxs_i))
                return results
            else:
                # IndexFlatL2: range_search uses squared L2 threshold
                # For cosine distance on unit vectors: L2² = 2 * cosine_dist
                l2_sq_threshold = 2.0 * radius if self.metric == "cosine" else radius
                
                lims, D, I = self.index.range_search(X_q, l2_sq_threshold)
                results = []
                for i in range(X_q.shape[0]):
                    start, end = int(lims[i]), int(lims[i + 1])
                    dists_i = D[start:end].copy()  # squared L2
                    idxs_i = I[start:end].copy()
                    # Convert squared L2 back to cosine distance
                    if self.metric == "cosine":
                        dists_i = dists_i / 2.0
                    results.append((dists_i, idxs_i))
                return results
        else:
            # sklearn radius_neighbors - use BATCH call (much faster than per-query loop)
            dists_list, idxs_list = self.index.radius_neighbors(
                X_query, radius=radius, return_distance=True
            )
            # sklearn returns list of arrays, convert to our format
            return [(dists_list[i], idxs_list[i]) for i in range(len(dists_list))]


def build_nn_index(
    X: np.ndarray,
    metric: str = "cosine",
    use_faiss: bool = None,
    faiss_index_type: str = "IP",  # "IP" (inner product, recommended) or "L2"
) -> NNIndex:
    """
    Build a nearest neighbor index. Uses FAISS if available and use_faiss is True/None.
    
    For cosine metric with FAISS:
    - "IP" (recommended): IndexFlatIP on L2-normalized vectors
      - Similarity threshold is intuitive: sim >= (1 - cosine_dist)
      - range_search is exact and efficient
    - "L2": IndexFlatL2, requires L2² <-> cosine_dist conversion
    
    Vectors should be L2-normalized for cosine metric.
    """
    if use_faiss is None:
        use_faiss = FAISS_AVAILABLE and X.shape[0] > 50000
    
    if use_faiss and FAISS_AVAILABLE:
        d = X.shape[1]
        X_f32 = X.astype(np.float32)
        
        if faiss_index_type == "IP":
            # IndexFlatIP for cosine similarity (on normalized vectors)
            # range_search threshold is similarity (1 - cosine_distance)
            index = faiss.IndexFlatIP(d)
        else:
            # IndexFlatL2 - need to convert L2² <-> cosine_distance
            index = faiss.IndexFlatL2(d)
        
        index.add(X_f32)
        return NNIndex(index=index, X_ref=X, metric=metric, use_faiss=True, 
                      faiss_index_type=faiss_index_type)
    else:
        nn = NearestNeighbors(metric=metric, algorithm="auto")
        nn.fit(X)
        return NNIndex(index=nn, X_ref=X, metric=metric, use_faiss=False,
                      faiss_index_type="sklearn")


def sample_embeddings(X: np.ndarray, n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Uniformly sample rows from X. Returns (sampled_X, indices)."""
    rng = np.random.default_rng(seed)
    if X.shape[0] <= n:
        return X, np.arange(X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx], idx


def compute_knn_distance_distribution(
    X: np.ndarray,
    k: int = 20,
    metric: str = "cosine",
    n_query: int = 20000,
    seed: int = 0,
    nn_index: NNIndex = None,
    X_ref: np.ndarray = None,
) -> np.ndarray:
    """
    Compute distribution D = {dist(x, x_(k))} using separate query and reference sets.
    
    This avoids the bias from using only a small sampled subset:
    - Query set Q (n_query): points for which we compute kNN distances
    - Reference set R: provided via nn_index (prebuilt) or X_ref
    
    Parameters:
    -----------
    X : np.ndarray
        Full embedding matrix to sample queries from
    nn_index : NNIndex, optional
        Prebuilt NN index on reference set. If provided, X_ref is ignored.
    X_ref : np.ndarray, optional
        Reference set to build index on (only used if nn_index is None)
    
    Returns:
        Distances to the k-th nearest neighbor for each query point.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    
    # Sample query set
    n_q = min(n_query, N)
    query_idx = rng.choice(N, size=n_q, replace=False)
    X_query = X[query_idx]
    
    # Build or reuse NN index
    if nn_index is None:
        if X_ref is None:
            raise ValueError("Either nn_index or X_ref must be provided")
        nn_index = build_nn_index(X_ref, metric=metric)
    
    # Query kNN (k+1 to handle potential self-matches if query overlaps with reference)
    dists, idxs = nn_index.query(X_query, k + 1)
    
    # For each query, find the k-th neighbor distance
    # If query point is in reference set, first neighbor might be itself (dist~0)
    kth_dists = []
    for i in range(n_q):
        d = dists[i]
        # Skip near-zero distances (self-matches)
        valid = d[d > 1e-8]
        if len(valid) >= k:
            kth_dists.append(valid[k - 1])
        elif len(valid) > 0:
            kth_dists.append(valid[-1])
        else:
            kth_dists.append(d[-1])
    
    return np.array(kth_dists)


def choose_radii_from_quantiles(D: np.ndarray, quantiles=(0.7, 0.8, 0.9)) -> dict:
    """Return dict {q: r_q} based on quantiles of D."""
    radii = {}
    for q in quantiles:
        radii[q] = float(np.quantile(D, q))
    return radii


# =============================================================================
# Greedy Covering Evaluation (Secondary Criterion for Radius Selection)
# =============================================================================

@dataclass
class SphereNeighborStats:
    """
    Statistics of sphere neighbor counts (posting length distribution proxy).
    
    For MULTI-ASSIGN mode:
        deg(c) = |{x : dist(x, c) <= r}| for sampled centers
        This is the exact posting length for multi-assignment indexing.
    
    For HARD-ASSIGN mode:
        deg(c) = |Voronoi cell of c| = number of points closest to c
        This is the posting length for hard-assignment (nearest center) indexing.
    
    IMPORTANT: When using a reference subset (X_ref) instead of full dataset (X_all),
    the raw degrees are scaled by (N_all / N_ref) to estimate full-dataset behavior.
    """
    radius: float
    quantile: float
    n_sampled_centers: int  # number of centers sampled
    assignment_mode: str  # "multi" or "hard"
    # Scale factor applied (N_all / N_ref), 1.0 if using full dataset
    scale_factor: float
    # Posting length distribution stats (SCALED to full dataset estimate)
    mean_degree: float  # avg |sphere| = avg posting length proxy
    median_degree: float
    std_degree: float
    max_degree: int
    p90_degree: float  # 90th percentile
    p99_degree: float  # 99th percentile
    # Raw (unscaled) stats for reference
    raw_mean_degree: float  # before scaling
    # Raw distribution for further analysis
    degree_histogram: Optional[np.ndarray] = None  # (optional) histogram counts


@dataclass
class CoveringStats:
    """
    Statistics from greedy covering at a given radius.
    
    NOTE: These stats use "new points covered" (disjoint increments) which is
    NOT the same as actual posting lengths. For posting length proxy, use
    SphereNeighborStats instead.
    
    This is kept for coverage estimation (what fraction of points are within
    radius r of at least one vocabulary center).
    """
    radius: float
    quantile: float
    n_centers: int  # vocabulary size V(r) estimate
    avg_points_per_center: float  # avg NEW points covered (not posting length!)
    median_points_per_center: float
    max_points_per_center: int
    # Coverage metrics
    covered_frac: float  # fraction of points covered by at least one sphere
    total_assignments: int  # total (point, center) pairs in disjoint cover
    avg_assignments_per_point: float  # should be ~1.0 for greedy


@dataclass
class AggregatedRadiusStats:
    """
    Aggregated statistics for a candidate radius, combining:
    1. Sphere neighbor stats (posting length proxy) - primary metrics
    2. Greedy covering stats (coverage estimation) - secondary metrics
    
    Reports statistics from a single evaluation run.
    
    IMPORTANT: Degree metrics are SCALED to estimate full-dataset behavior.
    If using subset-based estimation with scale_factor > 1, the reported
    mean_degree, p99_degree, max_degree are scaled estimates.
    """
    radius: float
    quantile: float
    assignment_mode: str  # "multi" or "hard"
    scale_factor: float  # N_full / N_ref used for degree scaling
    
    # === Sphere neighbor stats (posting length proxy) - PRIMARY ===
    # Note: These are SCALED estimates for full dataset
    mean_degree_mean: float  # avg posting length proxy (SCALED)
    p99_degree_mean: float  # 99th percentile posting length (SCALED)
    max_degree_mean: float  # max posting length (SCALED)
    
    # === Greedy covering stats (coverage estimation) - SECONDARY ===
    # Note: Coverage is estimated on subset; use as sanity check only
    n_centers_mean: float  # vocabulary size V(r) estimate
    covered_frac_mean: float  # coverage fraction (on subset)
    
    # === Constraint checks ===
    meets_coverage_threshold: bool  # covered_frac >= min_coverage
    is_recommended: bool  # meets coverage constraint


# Keep old name as alias for backward compatibility
AggregatedCoveringStats = AggregatedRadiusStats


# =============================================================================
# Sphere Neighbor Distribution (Posting Length Proxy) - 改法 A
# =============================================================================

def compute_sphere_neighbor_stats(
    X: np.ndarray,
    radius: float,
    quantile: float,
    nn_index: NNIndex,
    n_sample_centers: int = 5000,
    seed: int = 0,
    N_full: int = None,  # Full dataset size for scaling (if nn_index is on subset)
    assignment_mode: str = "multi",  # "multi" or "hard"
) -> SphereNeighborStats:
    """
    Compute posting length distribution proxy for radius selection.
    
    Supports two assignment modes:
    
    MULTI-ASSIGN (default):
        deg(c) = |{x : dist(x, c) <= r}|
        Each point can be assigned to multiple spheres it falls within.
        Use this if your final indexing allows multi-assignment or top-m assignment.
    
    HARD-ASSIGN:
        deg(c) = |Voronoi cell of c| = |{x : c = argmin_c' dist(x, c')}|
        Each point is assigned only to its nearest center.
        Use this if your final indexing uses hard (nearest center) assignment.
    
    SCALING for subset-based estimation:
        When nn_index is built on a subset X_ref (N_ref points) but you want to
        estimate behavior on full dataset X_all (N_full points), we scale:
            deg_estimated = deg_ref * (N_full / N_ref)
        This gives unbiased estimates assuming uniform sampling.
    
    Parameters:
    -----------
    X : np.ndarray
        Embedding matrix to sample centers from (can be full or subset)
    radius : float
        Sphere radius (cosine distance threshold)
    nn_index : NNIndex
        Prebuilt NN index (may be on subset X_ref)
    n_sample_centers : int
        Number of random points to sample as centers
    seed : int
        Random seed for reproducibility
    N_full : int, optional
        Full dataset size. If provided and different from nn_index size,
        degrees will be scaled by (N_full / N_ref).
    assignment_mode : str
        "multi": sphere membership proxy (for multi/top-m assignment)
        "hard": Voronoi cell size proxy (for hard assignment)
    
    Returns:
    --------
    SphereNeighborStats with (scaled) distribution statistics
    """
    rng = np.random.default_rng(seed)
    N_index = nn_index.X_ref.shape[0]  # Size of the indexed reference set
    N_centers_source = X.shape[0]
    
    # Compute scale factor for degree estimation
    if N_full is None:
        N_full = N_index
    scale_factor = N_full / N_index if N_index > 0 else 1.0
    
    # Sample centers from X (could be full dataset or subset)
    n_centers = min(n_sample_centers, N_centers_source)
    center_idxs = rng.choice(N_centers_source, size=n_centers, replace=False)
    X_centers = X[center_idxs]
    
    if assignment_mode == "hard":
        # HARD ASSIGNMENT: Estimate Voronoi cell sizes
        # Sample points and find their nearest center, then count
        degrees = _compute_hard_assignment_degrees(
            X_centers, nn_index, n_sample_points=min(50000, N_index), seed=seed
        )
    else:
        # MULTI ASSIGNMENT: Count neighbors within radius
        results = nn_index.radius_neighbors(X_centers, radius)
        # Compute degree (neighbor count) for each center
        # Exclude self (distance ~0) for accurate count
        degrees = np.array([
            np.sum(dists > 1e-8)  # exclude self-match
            for dists, idxs in results
        ])
    
    if len(degrees) == 0 or degrees.sum() == 0:
        return SphereNeighborStats(
            radius=radius, quantile=quantile, n_sampled_centers=0,
            assignment_mode=assignment_mode, scale_factor=scale_factor,
            mean_degree=0, median_degree=0, std_degree=0, max_degree=0,
            p90_degree=0, p99_degree=0,
            raw_mean_degree=0
        )
    
    # Store raw mean before scaling
    raw_mean = float(degrees.mean())
    
    # SCALE degrees to estimate full-dataset behavior
    # Only magnitude metrics need scaling; shape metrics are invariant
    scaled_degrees = degrees * scale_factor
    
    return SphereNeighborStats(
        radius=radius,
        quantile=quantile,
        n_sampled_centers=n_centers,
        assignment_mode=assignment_mode,
        scale_factor=scale_factor,
        # Scaled metrics (full-dataset estimate)
        mean_degree=float(scaled_degrees.mean()),
        median_degree=float(np.median(scaled_degrees)),
        std_degree=float(scaled_degrees.std()),
        max_degree=int(scaled_degrees.max()),
        p90_degree=float(np.percentile(scaled_degrees, 90)),
        p99_degree=float(np.percentile(scaled_degrees, 99)),
        raw_mean_degree=raw_mean,
    )


def _compute_hard_assignment_degrees(
    X_centers: np.ndarray,
    nn_index: NNIndex,
    n_sample_points: int = 50000,
    seed: int = 0,
) -> np.ndarray:
    """
    Compute Voronoi cell sizes for hard assignment proxy.
    
    For each sampled point from the reference set, find its nearest center,
    then count how many points each center attracts.
    
    This approximates the posting length distribution for hard-assignment indexing
    where each point is assigned only to its nearest vocabulary center.
    
    Parameters:
    -----------
    X_centers : np.ndarray
        The candidate centers (shape: n_centers x d)
    nn_index : NNIndex  
        Index on reference set (used to sample points)
    n_sample_points : int
        Number of points to sample for assignment counting
    seed : int
        Random seed
    
    Returns:
    --------
    np.ndarray: Degree (attraction count) for each center
    """
    rng = np.random.default_rng(seed)
    X_ref = nn_index.X_ref
    N_ref = X_ref.shape[0]
    
    # Sample points from reference set
    n_sample = min(n_sample_points, N_ref)
    sample_idxs = rng.choice(N_ref, size=n_sample, replace=False)
    X_sample = X_ref[sample_idxs]
    
    # Build a temporary index on centers to find nearest center for each point
    center_index = build_nn_index(X_centers, metric="cosine", use_faiss=FAISS_AVAILABLE,
                                   faiss_index_type="IP")
    
    # Find nearest center for each sampled point
    dists, nearest_center_ids = center_index.query(X_sample, k=1)
    nearest_center_ids = nearest_center_ids.flatten()
    
    # Count how many points each center attracts
    n_centers = X_centers.shape[0]
    degrees = np.bincount(nearest_center_ids, minlength=n_centers).astype(float)
    
    # Scale to match the reference set size (since we only sampled n_sample points)
    degrees = degrees * (N_ref / n_sample)
    
    return degrees


# NOTE: The exact greedy_covering() function has been REMOVED.
# It had O(N^3) complexity due to nested loops with linear search:
#   for neighbor_idx in neighbors_list[best_idx]:
#       for j in range(N):
#           if neighbor_idx in neighbors_list[j]:  # O(N) linear search
# This would freeze on N > 10k. Use fast_greedy_covering() instead.


def random_covering(
    X: np.ndarray,
    radius: float,
    nn_index: NNIndex,
    max_centers: int = 10000,
    target_coverage: float = 0.99,
    seed: int = 0,
) -> Tuple[np.ndarray, List[int], int]:
    """
    Fast random covering for coverage estimation (sanity check only).
    
    Instead of greedy (picking highest-density center each iteration),
    simply picks a random uncovered point as center. This is O(max_centers)
    range_search calls instead of O(max_centers * n_candidates).
    
    Trade-off:
    - Will OVERESTIMATE V(r) compared to greedy (more centers needed)
    - This is conservative: if random covering achieves 95% coverage,
      greedy covering will achieve at least that with fewer centers
    - Much faster: only 1 range_search per iteration
    
    Use this for coverage sanity check, NOT for accurate V(r) estimation.
    
    Complexity: O(max_centers * avg_neighbors) - much faster than greedy.
    
    Returns:
        center_indices: indices of selected centers  
        center_sizes: number of NEW points covered by each center
        n_covered: total number of unique points covered
    """
    N = X.shape[0]
    covered = np.zeros(N, dtype=bool)
    center_indices = []
    center_sizes = []
    
    rng = np.random.default_rng(seed)
    
    for _ in range(max_centers):
        # Check if we've reached target coverage
        current_coverage = covered.sum() / N
        if current_coverage >= target_coverage:
            break
        
        # Pick a random uncovered point as center
        uncovered_idx = np.where(~covered)[0]
        if len(uncovered_idx) == 0:
            break
        
        center_idx = rng.choice(uncovered_idx)
        
        # Find neighbors within radius (single point query)
        results = nn_index.radius_neighbors(X[center_idx:center_idx + 1], radius)
        dists, idxs = results[0]
        
        # Count NEW points covered (excluding already covered)
        new_covered_mask = ~covered[idxs]
        new_covered_idxs = idxs[new_covered_mask]
        n_new = len(new_covered_idxs)
        
        if n_new == 0:
            # This center covers nothing new, but we still count it
            # (edge case: isolated point)
            center_indices.append(center_idx)
            center_sizes.append(1)  # at least covers itself
            covered[center_idx] = True
        else:
            center_indices.append(center_idx)
            center_sizes.append(n_new)
            covered[new_covered_idxs] = True
    
    n_covered = int(covered.sum())
    return np.array(center_indices), center_sizes, n_covered


# Keep old name as alias for backward compatibility
fast_greedy_covering = random_covering


def evaluate_radius_with_covering(
    X: np.ndarray,
    radius: float,
    quantile: float,
    nn_index: NNIndex,
    n_sample: int = 50000,
    seed: int = 0,
) -> CoveringStats:
    """
    Evaluate a candidate radius by running random covering on a subset.
    
    PURPOSE: COVERAGE SANITY CHECK ONLY
    - Checks if radius r is large enough to cover most points
    - Uses random covering (not greedy) for speed
    - V(r) estimate is UPPER BOUND (random covering needs more centers than greedy)
    
    For posting length statistics, use compute_sphere_neighbor_stats() instead.
    
    Parameters:
    -----------
    nn_index : NNIndex
        Prebuilt NN index (not used directly - we build subset index)
    n_sample : int
        Subset size for coverage estimation
    
    Returns:
    --------
    CoveringStats with coverage metrics (V(r) is upper bound estimate)
    """
    # Sample subset for efficiency
    X_sub, sub_idx = sample_embeddings(X, n_sample, seed=seed)
    N = X_sub.shape[0]
    
    # Build index on the subset for greedy covering
    # Using FAISS IndexFlatIP - accurate range_search for cosine similarity
    sub_nn_index = build_nn_index(X_sub, metric="cosine", use_faiss=FAISS_AVAILABLE,
                                   faiss_index_type="IP")
    
    # Run fast greedy covering
    centers, sizes, n_covered = fast_greedy_covering(X_sub, radius, sub_nn_index, seed=seed)
    
    sizes = np.array(sizes)
    n_centers = len(centers)
    total_assignments = int(sizes.sum())  # For disjoint cover, equals n_covered
    
    # Compute statistics (note: these are on disjoint covering, NOT posting lengths)
    if n_centers == 0:
        return CoveringStats(
            radius=radius, quantile=quantile, n_centers=0,
            avg_points_per_center=0, median_points_per_center=0,
            max_points_per_center=0,
            covered_frac=0, total_assignments=0, avg_assignments_per_point=0
        )
    
    # Coverage metrics
    covered_frac = n_covered / N
    avg_assignments = total_assignments / (n_covered + 1e-10)
    
    return CoveringStats(
        radius=radius,
        quantile=quantile,
        n_centers=n_centers,
        avg_points_per_center=sizes.mean(),
        median_points_per_center=float(np.median(sizes)),
        max_points_per_center=int(sizes.max()),
        covered_frac=covered_frac,
        total_assignments=total_assignments,
        avg_assignments_per_point=avg_assignments
    )


def evaluate_multiple_radii(
    X: np.ndarray,
    radii: Dict[float, float],
    nn_index: NNIndex,
    n_sample: int = 50000,
    n_sample_centers: int = 5000,
    seed: int = 0,
    min_coverage: float = 0.95,
    assignment_mode: str = "multi",  # "multi" or "hard"
    skip_covering: bool = False,  # If True, skip covering evaluation for speed
    skip_sphere_stats: bool = False,  # If True, skip sphere neighbor stats (only compute coverage)
) -> Dict[float, AggregatedRadiusStats]:
    """
    Evaluate multiple candidate radii with multiple runs for stability.
    
    Uses TWO complementary evaluation methods:
    1. Sphere neighbor stats: Direct posting length proxy
       - For MULTI mode: deg(c) = |{x : dist(x,c) <= r}|
       - For HARD mode: deg(c) = |Voronoi cell of c|
       - Primary metric for avg posting length
    2. Greedy covering: Coverage estimation (sanity check)
       - Estimates vocabulary size V(r) and coverage fraction
       - Note: Coverage is on SUBSET, use as sanity check only
    
    SCALING: When nn_index is on subset (N_ref), degree metrics are scaled
    by (N_full / N_ref) to estimate full-dataset behavior.
    
    Parameters:
    -----------
    X : np.ndarray
        Full embedding matrix (N_full x d)
    nn_index : NNIndex
        Prebuilt NN index (may be on subset N_ref < N_full)
    n_sample : int
        Sample size for greedy covering (for V(r) estimation)
    n_sample_centers : int
        Number of centers to sample for sphere neighbor stats
        If 0 or skip_sphere_stats=True, sphere stats will be skipped (faster if only need coverage)
    min_coverage : float
        Minimum required coverage fraction (default 0.95)
        NOTE: This is subset coverage - use as sanity check, not exact threshold
    assignment_mode : str
        "multi": Use sphere membership for posting proxy (multi/top-m assignment)
        "hard": Use Voronoi cell size for posting proxy (hard assignment)
        NOTE: Only used if skip_sphere_stats=False. Coverage calculation does not depend on this.
    skip_sphere_stats : bool
        If True, skip sphere neighbor stats computation (only compute coverage).
        Useful when only need to find radius that meets coverage constraint.
        Coverage calculation does not depend on assignment_mode, so this can save computation.
    
    Returns:
        Dict mapping quantile -> AggregatedRadiusStats with mean±std and recommendations
    """
    results = {}
    N_full = X.shape[0]
    N_ref = nn_index.X_ref.shape[0]
    scale_factor = N_full / N_ref
    
    mode_str = f"{assignment_mode}-assign" if not skip_sphere_stats else "coverage-only"
    if scale_factor > 1.01 and not skip_sphere_stats:  # Only show if actually scaling
        print(f"  NOTE: Degrees scaled by {scale_factor:.1f}x (N_full={N_full}, N_ref={N_ref})")
    
    for q, r in radii.items():
        print(f"  Evaluating radius r={r:.4f} (q={q:.0%}, {mode_str})")

    
        # 1. Sphere neighbor stats (primary - posting length proxy)
        # Skip if skip_sphere_stats=True or n_sample_centers=0 (coverage-only mode)
        if skip_sphere_stats or n_sample_centers == 0:
            # Create dummy sphere stats (coverage doesn't depend on assignment_mode)
            sphere_stats = SphereNeighborStats(
                radius=r, quantile=q, n_sampled_centers=0,
                assignment_mode=assignment_mode, scale_factor=scale_factor,
                mean_degree=0, median_degree=0, std_degree=0, max_degree=0,
                p90_degree=0, p99_degree=0, raw_mean_degree=0
            )
        else:
            # Pass N_full for proper scaling
            sphere_stats = compute_sphere_neighbor_stats(
                X, r, q, nn_index, 
                n_sample_centers=n_sample_centers, 
                seed=seed,
                N_full=N_full,
                assignment_mode=assignment_mode,
            )
        
        # 2. Greedy covering stats (secondary - coverage estimation)
        # This is on subset only - use as sanity check
        # Skip if fast_mode or n_sample is 0
        if not skip_covering and n_sample > 0:
            covering_stats = evaluate_radius_with_covering(
                X, r, q, nn_index, n_sample=n_sample, seed=seed
            )
        else:
            # Create dummy covering stats if skipped
            covering_stats = CoveringStats(
                radius=r, quantile=q, n_centers=0,
                avg_points_per_center=0, median_points_per_center=0,
                max_points_per_center=0,
                covered_frac=0.0,
                total_assignments=0, avg_assignments_per_point=0
            )
        
        # Check constraints: only coverage is used
        # For coverage: if covering was skipped, assume it meets threshold (we can't verify)
        if skip_covering or n_sample == 0:
            meets_coverage = True  # Assume coverage OK if we skipped evaluation
        else:
            meets_coverage = covering_stats.covered_frac >= min_coverage
        is_recommended = meets_coverage
        
        agg_stats = AggregatedRadiusStats(
            radius=r,
            quantile=q,
            assignment_mode=assignment_mode,
            scale_factor=scale_factor,
            # Sphere neighbor stats (primary) - SCALED
            mean_degree_mean=sphere_stats.mean_degree,
            p99_degree_mean=sphere_stats.p99_degree,
            max_degree_mean=float(sphere_stats.max_degree),
            # Covering stats (secondary) - subset only
            n_centers_mean=float(covering_stats.n_centers),
            covered_frac_mean=covering_stats.covered_frac,  # Use covered_frac, not avg_points_per_center
            # Constraint checks
            meets_coverage_threshold=meets_coverage,
            is_recommended=is_recommended,
        )
        results[q] = agg_stats
        
        # Print summary
        status = "✓ RECOMMENDED" if is_recommended else ""
        if not meets_coverage:
            status = "✗ LOW COVERAGE"
        
        if skip_sphere_stats:
            # Coverage-only mode: don't show posting length stats
            print(f"    V(r)≈{agg_stats.n_centers_mean:.0f}, "
                  f"covered={agg_stats.covered_frac_mean:.1%} {status}")
        else:
            # Full mode: show both coverage and posting length stats
            print(f"    V(r)≈{agg_stats.n_centers_mean:.0f}, "
                  f"avg_deg={agg_stats.mean_degree_mean:.1f}, "
                  f"covered={agg_stats.covered_frac_mean:.1%} {status}")
    
    return results


def recommend_radius(
    agg_stats: Dict[float, AggregatedRadiusStats],
) -> Optional[float]:
    """
    Recommend the best radius based on coverage constraint.
    
    Strategy: Among radii that meet coverage constraint, choose the SMALLEST r
    (leads to finer vocabulary, more sparse representations).
    
    
    Returns:
        Recommended quantile key, or None if no radius meets coverage constraint
    """
    candidates = [(q, s) for q, s in agg_stats.items() if s.is_recommended]
    
    if not candidates:
        # Fallback: try to find one that at least meets coverage
        coverage_ok = [(q, s) for q, s in agg_stats.items() if s.meets_coverage_threshold]
        if coverage_ok:
            # Pick the smallest radius among coverage-ok
            best = min(coverage_ok, key=lambda x: x[1].radius)
            print(f"WARNING: No radius fully meets coverage threshold. "
                  f"Fallback to q={best[0]:.0%} (r={best[1].radius:.4f})")
            return best[0]
        else:
            print("WARNING: No radius meets coverage threshold. Consider using larger radii.")
            return None
    
    # Among recommended, pick smallest r (finest vocabulary)
    best = min(candidates, key=lambda x: x[1].radius)
    return best[0]


def project_to_2d(
    X: np.ndarray,
    seed: int = 0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """
    2D projection for visualization using UMAP.
    Note: UMAP distorts local distances; projections are for intuition only.
    """
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        metric="cosine",
    )
    Z = reducer.fit_transform(X)
    return Z


def get_neighbors_within_radius(
    center_idx: int,
    X: np.ndarray,
    r_high: float,
    nn_index: NNIndex,
    max_neighbors: int = 5000,
) -> np.ndarray:
    """
    Find indices of neighbors within high-dimensional radius r_high.
    Uses prebuilt nn_index for efficiency.
    """
    results = nn_index.radius_neighbors(X[center_idx:center_idx + 1], r_high, max_neighbors)
    dists, idxs = results[0]
    # Exclude self (distance ~0)
    mask = dists > 1e-8
    return idxs[mask]


def estimate_projected_radius_for_center(
    center_idx: int,
    X: np.ndarray,
    Z: np.ndarray,
    r_high: float,
    nn_index: NNIndex,
) -> float:
    """
    Estimate a 2D radius to draw for visualization.
    Returns median 2D distance to high-dimensional neighbors.
    
    Note: This is for illustration only; UMAP distorts local geometry.
    """
    neighbor_idxs = get_neighbors_within_radius(center_idx, X, r_high, nn_index)
    
    if len(neighbor_idxs) < 5:
        return 0.0
    
    dz = Z[neighbor_idxs] - Z[center_idx]
    d2 = np.sqrt((dz ** 2).sum(axis=1))
    return float(np.median(d2))


def pick_density_aware_centers(
    X: np.ndarray,
    r_high: float,
    nn_index: NNIndex,
    n_centers: int = 8,
    candidate_pool: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """
    Pick illustrative centers using density heuristic.
    Samples candidate_pool points and chooses those with most neighbors within r_high.
    Uses prebuilt nn_index for efficiency.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    pool = min(candidate_pool, N)
    cand_idx = rng.choice(N, size=pool, replace=False)

    # Batch query for efficiency
    results = nn_index.radius_neighbors(X[cand_idx], r_high)
    counts = np.array([len(idxs) for _, idxs in results])
    
    top = np.argsort(-counts)[:n_centers]
    return cand_idx[top]


def plot_panel_b_knn_distribution(
    D: np.ndarray, 
    radii: dict, 
    radius_stats: Dict[float, AggregatedRadiusStats] = None,
    out_path: str = None
):
    """
    Plot kNN distance distribution with quantile-based radii.
    If radius_stats provided, adds a secondary panel showing sphere neighbor metrics.
    """
    if radius_stats is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig, ax1 = plt.subplots(figsize=(6.5, 3.5))
    
    # Panel 1: kNN distribution
    ax1.hist(D, bins=60, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel("Distance to k-th nearest neighbor")
    ax1.set_ylabel("Count")

    # Mark quantile-based radii
    colors = plt.cm.tab10(np.linspace(0, 1, len(radii)))
    for (q, r), color in zip(radii.items(), colors):
        ax1.axvline(r, linestyle="--", linewidth=1.5, color=color)
        ax1.text(r, ax1.get_ylim()[1] * 0.95, f"q={int(q*100)}%\nr={r:.3f}",
                rotation=90, va="top", ha="right", fontsize=9)

    ax1.set_title("kNN distance distribution (for radius selection)")
    
    # Panel 2: Sphere neighbor statistics (posting length proxy)
    if radius_stats is not None:
        qs = sorted(radius_stats.keys())
        vocab_sizes = [radius_stats[q].n_centers_mean for q in qs]
        # Use sphere neighbor stats for posting length 
        avg_degrees = [radius_stats[q].mean_degree_mean for q in qs]
        recommended = [radius_stats[q].is_recommended for q in qs]
        
        x = np.arange(len(qs))
        width = 0.35
        
        # Bar plots
        bars1 = ax2.bar(x - width/2, vocab_sizes, width, 
                       label='V(r) (vocab size)', color='steelblue')
        bars2 = ax2.bar(x + width/2, avg_degrees, width,
                       label='Avg sphere deg', color='coral')
        
        ax2.set_xlabel('Quantile')
        ax2.set_ylabel('Count')
        ax2.set_xticks(x)
        
        # Mark recommended with star
        xlabels = []
        for i, q in enumerate(qs):
            label = f'{int(q*100)}%\nr={radii[q]:.3f}'
            if recommended[i]:
                label += '\n★'
            xlabels.append(label)
        ax2.set_xticklabels(xlabels)
        
        # Legend
        ax2.legend(loc='upper right', fontsize=8)
        
        ax2.set_title("Sphere neighbor stats (posting length proxy)")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    return fig


def plot_panel_a_projection_with_spheres(
    X: np.ndarray,
    Z: np.ndarray,
    radii: dict,
    nn_index: NNIndex,
    centers_per_r: int = 8,
    seed: int = 0,
    out_path: str = None,
    viz_mode: str = "neighbors",  # "circles", "neighbors", "both"
):
    """
    Make a 1xN panel showing 2D projection with illustrative coverage.
    
    Visualization style:
    - Background: grayscale scatter (lightgray, low alpha)
    - Sphere neighbors: high-saturation colors, one per center
    - Centers: black stars with white edge
    
    Parameters:
    -----------
    viz_mode: str
        How to show coverage:
        - "circles": Draw circles (illustrative only, UMAP distorts geometry)
        - "neighbors": Highlight actual high-d neighbors with different colors
        - "both": Show both circles and highlighted neighbors
    
    Note: Circles are illustrative only. UMAP projection distorts local distances,
    so circle radii do not correspond to actual high-dimensional geometry.
    """
    qs = list(radii.keys())
    assert len(qs) in (2, 3, 4), "Use 2-4 radii for a compact panel."

    fig, axes = plt.subplots(1, len(qs), figsize=(4.8 * len(qs), 4.2))
    if len(qs) == 1:
        axes = [axes]

    # Generate enough distinct colors for centers_per_r
    # Using tab20 colormap for up to 20 distinct colors, fallback to hsv for more
    if centers_per_r <= 20:
        cmap = plt.cm.tab20
        sphere_colors = [cmap(i / 20) for i in range(centers_per_r)]
    else:
        cmap = plt.cm.hsv
        sphere_colors = [cmap(i / centers_per_r) for i in range(centers_per_r)]

    for ax, q in zip(axes, qs):
        r_high = radii[q]

        # Background: grayscale scatter
        ax.scatter(Z[:, 0], Z[:, 1], s=2, alpha=0.15, c='lightgray')

        # Pick illustrative centers (density-aware)
        center_idxs = pick_density_aware_centers(
            X, r_high, nn_index, n_centers=centers_per_r, seed=seed + int(q * 1000)
        )

        for i, ci in enumerate(center_idxs):
            color = sphere_colors[i % len(sphere_colors)]
            
            # Get actual high-d neighbors
            neighbor_idxs = get_neighbors_within_radius(ci, X, r_high, nn_index)
            
            if viz_mode in ("neighbors", "both"):
                # Highlight neighbors with sphere color (alpha=0.5)
                if len(neighbor_idxs) > 0:
                    ax.scatter(Z[neighbor_idxs, 0], Z[neighbor_idxs, 1], 
                              s=10, alpha=0.5, c=color, edgecolors='none')
            
            if viz_mode in ("circles", "both"):
                # Draw illustrative circle
                r2d = estimate_projected_radius_for_center(ci, X, Z, r_high, nn_index)
                if r2d > 0:
                    circle = plt.Circle((Z[ci, 0], Z[ci, 1]), r2d, 
                                       fill=False, linewidth=1.5, 
                                       color=color, linestyle='--', alpha=0.8)
                    ax.add_patch(circle)
            
            # Center: colored star with black edge (same color as neighbors)
            ax.scatter(Z[ci, 0], Z[ci, 1], s=100, c=color, marker='*', 
                      edgecolors='black', linewidths=0.5, zorder=10)

        ax.set_title(f"q={int(q*100)}%, r={r_high:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])

    # Add caption noting circles are illustrative
    caption = "Note: UMAP distorts local geometry. "
    if viz_mode == "circles":
        caption += "Circles are illustrative only and do not represent true high-dimensional coverage."
    elif viz_mode == "neighbors":
        caption += "Highlighted points are actual high-dimensional neighbors within radius r."
    else:
        caption += "Dashed circles are illustrative; colored points are actual high-d neighbors."
    
    fig.suptitle("Effect of radius r on vocabulary granularity", y=1.02, fontsize=12)
    fig.text(0.5, -0.02, caption, ha='center', fontsize=9, style='italic', wrap=True)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def make_radius_selection_figures(
    X: np.ndarray,
    out_dir: str = "./fig_radius",
    normalize_vectors: bool = True,
    metric: str = "cosine",
    k: int = 20,
    quantiles=(0.7, 0.8, 0.9),
    seed: int = 0,
    n_query: int = 20000,
    n_reference: int = 200000,
    n_plot_sample: int = 20000,
    n_covering_sample: int = 50000,
    viz_mode: str = "neighbors",  # "circles", "neighbors", "both"
    run_covering_eval: bool = True,
    assignment_mode: str = "multi",  # "multi" or "hard"
    # Optimization parameters for faster evaluation
    n_sample_centers: int = 2000,  # Reduced from 5000 for speed
    fast_mode: bool = False,  # If True, skip covering evaluation (only sphere stats)
):
    """
    End-to-end script for radius selection:
    
    1) (optional) L2-normalize embeddings for cosine geometry
    2) Compute kNN distance distribution D using separate query/reference sets
       (fixes bias from using small self-contained subset)
    3) Pick candidate radii by quantiles
    4) (optional) Evaluate radii via sphere neighbor stats + greedy covering
    5) Project subset to 2D for visualization
    6) Save figures
    
    SCALING: When n_reference < N (full dataset), degree metrics are scaled
    by (N / n_reference) to estimate full-dataset posting lengths.
    
    Parameters:
    -----------
    n_query : int
        Number of query points for kNN distribution (default 20k)
    n_reference : int
        Number of reference points for NN index (default 200k)
        Degree estimates will be scaled by (N / n_reference) if n_reference < N
    n_covering_sample : int
        Number of points for greedy covering evaluation (default 50k)
        Coverage is on this subset only - use as sanity check
    viz_mode : str
        Visualization mode: "circles", "neighbors", or "both"
        - "neighbors" recommended: shows actual high-d neighbors
        - "circles" are illustrative only due to UMAP distortion
    run_covering_eval : bool
        If True, runs radius evaluation (sphere neighbor stats, optionally covering)
    fast_mode : bool
        If True, skip covering evaluation and only compute sphere neighbor stats (faster)
        If False and run_covering_eval=True, run full evaluation (sphere stats + covering)
    assignment_mode : str
        "multi": Posting proxy = sphere membership count (for multi/top-m assignment)
                 deg(c) = |{x : dist(x, c) <= r}|
        "hard":  Posting proxy = Voronoi cell size (for hard/nearest assignment)
                 deg(c) = |{x : c = argmin_c' dist(x, c')}|
        Choose based on your final indexing strategy.
    """
    os.makedirs(out_dir, exist_ok=True)

    print("Preprocessing embeddings...")
    X_use = X.astype(np.float32, copy=False)
    if normalize_vectors:
        X_use = normalize(X_use, norm="l2")

    # Build global NN index for efficiency (used across multiple functions)
    print(f"Building NN index on {min(n_reference, X_use.shape[0])} reference points...")
    X_ref, _ = sample_embeddings(X_use, n_reference, seed=seed)
    global_nn_index = build_nn_index(X_ref, metric=metric)

    # Compute kNN distribution using the prebuilt global_nn_index
    print(f"Computing kNN distribution (k={k}, query={n_query}, ref={X_ref.shape[0]})...")
    D = compute_knn_distance_distribution(
        X_use, k=k, metric=metric, n_query=n_query, seed=seed,
        nn_index=global_nn_index, X_ref=X_ref
    )
    radii = choose_radii_from_quantiles(D, quantiles=quantiles)
    print(f"Candidate radii from quantiles: {radii}")

    # Optional: evaluate radii via sphere neighbor stats + greedy covering
    radius_stats = None
    recommended_q = None
    if run_covering_eval:
        N_full = X_use.shape[0]
        N_ref = X_ref.shape[0]
        scale_factor = N_full / N_ref
        
        mode_str = "FAST" if fast_mode else "FULL"
        print(f"\nEvaluating radii ({mode_str} mode, {assignment_mode}-assign)...")
        print(f"  Full dataset: N={N_full}, Reference subset: N_ref={N_ref}")
        if scale_factor > 1.01:
            print(f"  Degree SCALING: x{scale_factor:.1f} (to estimate full-dataset posting lengths)")
        print(f"  Sphere neighbor sample: {n_sample_centers} centers")
        if fast_mode:
            print(f"  [FAST MODE] Skipping covering evaluation (sphere stats only)")
        else:
            print(f"  Covering sample: {n_covering_sample} points (for subset coverage sanity check)")
        print(f"  Constraint: coverage >= 95%")
        
        radius_stats = evaluate_multiple_radii(
            X_use, radii, global_nn_index,
            n_sample=n_covering_sample if not fast_mode else 0,  # Skip if fast_mode
            n_sample_centers=n_sample_centers,
            seed=seed,
            min_coverage=0.95,  # Relaxed - subset coverage is approximate
            assignment_mode=assignment_mode,
            skip_covering=fast_mode,  # Skip covering evaluation in fast mode
        )
        
        # Get recommendation
        recommended_q = recommend_radius(radius_stats)
        if recommended_q is not None:
            print(f"\n>>> RECOMMENDED: q={int(recommended_q*100)}% (r={radii[recommended_q]:.4f})")
        
        # Save stats to file
        stats_path = os.path.join(out_dir, "radius_stats.txt")
        with open(stats_path, 'w') as f:
            f.write("Radius Selection Statistics\n")
            f.write("=" * 70 + "\n")
            f.write(f"Assignment mode: {assignment_mode}\n")
            f.write(f"Full dataset: N={N_full}, Reference subset: N_ref={N_ref}\n")
            f.write(f"Degree scale factor: {scale_factor:.2f}x\n")
            f.write(f"Sphere neighbor sample: 5000 centers, Covering sample: {n_covering_sample}\n")
            f.write(f"Runs per radius: 3\n")
            f.write(f"Constraint: coverage >= 95%\n")
            f.write("\n")
            if assignment_mode == "multi":
                f.write("NOTE: MULTI-ASSIGN mode\n")
                f.write("      deg(c) = |{x : dist(x,c) <= r}| (sphere membership count)\n")
                f.write("      Use this if final indexing allows multi-assignment\n")
            else:
                f.write("NOTE: HARD-ASSIGN mode\n")
                f.write("      deg(c) = |Voronoi cell of c| (nearest center assignment)\n")
                f.write("      Use this if final indexing uses hard assignment\n")
            f.write(f"      Degree metrics are SCALED by {scale_factor:.2f}x to estimate full-dataset\n")
            f.write("      V(r) estimate uses greedy covering on subset (approximate)\n")
            f.write("=" * 70 + "\n\n")
            
            for q, stats in radius_stats.items():
                status = "✓ RECOMMENDED" if stats.is_recommended else ""
                if not stats.meets_coverage_threshold:
                    status = "✗ LOW COVERAGE"
                
                f.write(f"Quantile {int(q*100)}% (r = {stats.radius:.4f}): {status}\n")
                f.write(f"  --- Posting Length Proxy ({stats.assignment_mode}-assign, scaled x{stats.scale_factor:.1f}) ---\n")
                f.write(f"  Mean degree (scaled):     {stats.mean_degree_mean:.1f}\n")
                f.write(f"  P99 degree (scaled):      {stats.p99_degree_mean:.1f}\n")
                f.write(f"  Max degree (scaled):      {stats.max_degree_mean:.0f}\n")
                f.write(f"  --- Coverage Stats (subset sanity check) ---\n")
                f.write(f"  Vocabulary size V(r):     {stats.n_centers_mean:.0f}\n")
                f.write(f"  Covered fraction (subset):{stats.covered_frac_mean:.1%}\n")
                f.write(f"  --- Constraint Checks ---\n")
                f.write(f"  Meets coverage (>=95%):   {'Yes' if stats.meets_coverage_threshold else 'No'}\n")
                f.write("\n")
            
            if recommended_q is not None:
                f.write(f">>> RECOMMENDED: q={int(recommended_q*100)}% (r={radii[recommended_q]:.4f})\n")
        
        print(f"Radius stats saved to {stats_path}")

    # Panel (b): distribution + quantile radii + radius stats
    print("\nGenerating Panel B (kNN distribution)...")
    fig_b = plot_panel_b_knn_distribution(
        D, radii, radius_stats=radius_stats,
        out_path=os.path.join(out_dir, "panel_b_knn_quantiles.png")
    )

    # Panel (a): 2D projection + coverage visualization
    print(f"\nGenerating Panel A (2D projection, n={n_plot_sample})...")
    Xp, plot_idx = sample_embeddings(X_use, n_plot_sample, seed=seed + 1)
    
    # Build NN index for plot subset
    plot_nn_index = build_nn_index(Xp, metric=metric)
    
    print("Computing UMAP projection...")
    Z = project_to_2d(Xp, seed=seed)
    
    fig_a = plot_panel_a_projection_with_spheres(
        X=Xp,
        Z=Z,
        radii=radii,
        nn_index=plot_nn_index,
        centers_per_r=20,
        seed=seed,
        out_path=os.path.join(out_dir, "panel_a_projection.png"),
        viz_mode=viz_mode,
    )

    print(f"\nFigures saved to {out_dir}/")
    
    return {
        "D": D, 
        "radii": radii, 
        "radius_stats": radius_stats,
        "recommended_quantile": recommended_q,
        "assignment_mode": assignment_mode,
        "fig_a": fig_a, 
        "fig_b": fig_b
    }


# ----------------------------
# Example usage (replace with your embeddings)
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select optimal radius for sparse retrieval vocabulary"
    )
    parser.add_argument(
        "--embeddings_path", 
        type=str, 
        default="./patent_contextual_spans_abstract2abstract.npy",
        help="Path to embeddings file (.npy or .npz). "
             "For .npz files, specify key with --embeddings_key (default: 'embeddings')"
    )
    parser.add_argument(
        "--embeddings_key",
        type=str,
        default="embeddings",
        help="Key name for embeddings in .npz file (default: 'embeddings')"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./fig_radius",
        help="Output directory for figures and stats (default: ./fig_radius)"
    )
    parser.add_argument(
        "--normalize_vectors",
        action="store_true",
        default=True,
        help="L2-normalize embeddings for cosine metric (default: True)"
    )
    parser.add_argument(
        "--no_normalize",
        dest="normalize_vectors",
        action="store_false",
        help="Skip L2 normalization"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Distance metric (default: cosine)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="k for kNN distance distribution (default: 20)"
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.7, 0.8, 0.9],
        help="Quantiles for candidate radii (default: 0.7 0.8 0.9)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--n_query",
        type=int,
        default=20000,
        help="Number of query points for kNN distribution (default: 20000)"
    )
    parser.add_argument(
        "--n_reference",
        type=int,
        default=200000,
        help="Reference set size for NN index (default: 200000)"
    )
    parser.add_argument(
        "--n_plot_sample",
        type=int,
        default=20000,
        help="Points for UMAP visualization (default: 20000)"
    )
    parser.add_argument(
        "--n_covering_sample",
        type=int,
        default=50000,
        help="Points for greedy covering evaluation (default: 50000)"
    )
    parser.add_argument(
        "--viz_mode",
        type=str,
        default="neighbors",
        choices=["circles", "neighbors", "both"],
        help="Visualization mode (default: neighbors)"
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="fast",
        choices=["none", "fast", "full"],
        help="Evaluation mode: 'none' = skip evaluation, 'fast' = sphere stats only (default), 'full' = sphere stats + covering (slower)"
    )
    parser.add_argument(
        "--assignment_mode",
        type=str,
        default="multi",
        choices=["multi", "hard"],
        help="Assignment mode: 'multi' for sphere membership, 'hard' for Voronoi (default: multi)"
    )
    parser.add_argument(
        "--n_sample_centers",
        type=int,
        default=2000,
        help="Number of centers to sample for sphere neighbor stats (default: 2000, increase for more accuracy but slower)"
    )
    
    args = parser.parse_args()
    
    # Load embeddings (support both .npy and .npz)
    print(f"Loading embeddings from: {args.embeddings_path}")
    if args.embeddings_path.endswith('.npz'):
        data = np.load(args.embeddings_path)
        if args.embeddings_key not in data:
            available_keys = list(data.keys())
            raise ValueError(
                f"Key '{args.embeddings_key}' not found in .npz file. "
                f"Available keys: {available_keys}"
            )
        X = data[args.embeddings_key]
        print(f"  Loaded embeddings with key '{args.embeddings_key}': shape {X.shape}")
    else:
        X = np.load(args.embeddings_path)
        print(f"  Loaded embeddings: shape {X.shape}")

    # Convert eval_mode to boolean flags for backward compatibility
    run_covering_eval = (args.eval_mode != "none")
    fast_mode = (args.eval_mode == "fast")
    
    results = make_radius_selection_figures(
        X,
        out_dir=args.out_dir,
        normalize_vectors=args.normalize_vectors,
        metric=args.metric,
        k=args.k,
        quantiles=tuple(args.quantiles),
        seed=args.seed,
        n_query=args.n_query,
        n_reference=args.n_reference,
        n_plot_sample=args.n_plot_sample,
        n_covering_sample=args.n_covering_sample,
        viz_mode=args.viz_mode,
        run_covering_eval=run_covering_eval,
        assignment_mode=args.assignment_mode,
        n_sample_centers=args.n_sample_centers,
        fast_mode=fast_mode,
    )
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Chosen radii: {results['radii']}")
    
    if results['radius_stats']:
        first_stats = list(results['radius_stats'].values())[0]
        print(f"\nRadius statistics ({first_stats.assignment_mode}-assign, scaled x{first_stats.scale_factor:.1f}):")
        print(f"  (Degree metrics scaled to estimate full-dataset posting lengths)")
        for q, stats in results['radius_stats'].items():
            status = "✓" if stats.is_recommended else "✗"
            print(f"  {status} q={int(q*100)}% (r={stats.radius:.4f}): "
                  f"V≈{stats.n_centers_mean:.0f}, "
                  f"avg_deg={stats.mean_degree_mean:.1f}, "
                  f"covered={stats.covered_frac_mean:.1%}")
        
        if results['recommended_quantile'] is not None:
            rq = results['recommended_quantile']
            print(f"\n>>> RECOMMENDED: q={int(rq*100)}% (r={results['radii'][rq]:.4f})")
        else:
            print("\n>>> WARNING: No radius fully meets coverage constraint!")
