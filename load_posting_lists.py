#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_posting_lists.py

Utility functions to load posting lists saved by build_centers_greedy.py or build_centers_kmeans.py.

Usage:
    from load_posting_lists import load_posting_lists
    
    # Load posting lists from .npz file
    posting_lists = load_posting_lists("centers_greedy_r0.399_posting_lists.npz")
    
    # posting_lists[i] is a numpy array of point indices assigned to center i
    # Example: Get all points assigned to center 0
    points_in_center_0 = posting_lists[0]
"""

import numpy as np
from typing import List


def load_posting_lists(posting_lists_path: str) -> List[np.ndarray]:
    """
    Load posting lists from .npz file saved by build_centers_greedy.py or build_centers_kmeans.py.
    
    Parameters:
    -----------
    posting_lists_path : str
        Path to the .npz file containing posting lists
        
    Returns:
    --------
    List[np.ndarray]
        List of numpy arrays, where posting_lists[i] contains the point indices
        assigned to center i. Each array has dtype=np.int64.
    """
    data = np.load(posting_lists_path)
    
    # Extract center indices from keys (e.g., "center_0", "center_1", ...)
    center_keys = sorted([k for k in data.keys() if k.startswith("center_")], 
                        key=lambda x: int(x.split("_")[1]))
    
    posting_lists = []
    for key in center_keys:
        posting_lists.append(data[key])
    
    return posting_lists


def get_posting_list_stats(posting_lists: List[np.ndarray]) -> dict:
    """
    Get statistics about posting lists.
    
    Parameters:
    -----------
    posting_lists : List[np.ndarray]
        List of posting lists (from load_posting_lists)
        
    Returns:
    --------
    dict
        Dictionary with statistics:
        - n_centers: number of centers
        - total_assignments: total number of point-center assignments
        - avg_posting_length: average posting list length
        - min_posting_length: minimum posting list length
        - max_posting_length: maximum posting list length
        - median_posting_length: median posting list length
    """
    lengths = [len(pl) for pl in posting_lists]
    
    return {
        "n_centers": len(posting_lists),
        "total_assignments": sum(lengths),
        "avg_posting_length": float(np.mean(lengths)),
        "min_posting_length": int(min(lengths)) if lengths else 0,
        "max_posting_length": int(max(lengths)) if lengths else 0,
        "median_posting_length": float(np.median(lengths)) if lengths else 0.0,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and display posting list statistics")
    parser.add_argument("posting_lists_path", type=str, help="Path to posting lists .npz file")
    args = parser.parse_args()
    
    posting_lists = load_posting_lists(args.posting_lists_path)
    stats = get_posting_list_stats(posting_lists)
    
    print(f"Posting Lists Statistics")
    print(f"=" * 50)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value:,}")
