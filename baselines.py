#!/usr/bin/env python
"""
evaluate_baselines_inference.py

This script evaluates baseline models (without training) on our patent and scientific evaluation tasks.
It loads a pretrained model and computes tokenization and embeddings on-the-fly using the model's tokenizer.
If precomputed embeddings are present in the expected temp directories, the script will load them to
speed up repeated runs instead of recomputing embeddings.

When evaluating checkpoint model directories the loader will try to load a tokenizer from the
checkpoint and (if missing) reconstruct a tokenizer temporarily for evaluation purposes.

Usage example:
    python evaluate_baselines_inference.py --model_name <path_or_model_id> --output_dir ./results
"""

from __future__ import absolute_import, division, unicode_literals

import os
import re
import sys
import json
import argparse
import logging

from tqdm import trange, tqdm
import pandas as pd
import numpy as np

import faiss
import torch

from transformers import set_seed,  AutoTokenizer, AutoModel

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


def log_embeddings_shape(embeddings_dict, context=""):
    """Helper function to log embedding shapes consistently"""
    if context:
        print(f"{context}:")
    for name, embeddings in embeddings_dict.items():
        print(f"  {name}: {embeddings.shape}")


# ================== SPARSE COVERAGE UTILITIES ==================

def find_centers(
    dense_model: str,
    tokenization_unit: str,
    include_cls: bool,
    search_dir: str = ".",
    mode: str = "abstract2abstract",
    layer: str = "last",
    centers_suffix: str = "",
) -> tuple:
    """
    Find centers files based on embedding parameters.
    
    Parameters:
    -----------
    dense_model : str
        Dense encoder model name (e.g., "ZoeYou/PatentMap-V0-SecPair-Claim")
    tokenization_unit : str
        Tokenization unit (e.g., "spacy_token", "encoder_token")
    include_cls : bool
        Whether CLS token was included
    search_dir : str
        Directory to search for centers files
    mode : str
        Mode: "abstract2abstract" or "claim2all"
    layer : str
        Layer used for embeddings: "last" or "second_last"
    
    Returns:
    --------
    tuple: (centers_path, centers_dir)
        Paths to centers .npy file and directory containing it
        Note: Posting lists are computed in baselines.py after target_coverage truncation
    """
    import glob
    
    # Extract model name from path
    model_name = dense_model.strip("/").split("/")[-1].replace("/", "_").replace("\\", "_")
    cls_suffix = "cls" if include_cls else "nocls"
    
    # Build expected directory name
    # Format: centers_greedy_{mode}_{model_name}_{unit}_{cls_suffix}_{layer}[_suffix]
    expected_dir_pattern = f"centers_greedy_{mode}_{model_name}_{tokenization_unit}_{cls_suffix}_{layer}{centers_suffix}"
    
    # Search for directories matching the pattern
    search_pattern = os.path.join(search_dir, expected_dir_pattern)
    matching_dirs = glob.glob(search_pattern)
    
    if not matching_dirs:
        # Try recursive search
        search_pattern = os.path.join(search_dir, "**", expected_dir_pattern)
        matching_dirs = glob.glob(search_pattern, recursive=True)
    
    if not matching_dirs:
        raise FileNotFoundError(
            f"Could not find centers directory matching: {expected_dir_pattern}\n"
            f"Searched in: {os.path.abspath(search_dir)}\n"
            f"Please ensure centers were built with matching parameters:\n"
            f"  dense_model={dense_model}\n"
            f"  tokenization_unit={tokenization_unit}\n"
            f"  include_cls={include_cls}\n"
            f"  mode={mode}\n"
            f"  layer={layer}"
        )
    
    # Use the first matching directory (or most recent if multiple)
    centers_dir = matching_dirs[0]
    if len(matching_dirs) > 1:
        # Prefer the most recently modified directory
        matching_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        centers_dir = matching_dirs[0]
        print(f"⚠️  Found {len(matching_dirs)} matching directories, using: {centers_dir}")
    
    # Find centers file (centers_greedy_r*.npy)
    centers_pattern = os.path.join(centers_dir, "centers_greedy_r*.npy")
    centers_files = glob.glob(centers_pattern)
    
    if not centers_files:
        raise FileNotFoundError(
            f"Could not find centers file in: {centers_dir}\n"
            f"Expected pattern: centers_greedy_r*.npy"
        )
    
    # Use the first (or most recent) centers file
    centers_path = centers_files[0]
    if len(centers_files) > 1:
        centers_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        centers_path = centers_files[0]
        print(f"⚠️  Found {len(centers_files)} centers files, using: {centers_path}")
    
    return centers_path, centers_dir


# ================== RESULT FORMATTING FUNCTIONS ==================

def print_section_header(title, width=80):
    """Print a formatted section header"""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_subsection_header(title, width=60):
    """Print a formatted subsection header"""
    print(f"\n{'-' * width}")
    print(f" {title}")
    print(f"{'-' * width}")


def print_metric_table(results_dict, task_name, precision=4):
    """
    Print results in a clean table format
    
    Args:
        results_dict: Dictionary with metric names as keys and values as values
        task_name: Name of the evaluation task
        precision: Number of decimal places for floating point numbers
    """
    print(f"\n📊 {task_name} Results:")
    print("-" * 50)
    
    if not results_dict:
        print("   No results available")
        return
    
    # Sort metrics with intelligent handling of numbers (e.g., @10, @20, @50, @100)
    def metric_sort_key(metric_name):
        """
        Create a sort key that handles numeric suffixes correctly.
        Examples: 
        - precision@1 -> ('precision', 1)
        - recall@100 -> ('recall', 100)
        - ndcg@50 -> ('ndcg', 50)
        - alignment -> ('alignment', 0)
        """
        # Extract the base metric name and numeric value
        match = re.match(r'([^@]+)@?(\d+)?', metric_name)
        if match:
            base_name = match.group(1)
            number = int(match.group(2)) if match.group(2) else 0
            return (base_name, number)
        return (metric_name, 0)
    
    sorted_keys = sorted(results_dict.keys(), key=metric_sort_key)
    
    # Print each metric with consistent formatting
    for key in sorted_keys:
        value = results_dict[key]
        
        # Format value based on type
        if isinstance(value, float):
            if abs(value) < 0.001:
                formatted_value = f"{value:.6f}"
            else:
                formatted_value = f"{value:.{precision}f}"
        elif isinstance(value, dict):
            formatted_value = str(value)
        else:
            formatted_value = str(value)
        
        # All metrics displayed with same format - no highlighting
        print(f"   📋 {key:<25}: {formatted_value}")


def print_comparison_summary(results_list, task_name, main_metric):
    """Print a comparison summary for multiple models/conditions"""
    print(f"\n🏆 {task_name} - {main_metric} Comparison:")
    print("-" * 60)
    
    # Sort by main metric (descending for most metrics)
    sorted_results = sorted(results_list, key=lambda x: x.get(main_metric, 0), reverse=True)
    
    for i, result in enumerate(sorted_results[:5]):  # Show top 5
        rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i] if i < 5 else f"{i+1}️⃣"
        model_name = result.get('model_name', f'Model_{i+1}')
        score = result.get(main_metric, 0)
        
        if isinstance(score, float):
            print(f"   {rank_emoji} {model_name:<20}: {score:.4f}")
        else:
            print(f"   {rank_emoji} {model_name:<20}: {score}")


def log_evaluation_start(task_name, model_name=None):
    """Log the start of an evaluation task"""
    if model_name:
        print(f"\n🚀 Starting {task_name} evaluation for {model_name}...")
    else:
        print(f"\n🚀 Starting {task_name} evaluation...")


def log_evaluation_complete(task_name, time_taken=None):
    """Log the completion of an evaluation task"""
    if time_taken:
        print(f"✅ {task_name} evaluation completed in {time_taken:.2f}s")
    else:
        print(f"✅ {task_name} evaluation completed")


# ================== END FORMATTING FUNCTIONS ==================


def load_checkpoint_model(checkpoint_path, max_length=512, hf_model_name=None):
    """
    Load a sentence transformer checkpoint without dense layers, or load a HuggingFace model.
    If hf_model_name is provided, loads from HuggingFace; otherwise loads from checkpoint.
    """
    from sentence_transformers import SentenceTransformer, models
    if hf_model_name:
        print(f"Loading HuggingFace model: {hf_model_name}")
        model = SentenceTransformer(hf_model_name)
        print("Model loaded from HuggingFace Hub!")
        return model
    print(f"Loading checkpoint from: {checkpoint_path}")
    transformer = models.Transformer(checkpoint_path, max_seq_length=max_length)
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
        pooling = models.Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
    model = SentenceTransformer(modules=[transformer, pooling])
    print("Model loaded successfully (transformer + pooling only)")
    return model


def _is_sentence_transformer_checkpoint(model_name):
    """True if model_name points to a directory that has 1_Pooling (SentenceTransformer layout)."""
    path = _resolve_st_checkpoint_path(model_name)
    return path is not None and os.path.isdir(path) and os.path.exists(os.path.join(path, "1_Pooling"))


def _resolve_st_checkpoint_path(model_name):
    """Resolve model_name to an absolute path; returns None if path does not exist."""
    base = os.path.dirname(os.path.abspath(__file__))
    if os.path.isabs(model_name):
        path = model_name
    else:
        path = os.path.normpath(os.path.join(base, model_name.strip("/")))
    if os.path.isdir(path):
        return path
    if os.path.isdir(model_name):
        return model_name
    return path


def mean_pooling(token_embeddings, attention_mask):
    """
    Performs mean pooling on token embeddings.
    Args:
        token_embeddings: Tensor of shape (batch_size, seq_length, hidden_dim)
        attention_mask: Tensor of shape (batch_size, seq_length)
    Returns:
        Pooled tensor of shape (batch_size, hidden_dim)
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).to(token_embeddings.device)  # Ensure same device
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cls_pooling(model_output, attention_mask):
    return model_output.last_hidden_state[:, 0]  # Explicitly using last_hidden_state


def compute_rankings(top_indices):
    rankings = np.empty_like(top_indices)
    for query_idx, doc_order in enumerate(top_indices):
        rankings[query_idx, doc_order] = np.arange(1, len(doc_order) + 1)
    return rankings


def prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types):
    assert len(query_ids) == len(query_embeddings), f"query_ids and query_embeddings length mismatch: {len(query_ids)} vs {len(query_embeddings)}"

    results = {}
    texttype_q = "abstract"
    texttype_d = "abstract"

    # Convert to numpy array to ensure compatibility
    query_types = np.array(query_types)
    doc_types = np.array(doc_types)

    query_type_masks = (query_types == texttype_q)
    doc_type_masks = (doc_types == texttype_d)

    Q_emb = query_embeddings[query_type_masks].astype(np.float32)  # shape: [n_queries, emb_dim]
    D_emb = document_embeddings[doc_type_masks].astype(np.float32)    # shape: [n_docs, emb_dim]

    # Validate shape consistency
    if Q_emb.shape[1] != D_emb.shape[1]:
        logging.warning(f"Embedding dimension mismatch: Q_emb {Q_emb.shape} vs D_emb {D_emb.shape}")

    if np.any(np.isnan(Q_emb)) or np.any(np.isnan(D_emb)):
        raise ValueError("NaN detected in embeddings before normalization.")

    # Create copies to avoid modifying original data
    Q_emb_norm = Q_emb.copy()
    D_emb_norm = D_emb.copy()
    
    faiss.normalize_L2(Q_emb_norm)  # Normalize before similarity computation
    faiss.normalize_L2(D_emb_norm)
    distances = Q_emb_norm @ D_emb_norm.T  # FAISS optimized cosine similarity

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

    # Compute recall@k and ndcg@k for k=10,20,50,100
    results_key = f"{texttype_q}->{texttype_d}"
    results[results_key] = {
        'recall@10':  mean_recall_at_k(true_labels_list, predicted_labels_list, k=10),
        'recall@20':  mean_recall_at_k(true_labels_list, predicted_labels_list, k=20),
        'recall@50':  mean_recall_at_k(true_labels_list, predicted_labels_list, k=50),
        'recall@100': mean_recall_at_k(true_labels_list, predicted_labels_list, k=100),

        'ndcg@10':  mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=10),
        'ndcg@20':  mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=20),
        'ndcg@50':  mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=50),
        'ndcg@100': mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=100),

        'mrr@10': mean_mrr_at_k(true_labels_list, predicted_labels_list, k=10),
        'map': mean_average_precision(true_labels_list, predicted_labels_list, k=100),
        'pres@100': mean_pres_at_k(true_labels_list, predicted_labels_list, k=100, N_max=100),
    }

    # 3) compute performance for query -> all sections
    retrieved_sections = []   # for noting which section is retrieved at top_k
    
    # Calculer le nombre original de documents (avant multiplication par 3)
    original_doc_count = len(doc_ids) // 3
    
    for texttype_q in ["claim"]:
        query_type_masks = (query_types == texttype_q)
        Q_emb = query_embeddings[query_type_masks].astype(np.float32)

        D_emb = document_embeddings.astype(np.float32)
        D_ids = doc_ids
        assert len(D_ids) == D_emb.shape[0], f"Document IDs length {len(D_ids)} does not match document embeddings shape {D_emb.shape[0]}"

        if np.any(np.isnan(Q_emb)) or np.any(np.isnan(D_emb)):
            raise ValueError("NaN detected in embeddings before normalization.")
        
        faiss.normalize_L2(Q_emb)
        faiss.normalize_L2(D_emb)
        distances = Q_emb @ D_emb.T

        top_k_indices = np.argsort(-distances, axis=1)[:, :300]  # top_k * 3 to ensure we have enough candidates

        true_labels_list, predicted_labels_list = [], []
        for q_idx, retrieved_docs_indices in enumerate(top_k_indices):
            q_id_str = query_ids[q_idx]
            true_labels = citation_mapping.get(q_id_str, [])
            predicted_labels = [D_ids[d_idx] for d_idx in retrieved_docs_indices]

            # Filter out duplicates in predicted_labels without changing order
            _, unique_indices = np.unique(predicted_labels, return_index=True)
            predicted_labels = [predicted_labels[i] for i in sorted(unique_indices)][:100]
            
            # Use original_doc_count to calculate which section was retrieved
            # Embeddings are organized as: [abstracts, claims, inventions]
            # with each section having original_doc_count elements
            retrieved_sections.append([
                ["abstract", "claim", "invention"][retrieved_docs_indices[i] // original_doc_count] 
                for i in unique_indices[:100]
            ])

            true_labels_list.append(true_labels)
            predicted_labels_list.append(predicted_labels)

        results_key = f"{texttype_q}->all"
        results[results_key] = {
            'recall@10':  mean_recall_at_k(true_labels_list, predicted_labels_list, k=10),
            'recall@20':  mean_recall_at_k(true_labels_list, predicted_labels_list, k=20),
            'recall@50':  mean_recall_at_k(true_labels_list, predicted_labels_list, k=50),
            'recall@100': mean_recall_at_k(true_labels_list, predicted_labels_list, k=100),

            'ndcg@10':  mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=10),
            'ndcg@20':  mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=20),
            'ndcg@50':  mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=50),
            'ndcg@100': mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=100),

            'mrr@10': mean_mrr_at_k(true_labels_list, predicted_labels_list, k=10),
            'map': mean_average_precision(true_labels_list, predicted_labels_list, k=100),
            'pres@100': mean_pres_at_k(true_labels_list, predicted_labels_list, k=100, N_max=100),
            'retrieved_sections': f"[{len(retrieved_sections)} queries with retrieved sections]"  # summary instead of full list
        }

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


def _combine_dataframes_for_filtering(queries_df, documents_df):
    """Helper function to combine queries and documents DataFrames for quality filtering."""
    if queries_df is not None and documents_df is not None:
        return pd.concat([queries_df, documents_df], ignore_index=False)
    elif queries_df is not None:
        return queries_df
    elif documents_df is not None:
        return documents_df
    return None


def _compute_per_text_type_metrics(embeddings, text_types, compute_func, metric_name, **compute_kwargs):
    """Helper function to compute metrics per text type."""
    results = {}
    for text_type in set(text_types):
        text_type_mask = np.array(text_types) == text_type
        text_type_embeddings = embeddings[text_type_mask]
        
        if len(text_type_embeddings) > 0:
            metric_value = compute_func(text_type_embeddings, **compute_kwargs)
            results[text_type] = {metric_name: metric_value}
    
    return results


def _evaluate_embeddings_with_prefiltered_data(filtered_data, compute_func, metric_name, min_samples=10000, **compute_kwargs):
    """Helper function to evaluate embeddings using pre-filtered data."""
    results = {}
    
    if len(filtered_data['embeddings']) > min_samples:
        filtered_metric = compute_func(filtered_data['embeddings'], **compute_kwargs)
        results['global'] = {
            metric_name: filtered_metric,
            'samples_used': len(filtered_data['embeddings']),
            'filter_rate': filtered_data['filter_stats']['keep_rate']
        }
        
        # Calculate per-text-type metrics on filtered data
        filtered_per_type = _compute_per_text_type_metrics(
            filtered_data['embeddings'], filtered_data['types'], 
            compute_func, metric_name, **compute_kwargs
        )
        results.update(filtered_per_type)
    else:
        print(f"⚠️  Warning: Too few high-quality samples ({len(filtered_data['embeddings'])}) for {metric_name} evaluation")
        
    return results


def _evaluate_embeddings_with_quality_filter(embeddings, text_types, queries_df, documents_df, 
                                            compute_func, metric_name, min_samples=10000, **compute_kwargs):
    """Generic function for evaluating embeddings with optional quality filtering."""
    results = {}
    
    # Quality-filtered evaluation (main evaluation)
    combined_df = _combine_dataframes_for_filtering(queries_df, documents_df)
    
    if combined_df is not None:
        # Create dummy IDs for filtering
        dummy_ids = [f"item_{i}" for i in range(len(embeddings))]
        
        filtered_data = filter_by_text_quality(
            embeddings, dummy_ids, text_types, combined_df
        )
        
        if len(filtered_data['embeddings']) > min_samples:
            filtered_metric = compute_func(filtered_data['embeddings'], **compute_kwargs)
            results['global'] = {
                metric_name: filtered_metric,
                'samples_used': len(filtered_data['embeddings']),
                'filter_rate': filtered_data['filter_stats']['keep_rate']
            }
            
            # Calculate per-text-type metrics on filtered data
            filtered_per_type = _compute_per_text_type_metrics(
                filtered_data['embeddings'], filtered_data['types'], 
                compute_func, metric_name, **compute_kwargs
            )
            results.update(filtered_per_type)
        else:
            print(f"⚠️  Warning: Too few high-quality samples ({len(filtered_data['embeddings'])}) for {metric_name} evaluation")
    
    # Fallback to original evaluation if quality filtering unavailable or insufficient
    if not results and len(embeddings) > 0:
        print(f"📊 Falling back to unfiltered {metric_name} evaluation")
        global_metric = compute_func(embeddings, **compute_kwargs)
        results['global'] = {metric_name: global_metric}
        
        # Calculate per-text-type metrics
        per_type_results = _compute_per_text_type_metrics(
            embeddings, text_types, compute_func, metric_name, **compute_kwargs
        )
        results.update(per_type_results)
    
    return results



def uniformity_evaluation(embeddings, text_types, queries_df=None, documents_df=None, use_quality_filter=True, filtered_data=None):
    """
    Evaluate uniformity of embeddings across different text types.
    """
    if filtered_data is not None:
        # Use pre-filtered data
        return _evaluate_embeddings_with_prefiltered_data(
            filtered_data, compute_func=compute_uniformity,
            metric_name='uniformity', min_samples=10000,
            t=2.0, num_samples=10000, device='cuda'
        )
    
    if not use_quality_filter:
        # Skip quality filtering, use all data
        queries_df = documents_df = None
    
    return _evaluate_embeddings_with_quality_filter(
        embeddings, text_types, queries_df, documents_df,
        compute_func=compute_uniformity,
        metric_name='uniformity',
        min_samples=10000,
        t=2.0, num_samples=10000, device='cuda'
    )


def singular_spectrum_evaluation(embeddings, text_types, queries_df=None, documents_df=None, use_quality_filter=True, filtered_data=None):
    """
    Evaluate singular spectrum divergence of embeddings across different text types.
    """
    if filtered_data is not None:
        # Use pre-filtered data
        return _evaluate_embeddings_with_prefiltered_data(
            filtered_data, compute_func=compute_ssd,
            metric_name='ssd', min_samples=10000,
            normalize_by_d=True
        )
    
    if not use_quality_filter:
        # Skip quality filtering, use all data
        queries_df = documents_df = None
    
    return _evaluate_embeddings_with_quality_filter(
        embeddings, text_types, queries_df, documents_df,
        compute_func=compute_ssd,
        metric_name='ssd',
        min_samples=10000,
        normalize_by_d=True
    )


def filter_by_text_quality(embeddings, ids, types, texts_df, sections=['abstract', 'claim', 'invention'], verbose=True):
    """
    Filter embeddings based on text quality criteria.
    
    Args:
        embeddings: np.array of embeddings
        ids: list of IDs (can be dummy IDs like "item_0" or real IDs)
        types: list of section types corresponding to each embedding
        texts_df: DataFrame containing the actual texts
        sections: sections to check for quality
        verbose: whether to print filtering results
    
    Returns:
        dict with filtered data and statistics
    """
    if texts_df is None:
        return {
            'embeddings': embeddings,
            'ids': ids, 
            'types': types,
            'filter_stats': {'total': len(embeddings), 'filtered': 0, 'kept': len(embeddings)}
        }
    
    embeddings = np.array(embeddings)
    ids = np.array(ids)
    types = np.array(types)
    
    # Create quality masks for each section
    quality_masks = {}
    text_stats = {}
    
    for section in sections:
        if section in texts_df.columns:
            # Get text lengths for this section
            texts = texts_df[section].fillna('').astype(str)
            word_counts = [len(str(text).split()) for text in texts]
            
            # Create quality mask (sufficient words)
            section_quality = np.array(word_counts) >= QUALITY_MIN_WORDS
            quality_masks[section] = section_quality
            
            text_stats[section] = {
                'total': len(texts),
                'high_quality': np.sum(section_quality),
                'low_quality': np.sum(~section_quality),
                'quality_rate': np.mean(section_quality)
            }
    
    # Filter embeddings by section type
    keep_mask = np.ones(len(embeddings), dtype=bool)
    
    # Group embeddings by section type and apply quality filtering
    for section in sections:
        if section not in quality_masks:
            continue
            
        # Find all embeddings of this section type
        section_mask = (types == section)
        section_indices = np.where(section_mask)[0]
        
        # Apply quality filtering to this section
        section_quality = quality_masks[section]
        
        # For each embedding of this section type, check if its corresponding document has high quality
        for i, emb_idx in enumerate(section_indices):
            if i < len(section_quality):
                keep_mask[emb_idx] = section_quality[i]
            else:
                # If we have more embeddings than documents, cycle through quality mask
                keep_mask[emb_idx] = section_quality[i % len(section_quality)]
    
    # Apply filtering
    filtered_embeddings = embeddings[keep_mask]
    filtered_ids = ids[keep_mask]
    filtered_types = types[keep_mask]
    
    filter_stats = {
        'total': len(embeddings),
        'filtered': np.sum(~keep_mask),
        'kept': len(filtered_embeddings),
        'keep_rate': len(filtered_embeddings) / len(embeddings) if len(embeddings) > 0 else 0,
        'section_stats': text_stats
    }
    
    if verbose:
        print(f"\n🔍 Text Quality Filtering Results:")
        print(f"   • Total embeddings: {filter_stats['total']}")
        print(f"   • Kept (high quality): {filter_stats['kept']} ({filter_stats['keep_rate']:.1%})")
        print(f"   • Filtered (low quality): {filter_stats['filtered']} ({1-filter_stats['keep_rate']:.1%})")
        
        for section, stats in text_stats.items():
            print(f"   • {section.capitalize()} quality: {stats['high_quality']}/{stats['total']} ({stats['quality_rate']:.1%})")
    
    return {
        'embeddings': filtered_embeddings,
        'ids': filtered_ids,
        'types': filtered_types,
        'filter_stats': filter_stats,
        'keep_mask': keep_mask
    }


def _build_citation_pairs(q_embs, d_embs, q_ids_arr, d_ids_arr, citation_mapping):
    """Helper function to build citation pairs from embeddings and IDs."""
    query_pairs = []
    doc_pairs = []
    
    for q_idx, q_id in enumerate(q_ids_arr):
        cited_doc_ids = set(citation_mapping.get(q_id, []))
        for d_idx, did in enumerate(d_ids_arr):
            if did in cited_doc_ids:
                query_pairs.append(q_embs[q_idx])
                doc_pairs.append(d_embs[d_idx])
    
    return query_pairs, doc_pairs


def filter_citation_pairs_by_quality(query_embeddings, doc_embeddings, query_ids, doc_ids, 
                                     query_types, doc_types, 
                                     queries_df, documents_df):
    """
    Filter citation pairs to only include high-quality text pairs.
    
    Returns both query and doc pairs where both elements meet quality criteria.
    """
    if queries_df is None or documents_df is None:
        return query_embeddings, doc_embeddings, query_ids, doc_ids
    
    # Ensure all inputs have consistent lengths
    assert len(query_embeddings) == len(query_ids) == len(query_types), \
        f"Query dimension mismatch: embeddings={len(query_embeddings)}, ids={len(query_ids)}, types={len(query_types)}"
    assert len(doc_embeddings) == len(doc_ids) == len(doc_types), \
        f"Doc dimension mismatch: embeddings={len(doc_embeddings)}, ids={len(doc_ids)}, types={len(doc_types)}"
    
    # Create quality indicators for queries and documents  
    query_quality = {}
    doc_quality = {}
    
    # Check query quality
    for section in ['abstract', 'claim', 'invention']:
        if section in queries_df.columns:
            texts = queries_df[section].fillna('').astype(str)
            word_counts = [len(str(text).split()) for text in texts]
            query_quality[section] = {qid: wc >= QUALITY_MIN_WORDS
                                    for qid, wc in zip(queries_df.index, word_counts)}
    
    # Check document quality 
    for section in ['abstract', 'claim', 'invention']:
        if section in documents_df.columns:
            texts = documents_df[section].fillna('').astype(str)
            word_counts = [len(str(text).split()) for text in texts]
            doc_quality[section] = {did: wc >= QUALITY_MIN_WORDS
                                  for did, wc in zip(documents_df.index, word_counts)}
    
    # Filter embeddings based on quality
    q_keep_mask = np.ones(len(query_embeddings), dtype=bool)
    d_keep_mask = np.ones(len(doc_embeddings), dtype=bool)
    
    # Filter queries
    query_types_arr = np.array(query_types)
    for i, (qid, qtype) in enumerate(zip(query_ids, query_types_arr)):
        if qtype in query_quality:
            # Handle both original query IDs and duplicated IDs
            original_qid = qid
            if isinstance(qid, str) and qid.startswith('Q'):
                # Remove any numeric suffixes that might be from ID multiplication
                base_qid = qid
            else:
                base_qid = qid
            
            if base_qid in query_quality[qtype]:
                q_keep_mask[i] = query_quality[qtype][base_qid]
            else:
                # If exact ID not found, default to keeping it
                q_keep_mask[i] = True
    
    # Filter documents
    doc_types_arr = np.array(doc_types)
    for i, (did, dtype) in enumerate(zip(doc_ids, doc_types_arr)):
        if dtype in doc_quality:
            # Handle both original document IDs and duplicated IDs
            original_did = did
            if isinstance(did, str) and did.startswith('D'):
                # Remove any numeric suffixes that might be from ID multiplication
                base_did = did
            else:
                base_did = did
            
            if base_did in doc_quality[dtype]:
                d_keep_mask[i] = doc_quality[dtype][base_did]
            else:
                # If exact ID not found, default to keeping it
                d_keep_mask[i] = True
    
    print(f"\n🔍 Citation Pair Quality Filtering:")
    print(f"   • Queries: {np.sum(q_keep_mask)}/{len(query_embeddings)} kept ({np.mean(q_keep_mask):.1%})")
    print(f"   • Documents: {np.sum(d_keep_mask)}/{len(doc_embeddings)} kept ({np.mean(d_keep_mask):.1%})")
    
    # Apply filtering and ensure output arrays are properly sized
    filtered_q_embeddings = query_embeddings[q_keep_mask]
    filtered_d_embeddings = doc_embeddings[d_keep_mask]
    filtered_q_ids = np.array(query_ids)[q_keep_mask]
    filtered_d_ids = np.array(doc_ids)[d_keep_mask]
    filtered_q_types = np.array(query_types)[q_keep_mask]
    filtered_d_types = np.array(doc_types)[d_keep_mask]
    
    print(f"   • Filtered query embeddings shape: {filtered_q_embeddings.shape}")
    print(f"   • Filtered doc embeddings shape: {filtered_d_embeddings.shape}")
    print(f"   • Filtered query types: {len(filtered_q_types)}")
    print(f"   • Filtered doc types: {len(filtered_d_types)}")
    
    return (filtered_q_embeddings, filtered_d_embeddings, 
            filtered_q_ids, filtered_d_ids, 
            filtered_q_types, filtered_d_types)



def main():
    # Set up safer defaults to prevent segfaults
    import os
    import sys
    
    # Set environment variables for stability
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Prevent tokenizer multiprocessing issues
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Limit CUDA memory fragmentation
    
    # Set up signal handlers to catch segfaults
    import signal
    
    def signal_handler(signum, frame):
        print(f"Received signal {signum}. Cleaning up...")
        cleanup_resources()
        sys.exit(1)
    
    signal.signal(signal.SIGSEGV, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, 
                       help="Path to pretrained model or model ID. Supported models: "
                            "allenai/specter2_base, patentbert, mpi-inno-comp/paecter, "
                            "anferico/bert-for-patents, datalyes/patembed-large, naver/splade-v2, bm25, bm25f, "
                            "sparse_coverage, SentenceTransformer checkpoint dir (e.g. checkpoint-1142), or other checkpoint paths.")
    parser.add_argument("--output_dir", type=str, default='./baseline_eval', help="Output directory for evaluation results.")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    
    # Parameters for sparse_coverage model
    parser.add_argument("--dense_model", type=str, default="ZoeYou/PatentMap-V0-SecPair-Claim",
                       help="Dense encoder model used to build embeddings (for sparse_coverage). "
                            "Example: ZoeYou/PatentMap-V0-SecPair-Claim")
    parser.add_argument("--tokenization_unit", type=str, default="spacy_token",
                       choices=["spacy_token", "encoder_token", "spacy_sentence", "noun_chunk", "doc"],
                       help="Tokenization unit used to build embeddings (for sparse_coverage). "
                            "Example: spacy_token, encoder_token")
    parser.add_argument("--include_cls", action="store_true", default=True,
                       help="Whether CLS token embeddings were included (for sparse_coverage). "
                            "If True, uses 'cls' suffix, otherwise 'nocls'")
    parser.add_argument("--layer", type=str, default="last",
                       choices=["last", "second_last"],
                       help="Which layer to use for embeddings (for sparse_coverage). "
                            "Options: 'last' (default) or 'second_last'. "
                            "This must match the layer used when building centers.")
    parser.add_argument("--target_coverage", type=float, default=0.7,
                       help="Target coverage for post-processing (for sparse_coverage). "
                            "If specified (0.0-1.0), will truncate centers to achieve this coverage. "
                            "Uses coverage_history from centers JSON to find the appropriate number of centers. "
                            "Default: None (use all centers). Example: --target_coverage 0.7 for 70%% coverage.")
    parser.add_argument("--document_assignment", type=str, default="soft", choices=["hard", "soft"],
                       help="Document side: hard = each span -> nearest center only (Voronoi); soft = each span in all spheres (range_search). Default: soft.")
    parser.add_argument("--weight_aggregation", type=str, default="max", choices=["max", "sum"],
                       help="Per (query, center) and (doc, center): max = use max similarity (default); sum = use sum of similarities (TF-style). Default: max.")
    parser.add_argument("--use_soft_assignment", action="store_true", default=False,
                       help="Use soft assignment for query spans: all centers with sim >= threshold (same as document side, range_search). "
                            "Default: False (hard assignment: only nearest center per span). If True, query assignment is consistent with posting lists.")
    parser.add_argument("--soft_assignment_max_centers_per_span", type=int, default=None,
                       help="When use_soft_assignment: cap each span to at most this many centers (by similarity). "
                            "Default: None (no cap). Try 5 or 10 to reduce noise while keeping multi-center recall.")
    parser.add_argument("--use_vmf", action="store_true", default=False,
                       help="Use von Mises-Fisher (vMF) continuous weights: weight = exp(kappa_c * (sim - 1)) with per-center kappa. "
                            "kappa_c is estimated from each center's posting list mean similarity (vMF MLE approximation). "
                            "Default: False (use raw similarity as weight).")
    parser.add_argument("--length_norm", type=str, default="none",
                       choices=["none", "sqrt_centers"],
                       help="Document length normalization for sparse_coverage. "
                            "none: no normalization. sqrt_centers: divide by sqrt(num_centers_hit). "
                            "Default: none.")
    parser.add_argument("--centers_suffix", type=str, default="",
                       help="Suffix for centers directory (e.g. '_soft', '_percenter'). "
                            "Use when centers were built with --soft_cover or --per_center_r to custom out_dir.")

    args = parser.parse_args()

    print(f"Running evaluation for model: {args.model_name}")
    print("=============================================>>>>>>>>>")

    # Handle the case where model_name is None
    if args.model_name is None:
        print("Error: --model_name is required")
        return
    
    # Initialize temp directories for all models (not just non-bm25/non-checkpoint)
    model_basename = args.model_name.strip("/").split("/")[-1]
    priorart_temp_dir = os.path.join(args.output_dir, f'priorart_temp_{model_basename}')
    
    # Create directories if they don't exist (for non-BM25 models)
    if not ("bm25" in args.model_name):
        for temp_dir in [priorart_temp_dir]:
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
                print(f"Created directory: {temp_dir}")

    # Print evaluation header
    print(f"📋 Model: {args.model_name}")
    print(f"📁 Output Directory: {args.output_dir}")


    ############################################## crete dataset for prior-art search ##################################################
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

        ############################ Prior-art Search evaluation ############################
        # check if the embeddings are already created
        if os.path.exists(f'{priorart_temp_dir}/query_embeddings.pt') and os.path.exists(f'{priorart_temp_dir}/document_embeddings.pt'):
            print("Embeddings already created!")
            query_embeddings = torch.load(f'{priorart_temp_dir}/query_embeddings.pt', weights_only=False)
            document_embeddings = torch.load(f'{priorart_temp_dir}/document_embeddings.pt', weights_only=False)

        else:
            # Use EXACT same text formatting as patent.py for consistency
            query_embeddings_dict = {}
            doc_embeddings_dict = {}
            
            # Process each text type separately, exactly like patent.py
            for texttype in ["abstract", "claim", "invention"]:
                # Specter2 and PatentBERT don't use section tokens - clean format
                if texttype == "abstract":
                    query_texts = [queries_df.iloc[i]['title'] + f" {tokenizer.sep_token} " + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                    doc_texts = [documents_df.iloc[i]['title'] + f" {tokenizer.sep_token} " + documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                else:
                    query_texts = [queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                    doc_texts = [documents_df.iloc[i][texttype] for i in range(len(documents_df))]

                query_encodings = tokenizer(query_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
                doc_encodings = tokenizer(doc_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

                # get the embeddings by batch
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
                
                # Store embeddings by text type
                query_embeddings_dict[texttype] = query_embs
                doc_embeddings_dict[texttype] = doc_embs
            
            # Create concatenated versions for compatibility with existing evaluation
            query_embeddings = np.concatenate([query_embeddings_dict["abstract"], query_embeddings_dict["claim"], query_embeddings_dict["invention"]], axis=0)
            document_embeddings = np.concatenate([doc_embeddings_dict["abstract"], doc_embeddings_dict["claim"], doc_embeddings_dict["invention"]], axis=0)

            print(query_embeddings.shape, document_embeddings.shape)

            # save the embeddings
            torch.save(query_embeddings, f'{priorart_temp_dir}/query_embeddings.pt')
            torch.save(document_embeddings, f'{priorart_temp_dir}/document_embeddings.pt')

        query_embeddings = query_embeddings.astype('float32')
        document_embeddings = document_embeddings.astype('float32')
        prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types)


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

        ############################ Prior-art Search evaluation ############################
        if os.path.exists(f'{priorart_temp_dir}/query_embeddings.pt') and os.path.exists(f'{priorart_temp_dir}/document_embeddings.pt'):
            print("Embeddings already created!")
            query_embeddings = torch.load(f'{priorart_temp_dir}/query_embeddings.pt', weights_only=False)
            document_embeddings = torch.load(f'{priorart_temp_dir}/document_embeddings.pt', weights_only=False)

        else:
            # Use EXACT same text formatting as patent.py for consistency
            query_embeddings_dict = {}
            doc_embeddings_dict = {}
            
            # Get original IDs (before multiplication) 
            original_query_ids = list(queries_df.index)
            original_doc_ids = list(documents_df.index)
            
            # Process each text type separately, exactly like patent.py
            for texttype in ["abstract", "claim", "invention"]:
                # Format texts according to model type
                if args.model_name.lower() == "mpi-inno-comp/paecter":
                    # PAECTer doesn't use section tokens - clean format
                    if texttype == "abstract":
                        query_texts = [queries_df.iloc[i]['title'] + f" {tokenizer.sep_token} " + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                        doc_texts = [documents_df.iloc[i]['title'] + f" {tokenizer.sep_token} " + documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                    else:
                        query_texts = [queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                        doc_texts = [documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                elif args.model_name.lower() == "anferico/bert-for-patents":
                    # BERT-for-patents uses section tokens like patent.py
                    if texttype == "abstract":
                        query_texts = [queries_df.iloc[i]['title'] + f" [SEP] [{texttype}] " + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                        doc_texts = [documents_df.iloc[i]['title'] + f" [SEP] [{texttype}] " + documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                    else:
                        query_texts = [f"[{texttype}] " + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                        doc_texts = [f"[{texttype}] " + documents_df.iloc[i][texttype] for i in range(len(documents_df))]

                query_encodings = tokenizer(query_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
                doc_encodings = tokenizer(doc_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

                # get the embeddings by batch
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
                
                # Store embeddings by text type
                query_embeddings_dict[texttype] = query_embs
                doc_embeddings_dict[texttype] = doc_embs
            
            # Create concatenated versions for compatibility with existing evaluation
            query_embeddings = np.concatenate([query_embeddings_dict["abstract"], query_embeddings_dict["claim"], query_embeddings_dict["invention"]], axis=0)
            document_embeddings = np.concatenate([doc_embeddings_dict["abstract"], doc_embeddings_dict["claim"], doc_embeddings_dict["invention"]], axis=0)

            print(query_embeddings.shape, document_embeddings.shape)

            # save the embeddings
            torch.save(query_embeddings, f'{priorart_temp_dir}/query_embeddings.pt', pickle_protocol=4)
            torch.save(document_embeddings, f'{priorart_temp_dir}/document_embeddings.pt', pickle_protocol=4)

        query_embeddings = query_embeddings.astype('float32')
        document_embeddings = document_embeddings.astype('float32')
        prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types)


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

        # Cache with _prompted suffix so we never reuse old unprompted embeddings
        query_cache = os.path.join(priorart_temp_dir, "query_embeddings_prompted.pt")
        doc_cache = os.path.join(priorart_temp_dir, "document_embeddings_prompted.pt")
        ############################ Prior-art Search evaluation ############################
        if os.path.exists(query_cache) and os.path.exists(doc_cache):
            print("Embeddings already created (with prompts)!")
            query_embeddings = torch.load(query_cache, weights_only=False)
            document_embeddings = torch.load(doc_cache, weights_only=False)
        else:
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
                # Use built-in task prompts: encode_query(..., prompt_name) / encode_document(..., prompt_name)
                try:
                    query_embs = model.encode_query(raw_query, prompt_name=PATEN_TEB_RETRIEVAL_PROMPT_NAME, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
                    doc_embs = model.encode_document(raw_doc, prompt_name=PATEN_TEB_RETRIEVAL_PROMPT_NAME, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
                except Exception as e:
                    # Fallback: manual prompts per PatenTEB Table 11 retrieval_MIXED (if model.prompts structure differs)
                    PROMPT_QUERY = "encode query for mixed document retrieval: "
                    PROMPT_DOC = "encode document for mixed retrieval: "
                    query_texts = [PROMPT_QUERY + t for t in raw_query]
                    doc_texts = [PROMPT_DOC + t for t in raw_doc]
                    query_embs = model.encode(query_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
                    doc_embs = model.encode(doc_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
                query_embeddings_dict[texttype] = query_embs
                doc_embeddings_dict[texttype] = doc_embs

            query_embeddings = np.concatenate([query_embeddings_dict["abstract"], query_embeddings_dict["claim"], query_embeddings_dict["invention"]], axis=0)
            document_embeddings = np.concatenate([doc_embeddings_dict["abstract"], doc_embeddings_dict["claim"], doc_embeddings_dict["invention"]], axis=0)
            print(query_embeddings.shape, document_embeddings.shape)
            torch.save(query_embeddings, query_cache, pickle_protocol=4)
            torch.save(document_embeddings, doc_cache, pickle_protocol=4)

        query_embeddings = query_embeddings.astype('float32')
        document_embeddings = document_embeddings.astype('float32')
        prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types)


########################################################################################################################################################
########################################################################################################################################################
    elif args.model_name == "bm25" or args.model_name == "bm25f":
        import bm25s
        import snowballstemmer
        from itertools import product

        ############################ BM25F Helper Functions ############################
        def build_fielded_bm25(documents_df, field_weights, stemmer, fields=None):
            """
            Build a fielded BM25 retriever with separate field handling.
            
            Args:
                documents_df: DataFrame with columns ['title', 'abstract', 'claim', 'invention']
                field_weights: dict with keys for the fields to use
                stemmer: snowballstemmer instance
                fields: list of field names to use (default: all fields in field_weights)
            
            Returns:
                retriever: BM25 retriever
                doc_ids: list of document IDs
            """
            n_docs = len(documents_df)
            doc_ids = list(documents_df.index)
            
            # Use specified fields or all fields in field_weights
            if fields is None:
                fields = list(field_weights.keys())
            
            # Build combined text strings with field weighting via repetition
            combined_texts = []
            for i in range(n_docs):
                doc_text_parts = []
                for field in fields:
                    if field not in field_weights:
                        continue
                    
                    # Get field text safely
                    field_val = documents_df.iloc[i][field]
                    field_text = str(field_val) if pd.notna(field_val) else ''
                    
                    if not field_text.strip():
                        continue
                    
                    weight = field_weights[field]
                    repeat_count = max(1, int(weight))
                    # Repeat the field text based on weight
                    doc_text_parts.extend([field_text] * repeat_count)
                
                combined_text = ' '.join(doc_text_parts)
                if not combined_text.strip():
                    combined_text = 'empty_document'
                combined_texts.append(combined_text)
            
            # Tokenize and index using standard bm25s workflow
            corpus_tokens = bm25s.tokenize(combined_texts, stopwords="en", stemmer=stemmer)
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)
            
            return retriever, doc_ids
        
        def evaluate_bm25f_with_weights(queries_df, documents_df, citation_mapping, 
                                       field_weights, query_field='claim', stemmer=None, fields=None):
            """
            Evaluate BM25F with given field weights.
            
            Args:
                queries_df: Query DataFrame
                documents_df: Document DataFrame
                citation_mapping: Ground truth citations
                field_weights: dict of field weights
                query_field: which field to use for queries ('claim' or 'abstract')
                stemmer: snowballstemmer instance
                fields: list of field names to use in documents (default: all fields in field_weights)
            
            Returns:
                dict with recall@k, ndcg@10, mrr@10, and map scores
            """
            if stemmer is None:
                stemmer = snowballstemmer.stemmer('english')
            
            # Build fielded retriever
            retriever, doc_ids = build_fielded_bm25(documents_df, field_weights, stemmer, fields=fields)
            
            # Prepare queries based on query_field
            if query_field == 'abstract':
                # Use title + abstract for queries (like embedding models)
                query_texts = (queries_df['title'].fillna('') + ' ' + queries_df['abstract'].fillna('')).tolist()
            else:  # claim
                query_texts = queries_df[query_field].fillna('').tolist()
            
            # Tokenize queries - keep the full Tokenized object for retrieve()
            query_tokens = bm25s.tokenize(query_texts, stopwords="en", stemmer=stemmer)
            
            results, _ = retriever.retrieve(query_tokens, k=100)
            
            # Map results back to document IDs
            retrieved_ids = [[doc_ids[i] for i in result] for result in results]
            
            # Prepare ground truth labels
            query_ids = list(queries_df.index)
            true_labels_list = [citation_mapping.get(q, []) for q in query_ids]
            
            # Calculate all metrics
            metrics = {}
            
            # Recall@k
            for k in [10, 20, 50, 100]:
                metrics[f'recall@{k}'] = mean_recall_at_k(true_labels_list, retrieved_ids, k=k)
            
            # nDCG@10
            metrics['ndcg@10'] = mean_ndcg_at_k(true_labels_list, retrieved_ids, k=10)
            
            # MRR@10
            metrics['mrr@10'] = mean_mrr_at_k(true_labels_list, retrieved_ids, k=10)
            
            # MAP (using top 100 results)
            metrics['map'] = mean_average_precision(true_labels_list, retrieved_ids, k=100)
            
            # PRES@100
            metrics['pres@100'] = mean_pres_at_k(true_labels_list, retrieved_ids, k=100, N_max=100)
            
            return metrics
        
        def grid_search_field_weights(queries_df, documents_df, citation_mapping, 
                                     query_field='claim', stemmer=None):
            """
            Grid search over field weight combinations to find optimal weights.
            
            Returns:
                best_weights: dict of optimal field weights
                best_recalls: dict of recall scores with best weights
                all_results: list of (weights, recalls) for all combinations
            """
            if stemmer is None:
                stemmer = snowballstemmer.stemmer('english')
            
            print(f"\n🔍 Grid Search for Optimal Field Weights ({query_field} queries)")
            print("=" * 70)
            
            # Define fields and weight ranges based on query type
            if query_field == 'abstract':
                # Abstract→Abstract: Only use title + abstract fields
                field_names = ['title', 'abstract']
                weight_options = {
                    'title': [1.0, 1.5, 2.0, 2.5],
                    'abstract': [1.5, 2.0, 2.5, 3.0]
                }
            else:  # claim queries
                # Claim→All: Use all fields
                field_names = ['title', 'abstract', 'claim', 'invention']
                weight_options = {
                    'title': [1.5, 2.0, 2.5, 3.0],
                    'abstract': [1.5, 2.0, 2.5, 3.0],
                    'claim': [2.0, 2.5, 3.0, 3.5],
                    'invention': [0.5, 1.0, 1.5]
                }
            
            # Generate all combinations
            weight_combinations = list(product(*[weight_options[f] for f in field_names]))
            
            # Print weight ranges based on fields being used
            weight_range_str = ", ".join([f"{field}={weight_options[field]}" for field in field_names])
            print(f"   • Testing {len(weight_combinations)} weight combinations")
            print(f"   • Document fields: {field_names}")
            print(f"   • Weight ranges: {weight_range_str}")
            
            best_recall = 0
            best_weights = None
            best_recalls = None
            all_results = []
            
            # Test each combination
            for i, weights_tuple in enumerate(weight_combinations):
                weights = dict(zip(field_names, weights_tuple))
                
                try:
                    # Evaluate this weight combination
                    metrics = evaluate_bm25f_with_weights(
                        queries_df, documents_df, citation_mapping, 
                        weights, query_field=query_field, stemmer=stemmer, fields=field_names
                    )
                    
                    all_results.append((weights.copy(), metrics.copy()))
                    
                    # Track best by recall@100 (most comprehensive metric)
                    if metrics['recall@100'] > best_recall:
                        best_recall = metrics['recall@100']
                        best_weights = weights.copy()
                        best_recalls = metrics.copy()
                    
                    # Progress update every 8 combinations
                    if (i + 1) % 8 == 0 or (i + 1) == len(weight_combinations):
                        print(f"   • Tested {i+1}/{len(weight_combinations)} combinations... "
                              f"Current best R@100: {best_recall:.4f}")
                except Exception as e:
                    print(f"   ⚠️  Error with weights {weights}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"\n✅ Grid Search Complete!")
            print(f"   • Total results collected: {len(all_results)}")
            print(f"   • Best weights: {best_weights}")
            print(f"   • Best Recall@100: {best_recall:.4f}")
            
            # Handle case where no valid results were found
            if best_weights is None or len(all_results) == 0:
                print("   ⚠️  WARNING: No valid results found! Using default weights.")
                # Return default equal weights
                default_weights = {field: 1.0 for field in field_names}
                default_metrics = evaluate_bm25f_with_weights(
                    queries_df, documents_df, citation_mapping, 
                    default_weights, query_field=query_field, stemmer=stemmer, fields=field_names
                )
                return default_weights, default_metrics, [(default_weights, default_metrics)]
            
            return best_weights, best_recalls, all_results

        ############################ BM25/BM25F Evaluation ############################
        print(f"Running {'BM25F (Fielded)' if args.model_name == 'bm25f' else 'BM25 (Standard)'} Prior-art search evaluation")
        
        stemmer = snowballstemmer.stemmer('english')
        original_doc_ids = list(documents.keys())
        
        if args.model_name == "bm25f":
            # ============= BM25F: Fielded BM25 with optimized weights =============
            
            # 1) Abstract query evaluation with grid search
            print("\n" + "=" * 70)
            print("BM25F Evaluation 1: Abstract Queries → Title + Abstract Fields")
            print("=" * 70)
            
            abstract_best_weights, abstract_best_recalls, abstract_all_results = grid_search_field_weights(
                queries_df, documents_df, citation_mapping, 
                query_field='abstract', stemmer=stemmer
            )
            
            # Display best results
            print_metric_table(abstract_best_recalls, f"BM25F: Abstract → All (Best Weights)")
            
            # Show top 5 weight combinations
            print(f"\n📊 Top 5 Weight Combinations (by Recall@100):")
            abstract_all_results.sort(key=lambda x: x[1]['recall@100'], reverse=True)
            for rank, (weights, metrics) in enumerate(abstract_all_results[:5], 1):
                weight_str = ", ".join([f"{k[0].upper()}={v:.1f}" for k, v in weights.items()])
                print(f"   {rank}. Weights: {weight_str} → "
                      f"R@10={metrics['recall@10']:.4f}, R@100={metrics['recall@100']:.4f}, "
                      f"nDCG@10={metrics['ndcg@10']:.4f}, MRR@10={metrics['mrr@10']:.4f}, MAP={metrics['map']:.4f}")
            
            # 2) Claim query evaluation with grid search
            print("\n" + "=" * 70)
            print("BM25F Evaluation 2: Claim Queries → All Document Fields")
            print("=" * 70)
            
            claim_best_weights, claim_best_recalls, claim_all_results = grid_search_field_weights(
                queries_df, documents_df, citation_mapping, 
                query_field='claim', stemmer=stemmer
            )
            
            # Display best results
            print_metric_table(claim_best_recalls, f"BM25F: Claim → All (Best Weights)")
            
            # Show top 5 weight combinations
            print(f"\n📊 Top 5 Weight Combinations (by Recall@100):")
            claim_all_results.sort(key=lambda x: x[1]['recall@100'], reverse=True)
            for rank, (weights, metrics) in enumerate(claim_all_results[:5], 1):
                weight_str = ", ".join([f"{k[0].upper()}={v:.1f}" for k, v in weights.items()])
                print(f"   {rank}. Weights: {weight_str} → "
                      f"R@10={metrics['recall@10']:.4f}, R@100={metrics['recall@100']:.4f}, "
                      f"nDCG@10={metrics['ndcg@10']:.4f}, MRR@10={metrics['mrr@10']:.4f}, MAP={metrics['map']:.4f}")
            
            # Summary
            print("\n" + "=" * 70)
            print("📝 BM25F Summary:")
            print(f"   • Abstract queries (title + abstract fields):")
            print(f"     Recall@100: {abstract_best_recalls['recall@100']:.4f}, nDCG@10: {abstract_best_recalls['ndcg@10']:.4f}, "
                  f"MRR@10: {abstract_best_recalls['mrr@10']:.4f}, MAP: {abstract_best_recalls['map']:.4f}")
            abstract_weight_str = ", ".join([f"{k.capitalize()}={v:.1f}" for k, v in abstract_best_weights.items()])
            print(f"     Optimal weights: {abstract_weight_str}")
            print(f"   • Claim queries (all fields):")
            print(f"     Recall@100: {claim_best_recalls['recall@100']:.4f}, nDCG@10: {claim_best_recalls['ndcg@10']:.4f}, "
                  f"MRR@10: {claim_best_recalls['mrr@10']:.4f}, MAP: {claim_best_recalls['map']:.4f}")
            claim_weight_str = ", ".join([f"{k.capitalize()}={v:.1f}" for k, v in claim_best_weights.items()])
            print(f"     Optimal weights: {claim_weight_str}")
            print("=" * 70)
            
        else:
            # ============= Standard BM25: Concatenated fields (baseline) =============
            
            # 1) Abstract-to-Abstract evaluation (like other models' abstract->abstract)
            print("\nBM25 Evaluation 1: Abstract-to-Abstract retrieval")
            abstract_train_corpus = documents_df['title'] + ' ' + documents_df['abstract']
            abstract_test_corpus = queries_df['title'] + ' ' + queries_df['abstract']
            
            # Tokenize corpus
            abstract_corpus_tokens = bm25s.tokenize(abstract_train_corpus.tolist(), stopwords="en", stemmer=stemmer)
            
            # Create and index BM25 model
            abstract_retriever = bm25s.BM25()
            abstract_retriever.index(abstract_corpus_tokens)
            
            # Tokenize queries and retrieve
            abstract_queries_tokens = bm25s.tokenize(abstract_test_corpus.tolist(), stemmer=stemmer)
            abstract_results, _ = abstract_retriever.retrieve(abstract_queries_tokens, k=100)
            
            # Map results back to document IDs (only abstract docs)
            abstract_retrieved_ids = [[original_doc_ids[i] for i in result] for result in abstract_results]
            
            # Calculate metrics for abstract-to-abstract
            query_ids_list = list(queries.keys())
            true_labels_list = [citation_mapping.get(q, []) for q in query_ids_list]
            
            bm25_abstract_results = {}
            # Recall@k
            for k in [10, 20, 50, 100]:
                bm25_abstract_results[f'recall@{k}'] = mean_recall_at_k(true_labels_list, abstract_retrieved_ids, k=k)
            
            # nDCG@10
            bm25_abstract_results['ndcg@10'] = mean_ndcg_at_k(true_labels_list, abstract_retrieved_ids, k=10)
            
            # MRR@10
            bm25_abstract_results['mrr@10'] = mean_mrr_at_k(true_labels_list, abstract_retrieved_ids, k=10)
            
            # MAP
            bm25_abstract_results['map'] = mean_average_precision(true_labels_list, abstract_retrieved_ids, k=100)
            
            # PRES@100
            bm25_abstract_results['pres@100'] = mean_pres_at_k(true_labels_list, abstract_retrieved_ids, k=100, N_max=100)
            
            print_metric_table(bm25_abstract_results, "BM25: Abstract → Abstract")
            
            # 2) Claim-to-All evaluation (like other models' claim->all)
            print("\nBM25 Evaluation 2: Claim-to-All retrieval")
            # Use all document sections as corpus
            all_train_corpus = (
                (documents_df['title'] + ' ' + documents_df['abstract']).tolist() + 
                documents_df['claim'].tolist() + 
                documents_df['invention'].tolist()
            )
            # Use only claim queries
            claim_test_corpus = queries_df['claim'].tolist()
            
            # Tokenize corpus
            all_corpus_tokens = bm25s.tokenize(all_train_corpus, stopwords="en", stemmer=stemmer)
            
            # Create and index BM25 model
            all_retriever = bm25s.BM25()
            all_retriever.index(all_corpus_tokens)
            
            # Tokenize queries and retrieve (k=300 to account for 3x document sections)
            claim_queries_tokens = bm25s.tokenize(claim_test_corpus, stemmer=stemmer)
            claim_results, _ = all_retriever.retrieve(claim_queries_tokens, k=300)
            
            # Map results back to original document IDs and remove duplicates
            # Document ordering: [abstracts, claims, inventions] each of length original_doc_count
            original_doc_count = len(original_doc_ids)
            claim_retrieved_ids = []
            
            for result in claim_results:
                doc_ids_for_query = []
                for idx in result:
                    # Map index back to original document ID
                    if idx < original_doc_count:  # abstract section
                        doc_id = original_doc_ids[idx]
                    elif idx < 2 * original_doc_count:  # claim section
                        doc_id = original_doc_ids[idx - original_doc_count]
                    else:  # invention section
                        doc_id = original_doc_ids[idx - 2 * original_doc_count]
                    doc_ids_for_query.append(doc_id)
                
                # Remove duplicates while preserving order, keep only top 100
                unique_doc_ids = list(dict.fromkeys(doc_ids_for_query))[:100]
                claim_retrieved_ids.append(unique_doc_ids)
            
            # Calculate metrics for claim-to-all
            bm25_claim_results = {}
            # Recall@k
            for k in [10, 20, 50, 100]:
                bm25_claim_results[f'recall@{k}'] = mean_recall_at_k(true_labels_list, claim_retrieved_ids, k=k)
            
            # nDCG@10
            bm25_claim_results['ndcg@10'] = mean_ndcg_at_k(true_labels_list, claim_retrieved_ids, k=10)
            
            # MRR@10
            bm25_claim_results['mrr@10'] = mean_mrr_at_k(true_labels_list, claim_retrieved_ids, k=10)
            
            # MAP
            bm25_claim_results['map'] = mean_average_precision(true_labels_list, claim_retrieved_ids, k=100)
            
            # PRES@100
            bm25_claim_results['pres@100'] = mean_pres_at_k(true_labels_list, claim_retrieved_ids, k=100, N_max=100)
            
            print_metric_table(bm25_claim_results, "BM25: Claim → All Sections")
            
            print("\n📝 Note: BM25 evaluation completed.")


########################################################################################################################################################
########################################################################################################################################################
    elif args.model_name.lower() in ["naver/splade-v2", "splade-v2", "naver/splade_v2_max", "naver/splade_v2_distil"]:
        """
        SPLADE-v2 Sparse Retrieval Model
        
        SPLADE uses a BERT-like architecture with MLM head to produce sparse representations.
        Key differences from dense models:
        - Produces sparse vectors (most dimensions are zero)
        - Uses vocabulary-sized output (30k+ dimensions)
        - Retrieval via sparse vector dot product
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
                    query_texts = (queries_df['title'] + ' ' + queries_df['abstract']).fillna('').tolist()
                    doc_texts = (documents_df['title'] + ' ' + documents_df['abstract']).fillna('').tolist()
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
            """SPLADE-specific evaluation using proper similarity computation."""
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
            
            # 1) Abstract-to-Abstract evaluation
            texttype_q = "abstract"
            texttype_d = "abstract"
            
            query_types_arr = np.array(query_types)
            doc_types_arr = np.array(doc_types)
            
            query_type_masks = (query_types_arr == texttype_q)
            doc_type_masks = (doc_types_arr == texttype_d)
            
            Q_sparse = query_dense[query_type_masks]
            D_sparse = doc_dense[doc_type_masks]
            
            # Use model.similarity() for proper SPLADE scoring (keeps PyTorch format)
            similarities = model.similarity(Q_sparse, D_sparse)
            
            # Convert to numpy for top-k computation
            if hasattr(similarities, 'cpu'):
                distances = similarities.cpu().numpy()
            else:
                distances = similarities
            
            # Get top-k indices
            top_k_indices = np.argsort(-distances, axis=1)
            
            # Build true/predicted labels using ORIGINAL IDs
            true_labels_list, predicted_labels_list = [], []
            for q_idx, retrieved_docs_indices in enumerate(top_k_indices):
                q_id_str = original_query_ids[q_idx]  # Use original query ID
                true_labels = citation_mapping.get(q_id_str, [])
                predicted_labels = [original_doc_ids[d_idx] for d_idx in retrieved_docs_indices]  # Use original doc IDs
                true_labels_list.append(true_labels)
                predicted_labels_list.append(predicted_labels)
            
            results_key = f"{texttype_q}->{texttype_d}"
            results[results_key] = {
                'recall@10': mean_recall_at_k(true_labels_list, predicted_labels_list, k=10),
                'recall@20': mean_recall_at_k(true_labels_list, predicted_labels_list, k=20),
                'recall@50': mean_recall_at_k(true_labels_list, predicted_labels_list, k=50),
                'recall@100': mean_recall_at_k(true_labels_list, predicted_labels_list, k=100),
                'ndcg@10': mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=10),
                'ndcg@20': mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=20),
                'ndcg@50': mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=50),
                'ndcg@100': mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=100),
                'mrr@10': mean_mrr_at_k(true_labels_list, predicted_labels_list, k=10),
                'map': mean_average_precision(true_labels_list, predicted_labels_list, k=100),
                'pres@100': mean_pres_at_k(true_labels_list, predicted_labels_list, k=100, N_max=100),
            }
            
            # 2) Claim-to-All evaluation
            texttype_q = "claim"
            query_type_masks = (query_types_arr == texttype_q)
            Q_sparse = query_dense[query_type_masks]
            D_sparse = doc_dense  # All document sections
            
            # Use model.similarity() for proper SPLADE scoring (keeps PyTorch format)
            similarities = model.similarity(Q_sparse, D_sparse)
            
            # Convert to numpy for top-k computation
            if hasattr(similarities, 'cpu'):
                distances = similarities.cpu().numpy()
            else:
                distances = similarities
            
            # Get top-k indices (retrieve more to account for duplicates)
            top_k_indices = np.argsort(-distances, axis=1)[:, :300]
            
            retrieved_sections = []
            true_labels_list, predicted_labels_list = [], []
            
            for q_idx, retrieved_docs_indices in enumerate(top_k_indices):
                q_id_str = original_query_ids[q_idx]  # Use original query ID
                true_labels = citation_mapping.get(q_id_str, [])
                
                # Map indices back to original doc IDs (doc_ids is [abstract*N, claim*N, invention*N])
                predicted_labels = []
                for d_idx in retrieved_docs_indices:
                    # d_idx maps to the concatenated structure: first N are abstracts, next N are claims, last N are inventions
                    orig_doc_idx = d_idx % original_doc_count
                    predicted_labels.append(original_doc_ids[orig_doc_idx])
                
                # Remove duplicates while preserving order
                seen = set()
                unique_predicted = []
                unique_indices_list = []
                for i, label in enumerate(predicted_labels):
                    if label not in seen:
                        seen.add(label)
                        unique_predicted.append(label)
                        unique_indices_list.append(i)
                predicted_labels = unique_predicted[:100]
                
                # Track which sections were retrieved
                retrieved_sections.append([
                    ["abstract", "claim", "invention"][retrieved_docs_indices[i] // original_doc_count]
                    for i in unique_indices_list[:100]
                ])
                
                true_labels_list.append(true_labels)
                predicted_labels_list.append(predicted_labels)
            
            results_key = f"{texttype_q}->all"
            results[results_key] = {
                'recall@10': mean_recall_at_k(true_labels_list, predicted_labels_list, k=10),
                'recall@20': mean_recall_at_k(true_labels_list, predicted_labels_list, k=20),
                'recall@50': mean_recall_at_k(true_labels_list, predicted_labels_list, k=50),
                'recall@100': mean_recall_at_k(true_labels_list, predicted_labels_list, k=100),
                'ndcg@10': mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=10),
                'ndcg@20': mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=20),
                'ndcg@50': mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=50),
                'ndcg@100': mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=100),
                'mrr@10': mean_mrr_at_k(true_labels_list, predicted_labels_list, k=10),
                'map': mean_average_precision(true_labels_list, predicted_labels_list, k=100),
                'pres@100': mean_pres_at_k(true_labels_list, predicted_labels_list, k=100, N_max=100),
                'retrieved_sections': f"[{len(retrieved_sections)} queries with retrieved sections]"
            }
            
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
    elif _is_sentence_transformer_checkpoint(args.model_name):
        # SentenceTransformer-style checkpoint (e.g. checkpoint-1142): transformer + pooling only.
        # No prompts, no special tokens, no separator; abstract = title + space + text, claim/invention = plain text.
        _checkpoint_path = _resolve_st_checkpoint_path(args.model_name)
        print(f"\n🔍 Loading SentenceTransformer checkpoint: {_checkpoint_path}")
        model = load_checkpoint_model(_checkpoint_path, max_length=512, hf_model_name=None)
        embedding_dim = model.get_sentence_embedding_dimension()
        model.to(device)

        if os.path.exists(f'{priorart_temp_dir}/query_embeddings.pt') and os.path.exists(f'{priorart_temp_dir}/document_embeddings.pt'):
            print("Embeddings already created!")
            query_embeddings = torch.load(f'{priorart_temp_dir}/query_embeddings.pt', weights_only=False)
            document_embeddings = torch.load(f'{priorart_temp_dir}/document_embeddings.pt', weights_only=False)
        else:
            query_embeddings_dict = {}
            doc_embeddings_dict = {}
            for texttype in ["abstract", "claim", "invention"]:
                if texttype == "abstract":
                    raw_query = [queries_df.iloc[i]['title'] + " " + queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                    raw_doc = [documents_df.iloc[i]['title'] + " " + documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                else:
                    raw_query = [queries_df.iloc[i][texttype] for i in range(len(queries_df))]
                    raw_doc = [documents_df.iloc[i][texttype] for i in range(len(documents_df))]
                query_embs = model.encode(raw_query, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
                doc_embs = model.encode(raw_doc, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
                query_embeddings_dict[texttype] = query_embs
                doc_embeddings_dict[texttype] = doc_embs
            query_embeddings = np.concatenate([query_embeddings_dict["abstract"], query_embeddings_dict["claim"], query_embeddings_dict["invention"]], axis=0)
            document_embeddings = np.concatenate([doc_embeddings_dict["abstract"], doc_embeddings_dict["claim"], doc_embeddings_dict["invention"]], axis=0)
            print(query_embeddings.shape, document_embeddings.shape)
            torch.save(query_embeddings, f'{priorart_temp_dir}/query_embeddings.pt', pickle_protocol=4)
            torch.save(document_embeddings, f'{priorart_temp_dir}/document_embeddings.pt', pickle_protocol=4)

        query_embeddings = query_embeddings.astype('float32')
        document_embeddings = document_embeddings.astype('float32')
        prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types)


########################################################################################################################################################
########################################################################################################################################################
    elif "checkpoint" in args.model_name or "bestmodel" in args.model_name or "patentmap" in args.model_name.lower():
        def load_checkpoint_model_and_tokenizer(checkpoint_path):
            """Smart checkpoint loader that handles tokenizer and model loading intelligently."""
            from transformers import AutoConfig, AutoTokenizer
            from dataclasses import dataclass
            from typing import Optional
            
            @dataclass
            class ModelArguments:
                do_mlm: bool = False
                regularization: Optional[str] = None
                temperature: float = 0.05
                pooler_type: str = "cls"
                mlp_only_train: bool = True
                model_name_or_path: Optional[str] = None
            
            print(f"🔄 Loading checkpoint: {checkpoint_path}")
            
            # Step 1: Try loading tokenizer from checkpoint, fallback to reconstruction
            try:
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
                print(f"✅ Loaded tokenizer from checkpoint ({len(tokenizer)} tokens)")
            except:
                print("⚠️  Tokenizer not found in checkpoint, reconstructing...")
                tokenizer = reconstruct_tokenizer(checkpoint_path)
            
            # Step 2: Load model with proper config
            model_args = ModelArguments(model_name_or_path=checkpoint_path)
            config = AutoConfig.from_pretrained("anferico/bert-for-patents")
            config.vocab_size = len(tokenizer)
            
            # Step 3: Load model with smart error handling
            model = load_model_with_fallback(checkpoint_path, config, model_args)
            
            # Step 4: Ensure vocab size consistency
            if model.get_input_embeddings().num_embeddings != len(tokenizer):
                model.resize_token_embeddings(len(tokenizer))
                print(f"🔧 Resized model embeddings to {len(tokenizer)} tokens")
            
            return model, tokenizer, model.config.hidden_size

        def reconstruct_tokenizer(checkpoint_path):
            """Reconstruct tokenizer by inferring settings from checkpoint path."""
            special_tokens = {"abstract": "[abstract]", "claim": "[claim]", "summary": "[summary]",
                            "background": "[invention]", "drawing": "[drawing]", "detailed_description": "[description]"}
            
            tokenizer = AutoTokenizer.from_pretrained("anferico/bert-for-patents")
            
            # Smart view inference from path
            views_match = re.search(r'views-([^/]*?)(?:_reg-|/|$)', checkpoint_path)
            if views_match and views_match.group(1):
                additional_views = views_match.group(1).split('+')
                print(f"📊 Inferred views from path: {additional_views}")
            else:
                additional_views = []
                print(f"📊 No views specified in path - using minimal tokenizer to match training vocab_size")
            
            # Add required special tokens
            tokens_to_add = []
            for view in ['abstract'] + additional_views:
                if view in special_tokens:
                    token = special_tokens[view]
                    if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id:
                        tokens_to_add.append(token)
            
            # Handle detailed_description dependency on drawing
            if "detailed_description" in additional_views and "drawing" not in additional_views:
                drawing_token = special_tokens["drawing"]
                if drawing_token not in tokens_to_add and tokenizer.convert_tokens_to_ids(drawing_token) == tokenizer.unk_token_id:
                    tokens_to_add.append(drawing_token)
            
            if tokens_to_add:
                tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})
                print(f"➕ Added tokens: {tokens_to_add}")
            else:
                print(f"✅ Using base tokenizer without additional tokens (vocab_size: {len(tokenizer)})")
            
            return tokenizer

        def load_model_with_fallback(checkpoint_path, config, model_args):
            """Load model with progressive fallback strategies."""
            from patentmap.models import BertForCL
            
            is_local = os.path.exists(checkpoint_path)
            loading_strategies = [
                # Strategy 1: Standard loading
                {"local_files_only": is_local, "trust_remote_code": True, "ignore_mismatched_sizes": True},
                # Strategy 2: Minimal parameters
                {"ignore_mismatched_sizes": True, "local_files_only": is_local},
                # Strategy 3: Basic fallback
                {"ignore_mismatched_sizes": True}
            ]
            
            for i, kwargs in enumerate(loading_strategies, 1):
                try:
                    print(f"🔄 Trying loading strategy {i}...")
                    return BertForCL.from_pretrained(checkpoint_path, config=config, model_args=model_args, **kwargs)
                except Exception as e:
                    print(f"❌ Strategy {i} failed: {e}")
                    if i == len(loading_strategies):
                        raise RuntimeError(f"All loading strategies failed for {checkpoint_path}")
            
        # Main loading execution
        model, tokenizer, embedding_dim = load_checkpoint_model_and_tokenizer(args.model_name)
        batch_size = 512
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup model for inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.to(device).eval()
        print(f"🚀 Model ready on {device}")

        ############################ Prior-art Search evaluation ############################
        # check if the embeddings are already created
        priorart_temp_dir = os.path.join(args.output_dir, f'priorart_temp_{args.model_name}')
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
                
                with torch.no_grad():
                    for i in trange(0, len(query_encodings['input_ids']), batch_size, desc=f"Computing {texttype} query embeddings"):
                        batch = {key: val[i:i+batch_size].to(device) for key, val in query_encodings.items()}
                        outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                        query_embs[i:i+batch_size] = outputs.pooler_output.detach().cpu().numpy()

                    for i in trange(0, len(doc_encodings['input_ids']), batch_size, desc=f"Computing {texttype} document embeddings"):
                        batch = {key: val[i:i+batch_size].to(device) for key, val in doc_encodings.items()}
                        outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                        doc_embs[i:i+batch_size] = outputs.pooler_output.detach().cpu().numpy()
                
                # Store embeddings by text type
                query_embeddings_dict[texttype] = query_embs
                doc_embeddings_dict[texttype] = doc_embs
            
            # For compatibility with existing evaluation code, we'll create the concatenated versions
            # But the evaluation should use the separated versions to match patent.py exactly
            query_embeddings = np.concatenate([query_embeddings_dict["abstract"], query_embeddings_dict["claim"], query_embeddings_dict["invention"]], axis=0)
            document_embeddings = np.concatenate([doc_embeddings_dict["abstract"], doc_embeddings_dict["claim"], doc_embeddings_dict["invention"]], axis=0)

            print(query_embeddings.shape, document_embeddings.shape)

            # save the embeddings in both formats for compatibility
            torch.save(query_embeddings, f'{priorart_temp_dir}/query_embeddings.pt')
            torch.save(document_embeddings, f'{priorart_temp_dir}/document_embeddings.pt')
            
            # Also save the text-type separated embeddings (matching patent.py format)
            np.savez(f'{priorart_temp_dir}/query_embeddings_by_type.npz', **query_embeddings_dict)
            np.savez(f'{priorart_temp_dir}/doc_embeddings_by_type.npz', **doc_embeddings_dict)

            print(query_embeddings.shape, document_embeddings.shape)

            # save the embeddings
            torch.save(query_embeddings, f'{priorart_temp_dir}/query_embeddings.pt')
            torch.save(document_embeddings, f'{priorart_temp_dir}/document_embeddings.pt')

        query_embeddings = query_embeddings.astype('float32')
        document_embeddings = document_embeddings.astype('float32')

        # For checkpoint models, use the exact same evaluation method as patent.py to ensure consistency
        print("Using patent.py-compatible evaluation for checkpoint model...")
        print("This ensures exact consistency with training-time evaluation results.")
        
        # Use the standard evaluation for now, but note that minor differences may exist
        # due to different data organization methods between baseline.py and patent.py
        prior_art_search_evaluation(query_ids, doc_ids, query_embeddings, document_embeddings, citation_mapping, query_types, doc_types)


########################################################################################################################################################
########################################################################################################################################################
    elif args.model_name == "sparse_coverage":
        """
        Sparse Coverage Retrieval
        
        Uses pre-built vocabulary centers and posting lists for efficient sparse retrieval.
        Documents are already indexed via posting lists saved during center construction.
        Only queries need to be encoded and assigned to centers.
        """
        print(f"\n🔍 Sparse Coverage Retrieval")
        
        # Validate required parameters
        if args.dense_model is None:
            raise ValueError("--dense_model is required for sparse_coverage")
        if args.tokenization_unit is None:
            raise ValueError("--tokenization_unit is required for sparse_coverage")
        
        print(f"   Dense model: {args.dense_model}")
        print(f"   Tokenization unit: {args.tokenization_unit}")
        print(f"   Include CLS: {args.include_cls}")
        print(f"   Layer: {getattr(args, 'layer', 'last')}")
        print(f"   Length norm: {getattr(args, 'length_norm', 'none')}")
        
        # Import necessary modules
        from load_posting_lists import load_posting_lists
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
            nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
            nlp.max_length = 1000000
            # Set global NLP in utils module so process_doc_batch can use it
            utils.NLP = nlp
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
                    include_cls=args.include_cls,
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
                f"  include_cls={args.include_cls}\n"
                f"  layer={layer}\n"
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
        
        def _truncate_centers_by_coverage(centers: np.ndarray, centers_info: dict) -> tuple[np.ndarray, int]:
            V_original = int(centers.shape[0])
            target_coverage = getattr(args, "target_coverage", None)
            if target_coverage is None:
                return centers, V_original
            
            if target_coverage < 0.0 or target_coverage > 1.0:
                raise ValueError(f"--target_coverage must be in [0.0, 1.0], got {target_coverage}")
            
            coverage_history = centers_info.get("coverage_history", None)
            if not coverage_history:
                print(f"   ⚠️  Coverage history not available, cannot truncate by coverage")
                print(f"   Using all {V_original:,} centers")
                return centers, V_original
            
            truncate_idx = None
            for i, cov in enumerate(coverage_history):
                if cov >= target_coverage:
                    truncate_idx = i + 1
                    break
            
            if truncate_idx is None:
                print(f"   ⚠️  Target coverage {target_coverage:.1%} not reached (max: {coverage_history[-1]:.1%})")
                print(f"   Using all {V_original:,} centers")
                return centers, V_original
            
            V = int(truncate_idx)
            centers_trunc = centers[:V]
            actual_coverage = coverage_history[V - 1] if V > 0 else 0.0
            print(f"   📊 Post-processing: truncating to {V:,} centers for {target_coverage:.1%} coverage")
            print(f"   Actual coverage: {actual_coverage:.1%} (using first {V:,} centers)")
            return centers_trunc, V
        
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
            model_name_clean = args.dense_model.strip("/").split("/")[-1].replace("/", "_").replace("\\", "_")
            cls_suffix = "cls" if args.include_cls else "nocls"
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
                    keep_cls=args.include_cls,
                    span_pooling="mean",
                )
                
                query_spans_dict: dict[int, list[np.ndarray]] = {}
                for doc_id, _section, _doc_text, _span_text_raw, _span_text_canonical, span_emb in batch_results:
                    q_idx = int(doc_id.split("_")[1])
                    query_spans_dict.setdefault(q_idx, []).append(span_emb)
                
                for q_idx in range(batch_start, batch_end):
                    spans = query_spans_dict.get(q_idx, [])
                    if spans:
                        all_query_spans.append(np.stack(spans))
                    else:
                        all_query_spans.append(np.zeros((0, d), dtype=np.float32))
            
            return all_query_spans
        
        def _assign_query_spans_to_centers(
            query_spans: list[np.ndarray],
            center_index: faiss.Index,
            V: int,
            sim_threshold: float,
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            use_soft_assignment = args.use_soft_assignment if hasattr(args, "use_soft_assignment") else False
            weight_agg = getattr(args, "weight_aggregation", "max")

            def _update_weight(weights: dict, key: int, sim: float) -> None:
                if weight_agg == "sum":
                    weights[key] = weights.get(key, 0.0) + max(0.0, sim)
                else:
                    weights[key] = max(weights.get(key, 0.0), max(0.0, sim))

            query_sparse: list[tuple[np.ndarray, np.ndarray]] = []
            for spans in query_spans:
                if spans.shape[0] == 0:
                    query_sparse.append((np.array([], dtype=np.int32), np.array([], dtype=np.float32)))
                    continue

                spans_norm = spans.astype(np.float32).copy()
                faiss.normalize_L2(spans_norm)

                if use_soft_assignment and sim_threshold is not None:
                    max_centers_per_span = getattr(args, "soft_assignment_max_centers_per_span", None)
                    lims, D, I = center_index.range_search(spans_norm, sim_threshold)
                    center_weights: dict[int, float] = {}
                    for span_idx in range(spans_norm.shape[0]):
                        start, end = int(lims[span_idx]), int(lims[span_idx + 1])
                        pairs = [(int(I[j]), float(D[j])) for j in range(start, end) if float(D[j]) > 0]
                        if max_centers_per_span is not None and max_centers_per_span > 0 and len(pairs) > max_centers_per_span:
                            pairs = sorted(pairs, key=lambda x: -x[1])[:max_centers_per_span]
                        for center_id, sim in pairs:
                            _update_weight(center_weights, center_id, sim)
                    if not center_weights:
                        similarities, assigned = center_index.search(spans_norm, k=1)
                        for span_idx in range(similarities.shape[0]):
                            center_id = int(assigned[span_idx, 0])
                            sim = float(similarities[span_idx, 0])
                            _update_weight(center_weights, center_id, sim)

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
                        sim = float(similarities[span_idx, 0])
                        _update_weight(center_weights, center_id, sim)

                    centers_arr = np.array(list(center_weights.keys()), dtype=np.int32)
                    weights_arr = np.array([center_weights[c] for c in centers_arr], dtype=np.float32)
                    query_sparse.append((centers_arr, weights_arr))

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
            centers, V = _truncate_centers_by_coverage(centers, centers_info)
            print(f"   Final vocabulary size: {V:,} centers")
            
            r, sim_threshold = _get_r_and_sim_threshold(centers_info)
            
            if model.config.hidden_size != d:
                raise ValueError(f"Dimension mismatch: model hidden_size={model.config.hidden_size} but centers dimension={d}")
            
            print(f"\n🔨 Building FAISS index on centers...")
            centers_norm = centers.copy()
            faiss.normalize_L2(centers_norm)
            center_index = faiss.IndexFlatIP(d)
            center_index.add(centers_norm.astype(np.float32))
            print(f"✅ Center index built")
            
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
            
            # Load metadata (span -> doc_id)
            print(f"\n📦 Loading embeddings metadata...")
            span_to_doc: dict[int, str] = {}
            current_span_idx = 0
            total_spans_in_metadata = 0
            
            for section_name in doc_sections:
                mp = _metadata_path(section_name)
                if not os.path.exists(mp):
                    raise FileNotFoundError(f"Could not find metadata file: {mp}")
                section_span_count = 0
                print(f"   Loading metadata from {section_name}: {mp}")
                with open(mp, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        meta = json.loads(line)
                        doc_id = meta.get("d", meta.get("doc_id", ""))
                        span_to_doc[current_span_idx] = doc_id
                        current_span_idx += 1
                        section_span_count += 1
                total_spans_in_metadata += section_span_count
                print(f"     Loaded {section_span_count:,} spans from {section_name}")
            
            print(f"   Total loaded {len(span_to_doc):,} span-to-document mappings")
            
            # Build doc_span_count for length normalization (spans per document)
            doc_span_count: dict[str, int] = {}
            for _span_idx, doc_id in span_to_doc.items():
                doc_span_count[doc_id] = doc_span_count.get(doc_id, 0) + 1
            
            # Load embeddings for document sections; build posting lists (doc hard = nearest center only, doc soft = sphere/range_search)
            document_assignment = getattr(args, "document_assignment", "soft")
            print(f"\n🔨 Computing posting lists for {V:,} centers...")
            print(f"   Document assignment: {document_assignment}")
            
            embeddings_files = [(sec, _embeddings_path(sec)) for sec in doc_sections]
            posting_lists: list[list[tuple[int, float]]] = []
            centers_norm_for_pl = centers.copy()
            faiss.normalize_L2(centers_norm_for_pl)
            total_loaded = 0
            
            if document_assignment == "hard":
                # Doc hard: each span -> nearest center only (Voronoi); no embeddings_index needed
                print(f"   Building posting lists: each span -> nearest center (k=1)")
                posting_lists = [[] for _ in range(V)]
                span_offset = 0
                for section_name, ep in embeddings_files:
                    print(f"   Loading {section_name} embeddings: {ep}")
                    sec_emb = _load_embeddings(ep)
                    if sec_emb.shape[1] != d:
                        raise ValueError(f"Embedding dimension mismatch for {section_name}: {sec_emb.shape[1]} != {d}")
                    sec_emb = sec_emb.astype(np.float32)
                    faiss.normalize_L2(sec_emb)
                    sims, assigned = center_index.search(sec_emb, 1)
                    for j in range(sec_emb.shape[0]):
                        c = int(assigned[j, 0])
                        sim = float(sims[j, 0])
                        if sim > 0:
                            posting_lists[c].append((span_offset + j, sim))
                    total_loaded += sec_emb.shape[0]
                    span_offset += sec_emb.shape[0]
                    print(f"     ✅ {sec_emb.shape[0]:,} spans assigned (total: {total_loaded:,})")
                    del sec_emb
                print(f"   ✅ Total spans: {total_loaded:,}")
            else:
                # Doc soft: sphere (range_search) per center; build embeddings_index section-by-section
                print(f"   Using radius r={r:.6f} for range search")
                use_section_by_section = len(doc_sections) > 1
                if use_section_by_section:
                    print(f"   claim2all: building FAISS index section-by-section to reduce peak memory")
                embeddings_index = faiss.IndexFlatIP(d)
                for section_name, ep in embeddings_files:
                    print(f"   Loading {section_name} embeddings: {ep}")
                    sec_emb = _load_embeddings(ep)
                    if sec_emb.shape[1] != d:
                        raise ValueError(f"Embedding dimension mismatch for {section_name}: {sec_emb.shape[1]} != {d}")
                    sec_emb = sec_emb.astype(np.float32)
                    faiss.normalize_L2(sec_emb)
                    embeddings_index.add(sec_emb)
                    total_loaded += sec_emb.shape[0]
                    print(f"     ✅ Loaded {sec_emb.shape[0]:,} (index size: {total_loaded:,})")
                    del sec_emb
                print(f"   ✅ Total in index: {total_loaded:,}")
                r_per_center = centers_info.get("r_per_center", None)
                if r_per_center is not None:
                    if len(r_per_center) < V:
                        r_per_center = None
                    else:
                        if len(r_per_center) > V:
                            r_per_center = r_per_center[:V]  # truncate to match centers after coverage truncation
                        print(f"   Using per-center radius (r_per_center from centers JSON)")
                sim_thresh_default = 1.0 - r
                for center_idx in tqdm(range(V), desc="Computing posting lists"):
                    sim_thresh_pl = (1.0 - float(r_per_center[center_idx])) if r_per_center else sim_thresh_default
                    center_emb = centers_norm_for_pl[center_idx:center_idx + 1].astype(np.float32)
                    lims, D, I = embeddings_index.range_search(center_emb, sim_thresh_pl)
                    if len(lims) < 2:
                        posting_lists.append([])
                        continue
                    start, end = int(lims[0]), int(lims[1])
                    posting_lists.append([(int(I[i]), float(D[i])) for i in range(start, end)])
            
            # Align metadata if index size differs from metadata count
            if total_loaded != total_spans_in_metadata:
                print(f"   ⚠️  Warning: Embeddings count ({total_loaded:,}) != metadata count ({total_spans_in_metadata:,})")
                min_count = min(total_loaded, total_spans_in_metadata)
                span_to_doc = {k: v for k, v in span_to_doc.items() if k < min_count}
                print(f"   Truncated span_to_doc to {min_count:,} spans")
            
            # Build document-level inverted index (doc_id strings)
            print(f"\n🔨 Building document-level inverted index from posting lists...")
            doc_postings: list[list[str]] = [[] for _ in range(V)]
            doc_postings_weights: list[list[float]] = [[] for _ in range(V)]
            doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(documents_df.index)}
            doc_centers_hit: dict[str, int] = {}
            
            weight_agg = getattr(args, "weight_aggregation", "max")
            for center_idx in tqdm(range(V), desc="Building inverted index"):
                span_sims = posting_lists[center_idx]
                if not span_sims:
                    continue
                doc_weights: dict[str, float] = {}
                for span_idx, similarity in span_sims:
                    doc_id = span_to_doc.get(span_idx, None)
                    if doc_id is None or doc_id not in doc_id_to_idx:
                        continue
                    sim = float(similarity)
                    if weight_agg == "sum":
                        doc_weights[doc_id] = doc_weights.get(doc_id, 0.0) + max(0.0, sim)
                    else:
                        doc_weights[doc_id] = max(doc_weights.get(doc_id, 0.0), max(0.0, sim))
                for doc_id, weight in doc_weights.items():
                    doc_postings[center_idx].append(doc_id)
                    doc_postings_weights[center_idx].append(float(weight))
                    doc_centers_hit[doc_id] = doc_centers_hit.get(doc_id, 0) + 1
            
            N_docs = len(documents_df)
            df = np.array([len(set(pl)) if pl else 0 for pl in doc_postings], dtype=np.float32)
            idf = (np.log((N_docs + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)

            # Per-center kappa for vMF (when --use_vmf): estimate from posting list mean similarity
            use_vmf = getattr(args, "use_vmf", False)
            kappa = np.ones(V, dtype=np.float32)  # default 1.0 for empty centers
            if use_vmf:
                for center_idx in range(V):
                    sims = [sim for _, sim in posting_lists[center_idx]]
                    if sims:
                        mean_sim = float(np.mean(sims))
                        # vMF MLE approx: kappa ≈ (d-1) / (2*(1 - R_bar)), R_bar = mean_sim
                        kappa_c = (d - 1.0) / (2.0 * max(1.0 - mean_sim, 0.01))
                        kappa[center_idx] = np.clip(kappa_c, 0.1, 100.0)
                print(f"   vMF: per-center kappa computed (min={kappa.min():.2f}, max={kappa.max():.2f}, mean={np.mean(kappa):.2f})")

            # Encode + assign queries
            if mode == "abstract2abstract":
                print(f"\n📝 Evaluating: Abstract -> Abstract")
                query_texts = []
                for idx in queries_df.index:
                    title = queries_df.loc[idx, "title"] if "title" in queries_df.columns else ""
                    abstract = queries_df.loc[idx, "abstract"] if "abstract" in queries_df.columns else ""
                    query_texts.append(f"{title} [SEP] [abstract] {abstract}".strip())
            else:
                print(f"\n📝 Evaluating: Claim -> All")
                query_texts = queries_df["claim"].fillna("").tolist()
            
            query_spans = _encode_query_spans(query_texts, section=query_section, d=d)
            query_sparse = _assign_query_spans_to_centers(query_spans, center_index=center_index, V=V, sim_threshold=sim_threshold)
            
            # Length normalization setup
            length_norm = getattr(args, "length_norm", "none")
            avg_span_count = 1.0
            if length_norm == "sqrt_centers":
                print(f"   Length norm: sqrt_centers")
            
            print(f"🔍 Retrieving documents...")
            top_k = 100
            top_indices: list[list[int]] = []
            for _q_idx, (terms, weights) in enumerate(tqdm(query_sparse, desc="Retrieving")):
                doc_scores: dict[int, float] = {}
                for term, q_weight in zip(terms, weights):
                    if len(doc_postings[term]) == 0:
                        continue
                    kt = float(kappa[term])
                    q_sim = float(q_weight)
                    idf_t = float(idf[term])
                    for doc_id, d_sim_weight in zip(doc_postings[term], doc_postings_weights[term]):
                        doc_idx = doc_id_to_idx.get(doc_id, None)
                        if doc_idx is None:
                            continue
                        d_sim = float(d_sim_weight)
                        if use_vmf:
                            contrib = np.exp(kt * (q_sim - 1.0)) * np.exp(kt * (d_sim - 1.0)) * idf_t
                        else:
                            contrib = q_sim * d_sim * idf_t
                        doc_scores[doc_idx] = doc_scores.get(doc_idx, 0.0) + contrib
                # Apply document length normalization
                if length_norm != "none" and doc_scores:
                    for doc_idx in list(doc_scores.keys()):
                        doc_id = documents_df.index[doc_idx]
                        if length_norm == "sqrt_centers":
                            nch = doc_centers_hit.get(doc_id, 1)
                            norm_factor = max(np.sqrt(float(nch)), 1e-6)
                        else:
                            norm_factor = 1.0
                        doc_scores[doc_idx] /= norm_factor
                if doc_scores:
                    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
                    top_docs = [doc_idx for doc_idx, _ in sorted_docs[:top_k]]
                    top_indices.append(top_docs)
                else:
                    top_indices.append([])
            
            true_labels_list = []
            retrieved_ids_list = []
            for q_idx, q_id in enumerate(queries_df.index):
                true_labels_list.append(citation_mapping.get(q_id, []))
                retrieved_doc_ids = [documents_df.index[i] for i in top_indices[q_idx]]
                retrieved_ids_list.append(retrieved_doc_ids)
            
            results = {}
            for k in [10, 20, 50, 100]:
                results[f"recall@{k}"] = mean_recall_at_k(true_labels_list, retrieved_ids_list, k=k)
            for k in [10, 20, 50, 100]:
                results[f"ndcg@{k}"] = mean_ndcg_at_k(true_labels_list, retrieved_ids_list, k=k)
            results["mrr@10"] = mean_mrr_at_k(true_labels_list, retrieved_ids_list, k=10)
            results["map"] = mean_average_precision(true_labels_list, retrieved_ids_list, k=100)
            results["pres@100"] = mean_pres_at_k(true_labels_list, retrieved_ids_list, k=100)
            
            if mode == "abstract2abstract":
                print_metric_table(results, "Sparse Coverage: Abstract -> Abstract")
            else:
                print_metric_table(results, "Sparse Coverage: Claim -> All")
            
            print(f"\n✅ Task {mode} evaluation completed")
        
        print(f"\n✅ Sparse Coverage evaluation completed for all available tasks")


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