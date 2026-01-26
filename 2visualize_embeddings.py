"""
Interactive web application for visualizing patent document embeddings.

This Flask app provides an interactive web interface for:
1. Auto-detecting embeddings directories
2. Loading embeddings and metadata from saved output directories
3. Selecting sections (abstract, claim, invention) and units
4. Selecting documents to highlight
5. Visualizing embeddings with interactive 2D plots
6. Viewing tokenized results

Usage:
    python app.py --port 5000
"""

import os
import json
import numpy as np
import argparse
from typing import List, Dict, Tuple, Optional
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
from transformers import AutoTokenizer, AutoModel
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import glob
import re
import torch
import spacy
from utils import (
    ensure_section_tokens,
    load_corpus,
    process_doc_batch,
    DEVICE,
)


app = Flask(__name__)
CORS(app)

# Global cache for loaded data
_data_cache = {}

# Global cache for models and tokenizers
_model_cache = {}
_tokenizer_cache = {}
_nlp_cache = None

# Global cache for dimensionality reduction results
_reduction_cache = {}
_reduction_cache_size_limit = 10  # Maximum number of cached results


def scan_embeddings_directories(base_dir: str = ".") -> List[Dict[str, str]]:
    """
    Scan for embeddings directories matching the pattern embeddings_*.
    
    Args:
        base_dir: Base directory to scan (default: current directory)
    
    Returns:
        List of dicts with 'path' and 'name' keys
    """
    embeddings_dirs = []
    base_path = os.path.abspath(base_dir)
    
    # Look for directories matching embeddings_* pattern
    pattern = os.path.join(base_path, "embeddings_*")
    found_dirs = glob.glob(pattern)
    
    for dir_path in found_dirs:
        if os.path.isdir(dir_path):
            dir_name = os.path.basename(dir_path)
            # Check if it contains at least one section file
            has_embeddings = False
            for section in ['abstract', 'claim', 'invention']:
                for unit in ['spacy_token', 'spacy_sentence', 'doc', 'noun_chunk', 'encoder_token']:
                    npy_file = os.path.join(dir_path, f"{section}_{unit}.npy")
                    npz_file = os.path.join(dir_path, f"{section}_{unit}.npz")
                    if os.path.exists(npy_file) or os.path.exists(npz_file):
                        has_embeddings = True
                        break
                if has_embeddings:
                    break
            
            if has_embeddings:
                embeddings_dirs.append({
                    'path': dir_path,
                    'name': dir_name,
                    'relative_path': os.path.relpath(dir_path, base_path)
                })
    
    # Sort by name
    embeddings_dirs.sort(key=lambda x: x['name'])
    return embeddings_dirs


def parse_embedding_dir_name(embeddings_dir: str) -> Optional[Dict[str, str]]:
    """
    Parse embeddings directory name to extract model info.
    
    Directory format: embeddings_{model_name}_{unit}_{cls}_{layer}
    Example: embeddings_PatentMap-V0-SecPair-Claim_spacy_token_cls_second_last
    
    Returns:
        Dict with keys: model_name, unit, cls, layer, or None if parsing fails
    """
    dir_name = os.path.basename(embeddings_dir.rstrip('/'))
    
    # Remove 'embeddings_' prefix
    if not dir_name.startswith('embeddings_'):
        return None
    
    parts = dir_name[len('embeddings_'):].split('_')
    
    # Known unit types (ordered by specificity - longer names first)
    valid_units = ["spacy_sentence", "spacy_token", "encoder_token", "noun_chunk", "doc"]
    valid_layers = ["second_last", "last"]  # Order matters: check longer first
    valid_cls = ["cls", "nocls"]
    
    # Find unit, cls, and layer from the end
    unit = None
    cls = None
    layer = None
    
    # Check for layer (could be multi-word like "second_last")
    # Check longer names first
    for l in valid_layers:
        layer_parts = l.split('_')
        if len(parts) >= len(layer_parts):
            # Check if the last N parts match the layer
            if parts[-len(layer_parts):] == layer_parts:
                layer = l
                parts = parts[:-len(layer_parts)]
                break
    
    # Check for cls (should be after layer is removed)
    if len(parts) >= 1 and parts[-1] in valid_cls:
        cls = parts[-1]
        parts = parts[:-1]
    
    # Find unit (could be multi-word like "spacy_sentence")
    # Check longer names first
    for u in valid_units:
        unit_parts = u.split('_')
        if len(parts) >= len(unit_parts):
            # Check if the last N parts match the unit
            if parts[-len(unit_parts):] == unit_parts:
                unit = u
                parts = parts[:-len(unit_parts)]
                break
    
    # Remaining parts form the model name
    model_name = '_'.join(parts) if parts else None
    
    # Debug: print parsing results
    if not all([model_name, unit, cls, layer]):
        print(f"Parsing failed for: {dir_name}")
        print(f"  Remaining parts: {parts}")
        print(f"  Found - model_name: {model_name}, unit: {unit}, cls: {cls}, layer: {layer}")
    
    if model_name and unit and cls and layer:
        return {
            'model_name': model_name,
            'unit': unit,
            'cls': cls,
            'layer': layer
        }
    
    return None


def detect_unit_from_directory(embeddings_dir: str) -> Optional[str]:
    """
    Automatically detect unit type from embeddings directory name.
    
    Directory format: embeddings_{model_name}_{unit}_{cls}_{layer}
    Example: embeddings_PatentMap-V0-SecPair-Claim_spacy_token_cls_second_last
    
    Returns:
        Unit type string if detected, None otherwise
    """
    parsed = parse_embedding_dir_name(embeddings_dir)
    if parsed:
        return parsed['unit']
    
    # Fallback to old method
    dir_name = os.path.basename(embeddings_dir.rstrip('/'))
    
    # Known unit types (ordered by specificity - longer names first)
    valid_units = ["spacy_sentence", "spacy_token", "encoder_token", "noun_chunk", "doc"]
    
    # Try to find unit type in directory name
    for unit in valid_units:
        pattern = f"_{unit}_"
        if pattern in dir_name:
            return unit
        if dir_name.endswith(f"_{unit}"):
            return unit
    
    # If not found in directory name, try to detect from files in directory
    if os.path.exists(embeddings_dir):
        for file in os.listdir(embeddings_dir):
            if (file.endswith('.npy') or file.endswith('.npz')) and '_metadata' not in file:
                for unit in valid_units:
                    if f"_{unit}." in file or f"_{unit}_" in file:
                        print(f"Detected unit '{unit}' from file: {file}")
                        return unit
    
    return None


def get_model_path_from_name(model_name: str) -> str:
    """
    Infer model path from model name.
    
    Common mappings:
    - PatentMap-V0-SecPair-Claim -> ZoeYou/PatentMap-V0-SecPair-Claim
    - bert-for-patents -> anferico/bert-for-patents
    - paecter -> (may need special handling)
    
    Returns:
        Model path (HuggingFace ID or local path)
    """
    # Common model name to path mappings
    model_mappings = {
        'PatentMap-V0-SecPair-Claim': 'ZoeYou/PatentMap-V0-SecPair-Claim',
        'bert-for-patents': 'anferico/bert-for-patents',
        'paecter': 'anferico/bert-for-patents',  # Fallback, may need adjustment
    }
    
    # Check if it's already a full path/ID
    if '/' in model_name:
        return model_name
    
    # Check mappings
    if model_name in model_mappings:
        return model_mappings[model_name]
    
    # Default: try to use as-is (might be a local path or HuggingFace ID)
    return model_name


def load_embeddings_and_metadata(embeddings_dir: str, sections: List[str], unit: str) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Dict]]]:
    """
    Load embeddings and metadata for specified sections.
    
    Args:
        embeddings_dir: Directory containing the embeddings
        sections: List of sections to load (e.g., ['abstract', 'claim'])
        unit: Unit type (e.g., 'spacy_token', 'encoder_token')
    
    Returns:
        embeddings_by_section: Dict mapping section -> embeddings array
        metadata_by_section: Dict mapping section -> list of metadata dicts
    """
    cache_key = f"{embeddings_dir}_{'_'.join(sorted(sections))}_{unit}"
    
    if cache_key in _data_cache:
        return _data_cache[cache_key]
    
    embeddings_by_section = {}
    metadata_by_section = {}
    
    for section in sections:
        # Try to find embeddings file (.npy or .npz)
        npy_file = os.path.join(embeddings_dir, f"{section}_{unit}.npy")
        npz_file = os.path.join(embeddings_dir, f"{section}_{unit}.npz")
        
        if os.path.exists(npz_file):
            data = np.load(npz_file)
            embeddings = data['embeddings']
        elif os.path.exists(npy_file):
            embeddings = np.load(npy_file)
        else:
            continue
        
        embeddings_by_section[section] = embeddings
        
        # Load metadata
        metadata_file = os.path.join(embeddings_dir, f"{section}_{unit}_metadata.jsonl")
        if os.path.exists(metadata_file):
            metadata = []
            with open(metadata_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        compact = json.loads(line)
                        meta = {
                            'doc_id': compact.get('d', ''),
                            'section': compact.get('s', section),
                            'span_text': compact.get('t', ''),
                            'span_text_raw': compact.get('r', ''),
                            'unit': compact.get('u', unit)
                        }
                        metadata.append(meta)
                    except json.JSONDecodeError:
                        continue
            metadata_by_section[section] = metadata
        else:
            # Create dummy metadata
            metadata_by_section[section] = [
                {'doc_id': f'unknown_{i}', 'section': section, 'span_text': '', 'span_text_raw': '', 'unit': unit}
                for i in range(len(embeddings))
            ]
    
    _data_cache[cache_key] = (embeddings_by_section, metadata_by_section)
    return embeddings_by_section, metadata_by_section


def _get_embeddings_hash(embeddings: np.ndarray) -> str:
    """
    Generate a hash for embeddings array to use as cache key.
    Uses a sample of embeddings for efficiency.
    """
    import hashlib
    # Use a sample of embeddings for hashing (faster)
    sample_size = min(1000, len(embeddings))
    if sample_size > 0:
        sample_indices = np.linspace(0, len(embeddings) - 1, sample_size, dtype=int)
        sample = embeddings[sample_indices]
    else:
        sample = embeddings
    
    # Create hash from sample and shape
    hash_obj = hashlib.md5(sample.tobytes())
    hash_obj.update(f"{embeddings.shape}".encode())
    return hash_obj.hexdigest()


def normalize_embeddings(embeddings: np.ndarray, norm: str = 'l2') -> np.ndarray:
    """
    Normalize embeddings using layer normalization.
    
    Args:
        embeddings: Input embeddings array of shape (n_samples, n_features)
        norm: Normalization type ('l2' for L2 normalization, 'none' for no normalization)
    
    Returns:
        Normalized embeddings array
    """
    if norm == 'none' or norm is None:
        return embeddings
    
    if norm == 'l2':
        # L2 normalization: normalize each sample to unit length
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms
        
        # Verify normalization: check that norms are approximately 1.0
        normalized_norms = np.linalg.norm(normalized, axis=1)
        avg_norm = np.mean(normalized_norms)
        if not np.allclose(normalized_norms, 1.0, atol=1e-6):
            print(f"Warning: L2 normalization may not be correct. Average norm: {avg_norm:.6f}")
        
        return normalized
    
    raise ValueError(f"Unknown normalization type: {norm}. Use 'l2' or 'none'.")


def reduce_dimensions(embeddings: np.ndarray, method: str = 'umap', n_components: int = 2, 
                     random_state: int = 42, n_neighbors: int = 15, min_dist: float = 0.1,
                     use_cache: bool = True, whiten: bool = False, normalize: str = 'none') -> np.ndarray:
    """
    Reduce embeddings to 2D for visualization with optional caching and normalization.
    
    Args:
        embeddings: Input embeddings array
        method: Reduction method ('umap', 'tsne', or 'pca')
        n_components: Number of output dimensions (default: 2)
        random_state: Random seed for reproducibility (not used for PCA)
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        use_cache: Whether to use cache for results
        whiten: Whether to whiten PCA components (default: False)
        normalize: Normalization type before reduction ('l2' for L2 normalization, 'none' for no normalization)
    
    Returns:
        2D embeddings array
    """
    global _reduction_cache, _reduction_cache_size_limit
    
    # Apply normalization if requested
    original_embeddings = embeddings.copy()  # Keep original for comparison
    if normalize and normalize != 'none':
        embeddings = normalize_embeddings(embeddings, norm=normalize)
        # Verify normalization was applied
        sample_norm_before = np.linalg.norm(original_embeddings[0])
        sample_norm_after = np.linalg.norm(embeddings[0])
        print(f"Applied {normalize} normalization to embeddings before reduction")
        print(f"  Sample vector norm: {sample_norm_before:.4f} -> {sample_norm_after:.4f}")
    else:
        print(f"No normalization applied (normalize='{normalize}')")
    
    # Create cache key if caching is enabled
    if use_cache:
        # Use normalized embeddings for cache key if normalization is applied
        emb_hash = _get_embeddings_hash(embeddings)
        cache_key = f"{emb_hash}_{method}_{n_components}_{random_state}_{n_neighbors}_{min_dist}_{whiten}_{normalize}"
        
        # Check cache
        if cache_key in _reduction_cache:
            print(f"Using cached reduction result for {method} (normalize={normalize})")
            return _reduction_cache[cache_key]
        else:
            print(f"Cache miss for {method} (normalize={normalize}), computing new result...")
    
    # Compute reduction
    if method == 'pca':
        from sklearn.decomposition import PCA
        
        # PCA is deterministic and fast, good for large datasets
        # Note: PCA doesn't use random_state, but we include it in cache key for consistency
        reducer = PCA(
            n_components=n_components,
            whiten=whiten,
            random_state=random_state  # For consistency, though PCA is deterministic
        )
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Print variance explained
        explained_variance = reducer.explained_variance_ratio_
        total_variance = explained_variance.sum()
        print(f"PCA: Explained variance ratio: {explained_variance} (total: {total_variance:.2%})")
        
        # Print statistics about input embeddings (help debug normalization)
        if normalize and normalize != 'none':
            input_norms = np.linalg.norm(embeddings, axis=1)
            print(f"  Input embeddings stats (after {normalize} normalization):")
            print(f"    Mean norm: {np.mean(input_norms):.6f}, Std: {np.std(input_norms):.6f}")
            print(f"    Min norm: {np.min(input_norms):.6f}, Max norm: {np.max(input_norms):.6f}")
        else:
            input_norms = np.linalg.norm(embeddings, axis=1)
            print(f"  Input embeddings stats (no normalization):")
            print(f"    Mean norm: {np.mean(input_norms):.6f}, Std: {np.std(input_norms):.6f}")
            print(f"    Min norm: {np.min(input_norms):.6f}, Max norm: {np.max(input_norms):.6f}")
        
        # Store in cache if enabled
        if use_cache:
            if len(_reduction_cache) >= _reduction_cache_size_limit:
                # Remove oldest entry (simple FIFO - remove first key)
                oldest_key = next(iter(_reduction_cache))
                del _reduction_cache[oldest_key]
            _reduction_cache[cache_key] = embeddings_2d
            print(f"Cached reduction result for {method}")
        
        return embeddings_2d
    
    if method == 'umap':
        if not UMAP_AVAILABLE:
            method = 'tsne'
        else:
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
                verbose=False
            )
            embeddings_2d = reducer.fit_transform(embeddings)
            
            # Store in cache if enabled
            if use_cache:
                if len(_reduction_cache) >= _reduction_cache_size_limit:
                    # Remove oldest entry (simple FIFO - remove first key)
                    oldest_key = next(iter(_reduction_cache))
                    del _reduction_cache[oldest_key]
                _reduction_cache[cache_key] = embeddings_2d
                print(f"Cached reduction result for {method}")
            
            return embeddings_2d
    
    if method == 'tsne':
        # For large datasets, use PCA first to speed up t-SNE
        if len(embeddings) > 10000:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50, random_state=random_state)
            embeddings_pca = pca.fit_transform(embeddings)
        else:
            embeddings_pca = embeddings
        
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=min(30, len(embeddings) - 1),
            n_iter=1000,
            verbose=0
        )
        embeddings_2d = reducer.fit_transform(embeddings_pca)
        
        # Store in cache if enabled
        if use_cache:
            if len(_reduction_cache) >= _reduction_cache_size_limit:
                # Remove oldest entry
                oldest_key = next(iter(_reduction_cache))
                del _reduction_cache[oldest_key]
            _reduction_cache[cache_key] = embeddings_2d
            print(f"Cached reduction result for {method}")
        
        return embeddings_2d
    
    raise ValueError(f"Unknown reduction method: {method}")


def find_embeddings_for_doc(metadata: List[Dict], embeddings: np.ndarray, doc_id: str) -> Tuple[List[int], List[Dict]]:
    """Find all embeddings and metadata entries for a given document."""
    indices = []
    selected_metadata = []
    
    for i, meta in enumerate(metadata):
        if meta.get('doc_id') == doc_id:
            indices.append(i)
            selected_metadata.append(meta)
    
    return indices, selected_metadata


def find_all_doc_embeddings(metadata: List[Dict], embeddings: np.ndarray, doc_ids: List[str]) -> Tuple[List[int], List[Dict]]:
    """
    Find all embeddings and metadata entries for multiple documents.
    
    Args:
        metadata: Full metadata list
        embeddings: Full embeddings array (for shape validation)
        doc_ids: List of document IDs to find
    
    Returns:
        (indices, selected_metadata)
    """
    indices = []
    selected_metadata = []
    
    for doc_id in doc_ids:
        for i, meta in enumerate(metadata):
            if meta.get('doc_id') == doc_id:
                indices.append(i)
                selected_metadata.append(meta)
    
    return indices, selected_metadata


def combine_section_embeddings(
    embeddings_by_section: Dict[str, np.ndarray],
    metadata_by_section: Dict[str, List[Dict]],
    sections: List[str]
) -> Tuple[np.ndarray, List[Dict], Dict[str, int]]:
    """
    Combine embeddings and metadata from multiple sections.
    
    Args:
        embeddings_by_section: Dict mapping section -> embeddings array
        metadata_by_section: Dict mapping section -> list of metadata dicts
        sections: List of sections to combine
    
    Returns:
        (combined_embeddings, combined_metadata, section_offsets)
        section_offsets: Dict mapping section -> starting index in combined array
    """
    all_embeddings = []
    all_metadata = []
    section_offsets = {}
    
    for section in sections:
        if section in embeddings_by_section:
            section_offsets[section] = len(all_embeddings)
            all_embeddings.append(embeddings_by_section[section])
            all_metadata.extend(metadata_by_section[section])
    
    if not all_embeddings:
        raise ValueError('No embeddings found in selected sections')
    
    return np.vstack(all_embeddings), all_metadata, section_offsets


def subsample_embeddings_preserve_selected(
    embeddings: np.ndarray,
    metadata: List[Dict],
    selected_indices: set,
    max_embeddings: int,
    random_seed: int = 42
) -> Tuple[np.ndarray, List[Dict], Dict[int, int]]:
    """
    Subsample embeddings while preserving selected indices.
    
    Args:
        embeddings: Full embeddings array
        metadata: Full metadata list
        selected_indices: Set of indices that must be preserved
        max_embeddings: Maximum number of embeddings to keep
        random_seed: Random seed for reproducibility
    
    Returns:
        (subsampled_embeddings, subsampled_metadata, index_mapping)
        index_mapping: Maps old_index -> new_index
    """
    if len(embeddings) <= max_embeddings:
        # No subsampling needed
        index_mapping = {i: i for i in range(len(embeddings))}
        return embeddings, metadata, index_mapping
    
    num_selected = len(selected_indices)
    num_available = max_embeddings - num_selected
    
    if num_available < 0:
        # If selected items exceed max, keep only selected
        indices_to_keep = sorted(list(selected_indices))
    else:
        # Randomly sample from non-selected indices
        all_indices = set(range(len(embeddings)))
        non_selected_indices = list(all_indices - selected_indices)
        
        np.random.seed(random_seed)
        if len(non_selected_indices) > num_available:
            sampled = np.random.choice(
                non_selected_indices,
                size=num_available,
                replace=False
            ).tolist()
        else:
            sampled = non_selected_indices
        
        indices_to_keep = sorted(list(selected_indices) + sampled)
    
    # Create mapping from old indices to new indices
    index_mapping = {old: new for new, old in enumerate(indices_to_keep)}
    
    # Apply subsampling
    subsampled_embeddings = embeddings[indices_to_keep]
    subsampled_metadata = [metadata[i] for i in indices_to_keep]
    
    return subsampled_embeddings, subsampled_metadata, index_mapping


def update_indices_after_subsampling(
    original_indices: List[int],
    index_mapping: Dict[int, int]
) -> List[int]:
    """
    Update indices list after subsampling.
    
    Args:
        original_indices: Original indices before subsampling
        index_mapping: Mapping from old_index -> new_index
    
    Returns:
        Updated indices after subsampling
    """
    updated = [index_mapping[i] for i in original_indices if i in index_mapping]
    return updated


def parse_visualization_params(data: dict) -> dict:
    """
    Parse visualization parameters with backward compatibility.
    
    Args:
        data: Request JSON data
    
    Returns:
        Parsed parameters dict
    """
    # Handle sections parameter (support both 'section' and 'sections')
    sections = data.get('sections', None)
    if sections is None:
        section = data.get('section', 'abstract')
        sections = [section] if isinstance(section, str) else section
    
    if isinstance(sections, str):
        sections = [sections]
    
    return {
        'embeddings_dir': data.get('embeddings_dir'),
        'sections': sections,
        'unit': data.get('unit'),
        'reduction': data.get('reduction', 'umap'),
        'n_neighbors': data.get('n_neighbors', 15),
        'min_dist': data.get('min_dist', 0.1),
        'max_embeddings': data.get('max_embeddings', 50000),
        'whiten': data.get('whiten', False),  # PCA parameter: whether to whiten components
        'normalize': data.get('normalize', 'none'),  # Normalization type: 'l2' or 'none'
    }


def format_text_for_hover(text: str, max_line_length: int = 60) -> str:
    """
    Format text for hover tooltip with automatic line breaks.
    
    Inserts <br> tags at word boundaries to make text wrap in hover tooltips.
    Note: Plotly's hover template supports HTML, so <br> tags will be rendered.
    
    Args:
        text: Text to format
        max_line_length: Maximum characters per line before wrapping
    
    Returns:
        Formatted text with <br> tags for line breaks (HTML escaped except for <br>)
    """
    if not text:
        return ''
    
    # Escape HTML special characters first to prevent XSS
    import html
    text = html.escape(text)
    
    # Split into words
    words = text.split()
    if not words:
        return text
    
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        # Add 1 for space if not first word in line
        space_length = 1 if current_line else 0
        
        if current_length + space_length + word_length <= max_line_length:
            # Add to current line
            current_line.append(word)
            current_length += space_length + word_length
        else:
            # Start new line
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_length
    
    # Add remaining words
    if current_line:
        lines.append(' '.join(current_line))
    
    # Join with <br> tags - these will be rendered as HTML by Plotly
    # Note: We use <br> (not &lt;br&gt;) because Plotly's hover template
    # supports HTML rendering for customdata when used in hovertemplate
    return '<br>'.join(lines)


def create_plotly_visualization(embeddings_2d: np.ndarray, 
                                metadata: List[Dict],
                                selected_indices: Optional[List[int]] = None,
                                selected_metadata: Optional[List[Dict]] = None,
                                title: str = "Embeddings Visualization",
                                sections: Optional[List[str]] = None) -> dict:
    """
    Create interactive Plotly visualization of embeddings.
    
    Returns:
        JSON-serializable dict for Plotly
    """
    fig = go.Figure()
    
    # Plot all embeddings in gray with transparency
    fig.add_trace(go.Scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color='gray',
            opacity=0.3
        ),
        name='All embeddings',
        hovertemplate='<b>All Embeddings</b><br>' +
                      'X: %{x:.2f}<br>' +
                      'Y: %{y:.2f}<br>' +
                      '<extra></extra>',
        showlegend=True
    ))
    
    # Highlight selected embeddings with colors
    if selected_indices and len(selected_indices) > 0:
        selected_embeddings = embeddings_2d[selected_indices]
        
        # Use different colors for different documents
        colors = px.colors.qualitative.Set3
        
        # Map section to symbol shape
        section_symbols = {
            'abstract': 'circle',
            'claim': 'square',
            'invention': 'diamond'
        }
        
        # Group by document and section
        doc_section_groups = {}
        cls_indices = []
        regular_indices_by_doc_section = {}
        
        for idx, i in enumerate(selected_indices):
            meta = selected_metadata[idx] if selected_metadata else metadata[i]
            span_text_raw = meta.get('span_text_raw', '')
            doc_id = meta.get('doc_id', 'unknown')
            section = meta.get('section', 'unknown')
            
            # Check if this is a CLS token
            if span_text_raw == '[CLS]':
                cls_indices.append((idx, doc_id, section))
            else:
                if doc_id not in regular_indices_by_doc_section:
                    regular_indices_by_doc_section[doc_id] = {}
                if section not in regular_indices_by_doc_section[doc_id]:
                    regular_indices_by_doc_section[doc_id][section] = []
                regular_indices_by_doc_section[doc_id][section].append((idx, meta))
        
        # Plot regular tokens grouped by document and section
        doc_colors = {}
        color_idx = 0
        
        for doc_id, sections_dict in regular_indices_by_doc_section.items():
            # Assign color to document
            if doc_id not in doc_colors:
                doc_colors[doc_id] = colors[color_idx % len(colors)]
                color_idx += 1
            doc_color = doc_colors[doc_id]
            
            # Plot each section with different shape
            for section, token_list in sections_dict.items():
                symbol = section_symbols.get(section, 'circle')
                section_x = []
                section_y = []
                section_hover_data = []
                
                for idx, meta in token_list:
                    emb_2d = selected_embeddings[idx]
                    section_x.append(emb_2d[0])
                    section_y.append(emb_2d[1])
                    # Get full text
                    span_text_full = meta.get('span_text_raw', meta.get('span_text', ''))
                    # Determine truncation based on unit type
                    unit_type = meta.get('unit', 'unknown')
                    if unit_type == 'spacy_sentence':
                        # For sentences, show full text (or up to 500 chars for very long sentences)
                        span_text = span_text_full[:500] if len(span_text_full) > 500 else span_text_full
                        if len(span_text_full) > 500:
                            span_text += '...'
                        # Format with line breaks (longer lines for sentences)
                        span_text = format_text_for_hover(span_text, max_line_length=70)
                    elif unit_type == 'doc':
                        # For doc-level, show up to 300 chars
                        span_text = span_text_full[:300] if len(span_text_full) > 300 else span_text_full
                        if len(span_text_full) > 300:
                            span_text += '...'
                        # Format with line breaks
                        span_text = format_text_for_hover(span_text, max_line_length=65)
                    else:
                        # For tokens/noun_chunks, keep shorter truncation (50 chars)
                        span_text = span_text_full[:50] if len(span_text_full) > 50 else span_text_full
                        # Format with line breaks (shorter lines for tokens)
                        span_text = format_text_for_hover(span_text, max_line_length=50)
                    section_hover_data.append({
                        'doc_id': doc_id,
                        'section': section,
                        'text': span_text
                    })
                
                if section_x:  # Only plot if there are tokens
                    # Prepare hover text with HTML formatting for each point
                    hovertext_list = []
                    for idx, hd in enumerate(section_hover_data):
                        # Build hover text with HTML line breaks in the text field
                        hover_text = (f"<b>Token</b><br>"
                                     f"Doc ID: {hd['doc_id']}<br>"
                                     f"Section: {hd['section']}<br>"
                                     f"Text: {hd['text']}<br>"
                                     f"X: {section_x[idx]:.2f}<br>"
                                     f"Y: {section_y[idx]:.2f}")
                        hovertext_list.append(hover_text)
                    
                    fig.add_trace(go.Scatter(
                        x=section_x,
                        y=section_y,
                        mode='markers',
                        marker=dict(
                            size=8,  # Smaller size for regular tokens
                            symbol=symbol,
                            color=doc_color,
                            opacity=0.8,
                            line=dict(width=1.5, color='black')
                        ),
                        name=f"{doc_id} - {section.upper()}",
                        hovertext=hovertext_list,  # Use hovertext for better HTML support
                        hovertemplate='%{hovertext}<extra></extra>',
                        # Enable HTML rendering in hover
                        hoverlabel=dict(
                            namelength=-1,
                            bgcolor='rgba(255, 255, 255, 0.95)',
                            bordercolor='rgba(0, 0, 0, 0.2)',
                            font_size=12
                        ),
                        showlegend=True
                    ))
        
        # Plot CLS tokens with star symbol, grouped by document and section
        if cls_indices:
            # Group CLS tokens by document and section
            cls_by_doc_section = {}
            for idx, doc_id, section in cls_indices:
                key = (doc_id, section)
                if key not in cls_by_doc_section:
                    cls_by_doc_section[key] = []
                cls_by_doc_section[key].append(idx)
            
            # Plot each document-section's CLS tokens
            for (doc_id, section), cls_idx_list in cls_by_doc_section.items():
                cls_x = [selected_embeddings[idx][0] for idx in cls_idx_list]
                cls_y = [selected_embeddings[idx][1] for idx in cls_idx_list]
                
                # Use document color if available, otherwise gold
                cls_color = doc_colors.get(doc_id, 'gold')
                
                fig.add_trace(go.Scatter(
                    x=cls_x,
                    y=cls_y,
                    mode='markers',
                    marker=dict(
                        size=18,  # Larger size for CLS tokens
                        symbol='star',  # Star symbol
                        color=cls_color,
                        opacity=0.9,
                        line=dict(width=2, color='darkorange')
                    ),
                    name=f'[CLS] {doc_id} - {section.upper()}',
                    hovertemplate='<b>[CLS] Token</b><br>' +
                                  f'Doc ID: {doc_id}<br>' +
                                  f'Section: {section}<br>' +
                                  'X: %{x:.2f}<br>' +
                                  'Y: %{y:.2f}<br>' +
                                  '<extra></extra>',
                    showlegend=True
                ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family="Arial, sans-serif")
        ),
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        hovermode='closest',
        template='plotly_white',
        width=900,
        height=700,
        # Add right margin to accommodate legend
        margin=dict(l=60, r=200, t=60, b=60),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,  # Slightly further right, but margin will accommodate it
            font=dict(size=10),
            # Allow legend to scroll if too long
            itemwidth=30,
            tracegroupgap=5
        )
    )
    
    return json.loads(json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder))


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/test', methods=['GET'])
def test():
    """Test endpoint to verify routing works."""
    return jsonify({'status': 'ok', 'message': 'API routing is working'})


@app.route('/api/scan_directories', methods=['GET', 'POST'])
def scan_directories():
    """Scan for embeddings directories."""
    print(f"Received {request.method} request to /api/scan_directories")
    
    # Handle both GET and POST requests
    if request.method == 'POST':
        data = request.json if request.is_json else {}
        base_dir = data.get('base_dir', '.')
    else:  # GET request
        base_dir = request.args.get('base_dir', '.')
    
    print(f"Scanning directories in: {base_dir}")
    
    try:
        directories = scan_embeddings_directories(base_dir)
        print(f"Found {len(directories)} directories")
        return jsonify({
            'directories': directories,
            'count': len(directories)
        })
    except Exception as e:
        import traceback
        print(f"Error scanning directories: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/detect_unit', methods=['POST'])
def detect_unit():
    """Detect unit type from embeddings directory name."""
    data = request.json
    embeddings_dir = data.get('embeddings_dir')
    
    if not embeddings_dir:
        return jsonify({'error': 'embeddings_dir is required'}), 400
    
    if not os.path.exists(embeddings_dir):
        return jsonify({'error': f'Directory not found: {embeddings_dir}'}), 404
    
    try:
        detected_unit = detect_unit_from_directory(embeddings_dir)
        if detected_unit:
            return jsonify({'unit': detected_unit, 'detected': True})
        else:
            return jsonify({'unit': 'spacy_token', 'detected': False, 'message': 'Could not auto-detect unit type, using default: spacy_token'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/load_embeddings', methods=['POST'])
def load_embeddings():
    """Load embeddings and return available sections and document IDs."""
    data = request.json
    embeddings_dir = data.get('embeddings_dir')
    sections = data.get('sections', ['abstract'])
    unit = data.get('unit', None)  # Allow None to auto-detect
    
    if not embeddings_dir:
        return jsonify({'error': 'embeddings_dir is required'}), 400
    
    if not os.path.exists(embeddings_dir):
        return jsonify({'error': f'Directory not found: {embeddings_dir}'}), 404
    
    try:
        # Auto-detect unit if not provided
        if unit is None:
            detected_unit = detect_unit_from_directory(embeddings_dir)
            if detected_unit:
                unit = detected_unit
                print(f"Auto-detected unit type: {unit}")
            else:
                unit = 'spacy_token'  # Default fallback
                print(f"Could not auto-detect unit type, using default: {unit}")
        
        embeddings_by_section, metadata_by_section = load_embeddings_and_metadata(
            embeddings_dir, sections, unit
        )
        
        if not embeddings_by_section:
            return jsonify({'error': 'No embeddings found'}), 404
        
        # Get unique document IDs for each section
        doc_ids_by_section = {}
        for section, metadata in metadata_by_section.items():
            doc_ids = sorted(list(set(m.get('doc_id', '') for m in metadata if m.get('doc_id'))))
            doc_ids_by_section[section] = doc_ids
        
        # Get statistics
        stats = {}
        for section, embeddings in embeddings_by_section.items():
            stats[section] = {
                'count': len(embeddings),
                'dimension': int(embeddings.shape[1]) if len(embeddings.shape) > 1 else 0
            }
        
        # Debug: print loaded sections
        print(f"Loaded sections: {list(embeddings_by_section.keys())}")
        print(f"Stats keys: {list(stats.keys())}")
        
        return jsonify({
            'sections': list(embeddings_by_section.keys()),
            'doc_ids_by_section': doc_ids_by_section,
            'stats': stats,
            'unit': unit  # Return the unit type used
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualize', methods=['POST'])
def visualize():
    """Generate visualization for selected embeddings."""
    data = request.json
    
    if not data.get('embeddings_dir'):
        return jsonify({'error': 'embeddings_dir is required'}), 400
    
    # Parse parameters (with backward compatibility)
    params = parse_visualization_params(data)
    sections = params['sections']
    unit = params.get('unit') or data.get('unit', 'spacy_token')
    doc_ids = data.get('doc_ids', [])
    
    # Debug: print received sections
    print(f"Received sections for visualization: {sections}")
    
    try:
        # Load embeddings for all selected sections
        embeddings_by_section, metadata_by_section = load_embeddings_and_metadata(
            params['embeddings_dir'], sections, unit
        )
        
        if not embeddings_by_section:
            return jsonify({'error': 'No sections found'}), 404
        
        # Combine embeddings from all sections using helper function
        embeddings, metadata, section_offsets = combine_section_embeddings(
            embeddings_by_section, metadata_by_section, sections
        )
        
        # Find selected document indices BEFORE subsampling to ensure they're preserved
        selected_doc_indices = set()
        if doc_ids:
            selected_indices_list, _ = find_all_doc_embeddings(metadata, embeddings, doc_ids)
            selected_doc_indices = set(selected_indices_list)
        
        # Subsample if too large, but preserve selected document indices
        embeddings, metadata, index_mapping = subsample_embeddings_preserve_selected(
            embeddings, metadata, selected_doc_indices, params['max_embeddings']
        )
        
        if len(selected_doc_indices) > 0:
            original_count = len(embeddings_by_section[list(embeddings_by_section.keys())[0]])
            print(f"Subsampled from {original_count} to {len(embeddings)} embeddings")
            print(f"Preserved {len(selected_doc_indices)} embeddings from selected documents")
        
        # Reduce dimensions (with caching)
        embeddings_2d = reduce_dimensions(
            embeddings,
            method=params['reduction'],
            n_neighbors=params['n_neighbors'],
            min_dist=params['min_dist'],
            whiten=params.get('whiten', False),
            normalize=params.get('normalize', 'none'),
            use_cache=True
        )
        
        # Find selected embeddings (after subsampling, need to update indices)
        selected_indices = []
        selected_metadata_list = []
        
        if doc_ids:
            # Update indices after subsampling
            updated_selected_indices = update_indices_after_subsampling(
                list(selected_doc_indices), index_mapping
            )
            
            # Get metadata for selected indices
            for idx in updated_selected_indices:
                if idx < len(metadata):
                    selected_indices.append(idx)
                    selected_metadata_list.append(metadata[idx])
            
            print(f"Found {len(selected_indices)} embeddings to highlight for {len(doc_ids)} document(s)")
        
        # Create visualization
        sections_str = ', '.join([s.upper() for s in sections])
        title = f"Embeddings Visualization: {sections_str} Section(s)"
        if doc_ids:
            title += f"<br>Highlighted: {len(doc_ids)} document(s) with {len(selected_indices)} tokens"
        
        plot_data = create_plotly_visualization(
            embeddings_2d,
            metadata,
            selected_indices=selected_indices if selected_indices else None,
            selected_metadata=selected_metadata_list if selected_metadata_list else None,
            title=title,
            sections=sections  # Pass sections info for shape differentiation
        )
        
        return jsonify({
            'plot': plot_data,
            'selected_count': len(selected_indices)
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


def compute_query_embeddings(query_text: str, section: str, embeddings_dir: str, 
                            model_path: str = None, unit: str = None, 
                            keep_cls: bool = True, layer: str = "last", 
                            max_length: int = 512) -> Tuple[np.ndarray, List[Dict]]:
    """
    Compute embeddings for a query text on-the-fly.
    
    Args:
        query_text: The query text to embed
        section: Section type ('abstract', 'claim', 'invention')
        embeddings_dir: Embeddings directory (used to infer parameters if not provided)
        model_path: Model path (inferred from embeddings_dir if not provided)
        unit: Unit type (inferred from embeddings_dir if not provided)
        keep_cls: Whether to keep CLS token
        layer: Which layer to use ('last' or 'second_last')
        max_length: Maximum sequence length
    
    Returns:
        embeddings: np.array of shape [N, hidden_dim]
        metadata: List of metadata dicts
    """
    # Parse embeddings_dir to get parameters if not provided
    if not model_path or not unit:
        parsed = parse_embedding_dir_name(embeddings_dir)
        if parsed:
            if not model_path:
                model_path = get_model_path_from_name(parsed['model_name'])
            if not unit:
                unit = parsed['unit']
            if not layer:
                layer = parsed['layer']
            if keep_cls is None:
                keep_cls = (parsed['cls'] == 'cls')
    
    if not model_path or not unit:
        raise ValueError("Could not infer model_path and unit from embeddings_dir")
    
    # Load or get cached model and tokenizer
    cache_key = f"{model_path}_{layer}"
    if cache_key not in _model_cache:
        print(f"Loading model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        ensure_section_tokens(tokenizer, model)
        model.to(DEVICE)
        model.eval()
        _model_cache[cache_key] = model
        _tokenizer_cache[cache_key] = tokenizer
    else:
        model = _model_cache[cache_key]
        tokenizer = _tokenizer_cache[cache_key]
    
    # Query text is already formatted in the calling function, so use as-is
    formatted_text = query_text
    
    # Load spaCy if needed and set it in utils module
    global _nlp_cache
    if unit != "encoder_token":
        if _nlp_cache is None:
            print("Loading spaCy model...")
            _nlp_cache = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
            _nlp_cache.max_length = 900000 + 10000
        # Set in utils module so process_doc_batch can use it
        import utils
        utils.NLP = _nlp_cache
    
    # Process the query (single document)
    doc_ids = ['QUERY']
    sections_list = [section]
    doc_texts = [formatted_text]
    
    # Use process_doc_batch to compute embeddings
    batch_results = process_doc_batch(
        doc_texts=doc_texts,
        doc_ids=doc_ids,
        sections=sections_list,
        unit=unit,
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        max_length=max_length,
        keep_cls=keep_cls,
        layer=layer
    )
    
    # Extract embeddings and metadata
    embeddings_list = []
    metadata_list = []
    
    for doc_id, sec, doc_text, span_text_raw, span_text_canonical, span_emb in batch_results:
        embeddings_list.append(span_emb)
        metadata_list.append({
            'doc_id': doc_id,
            'section': sec,
            'span_text_raw': span_text_raw,
            'span_text': span_text_canonical,
            'unit': unit
        })
    
    if not embeddings_list:
        raise ValueError("No embeddings were generated for the query")
    
    embeddings = np.vstack(embeddings_list)
    return embeddings, metadata_list


def load_citation_mapping(mapping_file: str) -> Dict[str, List[str]]:
    """
    Load citation mapping from gold.json file.
    
    Args:
        mapping_file: Path to gold.json file
    
    Returns:
        Dict mapping query_id -> list of cited document IDs
    """
    with open(mapping_file, 'r') as f:
        raw_citations = json.load(f)
    
    # Convert to simple mapping: query_id -> [cited_doc_ids]
    citation_mapping = {}
    for query_id, cited_list in raw_citations.items():
        cited_doc_ids = []
        for cited_info in cited_list:
            cited_id = cited_info.get('cited_id', '')
            if cited_id:
                cited_doc_ids.append(cited_id)
        citation_mapping[query_id] = cited_doc_ids
    
    return citation_mapping


@app.route('/api/load_queries', methods=['POST'])
def load_queries():
    """Load queries from queries.json file."""
    data = request.json
    queries_file = data.get('queries_file', './downstream/perf200/content/queries.json')
    
    if not os.path.exists(queries_file):
        return jsonify({'error': f'Queries file not found: {queries_file}'}), 404
    
    try:
        queries = load_corpus(queries_file)
        query_ids = sorted(list(queries.keys()))
        
        # Get sections available for each query
        query_info = {}
        for qid in query_ids:
            query_info[qid] = {
                'id': qid,
                'sections': []
            }
            query_doc = queries[qid]
            if query_doc.get('abstract'):
                query_info[qid]['sections'].append('abstract')
            if query_doc.get('claim'):
                query_info[qid]['sections'].append('claim')
            if query_doc.get('invention'):
                query_info[qid]['sections'].append('invention')
        
        return jsonify({
            'queries': query_info,
            'query_ids': query_ids,
            'count': len(query_ids)
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/load_citations', methods=['POST'])
def load_citations():
    """Load citation mapping from gold.json file."""
    data = request.json
    mapping_file = data.get('mapping_file', './downstream/perf200/mapping/gold.json')
    
    if not os.path.exists(mapping_file):
        return jsonify({'error': f'Mapping file not found: {mapping_file}'}), 404
    
    try:
        citation_mapping = load_citation_mapping(mapping_file)
        return jsonify({
            'citations': citation_mapping,
            'count': len(citation_mapping)
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/visualize_query_citations', methods=['POST'])
def visualize_query_citations():
    """Visualize query and its cited documents."""
    data = request.json
    embeddings_dir = data.get('embeddings_dir')
    query_id = data.get('query_id')
    queries_file = data.get('queries_file', './downstream/perf200/content/queries.json')
    mapping_file = data.get('mapping_file', './downstream/perf200/mapping/gold.json')
    
    if not embeddings_dir or not query_id:
        return jsonify({'error': 'embeddings_dir and query_id are required'}), 400
    
    # Parse visualization parameters (with backward compatibility)
    params = parse_visualization_params(data)
    sections = params['sections']
    
    if not sections:
        return jsonify({'error': 'At least one section must be specified'}), 400
    
    # Debug: print received sections
    print(f"Received sections for query-citations visualization: {sections}")
    
    try:
        # Parse embeddings_dir to get parameters
        parsed = parse_embedding_dir_name(embeddings_dir)
        if not parsed:
            dir_name = os.path.basename(embeddings_dir.rstrip('/'))
            return jsonify({
                'error': f'Could not parse embeddings directory name: {dir_name}',
                'hint': 'Expected format: embeddings_{model_name}_{unit}_{cls}_{layer}'
            }), 400
        
        unit = parsed['unit']
        keep_cls = (parsed['cls'] == 'cls')
        layer = parsed['layer']
        
        # Load queries
        queries = load_corpus(queries_file)
        if query_id not in queries:
            return jsonify({'error': f'Query {query_id} not found'}), 404
        
        query_doc = queries[query_id]
        
        # Helper function to format query text for a section
        def format_query_text_for_section(section: str) -> str:
            """Format query text for a given section."""
            if section == 'abstract':
                title = query_doc.get('title', '').strip()
                abstract = query_doc.get('abstract', '').strip()
                if not abstract:
                    raise ValueError(f'Query {query_id} has no {section} section')
                return f"{title} [SEP] [abstract] {abstract}".strip() if title else f"[abstract] {abstract}".strip()
            elif section == 'claim':
                query_text = query_doc.get('claim', '').strip()
                if not query_text:
                    raise ValueError(f'Query {query_id} has no {section} section')
                return f"[claim] {query_text}".strip()
            elif section == 'invention':
                query_text = query_doc.get('invention', '').strip()
                if not query_text:
                    raise ValueError(f'Query {query_id} has no {section} section')
                return f"[invention] {query_text}".strip()
            else:
                raise ValueError(f'Invalid section: {section}')
        
        # Load citation mapping
        citation_mapping = load_citation_mapping(mapping_file)
        cited_doc_ids = citation_mapping.get(query_id, [])
        
        if not cited_doc_ids:
            return jsonify({'error': f'Query {query_id} has no cited documents'}), 404
        
        print(f"Computing embeddings for query {query_id} (sections: {sections})")
        print(f"Found {len(cited_doc_ids)} cited documents")
        
        # Compute query embeddings for each section
        all_query_embeddings = []
        all_query_metadata = []
        available_sections = []
        
        for section in sections:
            try:
                query_text = format_query_text_for_section(section)
                query_embeddings, query_metadata = compute_query_embeddings(
                    query_text=query_text,
                    section=section,
                    embeddings_dir=embeddings_dir,
                    unit=unit,
                    keep_cls=keep_cls,
                    layer=layer
                )
                all_query_embeddings.append(query_embeddings)
                all_query_metadata.extend(query_metadata)
                available_sections.append(section)
            except ValueError as e:
                # Skip sections that don't exist for this query
                print(f"Warning: {str(e)}, skipping section {section}")
                continue
        
        if not available_sections:
            return jsonify({'error': f'Query {query_id} has no valid sections from {sections}'}), 404
        
        # Update sections to only include available ones
        sections = available_sections
        
        # Combine query embeddings from all sections
        if all_query_embeddings:
            query_embeddings = np.vstack(all_query_embeddings)
        else:
            return jsonify({'error': 'No query embeddings were generated'}), 500
        
        # Load document embeddings for ALL selected sections (ALL documents from the dataset)
        embeddings_by_section, metadata_by_section = load_embeddings_and_metadata(
            embeddings_dir, sections, unit
        )
        
        if not embeddings_by_section:
            return jsonify({'error': f'No sections found in embeddings'}), 404
        
        # Combine document embeddings from all sections using helper function
        doc_embeddings, doc_metadata, section_offsets = combine_section_embeddings(
            embeddings_by_section, metadata_by_section, sections
        )
        
        # Find embeddings for cited documents across all sections using helper function
        cited_indices_in_full_dataset, cited_metadata_list = find_all_doc_embeddings(
            doc_metadata, doc_embeddings, cited_doc_ids
        )
        
        if not cited_indices_in_full_dataset:
            return jsonify({'error': f'No embeddings found for cited documents'}), 404
        
        # Combine query embeddings with ALL document embeddings from the dataset
        # This ensures dimensionality reduction is computed on the full corpus for better context
        all_embeddings = np.vstack([query_embeddings, doc_embeddings])
        all_metadata = all_query_metadata + doc_metadata
        
        # Create indices for visualization
        # Query indices are at the beginning (0 to len(query_embeddings)-1)
        query_indices = list(range(len(query_embeddings)))
        # Cited document indices are offset by query length
        cited_indices_in_combined = [len(query_embeddings) + i for i in cited_indices_in_full_dataset]
        
        # Subsample if needed (preserve query and cited documents) using helper function
        selected_indices_set = set(query_indices + cited_indices_in_combined)
        all_embeddings, all_metadata, index_mapping = subsample_embeddings_preserve_selected(
            all_embeddings, all_metadata, selected_indices_set, params['max_embeddings']
        )
        
        # Update indices after subsampling
        query_indices = update_indices_after_subsampling(query_indices, index_mapping)
        cited_indices_in_combined = update_indices_after_subsampling(cited_indices_in_combined, index_mapping)
        
        print(f"Computing dimensionality reduction on {len(all_embeddings)} embeddings "
              f"(query: {len(query_indices)}, cited: {len(cited_indices_in_combined)}, "
              f"other documents: {len(all_embeddings) - len(query_indices) - len(cited_indices_in_combined)})")
        
        # Reduce dimensions on the full dataset (query + all documents) with caching
        embeddings_2d = reduce_dimensions(
            all_embeddings,
            method=params['reduction'],
            n_neighbors=params['n_neighbors'],
            min_dist=params['min_dist'],
            whiten=params.get('whiten', False),
            normalize=params.get('normalize', 'none'),
            use_cache=True
        )
        
        # Create visualization with query and cited documents highlighted
        selected_indices = query_indices + cited_indices_in_combined
        selected_metadata = [all_metadata[i] for i in selected_indices]
        
        # Create title with all sections
        sections_str = ', '.join([s.upper() for s in sections])
        title = f"Query {query_id} ({sections_str}) and {len(cited_doc_ids)} Cited Documents"
        
        plot_data = create_plotly_visualization(
            embeddings_2d,
            all_metadata,
            selected_indices=selected_indices,
            selected_metadata=selected_metadata,
            title=title,
            sections=sections  # Pass all sections for shape differentiation
        )
        
        return jsonify({
            'plot': plot_data,
            'query_count': len(query_indices),
            'cited_count': len(cited_indices_in_combined),
            'cited_doc_ids': cited_doc_ids,
            'sections': sections  # Return the sections that were actually used
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/tokenized_results', methods=['POST'])
def tokenized_results():
    """Get tokenized results for selected documents."""
    data = request.json
    embeddings_dir = data.get('embeddings_dir')
    section = data.get('section', 'abstract')
    unit = data.get('unit', 'spacy_token')
    doc_ids = data.get('doc_ids', [])
    model_path = data.get('model_path', 'ZoeYou/PatentMap-V0-SecPair-Claim')
    
    if not embeddings_dir or not doc_ids:
        return jsonify({'error': 'embeddings_dir and doc_ids are required'}), 400
    
    try:
        # Load metadata
        embeddings_by_section, metadata_by_section = load_embeddings_and_metadata(
            embeddings_dir, [section], unit
        )
        
        if section not in metadata_by_section:
            return jsonify({'error': f'Section {section} not found'}), 404
        
        metadata = metadata_by_section[section]
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        results = {}
        for doc_id in doc_ids:
            doc_metadata = [m for m in metadata if m.get('doc_id') == doc_id]
            
            if not doc_metadata:
                results[doc_id] = {'error': f'No metadata found for document: {doc_id}'}
                continue
            
            tokenized_units = []
            for i, meta in enumerate(doc_metadata, 1):
                span_raw = meta.get('span_text_raw', '')
                span_canonical = meta.get('span_text', '')
                unit_type = meta.get('unit', 'unknown')
                
                # Tokenize the raw span text
                tokens = tokenizer.tokenize(span_raw) if span_raw else []
                
                tokenized_units.append({
                    'index': i,
                    'unit_type': unit_type,
                    'raw_text': span_raw,
                    'canonical_text': span_canonical,
                    'tokens': tokens,
                    'token_count': len(tokens)
                })
            
            results[doc_id] = {
                'unit_count': len(tokenized_units),
                'units': tokenized_units
            }
        
        return jsonify(results)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactive embeddings visualization web app")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", 
                       help="Host to bind to. Use '0.0.0.0' for network access, '127.0.0.1' for SSH forwarding")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--remote", action="store_true", 
                       help="Shortcut: bind to 0.0.0.0 for network access (NOT recommended for SSH forwarding)")
    parser.add_argument("--ssh-forward", action="store_true",
                       help="Optimized for SSH port forwarding: bind to 127.0.0.1 (recommended for SSH)")
    
    args = parser.parse_args()
    
    # Priority: --ssh-forward > --remote > --host
    if args.ssh_forward:
        args.host = "127.0.0.1"
    elif args.remote:
        args.host = "0.0.0.0"
    
    # Get hostname for better SSH forwarding instructions
    import socket
    try:
        hostname = socket.gethostname()
        hostname_fqdn = socket.getfqdn()
    except:
        hostname = "unknown"
        hostname_fqdn = "unknown"
    
    print(f"\n{'='*70}")
    print(f"Starting interactive embeddings visualization server...")
    print(f"{'='*70}")
    
    if args.host == "0.0.0.0":
        print(f"\n🌐 Server bound to 0.0.0.0 (accessible from network)")
        print(f"   Local access: http://localhost:{args.port}")
        print(f"   Remote access: http://<node-ip>:{args.port}")
        print(f"\n📡 For SSH port forwarding:")
        print(f"   On your LOCAL machine, run:")
        if hostname != "unknown":
            print(f"   ssh -L {args.port}:localhost:{args.port} $(whoami)@{hostname}")
            print(f"   Or if on SLURM cluster:")
            print(f"   ssh -L {args.port}:{hostname}:{args.port} $(whoami)@<login-node>")
        else:
            print(f"   ssh -L {args.port}:localhost:{args.port} <username>@<cluster-address>")
        print(f"   Then open: http://localhost:{args.port}")
        print(f"\n⚠️  Note: For SSH forwarding, consider using --ssh-forward instead of --remote")
    else:
        print(f"\n🔒 Server bound to {args.host} (localhost only - perfect for SSH forwarding)")
        print(f"   Access at: http://localhost:{args.port}")
        print(f"\n📡 For SSH port forwarding:")
        print(f"   1. Keep this server running on the remote machine")
        print(f"   2. On your LOCAL machine, open a NEW terminal and run:")
        if hostname != "unknown":
            print(f"      Option A - Direct connection (if accessible):")
            print(f"      ssh -L {args.port}:localhost:{args.port} $(whoami)@{hostname}")
            print(f"\n      Option B - SLURM cluster (two-hop via login node):")
            print(f"      Method 1 - Using ProxyJump (recommended, one command):")
            print(f"      ssh -L {args.port}:localhost:{args.port} -J $(whoami)@<login-node> $(whoami)@{hostname}")
            print(f"\n      Method 2 - Two separate commands:")
            print(f"      Step 1: ssh -L {args.port}:localhost:{args.port} $(whoami)@<login-node>")
            print(f"      Step 2: (in another terminal or background) ssh -L {args.port}:localhost:{args.port} {hostname}")
            print(f"\n      Example for CLEPS cluster:")
            print(f"      ssh -L {args.port}:localhost:{args.port} -J yzuo@cleps yzuo@{hostname}")
        else:
            print(f"      ssh -L {args.port}:localhost:{args.port} <username>@<cluster-address>")
        print(f"\n   3. Open browser on LOCAL machine: http://localhost:{args.port}")
        print(f"\n💡 Current hostname: {hostname}")
        print(f"   If connection fails, check:")
        print(f"   - SSH port forwarding is active (check with: ps aux | grep ssh)")
        print(f"   - For SLURM: you need TWO hops (login node -> compute node)")
        print(f"   - Server is bound to 127.0.0.1, so forward to 'localhost', not hostname")
        print(f"   - Firewall is not blocking the port")
    
    print(f"{'='*70}\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
