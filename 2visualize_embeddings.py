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
from transformers import AutoTokenizer
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import glob


app = Flask(__name__)
CORS(app)

# Global cache for loaded data
_data_cache = {}


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


def detect_unit_from_directory(embeddings_dir: str) -> Optional[str]:
    """
    Automatically detect unit type from embeddings directory name.
    
    Directory format: embeddings_{model_name}_{unit}_{cls}_{layer}
    Example: embeddings_PatentMap-V0-SecPair-Claim_spacy_token_cls_second_last
    
    Returns:
        Unit type string if detected, None otherwise
    """
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


def reduce_dimensions(embeddings: np.ndarray, method: str = 'umap', n_components: int = 2, 
                     random_state: int = 42, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """
    Reduce embeddings to 2D for visualization.
    """
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
                    span_text = meta.get('span_text_raw', meta.get('span_text', ''))[:50]
                    section_hover_data.append({
                        'doc_id': doc_id,
                        'section': section,
                        'text': span_text
                    })
                
                if section_x:  # Only plot if there are tokens
                    # Prepare customdata as list of lists for Plotly
                    customdata_list = [[hd['doc_id'], hd['section'], hd['text']] for hd in section_hover_data]
                    
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
                        hovertemplate='<b>Token</b><br>' +
                                      'Doc ID: %{customdata[0]}<br>' +
                                      'Section: %{customdata[1]}<br>' +
                                      'Text: %{customdata[2]}<br>' +
                                      'X: %{x:.2f}<br>' +
                                      'Y: %{y:.2f}<br>' +
                                      '<extra></extra>',
                        customdata=customdata_list,
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
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            font=dict(size=10)
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
    embeddings_dir = data.get('embeddings_dir')
    sections = data.get('sections', ['abstract'])  # Now accepts multiple sections
    unit = data.get('unit', 'spacy_token')
    doc_ids = data.get('doc_ids', [])
    reduction = data.get('reduction', 'umap')
    n_neighbors = data.get('n_neighbors', 15)
    min_dist = data.get('min_dist', 0.1)
    max_embeddings = data.get('max_embeddings', 50000)
    
    if not embeddings_dir:
        return jsonify({'error': 'embeddings_dir is required'}), 400
    
    # Ensure sections is a list
    if isinstance(sections, str):
        sections = [sections]
    
    # Debug: print received sections
    print(f"Received sections for visualization: {sections}")
    
    try:
        # Load embeddings for all selected sections
        embeddings_by_section, metadata_by_section = load_embeddings_and_metadata(
            embeddings_dir, sections, unit
        )
        
        if not embeddings_by_section:
            return jsonify({'error': 'No sections found'}), 404
        
        # Combine embeddings from all sections, preserving section information
        all_embeddings = []
        all_metadata = []
        section_offsets = {}  # Track where each section starts in combined array
        
        for section in sections:
            if section in embeddings_by_section:
                section_embeddings = embeddings_by_section[section]
                section_metadata = metadata_by_section[section]
                
                section_offsets[section] = len(all_embeddings)
                all_embeddings.append(section_embeddings)
                all_metadata.extend(section_metadata)
        
        if not all_embeddings:
            return jsonify({'error': 'No embeddings found in selected sections'}), 404
        
        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)
        metadata = all_metadata
        
        # Find selected document indices BEFORE subsampling to ensure they're preserved
        selected_doc_indices = set()
        if doc_ids:
            for doc_id in doc_ids:
                for i, meta in enumerate(metadata):
                    if meta.get('doc_id') == doc_id:
                        selected_doc_indices.add(i)
        
        # Subsample if too large, but preserve selected document indices
        subsample_mapping = {}  # Maps old index -> new index
        if len(embeddings) > max_embeddings:
            # Calculate how many embeddings we need to reserve for selected documents
            num_selected = len(selected_doc_indices)
            num_available = max_embeddings - num_selected
            
            if num_available < 0:
                # If selected documents need more than max_embeddings, use all selected
                indices_to_keep = sorted(list(selected_doc_indices))
            else:
                # Randomly sample from non-selected indices
                all_indices = set(range(len(embeddings)))
                non_selected_indices = list(all_indices - selected_doc_indices)
                
                if len(non_selected_indices) > num_available:
                    sampled_indices = np.random.choice(
                        non_selected_indices, 
                        size=num_available, 
                        replace=False
                    ).tolist()
                else:
                    sampled_indices = non_selected_indices
                
                # Combine selected and sampled indices
                indices_to_keep = sorted(list(selected_doc_indices) + sampled_indices)
            
            # Create mapping from old indices to new indices
            for new_idx, old_idx in enumerate(indices_to_keep):
                subsample_mapping[old_idx] = new_idx
            
            # Apply subsampling
            embeddings = embeddings[indices_to_keep]
            metadata = [metadata[i] for i in indices_to_keep]
            print(f"Subsampled from {len(embeddings_by_section[list(embeddings_by_section.keys())[0]])} to {len(embeddings)} embeddings")
            print(f"Preserved {len(selected_doc_indices)} embeddings from selected documents")
        
        # Reduce dimensions
        embeddings_2d = reduce_dimensions(
            embeddings,
            method=reduction,
            n_neighbors=n_neighbors,
            min_dist=min_dist
        )
        
        # Find selected embeddings (after subsampling, indices are already correct)
        selected_indices = []
        selected_metadata_list = []
        
        if doc_ids:
            for doc_id in doc_ids:
                indices, meta_list = find_embeddings_for_doc(metadata, embeddings, doc_id)
                selected_indices.extend(indices)
                selected_metadata_list.extend(meta_list)
            
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
                       help="Host to bind to. Use '0.0.0.0' for remote access (e.g., SSH/cluster)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--remote", action="store_true", 
                       help="Shortcut: bind to 0.0.0.0 for remote access (useful for SSH/cluster)")
    
    args = parser.parse_args()
    
    # If --remote flag is set, override host
    if args.remote:
        args.host = "0.0.0.0"
    
    print(f"\n{'='*70}")
    print(f"Starting interactive embeddings visualization server...")
    print(f"{'='*70}")
    
    if args.host == "0.0.0.0":
        print(f"\n🌐 Server bound to 0.0.0.0 (accessible from network)")
        print(f"   Local access: http://localhost:{args.port}")
        print(f"   Remote access: http://<node-ip>:{args.port}")
        print(f"\n📡 For SSH port forwarding, run on your local machine:")
        print(f"   ssh -L {args.port}:localhost:{args.port} <username>@<cluster-address>")
        print(f"   Then open: http://localhost:{args.port}")
    else:
        print(f"\n🔒 Server bound to {args.host} (localhost only)")
        print(f"   Access at: http://localhost:{args.port}")
        print(f"\n💡 For remote access via SSH, use:")
        print(f"   1. Run with: python app.py --remote --port {args.port}")
        print(f"   2. On local machine: ssh -L {args.port}:localhost:{args.port} <username>@<cluster>")
        print(f"   3. Open browser: http://localhost:{args.port}")
    
    print(f"{'='*70}\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
