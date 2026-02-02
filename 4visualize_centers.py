"""
Interactive web app to visualize center-building results.

- Scan centers_greedy_* directories
- Sort centers by: sphere size (n_points), radius r, overlap with other centers
- View original span text for embeddings that fall into each center's sphere

Usage:
    python 4visualize_centers.py --port 5001
"""

import os
import json
import glob
import argparse
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Reuse from 2visualize_embeddings
from importlib.util import spec_from_file_location, module_from_spec
_viz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2visualize_embeddings.py")
_spec = spec_from_file_location("viz_embeddings", _viz_path)
_viz = module_from_spec(_spec)
_spec.loader.exec_module(_viz)
load_embeddings_and_metadata = _viz.load_embeddings_and_metadata
parse_embedding_dir_name = _viz.parse_embedding_dir_name

app = Flask(__name__)
CORS(app)

_data_cache: Dict[str, Any] = {}


def scan_centers_directories(base_dir: str = ".") -> List[Dict[str, str]]:
    """Scan for centers_greedy_* directories that contain centers_greedy_r*.npy."""
    base_path = os.path.abspath(base_dir)
    pattern = os.path.join(base_path, "centers_greedy_*")
    found = glob.glob(pattern)
    out = []
    for path in found:
        if not os.path.isdir(path):
            continue
        npy_files = glob.glob(os.path.join(path, "centers_greedy_r*.npy"))
        if not npy_files:
            continue
        name = os.path.basename(path)
        out.append({
            "path": path,
            "name": name,
            "relative_path": os.path.relpath(path, base_path),
        })
    out.sort(key=lambda x: x["name"])
    return out


def parse_centers_dir_name(centers_dir: str) -> Optional[Dict[str, str]]:
    """
    Parse centers directory name to get mode and embeddings base name.
    Format: centers_greedy_{mode}_{model_name}_{unit}_{cls}_{layer}[_suffix]
    Returns: mode, model_short (for embeddings_dir = embeddings_ + model_short), unit, etc.
    """
    name = os.path.basename(centers_dir.rstrip("/"))
    if not name.startswith("centers_greedy_"):
        return None
    rest = name[len("centers_greedy_"):]
    # Mode
    if rest.startswith("abstract2abstract_"):
        mode = "abstract2abstract"
        model_short = rest[len("abstract2abstract_"):]
    elif rest.startswith("claim2all_"):
        mode = "claim2all"
        model_short = rest[len("claim2all_"):]
    else:
        return None
    # Strip known suffixes to get embeddings base (model_unit_cls_layer)
    for suffix in ["_soft_percenter", "_soft", "_percenter"]:
        if model_short.endswith(suffix):
            model_short = model_short[: -len(suffix)]
            break
    # Strip adaptive suffix so embeddings_dir = base (e.g. embeddings_Model_unit_cls_last)
    if "_adapt_" in model_short:
        idx = model_short.find("_adapt_")
        model_short = model_short[:idx]
    # embeddings_dir name (no path): embeddings_{model_unit_cls_layer}
    embeddings_basename = f"embeddings_{model_short}"
    parsed = parse_embedding_dir_name(embeddings_basename)
    if not parsed:
        return {"mode": mode, "embeddings_basename": embeddings_basename, "unit": None, "model_name": None, "layer": "last"}
    parsed["mode"] = mode
    parsed["embeddings_basename"] = embeddings_basename
    return parsed


def get_embeddings_dir_from_centers(centers_dir: str, search_dir: str = ".") -> Optional[str]:
    """Resolve embeddings directory path from a centers directory."""
    parsed = parse_centers_dir_name(centers_dir)
    if not parsed or "embeddings_basename" not in parsed:
        return None
    base = parsed["embeddings_basename"]
    # Prefer same parent as centers dir
    parent = os.path.dirname(os.path.abspath(centers_dir))
    path1 = os.path.join(parent, base)
    if os.path.isdir(path1):
        return path1
    path2 = os.path.join(os.path.abspath(search_dir), base)
    if os.path.isdir(path2):
        return path2
    return path2  # Return expected path even if missing (caller can error)


def load_centers_and_stats(centers_dir: str) -> Tuple[np.ndarray, dict]:
    """Load centers matrix and JSON stats from a centers directory."""
    npy_files = sorted(glob.glob(os.path.join(centers_dir, "centers_greedy_r*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"No centers_greedy_r*.npy in {centers_dir}")
    centers_path = npy_files[-1]  # Prefer latest if multiple
    centers = np.load(centers_path).astype(np.float32)
    json_path = centers_path.replace(".npy", ".json")
    stats = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            stats = json.load(f)
    return centers, stats


def load_embeddings_and_metadata_for_mode(
    embeddings_dir: str, mode: str, unit: str
) -> Tuple[np.ndarray, List[dict]]:
    """Load concatenated embeddings and metadata for the given mode (sections)."""
    if mode == "abstract2abstract":
        sections = ["abstract"]
    elif mode == "claim2all":
        sections = ["abstract", "claim", "invention"]
    else:
        sections = ["abstract"]
    embeddings_by_section, metadata_by_section = load_embeddings_and_metadata(
        embeddings_dir, sections, unit
    )
    if not embeddings_by_section:
        raise FileNotFoundError(f"No embeddings in {embeddings_dir} for sections {sections}")
    all_emb = []
    all_meta = []
    for sec in sections:
        if sec in embeddings_by_section:
            all_emb.append(embeddings_by_section[sec])
            all_meta.extend(metadata_by_section.get(sec, []))
    X = np.vstack(all_emb)
    return X, all_meta


def compute_center_stats(
    centers: np.ndarray,
    embeddings: np.ndarray,
    metadata: List[dict],
    stats: dict,
    compute_overlap: bool = False,
    max_centers_for_overlap: int = 2000,
) -> List[dict]:
    """
    For each center: n_points (sphere size), r (radius), optionally overlap.
    Returns list of {center_idx, n_points, r, overlap_ratio (if computed)}.
    """
    import faiss
    V, d = centers.shape
    N = embeddings.shape[0]
    r = float(stats.get("r", 0.0))
    sim_threshold = float(stats.get("sim_threshold", 1.0 - r))
    r_per_center = stats.get("r_per_center")
    if r_per_center is not None and len(r_per_center) == V:
        pass
    else:
        r_per_center = None

    # Normalize and build index on embeddings
    embeddings_norm = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(embeddings_norm)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings_norm)

    centers_norm = centers.astype(np.float32).copy()
    faiss.normalize_L2(centers_norm)

    result = []
    posting_lists: List[List[int]] = []  # center_idx -> list of point indices in sphere

    for c in range(V):
        sim_th = (1.0 - float(r_per_center[c])) if r_per_center is not None else sim_threshold
        lims, D, I = index.range_search(centers_norm[c : c + 1].astype(np.float32), sim_th)
        if len(lims) >= 2:
            start, end = int(lims[0]), int(lims[1])
            indices = [int(I[i]) for i in range(start, end)]
        else:
            indices = []
        n_points = len(indices)
        r_c = (1.0 - sim_th) if r_per_center is None else float(r_per_center[c])
        entry = {"center_idx": c, "n_points": n_points, "r": r_c}
        result.append(entry)
        if compute_overlap and c < max_centers_for_overlap:
            posting_lists.append(indices)
        else:
            posting_lists.append(indices if compute_overlap else [])

    if compute_overlap and posting_lists:
        # point_idx -> number of centers containing it (only for first max_centers_for_overlap centers)
        from collections import defaultdict
        point_count: Dict[int, int] = defaultdict(int)
        for indices in posting_lists:
            for idx in indices:
                point_count[idx] += 1
        for c, indices in enumerate(posting_lists):
            if not indices:
                result[c]["overlap_ratio"] = 0.0
                result[c]["overlap_count"] = 0
                continue
            overlap_count = sum(point_count[idx] - 1 for idx in indices)
            result[c]["overlap_count"] = overlap_count
            result[c]["overlap_ratio"] = overlap_count / max(len(indices), 1)
        # Centers beyond max_centers_for_overlap: no overlap computed
        for c in range(len(posting_lists), V):
            result[c]["overlap_ratio"] = None
            result[c]["overlap_count"] = None
        # Clear large lists to free memory
        for c in range(len(posting_lists)):
            posting_lists[c] = []

    return result


def get_spans_in_sphere(
    center_idx: int,
    centers: np.ndarray,
    embeddings: np.ndarray,
    metadata: List[dict],
    stats: dict,
    max_spans: int = 500,
) -> List[dict]:
    """Return list of {doc_id, section, span_text_raw, span_text, similarity} for points in center's sphere."""
    import faiss
    V, d = centers.shape
    r = float(stats.get("r", 0.0))
    sim_threshold = float(stats.get("sim_threshold", 1.0 - r))
    r_per_center = stats.get("r_per_center")
    sim_th = (1.0 - float(r_per_center[center_idx])) if r_per_center and len(r_per_center) > center_idx else sim_threshold

    embeddings_norm = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(embeddings_norm)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings_norm)
    centers_norm = centers.astype(np.float32).copy()
    faiss.normalize_L2(centers_norm)

    lims, D, I = index.range_search(centers_norm[center_idx : center_idx + 1].astype(np.float32), sim_th)
    if len(lims) < 2:
        return []
    start, end = int(lims[0]), int(lims[1])
    out = []
    for i in range(start, min(end, start + max_spans)):
        idx = int(I[i])
        sim = float(D[i])
        if idx >= len(metadata):
            continue
        m = metadata[idx]
        out.append({
            "doc_id": m.get("doc_id", ""),
            "section": m.get("section", ""),
            "span_text_raw": m.get("span_text_raw", "")[:500],
            "span_text": m.get("span_text", "")[:500],
            "similarity": round(sim, 4),
        })
    return out


# --------------- Flask routes ---------------

@app.route("/")
def index():
    return render_template("centers.html")


@app.route("/api/scan_centers", methods=["GET", "POST"])
def api_scan_centers():
    base_dir = request.json.get("base_dir", ".") if request.is_json else request.args.get("base_dir", ".")
    try:
        dirs = scan_centers_directories(base_dir)
        return jsonify({"directories": dirs, "count": len(dirs)})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/load_centers", methods=["POST"])
def api_load_centers():
    data = request.json or {}
    centers_dir = data.get("centers_dir")
    sort_by = data.get("sort_by", "n_points_desc")
    compute_overlap = data.get("compute_overlap", False)

    if not centers_dir or not os.path.isdir(centers_dir):
        return jsonify({"error": "centers_dir required and must exist"}), 400

    cache_key = f"{centers_dir}_{compute_overlap}"
    if cache_key in _data_cache:
        cached = dict(_data_cache[cache_key])
        center_list = list(cached.get("center_list", []))
        if sort_by == "n_points_desc":
            center_list = sorted(center_list, key=lambda x: (-x["n_points"], x["center_idx"]))
        elif sort_by == "n_points_asc":
            center_list = sorted(center_list, key=lambda x: (x["n_points"], x["center_idx"]))
        elif sort_by == "r_asc":
            center_list = sorted(center_list, key=lambda x: (x["r"], x["center_idx"]))
        elif sort_by == "r_desc":
            center_list = sorted(center_list, key=lambda x: (-x["r"], x["center_idx"]))
        elif sort_by == "overlap_desc" and center_list and center_list[0].get("overlap_ratio") is not None:
            center_list = sorted(center_list, key=lambda x: (-(x.get("overlap_ratio") or 0), x["center_idx"]))
        elif sort_by == "overlap_asc" and center_list and center_list[0].get("overlap_ratio") is not None:
            center_list = sorted(center_list, key=lambda x: (x.get("overlap_ratio") or 0, x["center_idx"]))
        elif sort_by == "center_idx":
            center_list = sorted(center_list, key=lambda x: x["center_idx"])
        cached["center_list"] = center_list[:5000]
        cached["sort_by"] = sort_by
        return jsonify(cached)

    try:
        parsed = parse_centers_dir_name(centers_dir)
        if not parsed:
            return jsonify({"error": f"Could not parse centers dir name: {os.path.basename(centers_dir)}"}), 400
        mode = parsed.get("mode", "abstract2abstract")
        unit = parsed.get("unit", "spacy_token")
        if not unit:
            return jsonify({"error": "Could not infer unit from centers dir name"}), 400

        embeddings_dir = get_embeddings_dir_from_centers(centers_dir, os.path.dirname(centers_dir))
        if not embeddings_dir or not os.path.isdir(embeddings_dir):
            return jsonify({"error": f"Embeddings dir not found: {embeddings_dir} (expected from centers dir)"}), 404

        centers, stats = load_centers_and_stats(centers_dir)
        embeddings, metadata = load_embeddings_and_metadata_for_mode(embeddings_dir, mode, unit)

        center_list = compute_center_stats(
            centers, embeddings, metadata, stats,
            compute_overlap=compute_overlap,
            max_centers_for_overlap=2000,
        )

        # Sort
        if sort_by == "n_points_desc":
            center_list.sort(key=lambda x: (-x["n_points"], x["center_idx"]))
        elif sort_by == "n_points_asc":
            center_list.sort(key=lambda x: (x["n_points"], x["center_idx"]))
        elif sort_by == "r_asc":
            center_list.sort(key=lambda x: (x["r"], x["center_idx"]))
        elif sort_by == "r_desc":
            center_list.sort(key=lambda x: (-x["r"], x["center_idx"]))
        elif sort_by == "overlap_desc" and center_list and "overlap_ratio" in center_list[0]:
            center_list.sort(key=lambda x: (-x.get("overlap_ratio", 0), x["center_idx"]))
        elif sort_by == "overlap_asc" and center_list and "overlap_ratio" in center_list[0]:
            center_list.sort(key=lambda x: (x.get("overlap_ratio", 0), x["center_idx"]))
        elif sort_by == "center_idx":
            center_list.sort(key=lambda x: x["center_idx"])

        # Cache with center_list in canonical order (center_idx) so we can re-sort by any sort_by
        center_list_by_idx = sorted(center_list, key=lambda x: x["center_idx"])
        payload = {
            "centers_dir": centers_dir,
            "mode": mode,
            "unit": unit,
            "n_centers": len(centers),
            "n_embeddings": len(embeddings),
            "stats": {k: v for k, v in stats.items() if k not in ["coverage_history", "r_per_center"]},
            "center_list": center_list[:5000],
            "sort_by": sort_by,
            "compute_overlap": compute_overlap,
        }
        _data_cache[cache_key] = {
            **payload,
            "center_list": center_list_by_idx[:5000],
        }
        return jsonify(payload)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/center_spans", methods=["POST"])
def api_center_spans():
    data = request.json or {}
    centers_dir = data.get("centers_dir")
    center_idx = data.get("center_idx")
    max_spans = data.get("max_spans", 500)

    if not centers_dir or center_idx is None:
        return jsonify({"error": "centers_dir and center_idx required"}), 400

    try:
        parsed = parse_centers_dir_name(centers_dir)
        mode = parsed.get("mode", "abstract2abstract")
        unit = parsed.get("unit", "spacy_token")
        embeddings_dir = get_embeddings_dir_from_centers(centers_dir, os.path.dirname(centers_dir))
        if not embeddings_dir or not os.path.isdir(embeddings_dir):
            return jsonify({"error": f"Embeddings dir not found: {embeddings_dir}"}), 404

        centers, stats = load_centers_and_stats(centers_dir)
        embeddings, metadata = load_embeddings_and_metadata_for_mode(embeddings_dir, mode, unit)

        if center_idx < 0 or center_idx >= len(centers):
            return jsonify({"error": f"center_idx must be in [0, {len(centers)-1}]"}), 400

        spans = get_spans_in_sphere(center_idx, centers, embeddings, metadata, stats, max_spans=max_spans)
        return jsonify({
            "centers_dir": centers_dir,
            "center_idx": center_idx,
            "spans": spans,
            "total_in_sphere": len(spans) if len(spans) < max_spans else f">{max_spans} (showing first {max_spans})",
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize center-building results")
    parser.add_argument("--port", type=int, default=5001, help="Port for the web app")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"Centers visualization: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
