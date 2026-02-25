"""
CLEF-IP 2013 Claims-to-Passages (EN) data loading.

Provides:
- load_clefip_en_topics(clefip_root) -> query_ids, query_texts
- load_clefip_en_qrels(clefip_root) -> qrels: {topic_id: [(doc_id, xpath), ...]}
- load_clefip_passage_corpus(clefip_root, qrels, doc_collection_root=None) -> passage_ids, passage_texts

Passage ID format: "doc_id::xpath" for uniqueness.
Documents are resolved from: tfiles/, tfam-docs/, and (if provided) doc_collection_root.
"""

import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional

# Default subdir names under clefip_root
TOPICS_DIR = "02_topics"
TEST_DIR = "clef-ip-2013-clms-psg-TEST"
TFILES_DIR = "tfiles"
TFAM_DOCS_DIR = "tfam-docs"
QRELS_DIR = "2013-clef-ip-clsm-to-psg-qrels"
QRELS_EN_FILE = "2013-clef-ip-QRELS-EN-claims-to-passages.txt"
TEST_META_FILE = "clef-ip-2013-clms-psg-TEST.txt"


def _text_of_element(el: ET.Element) -> str:
    """Recursively collect all text content of an element."""
    parts = []
    if el.text:
        parts.append(el.text)
    for child in el:
        parts.append(_text_of_element(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def _xpath_to_element(root: ET.Element, xpath: str) -> Optional[ET.Element]:
    """
    Resolve a simple XPath to an element. Supports:
    /patent-document/abstract/p, /patent-document/description/p[19], /patent-document/claims/claim[1]
    Path is relative to root; root may be <patent-document>.
    """
    # Normalize: remove leading /patent-document/ if present
    path = xpath.strip()
    if path.startswith("/patent-document/"):
        path = path[len("/patent-document/"):]
    if path.startswith("/"):
        path = path[1:]
    if not path:
        return root
    steps = path.split("/")
    current = root
    for step in steps:
        if not step:
            continue
        # step might be "p[19]" or "claim[1]" or "claim"
        match = re.match(r"^(\w+)(?:\[(\d+)\])?$", step)
        if not match:
            return None
        tag, idx = match.group(1), match.group(2)
        # Handle namespaced tags: in our XML tags are like claim, claim-text, description
        children = list(current)
        # Match by tag local name (strip namespace if any)
        candidates = [c for c in children if (c.tag.split("}")[-1] if "}" in c.tag else c.tag) == tag]
        if not candidates:
            return None
        if idx is not None:
            i = int(idx) - 1  # 1-based in XPath
            if i < 0 or i >= len(candidates):
                return None
            current = candidates[i]
        else:
            current = candidates[0]
    return current


def _get_passage_text_from_xml(xml_path: str, xpath: str) -> Optional[str]:
    """Load XML file and return text at xpath, or None if not found."""
    if not os.path.isfile(xml_path):
        return None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        el = _xpath_to_element(root, xpath)
        if el is None:
            return None
        return _text_of_element(el)
    except Exception:
        return None


def _tag_local(el: ET.Element) -> str:
    """Return local tag name without namespace."""
    return el.tag.split("}")[-1] if "}" in el.tag else el.tag


def _collect_passages_from_root(root: ET.Element) -> List[Tuple[str, str]]:
    """
    Yield (xpath, text) for each passage in patent-document root.
    XPath format: /patent-document/abstract[1]/p[1], /patent-document/description/p[1], /patent-document/claims/claim[1], etc.
    """
    out = []
    for abs_i, abs_el in enumerate([c for c in root if _tag_local(c) == "abstract"], start=1):
        for p_j, p_el in enumerate([c for c in abs_el if _tag_local(c) == "p"], start=1):
            xpath = f"/patent-document/abstract[{abs_i}]/p[{p_j}]"
            text = _text_of_element(p_el)
            if text:
                out.append((xpath, text))
    for desc_el in [c for c in root if _tag_local(c) == "description"]:
        for p_j, p_el in enumerate([c for c in desc_el if _tag_local(c) == "p"], start=1):
            xpath = f"/patent-document/description/p[{p_j}]"
            text = _text_of_element(p_el)
            if text:
                out.append((xpath, text))
    for claims_el in [c for c in root if _tag_local(c) == "claims"]:
        for c_j, claim_el in enumerate([c for c in claims_el if _tag_local(c) == "claim"], start=1):
            xpath = f"/patent-document/claims/claim[{c_j}]"
            text = _text_of_element(claim_el)
            if text:
                out.append((xpath, text))
    return out


def _parse_test_txt(path: str) -> List[Dict[str, str]]:
    """Parse clef-ip-2013-clms-psg-TEST.txt into list of topic dicts (tid, tfile, tclaims, tfam-docs)."""
    topics = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    blocks = content.strip().split("\n\n")
    for block in blocks:
        d = {}
        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("<tid>"):
                d["tid"] = line.replace("<tid>", "").replace("</tid>", "").strip()
            elif line.startswith("<tfile>"):
                d["tfile"] = line.replace("<tfile>", "").replace("</tfile>", "").strip()
            elif line.startswith("<tclaims>"):
                raw = line.replace("<tclaims>", "").replace("</tclaims>", "").strip()
                d["tclaims"] = [x.strip() for x in raw.split() if x.strip()]
            elif line.startswith("<tfam-docs>"):
                raw = line.replace("<tfam-docs>", "").replace("</tfam-docs>", "").strip()
                d["tfam-docs"] = raw
        if d.get("tid") and d.get("tfile"):
            if "tclaims" not in d:
                d["tclaims"] = []
            topics.append(d)
    return topics


def load_clefip_en_topics(clefip_root: str) -> Tuple[List[str], List[str]]:
    """
    Load EN topic (query) IDs and their claim texts from tfiles.
    Returns (query_ids, query_texts). Only topics that have at least one claim text are included.
    """
    test_dir = os.path.join(clefip_root, TOPICS_DIR, TEST_DIR)
    meta_path = os.path.join(test_dir, TEST_META_FILE)
    tfiles_dir = os.path.join(test_dir, TFILES_DIR)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"CLEF-IP topics meta not found: {meta_path}")
    topics_meta = _parse_test_txt(meta_path)
    query_ids = []
    query_texts = []
    for t in topics_meta:
        tid = t["tid"]
        tfile = t["tfile"]
        tclaims = t.get("tclaims") or []
        if not tclaims:
            continue
        xml_path = os.path.join(tfiles_dir, tfile)
        if not os.path.isfile(xml_path):
            continue
        texts = []
        for xpath in tclaims:
            text = _get_passage_text_from_xml(xml_path, xpath)
            if text:
                texts.append(text)
        if not texts:
            continue
        query_ids.append(tid)
        query_texts.append(" ".join(texts))
    return query_ids, query_texts


def load_clefip_en_qrels(clefip_root: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Load EN qrels: topic_id -> list of (doc_id, xpath).
    """
    qrels_path = os.path.join(clefip_root, TOPICS_DIR, QRELS_DIR, QRELS_EN_FILE)
    if not os.path.isfile(qrels_path):
        raise FileNotFoundError(f"CLEF-IP EN qrels not found: {qrels_path}")
    qrels = {}
    with open(qrels_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            topic_id = parts[0].strip()
            doc_id = parts[1].strip()
            xpath = parts[2].strip()
            qrels.setdefault(topic_id, []).append((doc_id, xpath))
    return qrels


def _resolve_doc_path(doc_id: str, clefip_root: str, doc_collection_root: Optional[str]) -> Optional[str]:
    """Return path to XML file for doc_id, or None if not found."""
    # 1) tfiles: doc_id + .xml
    tfiles_dir = os.path.join(clefip_root, TOPICS_DIR, TEST_DIR, TFILES_DIR)
    p = os.path.join(tfiles_dir, doc_id + ".xml")
    if os.path.isfile(p):
        return p
    # 2) tfam-docs
    tfam_dir = os.path.join(clefip_root, TOPICS_DIR, TEST_DIR, TFAM_DOCS_DIR)
    p = os.path.join(tfam_dir, doc_id + ".xml")
    if os.path.isfile(p):
        return p
    # 3) doc_collection_root (extracted 01): try subdirs then flat
    if doc_collection_root and os.path.isdir(doc_collection_root):
        for sub in ("ep0", "ep1", "wo", "ep", "clef-ip-2012-ep0", "clef-ip-2012-ep1", "clef-ip-2012-wo"):
            d = os.path.join(doc_collection_root, sub)
            if os.path.isdir(d):
                p = os.path.join(d, doc_id + ".xml")
                if os.path.isfile(p):
                    return p
        for sub in os.listdir(doc_collection_root):
            d = os.path.join(doc_collection_root, sub)
            if os.path.isdir(d):
                p = os.path.join(d, doc_id + ".xml")
                if os.path.isfile(p):
                    return p
        p = os.path.join(doc_collection_root, doc_id + ".xml")
        if os.path.isfile(p):
            return p
    return None


def load_clefip_passage_corpus(
    clefip_root: str,
    qrels: Dict[str, List[Tuple[str, str]]],
    doc_collection_root: Optional[str] = None,
    only_queries_in_qrels: bool = True,
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """
    Build passage corpus from qrels: unique (doc_id, xpath) and resolve text from XMLs.
    Returns (passage_ids, passage_texts, qrels_passage_ids).
    - passage_ids: list of "doc_id::xpath"
    - passage_texts: list of text for each passage (empty string if unresolved)
    - qrels_passage_ids: topic_id -> list of passage_ids that are relevant (and that we resolved)
    """
    # Unique (doc_id, xpath)
    seen = set()
    order = []
    for topic_id, pairs in qrels.items():
        for (doc_id, xpath) in pairs:
            key = (doc_id, xpath)
            if key not in seen:
                seen.add(key)
                order.append(key)
    pid_to_text = {}
    for doc_id, xpath in order:
        pid = f"{doc_id}::{xpath}"
        xml_path = _resolve_doc_path(doc_id, clefip_root, doc_collection_root)
        if xml_path:
            text = _get_passage_text_from_xml(xml_path, xpath)
            if text:
                pid_to_text[pid] = text
    passage_ids = list(pid_to_text.keys())
    passage_texts = [pid_to_text[pid] for pid in passage_ids]
    resolved = set(passage_ids)
    qrels_passage_ids = {}
    for topic_id, pairs in qrels.items():
        lst = [f"{doc_id}::{xpath}" for (doc_id, xpath) in pairs if f"{doc_id}::{xpath}" in resolved]
        if lst:
            qrels_passage_ids[topic_id] = lst
    return passage_ids, passage_texts, qrels_passage_ids


def load_clefip_en_for_eval(
    clefip_root: str,
    doc_collection_root: Optional[str] = None,
) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, List[str]]]:
    """
    One-shot load for evaluation: topics (filtered to those in EN qrels with at least one resolved passage),
    passage corpus, and qrels as passage_id lists.
    Returns (query_ids, query_texts, passage_ids, passage_texts, qrels_passage_ids).

    EN qrels reference doc_ids from the full CLEF-IP document collection (01). To resolve passage text,
    set doc_collection_root to the path of the **extracted** 01 collection (after extracting the 7z archives
    from 01_document_collection/). Without it, no passages are resolved and this raises RuntimeError.
    """
    qrels = load_clefip_en_qrels(clefip_root)
    query_ids_all, query_texts_all = load_clefip_en_topics(clefip_root)
    passage_ids, passage_texts, qrels_passage_ids = load_clefip_passage_corpus(
        clefip_root, qrels, doc_collection_root=doc_collection_root
    )
    if not passage_ids:
        raise RuntimeError(
            "No CLEF-IP passages could be resolved. EN qrels reference the full document collection (01). "
            "Extract the 7z archives in clefip2013/01_document_collection/ and pass the extracted folder path "
            "as doc_collection_root (e.g. --clefip_doc_root /path/to/extracted/01)."
        )
    query_id_set = set(qrels_passage_ids.keys())
    query_ids = [q for q in query_ids_all if q in query_id_set]
    query_texts = [query_texts_all[query_ids_all.index(q)] for q in query_ids]
    return query_ids, query_texts, passage_ids, passage_texts, qrels_passage_ids


def load_clefip_en_demo(clefip_root: str) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, List[str]]]:
    """
    Demo mode: build passage corpus only from tfiles/ and tfam-docs/ (no 01 collection).
    Synthetic qrels: for each topic, relevant passages = all passages from that topic's tfile.
    Use when --clefip_doc_root is not set, to run the pipeline without extracting 01.
    Returns (query_ids, query_texts, passage_ids, passage_texts, qrels_passage_ids).
    """
    test_dir = os.path.join(clefip_root, TOPICS_DIR, TEST_DIR)
    tfiles_dir = os.path.join(test_dir, TFILES_DIR)
    tfam_dir = os.path.join(test_dir, TFAM_DOCS_DIR)
    meta_path = os.path.join(test_dir, TEST_META_FILE)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"CLEF-IP topics meta not found: {meta_path}")
    topics_meta = _parse_test_txt(meta_path)
    # Build passage corpus from all XMLs in tfiles and tfam-docs
    pid_to_text = {}
    doc_ids_seen = set()
    for subdir, dir_path in [(TFILES_DIR, tfiles_dir), (TFAM_DOCS_DIR, tfam_dir)]:
        if not os.path.isdir(dir_path):
            continue
        for fn in os.listdir(dir_path):
            if not fn.lower().endswith(".xml"):
                continue
            doc_id = fn[:-4]
            xml_path = os.path.join(dir_path, fn)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except Exception:
                continue
            for xpath, text in _collect_passages_from_root(root):
                pid = f"{doc_id}::{xpath}"
                pid_to_text[pid] = text
                doc_ids_seen.add(doc_id)
    passage_ids = list(pid_to_text.keys())
    passage_texts = [pid_to_text[pid] for pid in passage_ids]
    if not passage_ids:
        raise RuntimeError(
            "Demo mode: no passages found under tfiles/ or tfam-docs/. Check that "
            f"{tfiles_dir} and/or {tfam_dir} exist and contain XML files."
        )
    # Synthetic qrels: topic -> passages from that topic's tfile
    qrels_passage_ids = {}
    for t in topics_meta:
        tid = t.get("tid")
        tfile = t.get("tfile", "")
        if not tid or not tfile:
            continue
        doc_id = tfile.replace(".xml", "") if tfile.endswith(".xml") else tfile
        relevant = [pid for pid in passage_ids if pid.startswith(doc_id + "::")]
        if relevant:
            qrels_passage_ids[tid] = relevant
    # Load query texts only for topics that have synthetic qrels
    query_ids_all, query_texts_all = load_clefip_en_topics(clefip_root)
    query_id_set = set(qrels_passage_ids.keys())
    query_ids = [q for q in query_ids_all if q in query_id_set]
    query_texts = [query_texts_all[query_ids_all.index(q)] for q in query_ids]
    return query_ids, query_texts, passage_ids, passage_texts, qrels_passage_ids
