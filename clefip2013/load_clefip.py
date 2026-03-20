"""
CLEF-IP 2013 Claims-to-Passages (EN) data loading.

Provides:
- load_clefip_en_topics(clefip_root) -> query_ids, query_texts
- load_clefip_en_qrels(clefip_root) -> qrels: {topic_id: [(doc_id, xpath), ...]}
- load_clefip_passage_corpus(clefip_root, qrels, doc_collection_root=None) -> passage_ids, passage_texts
- build_full_passage_corpus_from_01(...) -> write corpus JSONL + ids.txt from full 01 collection
- load_clefip_en_for_eval_full_corpus(...) -> (query_ids, query_texts, corpus_path, ids_path, n_passages, qrels_passage_ids)

Passage ID format: "doc_id::xpath" for uniqueness.
Documents are resolved from: tfiles/, tfam-docs/, and (if provided) doc_collection_root.
"""

import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Callable

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


def _get_root_lang(root: ET.Element) -> str:
    """Return the lang attribute of the patent-document root element (uppercased), or 'unknown'."""
    return root.get("lang", "unknown").upper()


def _get_lang_from_xml_file(xml_path: str) -> Optional[str]:
    """Read only the first 512 bytes of an XML file to extract the patent-document lang attribute."""
    try:
        with open(xml_path, "rb") as f:
            chunk = f.read(512).decode("utf-8", errors="replace")
        m = re.search(r'<patent-document[^>]+\blang="([^"]+)"', chunk)
        return m.group(1).upper() if m else None
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


def _build_doc_collection_index(doc_collection_root: str) -> Dict[str, str]:
    """One-time walk of doc_collection_root: build doc_id -> absolute path for every .xml. Use for fast lookup."""
    index: Dict[str, str] = {}
    for root, _dirs, files in os.walk(doc_collection_root):
        for f in files:
            if f.endswith(".xml"):
                doc_id = f[:-4]  # strip .xml
                index[doc_id] = os.path.join(root, f)
    return index


def _resolve_doc_path(
    doc_id: str,
    clefip_root: str,
    doc_collection_root: Optional[str],
    doc_collection_index: Optional[Dict[str, str]] = None,
) -> Optional[str]:
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
    # 3) doc_collection_root (extracted 01): use prebuilt index if provided (fast); else try direct then walk
    if doc_collection_root and os.path.isdir(doc_collection_root):
        if doc_collection_index is not None:
            return doc_collection_index.get(doc_id)
        for sub in ("ep0", "ep1", "wo", "ep", "EP", "clef-ip-2012-ep0", "clef-ip-2012-ep1", "clef-ip-2012-wo"):
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
        for root, _dirs, files in os.walk(doc_collection_root):
            if (doc_id + ".xml") in files:
                return os.path.join(root, doc_id + ".xml")
    return None


def load_clefip_passage_corpus(
    clefip_root: str,
    qrels: Dict[str, List[Tuple[str, str]]],
    doc_collection_root: Optional[str] = None,
    only_queries_in_qrels: bool = True,
    lang_filter: Optional[str] = "EN",
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """
    Build passage corpus from qrels: unique (doc_id, xpath) and resolve text from XMLs.
    Returns (passage_ids, passage_texts, qrels_passage_ids).
    - passage_ids: list of "doc_id::xpath"
    - passage_texts: list of text for each passage (empty string if unresolved)
    - qrels_passage_ids: topic_id -> list of passage_ids that are relevant (and that we resolved)
    If lang_filter is set (default "EN"), passages from documents with a different lang attribute
    are excluded from both the corpus and qrels_passage_ids.
    """
    lang_filter_upper = lang_filter.upper() if lang_filter else None
    # Unique (doc_id, xpath)
    seen = set()
    order = []
    for topic_id, pairs in qrels.items():
        for (doc_id, xpath) in pairs:
            key = (doc_id, xpath)
            if key not in seen:
                seen.add(key)
                order.append(key)
    # One-time index for doc_collection_root so we don't os.walk per doc_id (slow on huge 01_extracted)
    doc_collection_index: Optional[Dict[str, str]] = None
    if doc_collection_root and os.path.isdir(doc_collection_root):
        print("Building doc_id -> path index for 01 collection (one-time walk)...", flush=True)
        doc_collection_index = _build_doc_collection_index(doc_collection_root)
        print(f"  Indexed {len(doc_collection_index):,} XML files.", flush=True)
    pid_to_text = {}
    skipped_lang = 0
    for doc_id, xpath in order:
        pid = f"{doc_id}::{xpath}"
        xml_path = _resolve_doc_path(doc_id, clefip_root, doc_collection_root, doc_collection_index=doc_collection_index)
        if xml_path:
            if lang_filter_upper:
                doc_lang = _get_lang_from_xml_file(xml_path)
                if doc_lang != lang_filter_upper:
                    skipped_lang += 1
                    continue
            text = _get_passage_text_from_xml(xml_path, xpath)
            if text:
                pid_to_text[pid] = text
    if lang_filter_upper and skipped_lang:
        print(f"  load_clefip_passage_corpus: skipped {skipped_lang} passages from non-{lang_filter_upper} docs.", flush=True)
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


# Default subdir for full 01 passage corpus cache (under clefip_root)
FULL_CORPUS_DIR = "01_passage_corpus"
FULL_CORPUS_DIR_EN = "01_passage_corpus_en"  # EN-only filtered corpus cache
CORPUS_JSONL = "passages.jsonl"
IDS_TXT = "passage_ids.txt"


def _filter_qrels_by_doc_lang(
    qrels_passage_ids: Dict[str, List[str]],
    doc_collection_root: str,
    clefip_root: str,
    lang_filter: str = "EN",
) -> Dict[str, List[str]]:
    """
    Filter qrels_passage_ids to only keep passages whose source document has the given lang.
    Looks up each unique doc_id in tfiles/, tfam-docs/, then the 01 collection (early exit).
    """
    # Collect unique doc_ids
    doc_ids_needed: set = set()
    for pids in qrels_passage_ids.values():
        for pid in pids:
            doc_ids_needed.add(pid.split("::")[0])

    doc_id_to_lang: Dict[str, str] = {}
    remaining = set(doc_ids_needed)

    # Fast lookup in tfiles / tfam-docs first
    for subdir_path in [
        os.path.join(clefip_root, TOPICS_DIR, TEST_DIR, TFILES_DIR),
        os.path.join(clefip_root, TOPICS_DIR, TEST_DIR, TFAM_DOCS_DIR),
    ]:
        if not os.path.isdir(subdir_path):
            continue
        for doc_id in list(remaining):
            p = os.path.join(subdir_path, doc_id + ".xml")
            if os.path.isfile(p):
                lang = _get_lang_from_xml_file(p)
                doc_id_to_lang[doc_id] = lang or "unknown"
                remaining.discard(doc_id)

    # Walk 01 collection for anything not found above (early exit once all resolved)
    if remaining and doc_collection_root and os.path.isdir(doc_collection_root):
        for root_dir, _dirs, files in os.walk(doc_collection_root):
            for fn in files:
                if not fn.endswith(".xml"):
                    continue
                stem = fn[:-4]
                if stem in remaining:
                    lang = _get_lang_from_xml_file(os.path.join(root_dir, fn))
                    doc_id_to_lang[stem] = lang or "unknown"
                    remaining.discard(stem)
            if not remaining:
                break

    if remaining:
        print(f"  [_filter_qrels_by_doc_lang] {len(remaining)} doc_ids not found; treating as non-{lang_filter}: {remaining}", flush=True)

    filtered: Dict[str, List[str]] = {}
    for topic_id, pids in qrels_passage_ids.items():
        kept = [pid for pid in pids if doc_id_to_lang.get(pid.split("::")[0], "unknown") == lang_filter]
        if kept:
            filtered[topic_id] = kept
    n_before = sum(len(v) for v in qrels_passage_ids.values())
    n_after = sum(len(v) for v in filtered.values())
    print(f"  qrels filtered to {lang_filter}-only docs: {n_before} -> {n_after} passages, "
          f"{len(qrels_passage_ids)} -> {len(filtered)} topics retained.", flush=True)
    return filtered


def build_full_passage_corpus_from_01(
    doc_collection_root: str,
    corpus_jsonl_path: str,
    ids_txt_path: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    lang_filter: Optional[str] = "EN",
) -> Tuple[int, int]:
    """
    Walk the full 01 document collection, extract passages from each XML, and write
    (passage_id, text) to corpus JSONL and passage_id per line to ids_txt.
    doc_id is taken from filename (stem of .xml).
    If lang_filter is set (default "EN"), only documents whose patent-document lang attribute
    matches (case-insensitive) are included; all others are skipped.
    Returns (num_passages, num_docs_processed).
    """
    if not os.path.isdir(doc_collection_root):
        raise FileNotFoundError(f"doc_collection_root not found: {doc_collection_root}")
    num_docs = 0
    num_passages = 0
    num_skipped_lang = 0
    lang_filter_upper = lang_filter.upper() if lang_filter else None
    with open(corpus_jsonl_path, "w", encoding="utf-8") as jf, open(ids_txt_path, "w", encoding="utf-8") as idf:
        for root_dir, _dirs, files in os.walk(doc_collection_root):
            for fn in files:
                if not fn.lower().endswith(".xml"):
                    continue
                xml_path = os.path.join(root_dir, fn)
                doc_id = os.path.splitext(fn)[0]
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                except Exception:
                    continue
                if lang_filter_upper and _get_root_lang(root) != lang_filter_upper:
                    num_skipped_lang += 1
                    continue
                for xpath, text in _collect_passages_from_root(root):
                    if not text:
                        continue
                    pid = f"{doc_id}::{xpath}"
                    jf.write(json.dumps({"pid": pid, "text": text}, ensure_ascii=False) + "\n")
                    idf.write(pid + "\n")
                    num_passages += 1
                num_docs += 1
                if progress_callback and num_docs % 5000 == 0:
                    progress_callback(num_docs, num_passages)
    if progress_callback:
        progress_callback(num_docs, num_passages)
    if lang_filter_upper:
        print(f"  lang_filter={lang_filter_upper}: kept {num_docs:,} docs, skipped {num_skipped_lang:,} non-{lang_filter_upper} docs.", flush=True)
    return num_passages, num_docs


def load_clefip_en_for_eval_full_corpus(
    clefip_root: str,
    doc_collection_root: str,
    corpus_dir: Optional[str] = None,
    rebuild_corpus: bool = False,
    lang_filter: Optional[str] = "EN",
) -> Tuple[List[str], List[str], str, str, int, Dict[str, List[str]]]:
    """
    Load topics and qrels, and ensure the full 01 passage corpus is built (or use cache).
    Returns (query_ids, query_texts, corpus_jsonl_path, ids_txt_path, num_passages, qrels_passage_ids).

    If lang_filter is set (default "EN"), only documents with that lang attribute are included
    in the corpus, and qrels_passage_ids is filtered to only reference those docs. This ensures
    that models which only support the filtered language are evaluated fairly.
    The EN-only corpus is cached in a separate directory (01_passage_corpus_en) to avoid
    accidentally reusing a full-corpus cache.
    """
    qrels = load_clefip_en_qrels(clefip_root)
    query_ids_all, query_texts_all = load_clefip_en_topics(clefip_root)
    # Build raw qrels_passage_ids (all doc_ids from qrels, no lang filter yet)
    qrels_passage_ids_raw: Dict[str, List[str]] = {}
    for topic_id, pairs in qrels.items():
        lst = [f"{doc_id}::{xpath}" for (doc_id, xpath) in pairs]
        if lst:
            qrels_passage_ids_raw[topic_id] = lst

    # Filter qrels to only include passages from docs with the desired language
    if lang_filter:
        lang_filter_upper = lang_filter.upper()
        print(f"  Filtering qrels to {lang_filter_upper}-only documents...", flush=True)
        qrels_passage_ids = _filter_qrels_by_doc_lang(
            qrels_passage_ids_raw, doc_collection_root, clefip_root, lang_filter=lang_filter_upper
        )
    else:
        qrels_passage_ids = qrels_passage_ids_raw

    query_id_set = set(qrels_passage_ids.keys())
    query_ids = [q for q in query_ids_all if q in query_id_set]
    query_texts = [query_texts_all[query_ids_all.index(q)] for q in query_ids]

    if corpus_dir is None:
        default_subdir = FULL_CORPUS_DIR_EN if lang_filter and lang_filter.upper() == "EN" else FULL_CORPUS_DIR
        corpus_dir = os.path.join(clefip_root, default_subdir)
    os.makedirs(corpus_dir, exist_ok=True)
    corpus_jsonl_path = os.path.join(corpus_dir, CORPUS_JSONL)
    ids_txt_path = os.path.join(corpus_dir, IDS_TXT)

    if rebuild_corpus or not os.path.isfile(corpus_jsonl_path) or not os.path.isfile(ids_txt_path):
        def _progress(n_docs: int, n_pass: int) -> None:
            print(f"  01 corpus: {n_docs:,} docs, {n_pass:,} passages", flush=True)
        lang_label = f"{lang_filter.upper()}-only " if lang_filter else ""
        print(f"Building full 01 {lang_label}passage corpus (this may take a long time)...", flush=True)
        build_full_passage_corpus_from_01(
            doc_collection_root, corpus_jsonl_path, ids_txt_path,
            progress_callback=_progress, lang_filter=lang_filter,
        )
        print(f"  Wrote {corpus_jsonl_path} and {ids_txt_path}", flush=True)

    # num_passages = line count of ids file
    with open(ids_txt_path, "r", encoding="utf-8") as f:
        num_passages = sum(1 for _ in f)
    return query_ids, query_texts, corpus_jsonl_path, ids_txt_path, num_passages, qrels_passage_ids


def load_clefip_en_for_eval_sampled_corpus(
    clefip_root: str,
    doc_collection_root: str,
    sample_size: int,
    corpus_dir: Optional[str] = None,
    rebuild_corpus: bool = False,
    lang_filter: Optional[str] = "EN",
    seed: int = 42,
) -> Tuple[List[str], List[str], str, str, int, Dict[str, List[str]]]:
    """
    Like load_clefip_en_for_eval_full_corpus but builds a **sampled** corpus.

    Sampling semantics (for sample_size > 0):
      **Document-level**: sample_size is the number of *documents* to include.
      All documents referenced in qrels are always included (guaranteed recall ceiling = 100%).
      The remaining slots are filled with reservoir-sampled EN documents from the full corpus.
      ALL passages from each selected document are kept (preserves document structure).

      This is more realistic than passage-level sampling: the candidate pool contains
      complete documents (with all their passages), which is how real retrieval works.
      It also makes two-stage retrieval (doc pre-rank → passage re-rank) meaningful.

      Example: --clefip_sample_size 10000 → ~10k docs → ~800k passages (at ~80 passages/doc).

    Special modes:
      -1 = qrels-only: only the exact passages referenced in qrels.
      -2 = qrels-docs: all passages from documents cited in qrels.

    The sampled corpus is cached under 01_passage_corpus_en_sample_<N>docs/.
    Returns (query_ids, query_texts, corpus_jsonl_path, ids_txt_path, num_passages, qrels_passage_ids).
    """
    import random

    qrels = load_clefip_en_qrels(clefip_root)
    query_ids_all, query_texts_all = load_clefip_en_topics(clefip_root)

    qrels_passage_ids_raw: Dict[str, List[str]] = {}
    for topic_id, pairs in qrels.items():
        lst = [f"{doc_id}::{xpath}" for (doc_id, xpath) in pairs]
        if lst:
            qrels_passage_ids_raw[topic_id] = lst

    if lang_filter:
        lang_filter_upper = lang_filter.upper()
        print(f"  Filtering qrels to {lang_filter_upper}-only documents...", flush=True)
        qrels_passage_ids = _filter_qrels_by_doc_lang(
            qrels_passage_ids_raw, doc_collection_root, clefip_root, lang_filter=lang_filter_upper
        )
    else:
        qrels_passage_ids = qrels_passage_ids_raw
        lang_filter_upper = None

    query_id_set = set(qrels_passage_ids.keys())
    query_ids = [q for q in query_ids_all if q in query_id_set]
    query_texts = [query_texts_all[query_ids_all.index(q)] for q in query_ids]

    if corpus_dir is None:
        if sample_size == -1:
            subdir = "01_passage_corpus_en_qrels_only" if (lang_filter_upper == "EN") else "01_passage_corpus_qrels_only"
        elif sample_size == -2:
            subdir = "01_passage_corpus_en_qrels_docs" if (lang_filter_upper == "EN") else "01_passage_corpus_qrels_docs"
        else:
            subdir = f"01_passage_corpus_en_sample_{sample_size}docs" if (lang_filter_upper == "EN") else f"01_passage_corpus_sample_{sample_size}docs"
        corpus_dir = os.path.join(clefip_root, subdir)
    os.makedirs(corpus_dir, exist_ok=True)
    corpus_jsonl_path = os.path.join(corpus_dir, CORPUS_JSONL)
    ids_txt_path = os.path.join(corpus_dir, IDS_TXT)

    if not rebuild_corpus and os.path.isfile(corpus_jsonl_path) and os.path.isfile(ids_txt_path):
        with open(ids_txt_path, "r", encoding="utf-8") as f:
            num_passages = sum(1 for _ in f)
        _cache_label = {-1: "qrels-only", -2: "qrels-docs (all passages from cited documents)"}.get(
            sample_size, f"sample {sample_size:,} docs"
        )
        print(f"  Sampled corpus cache found: {num_passages:,} passages ({_cache_label})", flush=True)
        return query_ids, query_texts, corpus_jsonl_path, ids_txt_path, num_passages, qrels_passage_ids

    # Collect all qrels-relevant passage ids (must always be in corpus)
    qrels_pids: set = set()
    for pids in qrels_passage_ids.values():
        qrels_pids.update(pids)
    n_qrels = len(qrels_pids)
    # doc_ids referenced by qrels (for -2 mode: keep all passages from these docs)
    qrels_doc_ids: set = {pid.split("::")[0] for pid in qrels_pids}
    if sample_size == -1:
        n_sample_extra_docs = 0
        print(f"  Building qrels-only corpus: {n_qrels:,} relevant passages (no random negatives)...", flush=True)
    elif sample_size == -2:
        n_sample_extra_docs = 0
        print(f"  Building qrels-docs corpus: all passages from {len(qrels_doc_ids):,} cited documents...", flush=True)
    else:
        n_sample_extra_docs = max(0, sample_size - len(qrels_doc_ids))
        print(f"  Building document-sampled corpus: {len(qrels_doc_ids):,} qrels docs (always included) "
              f"+ up to {n_sample_extra_docs:,} random EN docs "
              f"(target total: {sample_size:,} docs)...", flush=True)

    # Prefer sampling from EN corpus JSONL if it already exists (fast sequential read).
    # Both qrels-only (n_sample_extra_docs==0) and sampled modes benefit from JSONL:
    #   - qrels-only: single scan to collect qrels passages (much faster than XML walk)
    #   - sampled (doc-level): Pass 1 scans passage_ids.txt to reservoir-sample doc_ids,
    #     Pass 2 scans passages.jsonl to collect all passages from selected docs.
    en_corpus_jsonl = os.path.join(clefip_root, FULL_CORPUS_DIR_EN, CORPUS_JSONL)
    en_corpus_ids = os.path.join(clefip_root, FULL_CORPUS_DIR_EN, IDS_TXT)
    rng = random.Random(seed)

    if os.path.isfile(en_corpus_jsonl):
        # --- Fast path: read from pre-built EN JSONL ---

        # Step 1: Determine the set of doc_ids to include.
        # For -1 and -2 modes, only qrels-related docs.
        # For sample_size > 0: qrels_doc_ids + reservoir-sampled additional doc_ids.
        selected_doc_ids: set = set(qrels_doc_ids)  # always include qrels docs

        if n_sample_extra_docs > 0 and os.path.isfile(en_corpus_ids):
            # Pass 1: scan passage_ids.txt to discover unique doc_ids and reservoir-sample.
            # This is much faster than scanning the full 61GB JSONL.
            print(f"  Pass 1: scanning {en_corpus_ids} to reservoir-sample {n_sample_extra_docs:,} doc_ids...", flush=True)
            reservoir_docs: list = []  # reservoir of non-qrels doc_ids
            n_docs_seen = 0
            prev_did = None
            with open(en_corpus_ids, "r", encoding="utf-8") as f:
                for line in f:
                    pid = line.strip()
                    if not pid:
                        continue
                    did = pid.split("::")[0]
                    if did == prev_did:
                        continue  # same doc as previous line, skip
                    prev_did = did
                    if did in qrels_doc_ids:
                        continue  # already included
                    n_docs_seen += 1
                    if len(reservoir_docs) < n_sample_extra_docs:
                        reservoir_docs.append(did)
                    else:
                        j = rng.randint(0, n_docs_seen - 1)
                        if j < n_sample_extra_docs:
                            reservoir_docs[j] = did
            selected_doc_ids.update(reservoir_docs)
            print(f"  Pass 1 done: {n_docs_seen:,} non-qrels docs seen, "
                  f"sampled {len(reservoir_docs):,} → total {len(selected_doc_ids):,} docs selected.", flush=True)
        elif n_sample_extra_docs > 0:
            # No passage_ids.txt available; fall back to scanning JSONL for doc_ids
            print(f"  Pass 1: scanning {en_corpus_jsonl} to reservoir-sample doc_ids (no ids.txt cache)...", flush=True)
            reservoir_docs = []
            n_docs_seen = 0
            prev_did = None
            with open(en_corpus_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Fast extraction: find "pid" value without full JSON parse
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    pid = obj.get("pid", "")
                    did = pid.split("::")[0]
                    if did == prev_did:
                        continue
                    prev_did = did
                    if did in qrels_doc_ids:
                        continue
                    n_docs_seen += 1
                    if len(reservoir_docs) < n_sample_extra_docs:
                        reservoir_docs.append(did)
                    else:
                        j = rng.randint(0, n_docs_seen - 1)
                        if j < n_sample_extra_docs:
                            reservoir_docs[j] = did
            selected_doc_ids.update(reservoir_docs)
            print(f"  Pass 1 done: {n_docs_seen:,} non-qrels docs seen, "
                  f"sampled {len(reservoir_docs):,} → total {len(selected_doc_ids):,} docs selected.", flush=True)

        # Step 2: Single-pass over JSONL to collect passages from selected docs.
        if sample_size == -2:
            print(f"  Scanning {en_corpus_jsonl}: collecting all passages from {len(qrels_doc_ids):,} cited doc_ids...", flush=True)
        elif n_sample_extra_docs > 0:
            print(f"  Pass 2: scanning {en_corpus_jsonl}: collecting all passages from {len(selected_doc_ids):,} selected docs...", flush=True)
        else:
            print(f"  Scanning {en_corpus_jsonl}: qrels lookup only...", flush=True)

        qrels_lines: Dict[str, str] = {}  # pid -> raw JSONL line (for qrels passages)
        selected_doc_lines: list = []  # all passages from selected docs (non-qrels ones)

        with open(en_corpus_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                pid = obj.get("pid", "")
                did = pid.split("::")[0]
                if pid in qrels_pids:
                    qrels_lines[pid] = line
                if sample_size == -2:
                    # -2 mode: collect all passages from qrels docs (non-qrels ones)
                    if did in qrels_doc_ids and pid not in qrels_pids:
                        selected_doc_lines.append(line)
                elif sample_size == -1:
                    # -1 mode: only qrels passages, stop early if all found
                    if len(qrels_lines) == n_qrels:
                        break
                elif did in selected_doc_ids and pid not in qrels_pids:
                    # doc-level sample: keep all passages from selected docs
                    selected_doc_lines.append(line)

        if sample_size == -2:
            print(f"  Found {len(qrels_lines):,}/{n_qrels:,} qrels passages "
                  f"+ {len(selected_doc_lines):,} other passages from cited docs.", flush=True)
        elif sample_size == -1:
            print(f"  Found {len(qrels_lines):,}/{n_qrels:,} qrels passages.", flush=True)
        else:
            print(f"  Found {len(qrels_lines):,}/{n_qrels:,} qrels passages "
                  f"+ {len(selected_doc_lines):,} passages from {len(selected_doc_ids):,} sampled docs.", flush=True)

        # Write sampled corpus
        with open(corpus_jsonl_path, "w", encoding="utf-8") as jf, open(ids_txt_path, "w", encoding="utf-8") as idf:
            # Write qrels passages first (deterministic order)
            for pid in sorted(qrels_pids):
                line = qrels_lines.get(pid)
                if line:
                    jf.write(line + "\n")
                    idf.write(pid + "\n")
            # Write remaining passages from selected docs (doc-level sample or -2 mode)
            for line in selected_doc_lines:
                try:
                    obj = json.loads(line)
                    jf.write(line + "\n")
                    idf.write(obj["pid"] + "\n")
                except Exception:
                    continue

    else:
        # --- Stream directly from XML files (slower, no pre-built cache needed) ---
        print(f"  EN corpus cache not found; streaming from XMLs...", flush=True)
        # First pass: collect qrels passages (need their text) and discover all doc_ids
        qrels_pid_to_text: Dict[str, str] = {}
        needed_doc_ids = {pid.split("::")[0] for pid in qrels_pids}

        # For doc-level sampling, we need to know all doc_ids first, then sample, then collect
        all_en_doc_ids: list = []  # all EN doc_ids encountered
        for root_dir, _dirs, files in os.walk(doc_collection_root):
            for fn in files:
                if not fn.endswith(".xml"):
                    continue
                stem = fn[:-4]
                try:
                    tree = ET.parse(os.path.join(root_dir, fn))
                    root_el = tree.getroot()
                except Exception:
                    continue
                if lang_filter_upper and _get_root_lang(root_el) != lang_filter_upper:
                    continue
                all_en_doc_ids.append(stem)
                if stem in needed_doc_ids:
                    for xpath, text in _collect_passages_from_root(root_el):
                        pid = f"{stem}::{xpath}"
                        if pid in qrels_pids and text:
                            qrels_pid_to_text[pid] = text
                    needed_doc_ids.discard(stem)

        # Reservoir-sample non-qrels doc_ids
        selected_doc_ids: set = set(qrels_doc_ids)
        if n_sample_extra_docs > 0:
            non_qrels_docs = [d for d in all_en_doc_ids if d not in qrels_doc_ids]
            rng.shuffle(non_qrels_docs)
            selected_doc_ids.update(non_qrels_docs[:n_sample_extra_docs])
            print(f"  Selected {len(selected_doc_ids):,} docs ({len(qrels_doc_ids):,} qrels + "
                  f"{min(n_sample_extra_docs, len(non_qrels_docs)):,} sampled).", flush=True)
        else:
            print("  Qrels-only/qrels-docs mode: no additional docs sampled.", flush=True)

        # Second pass: collect all passages from selected docs (excluding qrels, already collected)
        selected_doc_passages: list = []  # (pid, text) for non-qrels passages from selected docs
        if sample_size > 0 or sample_size == -2:
            docs_to_scan = selected_doc_ids - set(qrels_pid_to_text.keys())  # avoid re-parsing
            for root_dir, _dirs, files in os.walk(doc_collection_root):
                for fn in files:
                    if not fn.endswith(".xml"):
                        continue
                    stem = fn[:-4]
                    if stem not in selected_doc_ids:
                        continue
                    try:
                        tree = ET.parse(os.path.join(root_dir, fn))
                        root_el = tree.getroot()
                    except Exception:
                        continue
                    for xpath, text in _collect_passages_from_root(root_el):
                        pid = f"{stem}::{xpath}"
                        if pid not in qrels_pids and text:
                            selected_doc_passages.append((pid, text))

        with open(corpus_jsonl_path, "w", encoding="utf-8") as jf, open(ids_txt_path, "w", encoding="utf-8") as idf:
            for pid in sorted(qrels_pids):
                text = qrels_pid_to_text.get(pid)
                if text:
                    jf.write(json.dumps({"pid": pid, "text": text}, ensure_ascii=False) + "\n")
                    idf.write(pid + "\n")
            for pid, text in selected_doc_passages:
                jf.write(json.dumps({"pid": pid, "text": text}, ensure_ascii=False) + "\n")
                idf.write(pid + "\n")

    with open(ids_txt_path, "r", encoding="utf-8") as f:
        num_passages = sum(1 for _ in f)
    print(f"  Wrote sampled corpus: {num_passages:,} passages -> {corpus_jsonl_path}", flush=True)
    return query_ids, query_texts, corpus_jsonl_path, ids_txt_path, num_passages, qrels_passage_ids


def load_clefip_en_demo(
    clefip_root: str,
    lang_filter: Optional[str] = "EN",
) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, List[str]]]:
    """
    Demo mode: build passage corpus only from tfiles/ and tfam-docs/ (no 01 collection).
    Synthetic qrels: for each topic, relevant passages = all passages from that topic's tfile.
    Use when --clefip_doc_root is not set, to run the pipeline without extracting 01.
    If lang_filter is set (default "EN"), only documents with that lang attribute are included.
    Returns (query_ids, query_texts, passage_ids, passage_texts, qrels_passage_ids).
    """
    lang_filter_upper = lang_filter.upper() if lang_filter else None
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
            if lang_filter_upper and _get_root_lang(root) != lang_filter_upper:
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
