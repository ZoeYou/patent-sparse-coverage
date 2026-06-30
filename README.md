# sparse_coverage
A sparse retrieval system for patent search based on **learned semantic centers**. Spans from patent documents are mapped to a discrete vocabulary of embedding-space centers; retrieval is then a sparse dot-product over these center IDs, combining the coverage of dense retrieval with the efficiency of sparse inverted-index search.

Evaluated on **CLEF-IP 2013** claims-to-passages retrieval. Baselines include BM25, SPLADE, ColBERT, and standard dense bi-encoders.

---

## Method overview

1. **Embed** patent spans (abstract, claims, description) using a pretrained bi-encoder ([`1create_N_embeddings.py`](1create_N_embeddings.py)).
2. **Build centers** — a discrete vocabulary of representative embedding directions — via the k-center greedy algorithm ([`2build_centers_kcenter.py`](2build_centers_kcenter.py)).
3. **Index** — assign document spans to centers; build an inverted index keyed by center ID.
4. **Retrieve** — map query spans to centers; score documents by weighted center overlap (IDF-weighted, with soft/hard span-to-center assignment); re-rank with two-stage passage retrieval for CLEF-IP 2013.

---

## Repository structure

| File / folder | Purpose |
|---|---|
| `0cache_spacy_spans.py` | Pre-tokenize corpus spans with spaCy and cache to disk |
| `1create_N_embeddings.py` | Embed spans with a bi-encoder; save to `embeddings/` |
| `2build_centers_kcenter.py` | Build center vocabulary via k-center greedy algorithm |
| `3evaluate.py` | Main evaluation script (sparse_coverage + baselines) |
| `utils.py` | Shared utilities (tokenization, encoder helpers, IDF) |
| `clefip2013/` | CLEF-IP 2013 data (topics, qrels, corpus) |

---

## Quickstart

### Dependencies

```bash
conda create -n patentmap python=3.9
conda activate patentmap
pip install -r requirements_web.txt   # or requirements_spaces.txt for HF Spaces
```

Requires FAISS (`faiss-gpu` recommended for large corpora).

### Cache spaCy spans

Required before embedding. Tokenizes corpus documents into spans (spacy tokens, noun chunks, encoder tokens) and caches them to disk.

```bash
python 0cache_spacy_spans.py \
    --data_dir ./clefip2013 \
    --cache_dir ./span_cache \
    --spacy_model sci_lg \
    --spacy_n_process 4
```

### Embed the corpus

```bash
python 1create_N_embeddings.py \
    --model_name ZoeYou/PatentMap-V0-SecPair-Claim \
    --tokenization_unit spacy_token
```

### Build centers

```bash
python 2build_centers_kcenter.py \
    --V 30000 \
    --embeddings_dir embeddings/PatentMap-V0-SecPair-Claim_spacy_token_cls_last_meanpool
```

### Evaluate

**Sparse Coverage on CLEF-IP:**
```bash
python 3evaluate.py \
    --model_name sparse_coverage \
    --clefip_root ./clefip2013 \
    --clefip_two_stage_topk_docs 100
```

**BM25 baseline:**
```bash
python 3evaluate.py --model_name bm25 --clefip_two_stage
```

**Dense baseline:**
```bash
python 3evaluate.py --model_name ZoeYou/PatentMap-V0-SecPair-Claim --clefip_two_stage
```

---

## Evaluation metrics (CLEF-IP)

Two-stage retrieval: Stage 1 derives a document ranking from passage scores (first-occurrence dedup); Stage 2 re-ranks passages within the top-K documents.

| Metric | Level | Notes |
|---|---|---|
| `magp` | Passage | Official CLEF-IP MAP(D): per-document AP averaged over relevant docs (Piroi et al. 2012) |
| `recall_passage@1000` | Passage | Passage recall at depth 1000 |
| `pres_doc@100` | Document | Official CLEF-IP PRES@100 |
| `recall_doc@100` | Document | Document recall at 100 |
| `ndcg_doc@10` | Document | Document NDCG at depth 10 |
| `map_doc` | Document | Untruncated document MAP |
| `mrr_doc` | Document | Untruncated document MRR (rank of first relevant doc) |

---

## Key hyperparameters

| Flag | Default | Description |
|---|---|---|
| `--document_assignment` | `soft` | `soft`: span → top-K centers by radius; `hard`: nearest center only |
| `--soft_assignment_max_centers_per_span` | `5` | Cap centers per span in soft mode |
| `--weight_aggregation` | `max` | `max` or `sum` aggregation over center hits |
| `--idf_exponent` | `2.0` | IDF power in scoring |
| `--length_norm_exponent` | `0.5` | Document length norm: divide by `doc_span_count^exp`; `0` disables |
| `--clefip_two_stage_topk_docs` | `100` | Stage-1 top-K documents for two-stage rerank |


