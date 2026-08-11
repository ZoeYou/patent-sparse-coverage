from __future__ import absolute_import, division, unicode_literals

from collections import Counter, defaultdict
import numpy as np
import torch
import re
import json
import inspect
from torch import optim
from torch.utils.data import Dataset
import faiss
import torch.nn as nn

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import normalize


def create_dictionary(sentences):
    words = {}
    for s in sentences:
        for word in s:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2
    # words['<UNK>'] = 1e9 + 1
    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


def remove_escape_and_decode(text):
    try:
        text = text.replace('\/', '/').replace('\"', '"')
        text = text.encode('utf-8').decode('unicode_escape')
        return text.encode('latin1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeError) as e:
        # If decoding fails, clean problematic escape sequences
        import re
        # Remove incomplete or malformed \xXX escape sequences
        text = re.sub(r'\\x[0-9a-fA-F]{0,1}(?![0-9a-fA-F])', '', text)
        # Remove other problematic escape sequences
        text = re.sub(r'\\[^nrtbfav"\'\\]', '', text)
        # Return cleaned text directly
        return text.replace('\/', '/').replace('\"', '"')


def load_corpus(corpus_path):
    """
    Load the patent corpus from the json file and return a dictionary of dictionaries (of list) with the application id as the key
    """
    documents = defaultdict(dict)

    # load json file
    corpus = []
    with open(corpus_path, 'r') as f:
        for line in f:
            corpus.append(json.loads(line))

    for doc in corpus:
        app_id = str(doc['dnum'])
        title = doc['Content']['title']
        if 'pa01' not in doc['Content']:    # we observed that in the document.json file, some documents do not have an abstract
            abstract = ""
        else:
            abstract = doc['Content']['pa01']

        # get the claims
        claims = []
        for key in doc['Content'].keys():
            if key.startswith('c-en-'):
                claims.append(doc['Content'][key])
            # if key == 'c-en-0001' and abstract == "":   # if the document has no abstract, use the first claim as the abstract
            #     abstract = doc['Content']['c-en-0001']
        claims = '\n'.join(claims)
        claims = remove_escape_and_decode(claims)

        # get the description
        description = []
        for key in doc['Content'].keys():
            if key.startswith('p0'):
                description.append(doc['Content'][key])
        description = ' '.join(description)
        description = remove_escape_and_decode(description)

        # replace " -->"+ by empty string
        abstract = abstract.replace(" -->", "")
        claims = claims.replace(" -->", "")
        description = description.replace(" -->", "")

        # trunk the abstract, claims and description to 500 words to avoid memory error
        abstract = " ".join(abstract.split(" ")[:500])
        claims = " ".join(claims.split(" ")[:500])
        description = " ".join(description.split(" ")[:500])

        documents[app_id] = {
            'title': title,
            'abstract': abstract,
            'claim': claims,
            'invention': description
        }
    return documents



def mean_recall_at_k(true_labels, predicted_labels, k=10):
    """
    Calculate the mean Recall@k for a list of recommendations.

    Parameters:
    true_labels : list of list
        True relevant items for each recommendation list.
    predicted_labels : list of list
        Predicted recommended items for each recommendation list.
    k : int
        Number of recommendations to consider.

    Returns:
    float
        Mean Recall@k value.
    """

    recalls_at_k = []

    for true, pred in zip(true_labels, predicted_labels):
        # Calculate Recall@k for each recommendation list
        true_set = set(true)
        if not true_set:
            print("Empty true set")
            continue
        k = min(k, len(pred))
        relevant_count = sum(1 for item in pred[:k] if item in true_set)
        recalls_at_k.append(relevant_count / len(true_set))

    # Calculate the mean Recall@k
    mean_recall = sum(recalls_at_k) / len(recalls_at_k)

    return mean_recall



def ndcg_at_k(true_labels, predicted_docs, k=10):
    """Single-query nDCG."""
    top_k_docs = predicted_docs[:k]
    rel = [1 if doc_id in true_labels else 0 for doc_id in top_k_docs]
    dcg = 0.0
    for i, r in enumerate(rel, start=1):
        dcg += r / np.log2(i + 1)

    # IDCG
    R = min(len(true_labels), k)
    idcg = 0.0
    for i in range(1, R + 1):
        idcg += 1.0 / np.log2(i + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg



def mean_ndcg_at_k(true_labels_list, predicted_labels_list, k=10):
    """
    Aggregates ndcg_at_k over multiple queries.
    Each entry in true_labels_list/predicted_labels_list
    corresponds to a *single* query.
    """
    scores = []
    for q_true_labels, q_pred_docs in zip(true_labels_list, predicted_labels_list):
        scores.append(ndcg_at_k(q_true_labels, q_pred_docs, k=k))
    return np.mean(scores) if scores else 0.0



def mrr_at_k(true_labels, predicted_docs, k=10):
    """
    Calculate Mean Reciprocal Rank (MRR) at k for a single query.
    
    Parameters:
    true_labels : list
        True relevant items for this query.
    predicted_docs : list
        Predicted recommended items for this query.
    k : int
        Number of recommendations to consider.
    
    Returns:
    float
        Reciprocal rank (0 if no relevant item in top k).
    """
    top_k_docs = predicted_docs[:k]
    true_set = set(true_labels)
    
    for i, doc_id in enumerate(top_k_docs, start=1):
        if doc_id in true_set:
            return 1.0 / i
    
    return 0.0



def mean_mrr_at_k(true_labels_list, predicted_labels_list, k=10):
    """
    Calculate Mean Reciprocal Rank (MRR) at k over multiple queries.
    
    Parameters:
    true_labels_list : list of list
        True relevant items for each query.
    predicted_labels_list : list of list
        Predicted recommended items for each query.
    k : int
        Number of recommendations to consider.
    
    Returns:
    float
        Mean MRR@k value.
    """
    scores = []
    for q_true_labels, q_pred_docs in zip(true_labels_list, predicted_labels_list):
        scores.append(mrr_at_k(q_true_labels, q_pred_docs, k=k))
    return np.mean(scores) if scores else 0.0



def average_precision(true_labels, predicted_docs, k=None):
    """
    Calculate Average Precision (AP) for a single query.
    
    Parameters:
    true_labels : list
        True relevant items for this query.
    predicted_docs : list
        Predicted recommended items for this query.
    k : int or None
        Number of recommendations to consider (None = all).
    
    Returns:
    float
        Average precision score.
    """
    if k is not None:
        predicted_docs = predicted_docs[:k]
    
    true_set = set(true_labels)
    if not true_set:
        return 0.0
    
    num_hits = 0
    sum_precisions = 0.0
    
    for i, doc_id in enumerate(predicted_docs, start=1):
        if doc_id in true_set:
            num_hits += 1
            precision_at_i = num_hits / i
            sum_precisions += precision_at_i
    
    if num_hits == 0:
        return 0.0
    
    return sum_precisions / len(true_set)



def mean_average_precision(true_labels_list, predicted_labels_list, k=None):
    """
    Calculate Mean Average Precision (MAP) over multiple queries.
    
    Parameters:
    true_labels_list : list of list
        True relevant items for each query.
    predicted_labels_list : list of list
        Predicted recommended items for each query.
    k : int or None
        Number of recommendations to consider (None = all).
    
    Returns:
    float
        MAP value.
    """
    scores = []
    for q_true_labels, q_pred_docs in zip(true_labels_list, predicted_labels_list):
        scores.append(average_precision(q_true_labels, q_pred_docs, k=k))
    return np.mean(scores) if scores else 0.0



def pres_at_k(true_labels, predicted_docs, k=100, N_max=None):
    """
    Calculate PRES (Patent Retrieval Evaluation Score) at k for a single query.
    
    PRES is a recall-oriented metric that measures how well a retrieval system
    retrieves all relevant documents early in the ranking, under an adjustable
    cutoff N_max (maximum rank user is willing to examine).
    
    Reference: Magdy & Jones, SIGIR 2010
    https://dl.acm.org/doi/epdf/10.1145/1835449.1835551
    
    Parameters:
    true_labels : list
        True relevant items for this query.
    predicted_docs : list
        Predicted recommended items for this query (ranked list).
    k : int
        Number of recommendations to consider (default: 100).
    N_max : int or None
        Maximum rank cutoff user checks. If None, uses k.
        Relevant documents beyond N_max are treated as missed (rank = N_max).
    
    Returns:
    float
        PRES score in [0, 1], where 1 is best (all relevant docs in top positions).
    """
    if N_max is None:
        N_max = k
    
    # Look at top max(k, N_max) documents to find all relevant ones within N_max
    # (if N_max > k, we need to look beyond k to find relevant docs up to N_max)
    top_docs = predicted_docs[:max(k, N_max)]
    true_set = set(true_labels)
    
    if not true_set:
        return 0.0
    
    n = len(true_set)  # Total number of relevant documents
    
    # Find ranks of relevant documents in top max(k, N_max) (1-based indexing)
    rel_ranks = []
    for rank, doc_id in enumerate(top_docs, start=1):
        if doc_id in true_set and rank <= N_max:
            rel_ranks.append(rank)
    
    # Sort ranks
    rel_ranks = sorted(rel_ranks)
    m = len(rel_ranks)  # Number of relevant docs retrieved in top k
    
    # For missing relevant docs (beyond top k or beyond N_max), treat as rank = N_max
    num_missing = n - m
    full_ranks = rel_ranks + [N_max] * num_missing
    
    # Compute sum of (r_i - i) for i = 1..n
    sum_diff = 0.0
    for i, r in enumerate(full_ranks, start=1):
        # Ensure r doesn't exceed N_max
        r_i = min(r, N_max)
        sum_diff += (r_i - i)
    
    # Denominator: n * (N_max - n)
    denom = n * (N_max - n)
    
    if denom <= 0:
        # Edge case: if N_max == n, denominator is 0
        # PRES is 1 if all relevant docs are in top n, else 0
        return 1.0 if sum_diff == 0 else 0.0
    
    pres = 1.0 - (sum_diff / denom)
    # Ensure in [0, 1]
    pres = max(0.0, min(1.0, pres))
    return pres



def mean_pres_at_k(true_labels_list, predicted_labels_list, k=100, N_max=None):
    """
    Calculate Mean PRES (Patent Retrieval Evaluation Score) at k over multiple queries.
    
    PRES is a recall-oriented metric that measures how well a retrieval system
    retrieves all relevant documents early in the ranking.
    
    Reference: Magdy & Jones, SIGIR 2010
    https://dl.acm.org/doi/epdf/10.1145/1835449.1835551
    
    Parameters:
    true_labels_list : list of list
        True relevant items for each query.
    predicted_labels_list : list of list
        Predicted recommended items for each query (ranked lists).
    k : int
        Number of recommendations to consider (default: 100).
    N_max : int or None
        Maximum rank cutoff user checks. If None, uses k.
    
    Returns:
    float
        Mean PRES@k value.
    """
    scores = []
    for q_true_labels, q_pred_docs in zip(true_labels_list, predicted_labels_list):
        scores.append(pres_at_k(q_true_labels, q_pred_docs, k=k, N_max=N_max))
    return np.mean(scores) if scores else 0.0



def citation_to_citing_to_cited_dict(citations):
    """
    Convert the list of citations to a dictionary mapping citing patents to cited patents.
    """
    # Initialize an empty dictionary to store the results
    citing_to_cited_dict = defaultdict(list)

    # Iterate over the items in the JSON list
    for citing_id, cited_info in citations.items():
        for cited_line in cited_info:
            # if cited_line['type'] in ['A', 'X', 'Y']:
            # Extract the cited patents from the cited_info
            cited_patent = cited_line['cited_id']

            if cited_patent not in citing_to_cited_dict[citing_id]:
                # Add the citing patent and its cited patents to the dictionary
                citing_to_cited_dict[citing_id].append(cited_patent)
        
    return citing_to_cited_dict



def compute_ssd(embeddings: np.ndarray, l2_normalize_rows: bool = True, normalize_by_d: bool = False) -> float:
    """
    embeddings: (n, d) matrix
    Returns SSD / log(d), where SSD = KL(p || Uniform_d) with p from singular values.
    """
    E = embeddings.astype(np.float64, copy=False)  # better numerical stability

    # 1) Optional: L2-normalize rows (consistent with cosine-space diagnostics)
    if l2_normalize_rows:
        norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
        E = E / norms

    # 2) Center columns (remove mean)
    E = E - E.mean(axis=0, keepdims=True)

    n, d = E.shape

    # 3) Compute SVD; squared singular values ~ eigenvalues of covariance
    #    (you can also use randomized SVD if n is huge)
    s = np.linalg.svd(E, compute_uv=False)  # length = min(n, d)
    # keep only the first d terms just in case
    s = s[:d]

    # 4) Normalize singular values
    p = normalize_singular_values(s)  # length d, sums to 1


    # 5) SSD = KL(p || Uniform_d) = sum p_i log(p_i / (1/d)) = log d - H(p)
    H = -np.sum(p * (np.log(p + 1e-12)))
    ssd = np.log(d) - H

    # 6) Normalize by log d for cross-d comparability
    if normalize_by_d:
        ssd = ssd / np.log(d)
    return float(ssd)



def normalize_singular_values(singular_values):
    total = np.sum(singular_values)
    if total == 0:
        raise ValueError("The sum of singular values is zero, likely due to all-zero embeddings.")
    p = singular_values / total
    return p



def compute_kl_divergence(p, q):
    # Avoid log(0) by computing only for non-zero p
    kl_div = np.sum(p[p > 0] * np.log(p[p > 0] / q[p > 0]))
    return kl_div



def safe_parse_labels(x):
    """Safely parse IPC labels from various formats"""
    if isinstance(x, str):
        # Handle string representation of list like "['A01B', 'C12N']"
        x = x.strip("[]")
        # Try comma separation first, then space separation
        if ',' in x:
            labels = [label.strip("' \"") for label in x.split(",") if label.strip()]
        else:
            labels = [label.strip("' \"") for label in x.split() if label.strip()]
        return [label for label in labels if label]  # Remove empty strings
    elif isinstance(x, list):
        return [str(label) for label in x if str(label).strip()]
    else:
        import logging
        logging.warning(f"Unexpected label format: {type(x)} - {x}")
        return []


def get_primary_label(labels):
    """Safely extract primary label with boundary checking"""
    if isinstance(labels, list) and len(labels) > 0:
        return label_process(labels[0])
    else:
        import logging
        logging.warning(f"Empty or invalid label list: {labels}")
        return None


def process_label_list(label_list):
    """Process a list of labels consistently"""
    processed = [label_process(label) for label in label_list]
    return [label for label in processed if label is not None]


def label_process(labels):
    """Process the IPC labels. Keep only the first 4 characters and remove duplicates."""
    if type(labels) == list and type(labels[0]) == str:
        labels = [label[:4] for label in labels]
        # Remove duplicates while preserving order
        return list(dict.fromkeys(labels))
    elif type(labels) == str:
        return labels[:4]
    else:
        assert False, "Invalid label type"




def preprocess_claims(claims):
    """Reorganize the claim set."""
    claims = re.split(r"(?:[\W]+\s(\d+-\d+|\d+)\.\s|^(1-\d+)\.\s|^(1)\.\s)", claims)
    claims_list = [
        claim.strip() + '.' for claim in claims
        if claim and re.search(r"\b(?:cancelled|canceled)\b", claim, re.IGNORECASE) is None and not re.match(r"\d+-\d+|\d+", claim)
    ]

    # if last claim has two periods, remove the last period
    if claims_list and claims_list[-1].endswith('..'):
        claims_list[-1] = claims_list[-1][:-1]

    return '\n'.join(claims_list)



def preprocess_full_description(description):
    """Remove the first subsection of the full description if it is a cross-reference."""
    start_heading_pattern = re.compile(r"^(?:[A-Z]|-|\s|\(|\))*\s")
    intext_first_heading_pattern = re.compile(r"\.\s[A-Z]{5,}\s")
    cross_reference_pattern = re.compile(r"RELATED APPLICATIONS?|CROSS[-|\s]REFERENCE|RELATED DOCUMENTS?|PRIORITY|RELATED?|SPONSORED RESEARCH|FUNDED RESEARCH|STATEMENT REGARDING|GOVERNMENT|AGREEMENT|PATENT APPLICATIONS?|COPYRIGHT|GOVERNMENT INTEREST|SEQUENCE LISTING|REFERENCE")

    # Remove the first subsection if it is a cross-reference
    start_heading = start_heading_pattern.match(description)
    if not start_heading:
        # remove the first subsection before first intext heading
        second_heading = intext_first_heading_pattern.search(description)
        if second_heading:
            position = second_heading.start()
            description = description[position:]
            description = description.lstrip(" .").strip()
        return description

    while description and start_heading:
        if cross_reference_pattern.search(start_heading.group()):
            # Remove the first subsection
            second_heading = intext_first_heading_pattern.search(description)
            if second_heading:
                position = second_heading.start()
                description = description[position:]
            else:
                break
        else:
            break
        description = description.lstrip(" .").strip()
        start_heading = start_heading_pattern.match(description)

    return description



def preprocess_summary(summary):
    # check if <EOH> is in the text
    if '<EOH>' in summary:
        parts = summary.split('<EOH>')
        if len(parts) > 1 and parts[1].strip():
            return parts[1].strip()
        else:
            # No actual text after <EOH>
            return ""
    return summary



def compute_uniformity(embeddings, t=2.0, num_samples=10000, device='cuda'):
    """
    Compute the uniformity metric by sampling a subset of all possible pairs.
    
    :param embeddings: numpy array of shape (N, d)
    :param t: Temperature parameter
    :param num_samples: Number of pairs to sample
    :return: Scalar uniformity score.
    """
    # Convert embeddings to a torch tensor if they are not already
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.from_numpy(embeddings)
    embeddings = embeddings.to(device)
    
    embeddings = normalize(embeddings.cpu().numpy(), norm='l2', axis=1)
    
    N = embeddings.shape[0]
    
    # Randomly sample pairs of indices
    idx1 = np.random.randint(0, N, size=num_samples)
    idx2 = np.random.randint(0, N, size=num_samples)
    
    # Compute distances for the sampled pairs
    diffs = embeddings[idx1] - embeddings[idx2]
    distances = np.sum(diffs ** 2, axis=1)
    distances = np.clip(distances, 1e-12, None)
    
    uniformity = np.log(np.mean(np.exp(-t * distances)))
    return uniformity



class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]



class LinearClassifier(nn.Module):     # we design a simple linear classifier (to prevent the model memorizing the data) as the probing task
    def __init__(self, input_dim, num_classes, dim_hidden=1024):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, dim_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim_hidden, num_classes)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)  # raw logits, no sigmoid
        return logits



class KNNClassifier(object):
    def __init__(self, n_neighbors=5, metric='cosine', use_gpu=False):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.use_gpu = use_gpu
        self.res = None  # Store GPU resources explicitly

    def fit(self, X_train, y_train):
        """
        Stores the training data and labels.
        """
        # Normalize data for cosine similarity
        if self.metric == 'cosine':
            X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-10)
        self.X_train = X_train
        self.y_train = y_train

        d = X_train.shape[1]  # Feature dimension
        if self.metric == 'cosine':
            # Use IndexFlatIP for inner product (cosine similarity)
            self.index = faiss.IndexFlatIP(d)
        else:
            # Use IndexFlatL2 for L2 distance
            self.index = faiss.IndexFlatL2(d)

        if self.use_gpu:
            self.res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)

        self.index.add(X_train.astype(np.float32))  # Add data to the FAISS index

    def predict_proba(self, X_test):
        """
        Finds the k-nearest neighbors and computes probabilities.
        """
        if self.metric == 'cosine':
            X_test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-10)

        # Search for k nearest neighbors
        _, indices = self.index.search(X_test.astype(np.float32), self.n_neighbors)

        # Gather the labels of the neighbors
        neighbors_labels = self.y_train[indices]  # Shape: (n_test, n_neighbors, n_classes)

        # Compute probabilities as the mean of neighbor labels
        probabilities = np.mean(neighbors_labels, axis=1)  # Shape: (n_test, n_classes)
        return probabilities

    def predict(self, X_test, threshold=None, top_k=None):
        probabilities = self.predict_proba(X_test)
        if top_k is not None:
            predictions = np.zeros_like(probabilities, dtype=int)
            top_k_indices = np.argsort(-probabilities, axis=1)[:, :top_k]
            for idx, label_idx in enumerate(top_k_indices):
                predictions[idx, label_idx] = 1
            return predictions
        else:
            if threshold is None:
                threshold = 0.5
            return (probabilities > threshold).astype(int)

    def tune_k_by_precision_at_k(self, X_train, y_train, X_val, y_val, candidate_k_list, precision_k=1):
        best_k = None
        best_precision = -1.0

        for k in candidate_k_list:
            self.n_neighbors = k
            self.fit(X_train, y_train)
            preds = self.predict(X_val, top_k=precision_k)
            precision_scores = []

            for true, pred in zip(y_val, preds):
                pred_indices = np.where(pred == 1)[0]
                true_indices = np.where(true == 1)[0]
                if len(pred_indices) == 0:
                    precision_scores.append(0.0)
                else:
                    precision_scores.append(len(set(pred_indices).intersection(true_indices)) / precision_k)

            avg_precision = np.mean(precision_scores)
            if avg_precision > best_precision:
                best_precision = avg_precision
                best_k = k
                best_predictions = preds

        self.n_neighbors = best_k
        return best_k, best_precision



def analyze_retrieved_sections_integrated(retrieved_sections, query_section="claim", print_results=True):
    """
    Analyze retrieved section distribution and integrate into existing evaluation flow.
    
    Args:
        retrieved_sections: List of lists containing section names for each query
        query_section: Query section name for labeling
        print_results: Whether to print analysis results
    
    Returns:
        Dictionary containing section distribution statistics
    """
    if not retrieved_sections:
        if print_results:
            print(f"[WARNING] No retrieved_sections data available for analysis")
        return {}
    
    analysis_results = {}
    
    # Analyze different top-k values
    for k in [10, 20, 50, 100]:
        # Collect sections at top-k for each query
        sections_at_k = []
        for query_results in retrieved_sections:
            if len(query_results) >= k:
                sections_at_k.extend(query_results[:k])
        
        if not sections_at_k:
            continue
            
        # Calculate section distribution
        section_counts = Counter(sections_at_k)
        total_retrieved = len(sections_at_k)
        
        section_stats = {}
        for section in ["abstract", "claim", "invention"]:
            count = section_counts.get(section, 0)
            percentage = (count / total_retrieved * 100) if total_retrieved > 0 else 0
            section_stats[section] = {
                'count': count,
                'percentage': percentage
            }
        
        analysis_results[f'top_{k}'] = section_stats
    
    # Print concise summary only once for all k values
    if print_results and analysis_results:
        print(f"\n=== Retrieved Sections Analysis Summary for {query_section}->all ===")
        
        # Create a compact table showing percentages across different k values
        sections = ["abstract", "claim", "invention"]
        k_values = [10, 20, 50, 100]
        
        # Print header
        print(f"{'Section':<12} " + " ".join([f"@{k:<6}" for k in k_values]))
        print("-" * 50)
        
        # Print each section's percentages
        for section in sections:
            row = f"{section:<12} "
            for k in k_values:
                key = f'top_{k}'
                if key in analysis_results and section in analysis_results[key]:
                    pct = analysis_results[key][section]['percentage']
                    row += f"{pct:>6.1f}% "
                else:
                    row += f"{'--':>6}  "
            print(row)
    
    return analysis_results



def compute_multilabel_metrics(y_true, y_pred, threshold=0.5):
    """
    Computes micro/macro F1, precision, and recall for multilabel classification.

    Args:
        y_true (np.ndarray): Binary ground truth matrix of shape (n_samples, n_classes).
        y_pred (np.ndarray): Probability matrix of shape (n_samples, n_classes).
        threshold (float): Threshold to binarize predictions.

    Returns:
        dict: Dictionary of computed metrics.
    """
    y_pred_bin = (y_pred >= threshold).astype(int)

    return {
        "f1_micro": f1_score(y_true, y_pred_bin, average="micro") * 100,
        "f1_macro": f1_score(y_true, y_pred_bin, average="macro") * 100,
        "precision_micro": precision_score(y_true, y_pred_bin, average="micro") * 100,
        "precision_macro": precision_score(y_true, y_pred_bin, average="macro") * 100,
        "recall_micro": recall_score(y_true, y_pred_bin, average="micro") * 100,
        "recall_macro": recall_score(y_true, y_pred_bin, average="macro") * 100,
    }

special_tokens_map = {"abstract": "[abstract]",
                        "claim": "[claim]",
                        "summary": "[summary]",
                        "background": "[invention]",
                        "drawing": "[drawing]",
                        "detailed_description": "[description]"}

def explode_multiview_sections(df, sections, label_column="ipcr_labels", keep_doc=6000):
    """
    Fast version using melt/explode-style operations to flatten multi-section patent documents.
    """
    # Keep only relevant columns
    keep_cols = [label_column, "application_number"] + sections
    df = df[keep_cols].copy()

    # Filter out rows with empty labels
    df = df[df[label_column].apply(lambda x: len(x) > 0)]

    # filter out rows with invalid sections
    for section in sections:
        df = df[df[section].apply(lambda x: isinstance(x, str) and len(x.split()) > 10)]

    # Limit to a maximum number of documents to avoid memory issues
    if len(df) > keep_doc:
        df = df.sample(keep_doc, random_state=42).reset_index(drop=True)
        
    # Melt to long format: each row becomes a (doc_id, section, content)
    melted = df.melt(
        id_vars=["application_number", label_column],
        value_vars=sections,
        var_name="section",
        value_name="section_text"
    )

    # # Drop invalid section_texts
    # melted = melted[melted["section_text"].apply(lambda x: isinstance(x, str) and len(x.split()) > 10)]

    # Final formatting
    melted.rename(columns={
        "application_number": "doc_id",
        label_column: "ipcr_labels",
        "section_text": "text"
    }, inplace=True)

    return melted[["doc_id", "text", "section", "ipcr_labels"]].reset_index(drop=True)


def sample_one_section_per_doc(df):
    return df.groupby("doc_id").apply(lambda x: x.sample(1)).reset_index(drop=True)

def split_train_test_for_multiview(df_exploded, seed, test_size=0.15, required_sections=None):
    """
    Splits a multi-view patent dataframe into a training set and a test set where
    the test set contains only patents that have all required sections.

    Args:
        df (pd.DataFrame): Exploded multi-section dataframe.
        test_size (float): Fraction of patents to use for test set.
        seed (int): Random seed.
        required_sections (List[str]): Required sections for test set (e.g., abstract, claim...).

    Returns:
        train_df, test_df (pd.DataFrame): Train and test splits.
    """
    import random
    random.seed(seed)

    if required_sections is None:
        required_sections = ["abstract", "claim", "summary", "background", "detailed_description"]

    # First, group by doc_id and check which ones have all required sections
    doc_to_sections = df_exploded.groupby("doc_id")["section"].apply(set)
    valid_test_doc_ids = doc_to_sections[doc_to_sections.apply(lambda x: set(required_sections).issubset(x))].index.tolist()

    # Sample test doc ids
    num_test = int(len(valid_test_doc_ids) * test_size)
    sampled_test_ids = set(random.sample(valid_test_doc_ids, num_test))

    # Split
    test_df = df_exploded[df_exploded["doc_id"].isin(sampled_test_ids)].reset_index(drop=True)
    train_df = df_exploded[~df_exploded["doc_id"].isin(sampled_test_ids)].reset_index(drop=True)

    # Then sample 1 section per doc in train
    train_df = sample_one_section_per_doc(train_df)

    return train_df, test_df

def compute_alignment(embeddings1, embeddings2):
    """
    Compute alignment metric using SimCSE standard (mean squared L2 distance).
    
    Args:
        embeddings1: torch.Tensor or numpy.ndarray of shape (N, d) - first set of embeddings
        embeddings2: torch.Tensor or numpy.ndarray of shape (N, d) - second set of embeddings  
    
    Returns:
        float: Mean squared L2 distance between paired embeddings (SimCSE alignment metric)
    """
    # Convert to numpy if needed
    if isinstance(embeddings1, torch.Tensor):
        embeddings1 = embeddings1.detach().cpu().numpy()
    if isinstance(embeddings2, torch.Tensor):
        embeddings2 = embeddings2.detach().cpu().numpy()

    all_embeddings = np.vstack([embeddings1, embeddings2])
    all_normalized = normalize(all_embeddings, norm='l2', axis=1)
    n = len(embeddings1)
    embeddings1 = all_normalized[:n]
    embeddings2 = all_normalized[n:]
    
    # Compute mean squared L2 distance (SimCSE standard)
    differences = embeddings1 - embeddings2
    squared_distances = np.sum(differences ** 2, axis=1)
    mean_squared_l2_distance = np.mean(squared_distances)
    
    return float(mean_squared_l2_distance)


def compute_intra_document_cohesion(embeddings_dict, sections=None, normalize_by_random=True, num_random_pairs=10000):
    """
    Compute intra-document cohesion: average distance between different sections 
    of the same document, optionally normalized by random baseline.
    
    Args:
        embeddings_dict: dict with keys as section names, values as numpy arrays
                        of shape (n_docs, embedding_dim)
        sections: list of section names to consider (default: ['abstract', 'claim', 'invention'])
        normalize_by_random: bool, whether to normalize by random pair distances (default: True)
        num_random_pairs: int, number of random pairs to sample for baseline (default: 10000)
    
    Returns:
        dict with cohesion statistics:
            - mean_cohesion: average intra-document distance across all documents
            - std_cohesion: standard deviation of intra-document distances  
            - cohesion_per_document: list of cohesion scores for each document
            - random_baseline: average distance between random embedding pairs (if normalize_by_random=True)
            - normalized_cohesion: cohesion normalized by random baseline (if normalize_by_random=True)
    """
    if sections is None:
        sections = ['abstract', 'claim', 'invention']
    
    # Validate inputs
    if not all(section in embeddings_dict for section in sections):
        raise ValueError(f"Missing sections in embeddings_dict. Required: {sections}")
    
    # Get number of documents (should be same across all sections)
    n_docs = embeddings_dict[sections[0]].shape[0]
    for section in sections:
        if embeddings_dict[section].shape[0] != n_docs:
            raise ValueError(f"Inconsistent number of documents across sections")
    
    # Compute random baseline if requested
    random_baseline = None
    if normalize_by_random:
        # Collect all embeddings for random sampling
        all_embeddings = []
        for section in sections:
            all_embeddings.append(embeddings_dict[section])
        all_embeddings = np.vstack(all_embeddings)  # Shape: (n_docs * n_sections, embedding_dim)
        
        # Sample random pairs and compute their cosine distances
        n_total = all_embeddings.shape[0]
        random_distances = []
        
        for _ in range(num_random_pairs):
            # Sample two random indices
            i, j = np.random.choice(n_total, size=2, replace=False)
            
            # Compute cosine distance between random pair
            emb_i = all_embeddings[i]
            emb_j = all_embeddings[j]
            
            # Normalize embeddings for cosine distance
            emb_i_norm = emb_i / (np.linalg.norm(emb_i) + 1e-8)
            emb_j_norm = emb_j / (np.linalg.norm(emb_j) + 1e-8)
            
            cosine_sim = np.dot(emb_i_norm, emb_j_norm)
            cosine_dist = 1 - cosine_sim
            random_distances.append(cosine_dist)
        
        random_baseline = np.mean(random_distances)
    
    cohesion_scores = []
    pairwise_section_scores = {}  # Will store scores for each section pair
    
    # Initialize pairwise section tracking
    section_pairs = []
    for i in range(len(sections)):
        for j in range(i + 1, len(sections)):
            pair_name = f"{sections[i]}-{sections[j]}"
            section_pairs.append((i, j, pair_name))
            pairwise_section_scores[pair_name] = []
    
    for doc_idx in range(n_docs):
        # Get embeddings for this document across all sections
        section_embeddings = []
        for section in sections:
            embedding = embeddings_dict[section][doc_idx]
            section_embeddings.append(embedding)
        
        # Calculate pairwise cosine distances between all section pairs
        pairwise_distances = []
        for i, j, pair_name in section_pairs:
            # Use cosine distance: 1 - cosine_similarity
            emb_i = section_embeddings[i]
            emb_j = section_embeddings[j]
            
            # Normalize embeddings for cosine distance
            emb_i_norm = emb_i / (np.linalg.norm(emb_i) + 1e-8)
            emb_j_norm = emb_j / (np.linalg.norm(emb_j) + 1e-8)
            
            cosine_sim = np.dot(emb_i_norm, emb_j_norm)
            cosine_dist = 1 - cosine_sim
            pairwise_distances.append(cosine_dist)
            
            # Store this distance for the specific section pair
            pairwise_section_scores[pair_name].append(cosine_dist)
        
        # Average distance for this document (global cohesion)
        doc_cohesion = np.mean(pairwise_distances)
        cohesion_scores.append(doc_cohesion)
    
    # Prepare results with both global and pairwise section metrics
    results = {
        'mean_cohesion': float(np.mean(cohesion_scores)),
        'std_cohesion': float(np.std(cohesion_scores)),
        'cohesion_per_document': cohesion_scores,
        'num_documents': n_docs
    }
    
    # Add pairwise section results
    pairwise_results = {}
    for pair_name, distances in pairwise_section_scores.items():
        pairwise_results[pair_name] = {
            'mean_cohesion': float(np.mean(distances)),
            'std_cohesion': float(np.std(distances)),
            'cohesion_per_document': distances
        }
    
    results['pairwise_sections'] = pairwise_results
    
    # Add random baseline and normalized metrics if requested
    if normalize_by_random and random_baseline is not None:
        results['random_baseline'] = float(random_baseline)
        
        # Global normalized cohesion: lower values indicate better cohesion relative to random
        # We use ratio: intra_doc_distance / random_distance
        # Values < 1.0 indicate better-than-random cohesion
        results['normalized_cohesion'] = float(np.mean(cohesion_scores) / random_baseline)
        results['cohesion_improvement'] = float(1.0 - (np.mean(cohesion_scores) / random_baseline))
        
        # Add normalized metrics for each section pair
        for pair_name, pair_metrics in pairwise_results.items():
            normalized_cohesion = float(pair_metrics['mean_cohesion'] / random_baseline)
            cohesion_improvement = float(1.0 - normalized_cohesion)
            
            results['pairwise_sections'][pair_name]['normalized_cohesion'] = normalized_cohesion
            results['pairwise_sections'][pair_name]['cohesion_improvement'] = cohesion_improvement
    
    return results