'''
Evaluation script for patent embeddings

IPC-Classification, IPC-KNN use the same dataset.
'''

from __future__ import absolute_import, division, unicode_literals

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
import pickle

from patenteval.utils import (
    load_corpus,
    citation_to_citing_to_cited_dict,
    mean_recall_at_k,
    label_process,
    compute_uniformity,
    compute_alignment,
    compute_ssd,
    TextDataset,
    LinearClassifier,
    KNNClassifier,
    analyze_retrieved_sections_integrated,
    compute_intra_document_cohesion,
)



import pandas as pd
import numpy as np
import re

from tqdm import tqdm
import random

from sklearn.preprocessing import MultiLabelBinarizer
import faiss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead."
)

# ignore UserWarining for unknow labels for IPC classification
warnings.filterwarnings("ignore", category=UserWarning)


class PriorArtEval(object):
    """
    Evaluate prior art retrieval. Given queries and documents (each possibly
    having multiple text fields like 'abstract', 'claim', 'invention'), we
    retrieve top-K documents for each query using cosine distance, then
    compute recall@k and ndcg@k.
    """
    def __init__(self, task_path, params):
        logging.debug('***** Transfer task : Prior Art *****\n\n')
        self.seed = params['seed']
        self.loadFile(task_path)

    def loadFile(self, fpath):
        """
        Load the prior art dataset (queries, documents, and gold citations).
        """
        # 1) Load the queries and documents from JSON
        queries = load_corpus(f"{fpath}/content/queries.json")
        documents = load_corpus(f"{fpath}/content/documents.json")

        # Convert dict_keys to lists so we can index them safely
        query_ids = list(queries.keys())       # e.g. ['Q1', 'Q2', 'Q3', ...]
        doc_ids = list(documents.keys())       # e.g. ['D1', 'D2', 'D3', ...]

        self.data = {
            'queries': queries,        # e.g. queries["abstract"] = [...], etc.
            'documents': documents,    # e.g. documents["claim"] = [...], etc.
            'query_ids': query_ids,
            'doc_ids': doc_ids
        }

        # 2) Load citation mappings (gold standard)
        citation_file = f"{fpath}/mapping/gold.json"
        with open(citation_file) as f:
            raw_citations = json.load(f)
        # Expect something like {query_id: [list_of_cited_doc_ids], ...}
        self.citation_mapping = citation_to_citing_to_cited_dict(raw_citations)

    def do_prepare(self, params, prepare):
        """
        Optional hook to prepare things before run(). 
        SentEval requires this signature even if not used.
        """
        return


    def generate_embeddings(self, texts, batcher, params):
        """
        Efficiently computes embeddings using DataLoader for batch processing.
        """
        dataset = TextDataset(texts)
        dataloader = DataLoader(
            dataset, 
            batch_size=params['batcher_batch_size'], 
            shuffle=False, 
            num_workers=0 if torch.distributed.is_initialized() else 4,
            pin_memory=True)
        
        # Determine embedding size (run once on a small sample)
        sample_embedding = batcher(params, [texts[0]])[0]
        embedding_dim = sample_embedding.shape[-1]
        
        # Preallocate memory for embeddings
        embeddings = np.zeros((len(texts), embedding_dim), dtype=np.float32)

        # Compute embeddings in batches with memory management
        # for i, batch in enumerate(tqdm(dataloader, desc="Computing embeddings for prior art retrieval", leave=False)):
        for i, batch in enumerate(dataloader):
            batch_embeddings = batcher(params, batch)
            start_idx = i * params['batcher_batch_size']
            end_idx = start_idx + len(batch_embeddings)
            embeddings[start_idx:end_idx] = batch_embeddings
            
            # Clean GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return embeddings


    def run(self, params, batcher):
        """
        1) Embed queries and documents for each text type.
        2) Compute top-k retrieval using cosine distances.
        3) Evaluate recall@k.
        """
        # Reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Add these lines for reproducibility
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        # Data references
        data = self.data
        query_ids = list(data['query_ids'])   # list of query keys
        doc_ids = list(data['doc_ids'])       # list of doc keys
        top_k = 100                           # retrieve up to top 100

        # We only want to embed each text type once, so we store them in dicts.
        query_embeddings = {}
        doc_embeddings = {}

        # The queries and documents objects are assumed to be something like:
        #   queries[texttype] = ["text1", "text2", ...] in parallel with query_ids
        #   documents[texttype] = ["doc_text1", "doc_text2", ...] in parallel with doc_ids
        # Make sure that's how your load_corpus is structured.

        # 1) Embed each text type once per query set and document set (in a loop of batches)
        for texttype in ["abstract", "claim", "invention"]:
            if texttype == "abstract":
                query_texts = [data['queries'][q_id]['title'] + f" [SEP] [{texttype}] " + data['queries'][q_id][texttype] for q_id in query_ids]
                doc_texts = [data['documents'][d_id]['title'] + f" [SEP] [{texttype}] " + data['documents'][d_id][texttype] for d_id in doc_ids]
            else:
                query_texts = [f"[{texttype}] " + data['queries'][q_id][texttype] for q_id in query_ids]
                doc_texts = [f"[{texttype}] " + data['documents'][d_id][texttype] for d_id in doc_ids]
            
            query_embeddings[texttype] = self.generate_embeddings(query_texts, batcher, params)
            doc_embeddings[texttype] = self.generate_embeddings(doc_texts, batcher, params)

        ## save the embeddings for later use
        embedding_path = os.path.join(params["model_output_path"], 'citation_embeddings')
        if not os.path.exists(embedding_path):
            os.makedirs(embedding_path)

        # save the embeddings
        if params['final_eval']:
            # save the embeddings with the current step information
            np.savez(os.path.join(embedding_path, 'query_embeddings_final.npz'), **query_embeddings)
            np.savez(os.path.join(embedding_path, 'doc_embeddings_final.npz'), **doc_embeddings)
        else:
            np.savez(os.path.join(embedding_path, f'query_embeddings_{params["current_step"]}.npz'), **query_embeddings)
            np.savez(os.path.join(embedding_path, f'doc_embeddings_{params["current_step"]}.npz'), **doc_embeddings)

            # Safely clean up old files
            try:
                for file in os.listdir(embedding_path):
                    if file.startswith('query_embeddings') and file != f'query_embeddings_{params["current_step"]}.npz':
                        if not file.endswith(f'_{params["current_step"]}.npz') and not file.endswith('_final.npz'):
                            file_path = os.path.join(embedding_path, file)
                            if os.path.exists(file_path):
                                os.remove(file_path)

                    elif file.startswith('doc_embeddings') and file != f'doc_embeddings_{params["current_step"]}.npz':
                        if not file.endswith(f'_{params["current_step"]}.npz') and not file.endswith('_final.npz'):
                            file_path = os.path.join(embedding_path, file)
                            if os.path.exists(file_path):
                                os.remove(file_path)
            except Exception as e:
                logging.warning(f"Error cleaning up old embedding files: {str(e)}")

        # save the query and doc ids for later use
        np.save(os.path.join(embedding_path, 'query_ids.npy'), query_ids)
        np.save(os.path.join(embedding_path, 'doc_ids.npy'), doc_ids)


        # 2) For each texttype1 (query) vs texttype2 (doc), compute retrieval
        results = {}
        for texttype_q in ["abstract", "claim", "invention"]:
            for texttype_d in ["abstract", "claim", "invention"]:
                Q_emb = query_embeddings[texttype_q].astype(np.float32)  # shape [num_queries, emb_dim]
                D_emb = doc_embeddings[texttype_d].astype(np.float32)    # shape [num_docs, emb_dim]

                # Validate shape consistency
                if Q_emb.shape[1] != D_emb.shape[1]:
                    logging.warning(f"Embedding dimension mismatch: Q_emb {Q_emb.shape} vs D_emb {D_emb.shape}")
                    continue

                if np.any(np.isnan(Q_emb)) or np.any(np.isnan(D_emb)):
                    raise ValueError("NaN detected in embeddings before normalization.")

                # Create copies to avoid modifying original data
                Q_emb_norm = Q_emb.copy()
                D_emb_norm = D_emb.copy()
                
                faiss.normalize_L2(Q_emb_norm)  # Normalize before similarity computation
                faiss.normalize_L2(D_emb_norm)
                distances = Q_emb_norm @ D_emb_norm.T  # FAISS optimized cosine similarity

                # Validate distance matrix shape
                expected_shape = (len(query_ids), len(doc_ids))
                if distances.shape != expected_shape:
                    logging.warning(f"Distance matrix shape {distances.shape} != expected {expected_shape}")
                    continue

                # For each query row, we get top_k doc indices (sorted ascending by distance)
                top_k_indices = np.argsort(-distances, axis=1)[:, :top_k]

                # Evaluate retrieval: we build lists of true labels & predicted labels
                true_labels_list, predicted_labels_list = [], []

                # We'll iterate over each query index
                for q_idx, retrieved_docs_indices in enumerate(top_k_indices):
                    # 1) The query ID string, e.g. 'Q1'
                    q_id_str = query_ids[q_idx]
                    # 2) The set of true doc IDs for that query, e.g. ['D3', 'D27']
                    #    Make sure your citation_mapping stores them as a set/list
                    true_labels = self.citation_mapping.get(q_id_str, [])

                    # 3) Convert doc indices to doc ID strings
                    predicted_labels = [doc_ids[d_idx] for d_idx in retrieved_docs_indices]

                    true_labels_list.append(true_labels)
                    predicted_labels_list.append(predicted_labels)

                # Compute recall@k
                results_key = f"{texttype_q}->{texttype_d}"
                results[results_key] = {
                    'recall@10':  mean_recall_at_k(true_labels_list, predicted_labels_list, k=10),
                    'recall@20':  mean_recall_at_k(true_labels_list, predicted_labels_list, k=20),
                    'recall@50':  mean_recall_at_k(true_labels_list, predicted_labels_list, k=50),
                    'recall@100': mean_recall_at_k(true_labels_list, predicted_labels_list, k=100),
                }

        # 3) compute performance for query -> all sections
        for texttype_q in ["abstract", "claim", "invention"]:
            retrieved_sections = []   # for noting which section is retrieved at top_k
            Q_emb = query_embeddings[texttype_q].astype(np.float32)
            D_emb = np.concatenate([doc_embeddings[tt].astype(np.float32) for tt in ["abstract", "claim", "invention"]], axis=0)
            D_ids = np.concatenate([np.array(doc_ids) for _ in ["abstract", "claim", "invention"]], axis=0)

            if np.any(np.isnan(Q_emb)) or np.any(np.isnan(D_emb)):
                raise ValueError("NaN detected in embeddings before normalization.")
            
            faiss.normalize_L2(Q_emb)
            faiss.normalize_L2(D_emb)
            distances = Q_emb @ D_emb.T

            top_k_indices = np.argsort(-distances, axis=1)[:, :top_k * 3]  # top_k * 3 to ensure we have enough candidates

            true_labels_list, predicted_labels_list = [], []
            for q_idx, retrieved_docs_indices in enumerate(top_k_indices):
                q_id_str = query_ids[q_idx]
                true_labels = self.citation_mapping.get(q_id_str, [])
                predicted_labels = [D_ids[d_idx] for d_idx in retrieved_docs_indices]

                # Filter out duplicates in predicted_labels without changing order
                _, unique_indices = np.unique(predicted_labels, return_index=True)
                predicted_labels = [predicted_labels[i] for i in sorted(unique_indices)][:top_k]
                retrieved_sections.append([
                    ["abstract", "claim", "invention"][retrieved_docs_indices[i] // len(doc_ids)] 
                    for i in sorted(unique_indices)[:top_k]
                ])

                true_labels_list.append(true_labels)
                predicted_labels_list.append(predicted_labels)

            results_key = f"{texttype_q}->all"
            results[results_key] = {
                'recall@10':  mean_recall_at_k(true_labels_list, predicted_labels_list, k=10),
                'recall@20':  mean_recall_at_k(true_labels_list, predicted_labels_list, k=20),
                'recall@50':  mean_recall_at_k(true_labels_list, predicted_labels_list, k=50),
                'recall@100': mean_recall_at_k(true_labels_list, predicted_labels_list, k=100),

                'retrieved_sections': retrieved_sections  # for analysis/debugging
            }

            # Compute section analysis for this specific query section
            section_analysis = analyze_retrieved_sections_integrated(
                retrieved_sections, 
                query_section=texttype_q, 
                print_results=False
            )
            results[results_key]['section_analysis'] = section_analysis

        return results


class IPC_ClassificationEval(object):
    """
    Evaluate the IPC classification task (probing task)
    """
    def __init__(self, task_path, params):
        logging.debug('***** Transfer task : IPC Classification *****\n\n')
        self.seed = params['seed']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loadFile(task_path, params)

    def loadFile(self, fpath, params):
        """
        read the csv file of patent dataset for IPC classification
        """
        train_file_path = fpath + f'/train_{params["max_input_len"]}.csv'
        test_file_path = fpath + f'/test_{params["max_input_len"]}.csv'

        # check if the csv file exists
        if os.path.exists(train_file_path) and os.path.exists(test_file_path):
            # Load the train and test datasets
            train_dataset = pd.read_csv(train_file_path)
            test_dataset = pd.read_csv(test_file_path)
        else:
            # Validate that input directory exists
            input_dir = '/home/yzuo/scratch/representation_learning/patentmapv1/data/preprocessed_data/'
            if not os.path.exists(input_dir):
                raise FileNotFoundError(f"Input directory not found: {input_dir}")

            random.seed(self.seed)  # Set the random seed for reproducibility
            # create output directory if it does not exist
            if not os.path.exists(fpath):
                os.makedirs(fpath)

            # Directory where feather files are stored
            # Read all feather datasets from the directory that contain patent files published between 2005 - 2009
            all_feather_files = [f for f in os.listdir(input_dir) if f.endswith('.feather') and any(year in f for year in ['2005', '2006', '2007', '2008', '2009'])]
            
            if not all_feather_files:
                raise FileNotFoundError(f"No feather files found in {input_dir} for years 2005-2009")

            df_final = pd.DataFrame()

            for file in tqdm(all_feather_files, desc="Processing files for 2005-2009", leave=False):
                try:
                    df = pd.read_feather(os.path.join(input_dir, file))
                    
                    # Validate that required columns exist - only need abstract and title
                    required_columns = ["title", "abstract", "ipcr_labels"]
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    if missing_cols:
                        logging.warning(f"Missing columns in {file}: {missing_cols}")
                        continue

                    # Format: title [SEP] abstract (without special section tokens)
                    df["text"] = df["title"] + " [SEP] " + df["abstract"]
                    df["section"] = "abstract"  # Mark all as abstract section
                    
                    # Keep only necessary columns
                    df = df[["text", "ipcr_labels", "section"]].copy()
                    
                    # Filter out entries with empty text or labels
                    df = df.dropna(subset=["text", "ipcr_labels"])
                    df = df[df["text"].str.strip() != ""]
                    df = df[df["ipcr_labels"].astype(str).str.strip() != ""]

                    df = df.sample(min(len(df), 10000), random_state=self.seed)  # sample 10000 patents from each file

                    # Check if there's still data after filtering
                    if len(df) == 0:
                        logging.warning(f"No valid data remaining in {file} after filtering")
                        continue

                    df_final = pd.concat([df_final, df], ignore_index=True)

                    # clear memory
                    del df
                    
                except Exception as e:
                    logging.error(f"Error processing {file}: {str(e)}")
                    continue

            train_dataset, test_dataset = train_test_split(df_final, test_size=0.15, random_state=self.seed)

            assert len(test_dataset) > 0, "Test dataset is empty. Please check the data split."

            # keep only the relevant columns (all entries are already abstract section)
            train_dataset = train_dataset[['text', 'ipcr_labels', 'section']]
            test_dataset = test_dataset[['text', 'ipcr_labels', 'section']]

            # save the train and test datasets
            train_dataset.to_csv(train_file_path, index=False)
            test_dataset.to_csv(test_file_path, index=False)

        if type(train_dataset['ipcr_labels'].iloc[0]) == str:
            # Robust parser that handles both space-separated and comma-separated formats
            # e.g., "['A' 'B' 'C']" or "['A', 'B', 'C']"
            parse_labels = lambda x: re.findall(r"'([^']+)'", x)
            train_dataset['ipcr_labels'] = train_dataset['ipcr_labels'].apply(parse_labels)
            test_dataset['ipcr_labels'] = test_dataset['ipcr_labels'].apply(parse_labels)
        else:   # numpy array
            train_dataset['ipcr_labels'] = train_dataset['ipcr_labels'].apply(lambda x: x.tolist())
            test_dataset['ipcr_labels'] = test_dataset['ipcr_labels'].apply(lambda x: x.tolist())

        # Load the MultiLabelBinarizer and number of classes
        self.mlb = MultiLabelBinarizer()
        all_labels = train_dataset['ipcr_labels'].apply(label_process).tolist()
        self.mlb.fit(all_labels)
        self.num_classes = len(self.mlb.classes_)

        logging.debug(f"Number of classes: {self.num_classes}")
        logging.debug(f"Classes: {self.mlb.classes_}")

        # extract the primary labels for evaluation (indicated by the first label in the list, which is the primary label)
        self.test_primary_labels = test_dataset['ipcr_labels'].apply(lambda x: x[0]).apply(label_process).tolist()

        # convert the labels to binary format
        self.train_labels = np.array(list(self.mlb.transform(train_dataset['ipcr_labels'].apply(label_process))))  # shape: (n_samples, n_classes)
        self.test_labels = np.array(list(self.mlb.transform(test_dataset['ipcr_labels'].apply(label_process))))

        self.train_text = train_dataset['text'].tolist()
        self.test_text = test_dataset['text'].tolist()

        self.train_section = train_dataset['section'].tolist()
        self.test_section = test_dataset['section'].tolist()

        # save the MultiLabelBinarizer, the primary labels and the labels (for KNN evaluation)
        pickle.dump(self.mlb, open(fpath + f'/mlb.pkl', 'wb'))
        np.save(fpath + f'/test_primary_labels.npy', self.test_primary_labels)
        np.save(fpath + f'/train_labels.npy', self.train_labels)
        np.save(fpath + f'/test_labels.npy', self.test_labels)


    def do_prepare(self, params, prepare):
        return


    def generate_embeddings(self, texts, params, batcher):
        dataset = TextDataset(texts)
        dataloader = DataLoader(
            dataset, 
            batch_size=params['batcher_batch_size'], 
            shuffle=False, 
            num_workers=0 if torch.distributed.is_initialized() else 4,
            pin_memory=True)

        # Preallocate memory
        embeddings = np.zeros((len(texts), params['embedding_dim']), dtype=np.float32)

        start_idx = 0
        # for batch in tqdm(dataloader, desc="Generating embeddings for IPC classification", leave=False):
        for batch in dataloader:
            batch_embeddings = batcher(params, batch)
            batch_size = batch_embeddings.shape[0]  # Get actual batch size

            embeddings[start_idx:start_idx + batch_size] = batch_embeddings
            start_idx += batch_size  # Move index for next batch

        return embeddings


    def train_evaluate(self, X_train, y_train, X_test, y_test, primary_true, ks = [1, 3, 5]):
        import copy
        assert not (len(X_train) == 0 or len(X_test) == 0), "No data to train or test on."
        
        # Validate data dimensions
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(f"Feature dimension mismatch: train {X_train.shape[1]} vs test {X_test.shape[1]}")
        
        # Check for NaN or inf in data
        if torch.any(torch.isnan(X_train)) or torch.any(torch.isinf(X_train)):
            logging.warning("NaN or inf detected in training data")
            X_train = torch.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.any(torch.isnan(X_test)) or torch.any(torch.isinf(X_test)):
            logging.warning("NaN or inf detected in test data")
            X_test = torch.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Convert data to PyTorch tensors and divide into batches
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=self.seed)
        
        # Create DataLoaders for mini-batch approach
        batch_size = 256
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        ipc_model = LinearClassifier(X_train.shape[1], y_train.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(ipc_model.parameters(), lr=3e-4, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()  # Correct pour des logits sans sigmoid
        
        # Initialize early stopping parameters
        best_val_loss = float('inf')
        patience = 5
        epochs_no_improve = 0
        best_state_dict = None
        num_epochs = 100
        
        for epoch in range(num_epochs):
            # Training mode
            ipc_model.train()
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = ipc_model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # Check if loss is NaN
                if torch.isnan(loss):
                    logging.warning(f"NaN loss detected at epoch {epoch}")
                    break
                    
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(ipc_model.parameters(), max_norm=1.0)
                optimizer.step()

            # Evaluation mode
            ipc_model.eval()
            with torch.no_grad():
                val_outputs = ipc_model(X_valid.to(self.device))
                val_loss = criterion(val_outputs, y_valid.to(self.device))
                
                if torch.isnan(val_loss):
                    logging.warning(f"NaN validation loss detected at epoch {epoch}")
                    break
            
            # Check if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0  # Reset counter
                best_state_dict = copy.deepcopy(ipc_model.state_dict())
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.debug(f"Early stopping at epoch {epoch} with best val loss: {best_val_loss}")
                    break
        
        # Load the best model state
        if best_state_dict is not None:
            ipc_model.load_state_dict(best_state_dict)
        
        # Evaluation on test set
        ipc_model.eval()
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        all_probs = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                logits = ipc_model(batch_x)
                probs = torch.sigmoid(logits)  # Apply sigmoid here to get probabilities
                all_probs.append(probs.cpu())
        
        # Concatenate all batches of probabilities
        probs = torch.cat(all_probs, dim=0).numpy()
        
        # Check if predicted probabilities are valid
        if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
            logging.warning("Invalid probabilities detected, using fallback")
            probs = np.random.rand(*probs.shape)
        
        # Calculate precision@k
        pred_topk = np.argsort(-probs, axis=1)
        metrics = {}
        
        for k in ks:
            topk_indices = pred_topk[:, :k]  # Extract top-k indices for each sample
            precision_at_k = np.mean([
                len(set(np.where(true == 1)[0]).intersection(set(pred_indices))) / k
                for true, pred_indices in zip(y_test.cpu().numpy(), topk_indices)
            ])
            metrics[f"precision@{k}"] = precision_at_k * 100
        
        # Primary label accuracy
        pred_primary = self.mlb.classes_[np.argmax(probs, axis=1)]
        accuracy_primary = accuracy_score(primary_true, pred_primary)
        metrics["accuracy_primary"] = accuracy_primary * 100
        
        return metrics

    def run(self, params, batcher):
        # Reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Add these lines for reproducibility
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        if not bool(params['final_eval']) and (params['eval_sample_train'] and params['eval_sample_test'] ):
            idx_train = np.random.choice(len(self.train_text), params['eval_sample_train'], replace=False)
            self.train_text = [self.train_text[i] for i in idx_train]

            idx_test = np.random.choice(len(self.test_text), params['eval_sample_test'], replace=False)
            self.test_text = [self.test_text[i] for i in idx_test]

            # save the indices for later use
            # create the directory if it does not exist
            embedding_dir = os.path.join(params["model_output_path"], 'ipc_embeddings')
            if not os.path.exists(embedding_dir):
                os.makedirs(embedding_dir)
            np.save(os.path.join(embedding_dir, f'idx_train_{params["current_step"]}.npy'), idx_train)
            np.save(os.path.join(embedding_dir, f'idx_test_{params["current_step"]}.npy'), idx_test)

        train_embeddings = self.generate_embeddings(self.train_text, params, batcher)
        test_embeddings = self.generate_embeddings(self.test_text, params, batcher)

        # Prepare embeddings and labels
        train_embeddings = np.array(train_embeddings)
        train_labels = np.array(self.train_labels)
        train_section = np.array(self.train_section)
        test_embeddings = np.array(test_embeddings)
        test_labels = np.array(self.test_labels)
        test_section = np.array(self.test_section)
        test_primary_labels = np.array(self.test_primary_labels)

        if not params['final_eval'] and (params['eval_sample_train'] is not None and params['eval_sample_test'] is not None):
            train_labels = train_labels[idx_train]
            train_section = train_section[idx_train]
            test_labels = test_labels[idx_test]
            test_section = test_section[idx_test]
            test_primary_labels = np.array(self.test_primary_labels)[idx_test]
            

        # save the embeddings and labels (for KNN evaluation), including current step information
        embedding_path = os.path.join(params["model_output_path"], 'ipc_embeddings')
        if not os.path.exists(embedding_path):
            os.makedirs(embedding_path)
        
        # save the embeddings
        np.save(os.path.join(embedding_path, f'train_embeddings_{params["current_step"]}.npy'), train_embeddings)
        np.save(os.path.join(embedding_path, f'test_embeddings_{params["current_step"]}.npy'), test_embeddings)

        # Save the processed labels and sections that correspond to the saved embeddings
        # Handle both torch tensors and numpy arrays
        train_labels_np = train_labels.cpu().numpy() if hasattr(train_labels, 'cpu') else train_labels
        test_labels_np = test_labels.cpu().numpy() if hasattr(test_labels, 'cpu') else test_labels
        
        np.save(os.path.join(embedding_path, f'train_labels_{params["current_step"]}.npy'), train_labels_np)
        np.save(os.path.join(embedding_path, f'test_labels_{params["current_step"]}.npy'), test_labels_np)
        np.save(os.path.join(embedding_path, f'train_section_{params["current_step"]}.npy'), train_section)
        np.save(os.path.join(embedding_path, f'test_section_{params["current_step"]}.npy'), test_section)
        np.save(os.path.join(embedding_path, f'test_primary_labels_{params["current_step"]}.npy'), test_primary_labels)

        # remove the other embeddings and labels that were saved in previous steps
        for file in os.listdir(embedding_path):
            if file.startswith('train_embeddings') and file != f'train_embeddings_{params["current_step"]}.npy':
                if not file.endswith(f'_{params["current_step"]}.npy'):
                    os.remove(os.path.join(embedding_path, file))        
            elif file.startswith('test_embeddings') and file != f'test_embeddings_{params["current_step"]}.npy':
                if not file.endswith(f'_{params["current_step"]}.npy'):
                    os.remove(os.path.join(embedding_path, file))
            elif file.startswith('train_labels') and file != f'train_labels_{params["current_step"]}.npy':
                if not file.endswith(f'_{params["current_step"]}.npy'):
                    os.remove(os.path.join(embedding_path, file))
            elif file.startswith('test_labels') and file != f'test_labels_{params["current_step"]}.npy':
                if not file.endswith(f'_{params["current_step"]}.npy'):
                    os.remove(os.path.join(embedding_path, file))
            elif file.startswith('train_section') and file != f'train_section_{params["current_step"]}.npy':
                if not file.endswith(f'_{params["current_step"]}.npy'):
                    os.remove(os.path.join(embedding_path, file))
            elif file.startswith('test_section') and file != f'test_section_{params["current_step"]}.npy':
                if not file.endswith(f'_{params["current_step"]}.npy'):
                    os.remove(os.path.join(embedding_path, file))
            elif file.startswith('test_primary_labels') and file != f'test_primary_labels_{params["current_step"]}.npy':
                if not file.endswith(f'_{params["current_step"]}.npy'):
                    os.remove(os.path.join(embedding_path, file))
            elif file.startswith('idx_train') and file != f'idx_train_{params["current_step"]}.npy':
                if not file.endswith(f'_{params["current_step"]}.npy'):
                    os.remove(os.path.join(embedding_path, file))
            elif file.startswith('idx_test') and file != f'idx_test_{params["current_step"]}.npy':
                if not file.endswith(f'_{params["current_step"]}.npy'):
                    os.remove(os.path.join(embedding_path, file))

        results = {}
        # 1) Train and evaluate the model on abstract section only
        # Filter to abstract section only to avoid data leakage from other sections of the same patent
        train_abstract_mask = np.array(train_section) == 'abstract'
        test_abstract_mask = np.array(test_section) == 'abstract'
        
        train_embeddings_abstract = train_embeddings[train_abstract_mask]
        train_labels_abstract = train_labels[train_abstract_mask]
        test_embeddings_abstract = test_embeddings[test_abstract_mask]
        test_labels_abstract = test_labels[test_abstract_mask]
        test_primary_labels_abstract = np.array(test_primary_labels)[test_abstract_mask]
        
        logging.info(f"IPC Classification: Using {len(train_embeddings_abstract)} train / {len(test_embeddings_abstract)} test abstracts (filtered from {len(train_embeddings)} / {len(test_embeddings)} total)")
        
        train_embeddings_t = torch.tensor(train_embeddings_abstract, dtype=torch.float32).to(self.device)
        train_labels_t = torch.tensor(train_labels_abstract, dtype=torch.float32).to(self.device)
        test_embeddings_t = torch.tensor(test_embeddings_abstract, dtype=torch.float32).to(self.device)
        test_labels_t = torch.tensor(test_labels_abstract, dtype=torch.float32).to(self.device)

        # Train and evaluate the model on abstract section only
        metrics = self.train_evaluate(train_embeddings_t, train_labels_t, test_embeddings_t, test_labels_t, test_primary_labels_abstract)
        results['abstract'] = metrics
        
        # # 2) Train and evaluate the model on each section
        # for section in ["abstract", "claim", "summary", "background", "detailed_description"]:
        #     section_indices = np.where(test_section == section)[0]
        #     train_indices = np.where(train_section == section)[0]
        #     if len(section_indices) == 0:
        #         continue
        #     X_train = train_embeddings[train_indices]
        #     y_train = train_labels[train_indices]
        #     X_test = test_embeddings[section_indices]
        #     y_test = test_labels[section_indices]
        #     primary_true = test_primary_labels[section_indices]

        #     metrics = self.train_evaluate(X_train, y_train, X_test, y_test, primary_true)
        #     results[section] = metrics

        # 

        logging.debug('Results: {}'.format(results))
        return results



class IPC_KNNEval(object):
    """
    Evaluate the IPC classification via KNN (using the same dataset as IPC_Classification)
    """
    def __init__(self, task_path, params):
        logging.debug('***** Transfer task : IPC KNN *****\n\n')
        self.seed = params['seed']
        self.loadFile(task_path, params)

    def loadFile(self, fpath, params):
        """
        Load the numpy files of embeddings and labels that were saved by IPC_ClassificationEval
        """
        embedding_path = os.path.join(params["model_output_path"], 'ipc_embeddings')
        
        # Load embeddings
        self.train_embeddings = np.load(os.path.join(embedding_path, f'train_embeddings_{params["current_step"]}.npy'), allow_pickle=True)
        self.test_embeddings = np.load(os.path.join(embedding_path, f'test_embeddings_{params["current_step"]}.npy'), allow_pickle=True)

        # Load the processed labels and sections that correspond to the embeddings
        # These were saved by IPC_ClassificationEval and are already properly sampled/filtered
        self.train_labels = np.load(os.path.join(embedding_path, f'train_labels_{params["current_step"]}.npy'), allow_pickle=True)
        self.test_labels = np.load(os.path.join(embedding_path, f'test_labels_{params["current_step"]}.npy'), allow_pickle=True)
        self.train_section = np.load(os.path.join(embedding_path, f'train_section_{params["current_step"]}.npy'), allow_pickle=True).tolist()
        self.test_section = np.load(os.path.join(embedding_path, f'test_section_{params["current_step"]}.npy'), allow_pickle=True).tolist()
        self.test_primary_labels = np.load(os.path.join(embedding_path, f'test_primary_labels_{params["current_step"]}.npy'), allow_pickle=True)

        # Load the MultiLabelBinarizer (this remains unchanged)
        self.mlb = pickle.load(open(fpath + f'/mlb.pkl', 'rb'))


    def do_prepare(self, params, prepare):
        return
    

    def evaluate_knn(self, X_train, y_train, X_test, y_test, y_true_primary):
        knn = KNNClassifier(metric='cosine')
        X_subtrain, X_val, y_subtrain, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=self.seed)
        best_k, _ = knn.tune_k_by_precision_at_k(X_subtrain, y_subtrain, X_val, y_val, candidate_k_list=[1, 3, 5, 10])
        
        knn.n_neighbors = best_k
        knn.fit(X_train, y_train)
        probabilities = knn.predict_proba(X_test)

        pred_topk = np.argsort(-probabilities, axis=1)
        precision_at_k = {}
        for k in [1, 3, 5]:
            topk_indices = pred_topk[:, :k]  # Extract top-k indices for each sample
            precision_at_k[f"precision@{k}"] = np.mean([
                len(set(np.where(true == 1)[0]).intersection(set(pred_indices))) / k
                for true, pred_indices in zip(y_test, topk_indices)
            ]) * 100

        y_pred_primary = self.mlb.classes_[np.argmax(probabilities, axis=1)]
        accuracy_primary = accuracy_score(y_true_primary, y_pred_primary) * 100

        return {"accuracy_primary": accuracy_primary, **precision_at_k}


    def run(self, params, batcher):
        results = {}

        # Filter to abstract section only to avoid data leakage from other sections of the same patent
        train_abstract_mask = np.array(self.train_section) == 'abstract'
        test_abstract_mask = np.array(self.test_section) == 'abstract'
        
        train_embeddings = self.train_embeddings[train_abstract_mask]
        train_labels = self.train_labels[train_abstract_mask]
        test_embeddings = self.test_embeddings[test_abstract_mask]
        test_labels = self.test_labels[test_abstract_mask]
        test_primary_labels = np.array(self.test_primary_labels)[test_abstract_mask]
        
        logging.info(f"IPC KNN: Using {len(train_embeddings)} train / {len(test_embeddings)} test abstracts (filtered from {len(self.train_embeddings)} / {len(self.test_embeddings)} total)")

        # Train and evaluate the model on abstract section only
        metrics = self.evaluate_knn(train_embeddings, train_labels, test_embeddings, test_labels, test_primary_labels)
        results['abstract'] = metrics

        # # 2) Train and evaluate the model on each section
        # section_list = ["abstract", "claim", "summary", "background", "detailed_description"]
        # for section in section_list:
        #     indices = [i for i, sec in enumerate(self.test_section) if sec == section]
        #     if not indices:
        #         continue
        #     X_test = self.test_embeddings[indices]
        #     y_test = self.test_labels[indices]

        #     assert len(X_test) > 0, f"No samples found for section {section} in test set."
        #     y_true_primary = [self.test_primary_labels[i] for i in indices]

        #     section_metrics = self.evaluate_knn(self.train_embeddings, self.train_labels, X_test, y_test, y_true_primary)
        #     results[section] = section_metrics

        logging.debug('Results: {}'.format(results))
        return results



class SingularSpectrumEval(object):
    def __init__(self, task_path, params):
        logging.debug('***** Transfer task : Singular Distribution Entropy *****\n\n')
        self.seed = params['seed']
        self.embedding_path = os.path.join(params["model_output_path"], 'citation_embeddings')
        self.sections = ["abstract", "claim", "invention"]
        self.current_step = params["current_step"]
        self.query_embeddings = self.load_citation_embeddings(prefix="query", if_final_eval=params['final_eval'])
        self.doc_embeddings = self.load_citation_embeddings(prefix="doc", if_final_eval=params['final_eval'])

    def load_citation_embeddings(self, prefix="query", if_final_eval=False):
        """
        Load citation embeddings from npz files.
        If if_final_eval is True, it will load the embeddings from the final evaluation step.
        """
        if if_final_eval:
            npz_file = os.path.join(self.embedding_path, f'{prefix}_embeddings_final.npz')
        else:
            npz_file = os.path.join(self.embedding_path, f'{prefix}_embeddings_{self.current_step}.npz')
            if not os.path.exists(npz_file):
                raise FileNotFoundError(f"Missing npz embedding file for {prefix}: {npz_file}")
        npz_data = np.load(npz_file, allow_pickle=True)
        return {key: npz_data[key] for key in npz_data.files}

    def do_prepare(self, params, prepare):
        return

    def run(self, params, batcher):
        results = {}

        for section in self.sections:
            all_embeddings = np.concatenate(
                [self.query_embeddings[section], self.doc_embeddings[section]], axis=0
            )
            results[section] = compute_ssd(all_embeddings, l2_normalize_rows=True)

        # Global across all sections
        all_embeddings = np.concatenate(
            [self.query_embeddings[sec] for sec in self.sections] +
            [self.doc_embeddings[sec] for sec in self.sections],
            axis=0
        )
        results['global'] = compute_ssd(all_embeddings, l2_normalize_rows=True)
        return results



class UniformityEval:
    def __init__(self, task_path, params):
        logging.debug('***** Transfer task : Uniformity *****\n\n')
        self.seed = params['seed']
        self.embedding_path = os.path.join(params["model_output_path"], 'citation_embeddings')
        self.sections = ["abstract", "claim", "invention"]
        self.current_step = params["current_step"]
        self.query_embeddings = self.load_citation_embeddings(prefix="query", if_final_eval=params['final_eval'])
        self.doc_embeddings = self.load_citation_embeddings(prefix="doc", if_final_eval=params['final_eval'])

    def load_citation_embeddings(self, prefix="query", if_final_eval=False):
        """
        Load citation embeddings from npz files.
        If if_final_eval is True, it will load the embeddings from the final evaluation step.
        """
        if if_final_eval:
            npz_file = os.path.join(self.embedding_path, f'{prefix}_embeddings_final.npz')
        else:
            # Load the embeddings for the current step
            npz_file = os.path.join(self.embedding_path, f'{prefix}_embeddings_{self.current_step}.npz')
            if not os.path.exists(npz_file):
                raise FileNotFoundError(f"Missing npz embedding file for {prefix}: {npz_file}")
        npz_data = np.load(npz_file, allow_pickle=True)
        return {key: npz_data[key] for key in npz_data.files}

    def run(self, params, batcher):
        results = {}
        for section in self.sections:
            # Compute the uniformity for each section
            all_embeddings = np.concatenate([
                self.query_embeddings[section].astype(np.float32),
                self.doc_embeddings[section].astype(np.float32)
            ], axis=0)
            results[section] = compute_uniformity(all_embeddings)

        # Global uniformity: all section x section combinations
        all_embeddings = np.concatenate([
            self.query_embeddings[sec].astype(np.float32)
            for sec in self.sections
        ] + [
            self.doc_embeddings[sec].astype(np.float32)
            for sec in self.sections
        ], axis=0)
        results['global'] = compute_uniformity(all_embeddings)

        return results



class AlignmentEval(object):
    def __init__(self, task_path, params):
        logging.debug('***** Transfer task : Alignment *****\n\n')
        self.seed = params['seed']
        self.embedding_path = os.path.join(params["model_output_path"], 'citation_embeddings')
        self.sections = ["abstract", "claim", "invention"]
        self.current_step = params["current_step"]

        # Load the query and document IDs
        self.query_ids = np.load(os.path.join(self.embedding_path, 'query_ids.npy'), allow_pickle=True)
        self.doc_ids = np.load(os.path.join(self.embedding_path, 'doc_ids.npy'), allow_pickle=True)

        # Load citation mappings (gold standard)
        citation_file = f"{task_path}/mapping/gold.json"
        with open(citation_file) as f:
            raw_citations = json.load(f)
        # Expect something like {query_id: [list_of_cited_doc_ids], ...}
        self.citation_mapping = citation_to_citing_to_cited_dict(raw_citations)

        self.query_embeddings = self.load_citation_embeddings(prefix="query", if_final_eval=params['final_eval'])
        self.doc_embeddings = self.load_citation_embeddings(prefix="doc", if_final_eval=params['final_eval'])

    def load_citation_embeddings(self, prefix="query", if_final_eval=False):
        """
        Load citation embeddings from npz files.
        If if_final_eval is True, it will load the embeddings from the final evaluation step.
        """
        if if_final_eval:
            npz_file = os.path.join(self.embedding_path, f'{prefix}_embeddings_final.npz')
        else:
            npz_file = os.path.join(self.embedding_path, f'{prefix}_embeddings_{self.current_step}.npz')
            if not os.path.exists(npz_file):
                raise FileNotFoundError(f"Missing npz embedding file for {prefix}: {npz_file}")
        npz_data = np.load(npz_file, allow_pickle=True)
        return {key: npz_data[key] for key in npz_data.files}

    def run(self, params, batcher):
        results = {}
        # mapping: index -> doc_id
        qid_list = list(self.query_ids)
        did_list = list(self.doc_ids)

        for section in self.sections:
            qe = self.query_embeddings[section].astype(np.float32)
            de = self.doc_embeddings[section].astype(np.float32)
            
            query_pairs = []
            doc_pairs = []

            for q_idx, q_id in enumerate(qid_list):
                cited_doc_ids = self.citation_mapping.get(q_id, [])
                for doc_id in cited_doc_ids:
                    for i, did in enumerate(did_list):
                        if did == doc_id:
                            query_pairs.append(qe[q_idx])
                            doc_pairs.append(de[i])

            results[section] = {
                "mean_alignment": float(compute_alignment(np.array(query_pairs), np.array(doc_pairs))),
            }

        # Global alignment: across all sections
        global_query_pairs = []
        global_doc_pairs = []
        for q_ids, q_id in enumerate(qid_list):
            cited_doc_ids = self.citation_mapping.get(q_id, [])
            for doc_id in cited_doc_ids:
                for i, did in enumerate(did_list):
                    if did == doc_id:
                        for section1 in self.sections:
                            for section2 in self.sections:
                                global_query_pairs.append(self.query_embeddings[section1][q_ids].astype(np.float32))
                                global_doc_pairs.append(self.doc_embeddings[section2][i].astype(np.float32))
        
        # Calculate global alignment scores using standardized function
        global_alignment = compute_alignment(np.array(global_query_pairs), np.array(global_doc_pairs))
        results['global'] = {
            "mean_alignment": float(global_alignment),
        }

        return results


class TopologyEval(object):
    """
    Evaluate the topological structure of embedding space, focusing on 
    intra-document section relationships (cohesion within documents).
    """
    
    def __init__(self, task_path, params):
        logging.debug('***** Transfer task : Topology Evaluation *****\n\n')
        self.seed = params['seed']
        self.embedding_path = os.path.join(params["model_output_path"], 'citation_embeddings')
        self.sections = ["abstract", "claim", "invention"]
        self.current_step = params["current_step"]
        
        # Load document embeddings from citation embeddings
        self.doc_embeddings = self.load_citation_embeddings(prefix="doc", 
                                                           if_final_eval=params['final_eval'])
        
        # Load document IDs
        self.doc_ids = np.load(os.path.join(self.embedding_path, 'doc_ids.npy'), 
                              allow_pickle=True)

    def load_citation_embeddings(self, prefix="doc", if_final_eval=False):
        """Load citation embeddings for all sections"""
        embeddings = {}
        
        for section in self.sections:
            if if_final_eval:
                embedding_file = f'{prefix}_embeddings_{section}_final.npy'
            else:
                embedding_file = f'{prefix}_embeddings_{section}_{self.current_step}.npy'
            
            embedding_path = os.path.join(self.embedding_path, embedding_file)
            
            if os.path.exists(embedding_path):
                embeddings[section] = np.load(embedding_path)
                logging.debug(f"Loaded {section} embeddings: {embeddings[section].shape}")
            else:
                logging.warning(f"Embedding file not found: {embedding_path}")
                # Return None if embeddings are missing - evaluation will be skipped
                return None
        
        return embeddings

    def run(self, params, batcher):
        """
        Evaluate intra-document cohesion across all sections
        """
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Check if we have the required embeddings
        if self.doc_embeddings is None:
            logging.warning("Document embeddings not found. Skipping topology evaluation.")
            return {
                'error': 'missing_embeddings',
                'message': 'Document embeddings not available for topology evaluation'
            }

        results = {}
        
        try:
            # Compute intra-document cohesion using the utility function with random baseline
            cohesion_results = compute_intra_document_cohesion(
                embeddings_dict=self.doc_embeddings,
                sections=self.sections,
                normalize_by_random=True,
                num_random_pairs=10000
            )
            
            # Store results for each section analysis
            results['global'] = {
                'mean_cohesion': cohesion_results['mean_cohesion'],
                'std_cohesion': cohesion_results['std_cohesion'],
                'num_documents': len(cohesion_results['cohesion_per_document'])
            }
            
            # Add normalized cohesion results if available
            if 'normalized_cohesion' in cohesion_results:
                results['global'].update({
                    'random_baseline': cohesion_results['random_baseline'],
                    'normalized_cohesion': cohesion_results['normalized_cohesion'],
                    'cohesion_improvement': cohesion_results['cohesion_improvement']
                })
            
            # Add per-section pair analysis for more detailed insights
            section_pairs = [
                ('abstract', 'claim'),
                ('abstract', 'invention'), 
                ('claim', 'invention')
            ]
            
            # Use the global random baseline from the main cohesion analysis for consistency
            global_random_baseline = cohesion_results.get('random_baseline')
            
            for section1, section2 in section_pairs:
                pair_cohesions = []
                
                # Compute actual pairwise distances for this section pair
                for doc_idx in range(len(self.doc_ids)):
                    emb1 = self.doc_embeddings[section1][doc_idx]
                    emb2 = self.doc_embeddings[section2][doc_idx]
                    
                    # Normalize for cosine distance
                    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
                    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
                    
                    cosine_sim = np.dot(emb1_norm, emb2_norm)
                    cosine_dist = 1 - cosine_sim
                    pair_cohesions.append(cosine_dist)
                
                pair_key = f"{section1}_{section2}"
                results[pair_key] = {
                    'mean_distance': float(np.mean(pair_cohesions)),
                    'std_distance': float(np.std(pair_cohesions))
                }
                
                # Add normalized metrics if global baseline is available
                if global_random_baseline is not None:
                    results[pair_key].update({
                        'random_baseline': float(global_random_baseline),
                        'normalized_distance': float(np.mean(pair_cohesions) / global_random_baseline),
                        'distance_improvement': float(1.0 - (np.mean(pair_cohesions) / global_random_baseline))
                    })
            
            logging.debug(f"Topology evaluation completed with {len(self.doc_ids)} documents")
            
        except Exception as e:
            logging.error(f"Error in topology evaluation: {str(e)}")
            results['error'] = str(e)
        
        return results