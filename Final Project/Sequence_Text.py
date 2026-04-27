import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import ast
from train_baselines import multi_hot, protein_centric_fmax

from sklearn.metrics import average_precision_score

def load_no_description_ids(path: str) -> set:
    """Load the set of UniProt IDs that had no UniProt description."""
    if not os.path.exists(path):
        print(f"  [warn] {path} not found — assuming no missing descriptions.")
        return set()
    with open(path) as f:
        ids = set(line.strip() for line in f if line.strip())
    print(f"  Loaded {len(ids)} no-description IDs from {path}")
    return ids

class SeqTextDataset(Dataset):
    def __init__(self, csv_path, seq_pkl_path, text_npy_path, label_to_idx, no_description_ids=None):
        # 1. Load the raw data files
        df_raw = pd.read_csv(csv_path)
        with open(seq_pkl_path, "rb") as f:
            self.seq_data = pickle.load(f)  # ESM-2 embeddings dictionary
        text_raw = np.load(text_npy_path)   # PubMedBERT embeddings numpy array

        # 2. FILTERING: Only keep IDs that exist as keys in the sequence dictionary
        # This prevents KeyErrors for IDs like 'P02745'
        mask = df_raw["uniprot_id"].isin(self.seq_data.keys())
        
        # 3. Synchronize all internal structures
        # reset_index is critical so idx 0 to N maps correctly after filtering
        self.df = df_raw[mask].reset_index(drop=True)
        self.text_data = text_raw[mask.values] # Row-aligned PubMedBERT vectors
        
        self.label_to_idx = label_to_idx
        self.no_description_ids = no_description_ids if no_description_ids else set()

        print(f"Dataset Initialized: {len(self.df)} valid proteins (Dropped {len(df_raw) - len(self.df)})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Now this lookup is guaranteed to be safe
        row = self.df.iloc[idx]
        uid = row["uniprot_id"]
        
        z_seq = torch.tensor(self.seq_data[uid], dtype=torch.float32)
        z_text = torch.tensor(self.text_data[idx], dtype=torch.float32)
        
        terms = ast.literal_eval(row["go_terms"]) if isinstance(row["go_terms"], str) else []
        y = torch.tensor(multi_hot(terms, self.label_to_idx), dtype=torch.float32)
        
        return uid, z_seq, z_text, y
    
############################################
# Model
############################################

class TextSeqMLP(nn.Module):
    def __init__(self, in_dim, output_dim, hidden = 256, dropout=0.2):
        super().__init__()
        self.model = nn.Sequential(nn.LayerNorm(in_dim), 
                                   nn.Linear(in_dim, hidden),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(hidden, output_dim))
    def forward(self, z_seq, z_text):
        x = torch.cat([z_seq, z_text], dim=-1)
        return self.model(x)

###############################################
# Inference Fallback  (val / test only)
###############################################
def apply_inference_fallback(target_ds: SeqTextDataset,
                              train_zs_norm: torch.Tensor,
                              train_zt: torch.Tensor) -> SeqTextDataset:
    """
    For every protein in target_ds whose UniProt ID is in no_description_ids,
    substitute the text embedding of its nearest neighbour in training sequence
    space (cosine similarity on ESM-2 embeddings).
 
    Called at inference time only — never on the training set.
    train_zs_norm : (N_train, seq_dim)  L2-normalised ESM-2 embeddings
    train_zt      : (N_train, text_dim) PubMedBERT embeddings
    """
    missing_indices = [
        i for i, row in target_ds.df.iterrows()
        if row["uniprot_id"] in target_ds.no_description_ids
    ]
 
    if not missing_indices:
        print("Fallback: no missing descriptions — skipping.")
        return target_ds
 
    print(f"Fallback: substituting text embeddings for {len(missing_indices)} proteins...")
 
    for i in missing_indices:
        uid = target_ds.df.iloc[i]["uniprot_id"]

        if uid not in target_ds.seq_data:
            print(f"Warning: Cannot apply fallback for {uid} - missing sequence embedding.")
            continue

        z_seq_query = torch.tensor(target_ds.seq_data[uid], dtype=torch.float32)
 
        # Cosine similarity against all training proteins
        q_norm = F.normalize(z_seq_query.unsqueeze(0), dim=-1)   # (1, seq_dim)
        sims   = (q_norm @ train_zs_norm.T).squeeze(0)           # (N_train,)
        nn_idx = torch.argmax(sims).item()
 
        target_ds.text_data[i] = train_zt[nn_idx].numpy()
        print(f"{uid} → nearest neighbour index {nn_idx} "
              f"(cosine sim={sims[nn_idx]:.3f})")
 
    return target_ds

def train_one_epoch(model, loader, optimizer, pos_weight, device):
    model.train()
    total_loss = 0.0
 
    for _, z_seq, z_text, y in loader:
        z_seq, z_text, y = z_seq.to(device), z_text.to(device), y.to(device)
 
        optimizer.zero_grad()
        pred = model(z_seq, z_text)
        loss   = F.binary_cross_entropy_with_logits(pred, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item() * z_seq.size(0)
 
    return total_loss / len(loader.dataset)
 
def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for _, z_seq, z_text, y in dataloader:
            z_seq, z_text = z_seq.to(device), z_text.to(device)
            pred = model(z_seq, z_text)
            pred = torch.sigmoid(pred)

            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    fmax, threshold = protein_centric_fmax(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred, average="micro")
        
    return {"fmax": fmax, "threshold":threshold, "aupr": aupr}