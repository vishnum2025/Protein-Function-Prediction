import os
import argparse
import ast
import json
import math
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from structure_encoder import StructureDataset, StructureEncoder


#Converts a GO-term cell from the CSV into a Python list.
def parse_go(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    return ast.literal_eval(x)


#Builds the GO label set using only terms that appear at least min_freq times.
def build_label_space(csv_path, min_freq):
    df = pd.read_csv(csv_path)
    counts = Counter()
    for terms in df["go_terms"].map(parse_go):
        counts.update(terms)
    return sorted([t for t, c in counts.items() if c >= min_freq])


#Converts a protein's GO terms into a multi-hot vector.
def multi_hot(terms, label_to_idx):
    y = np.zeros(len(label_to_idx), dtype=np.float32)
    for t in terms:
        if t in label_to_idx:
            y[label_to_idx[t]] = 1.0
    return y


#Computes positive-class weights to reduce the effect of GO label imbalance.
def class_pos_weight(csv_path, label_to_idx, cap=10.0):
    df = pd.read_csv(csv_path)
    n = len(df)
    pos = np.zeros(len(label_to_idx), dtype=np.float32)

    for terms in df["go_terms"].map(parse_go):
        for t in terms:
            if t in label_to_idx:
                pos[label_to_idx[t]] += 1

    pos = np.clip(pos, 1, None)
    w = np.sqrt(np.clip((n - pos) / pos, 1.0, None))
    w = np.clip(w, 1.0, cap)
    return torch.tensor(w, dtype=torch.float32)


class GOOntology:
    #Loads the GO ontology file and stores namespaces (MF, BP, CC)/parent relationships.
    def __init__(self, obo_path):
        self.namespace = {}
        self.parents = defaultdict(set)
        self.cache = {}

        if not os.path.exists(obo_path):
            raise FileNotFoundError(
                f"Missing {obo_path}. Download with:\n"
                f"curl -L http://purl.obolibrary.org/obo/go/go-basic.obo -o {obo_path}"
            )

        self._load_obo(obo_path)

    #Parses go-basic.obo to collect GO term namespaces and parent terms.
    def _load_obo(self, path):
        current_id = None
        current_ns = None
        current_parents = set()
        obsolete = False
        in_term = False

        #Saves the current GO term after finishing a block.
        def save():
            if current_id and current_ns and not obsolete:
                self.namespace[current_id] = current_ns
                self.parents[current_id] = set(current_parents)

        with open(path) as f:
            for line in f:
                line = line.strip()

                if line == "[Term]":
                    save()
                    current_id = None
                    current_ns = None
                    current_parents = set()
                    obsolete = False
                    in_term = True
                    continue

                if line.startswith("[") and line != "[Term]":
                    save()
                    in_term = False
                    continue

                if not in_term:
                    continue

                if line.startswith("id: GO:"):
                    current_id = line.replace("id: ", "")

                elif line.startswith("namespace:"):
                    current_ns = line.replace("namespace: ", "")

                elif line.startswith("is_a: GO:"):
                    parent = line.split("is_a: ")[1].split(" ! ")[0]
                    current_parents.add(parent)

                elif line == "is_obsolete: true":
                    obsolete = True

        save()

    #Maps each GO term to MF, BP, or CC.
    def short_namespace(self, term):
        ns = self.namespace.get(term)
        if ns == "molecular_function":
            return "MF"
        if ns == "biological_process":
            return "BP"
        if ns == "cellular_component":
            return "CC"
        return None

    #Recursively finds all ancestor terms for a GO term.
    def ancestors(self, term):
        if term in self.cache:
            return self.cache[term]

        out = set()
        for p in self.parents.get(term, []):
            if p in self.namespace:
                out.add(p)
                out.update(self.ancestors(p))

        self.cache[term] = out
        return out

    # Adds each GO term's ancestors so semantic metrics can use the GO hierarchy.
    def expand(self, terms):
        out = set()
        for t in terms:
            if t in self.namespace:
                out.add(t)
                out.update(self.ancestors(t))
        return out


#Computes information content for GO terms based on their frequency in training data.
def information_content(train_csv, ontology):
    df = pd.read_csv(train_csv)
    counts = Counter()

    for terms in df["go_terms"].map(parse_go):
        for t in ontology.expand(terms):
            counts[t] += 1

    n = max(len(df), 1)
    return {t: -math.log(c / n) for t, c in counts.items()}


#Computes protein-centric Fmax by trying thresholds from 0.01 to 0.99.
def fmax_score(y_true, y_prob):
    best_f = 0.0
    best_t = 0.0
    eps = 1e-9

    for t in np.linspace(0.01, 0.99, 99):
        pred = (y_prob >= t).astype(np.int32)

        tp = (pred * y_true).sum(axis=1)
        fp = (pred * (1 - y_true)).sum(axis=1)
        fn = ((1 - pred) * y_true).sum(axis=1)

        has_pred = pred.sum(axis=1) > 0
        if not has_pred.any():
            continue

        precision = (tp[has_pred] / (tp[has_pred] + fp[has_pred] + eps)).mean()
        recall = (tp / (tp + fn + eps)).mean()
        f = 2 * precision * recall / (precision + recall + eps)

        if f > best_f:
            best_f = float(f)
            best_t = float(t)

    return best_f, best_t


#Computes micro-AUPR for the predicted GO probabilities.
def aupr_score(y_true, y_prob):
    if y_true.sum() == 0:
        return None
    return float(average_precision_score(y_true, y_prob, average="micro"))


#Computes Smin by measuring remaining uncertainty and misinformation over thresholds.
def smin_score(y_true, y_prob, labels, ontology, ic):
    best_s = None
    best_t = None

    for t in np.linspace(0.01, 0.99, 99):
        ru_total = 0.0
        mi_total = 0.0

        for i in range(y_true.shape[0]):
            true_terms = [labels[j] for j in range(len(labels)) if y_true[i, j] == 1]
            pred_terms = [labels[j] for j in range(len(labels)) if y_prob[i, j] >= t]

            true_expanded = ontology.expand(true_terms)
            pred_expanded = ontology.expand(pred_terms)

            ru_total += sum(ic.get(term, 0.0) for term in true_expanded - pred_expanded)
            mi_total += sum(ic.get(term, 0.0) for term in pred_expanded - true_expanded)

        ru = ru_total / max(y_true.shape[0], 1)
        mi = mi_total / max(y_true.shape[0], 1)
        s = math.sqrt(ru ** 2 + mi ** 2)

        if best_s is None or s < best_s:
            best_s = float(s)
            best_t = float(t)

    return best_s, best_t


#Evaluates Fmax, AUPR, and Smin separately for MF, BP, and CC.
def evaluate_namespace(y_true, y_prob, labels, ontology, ic):
    results = {}

    for ns in ["MF", "BP", "CC"]:
        idx = [i for i, term in enumerate(labels) if ontology.short_namespace(term) == ns]

        if not idx:
            results[ns] = {"fmax": None, "aupr": None, "smin": None, "num_terms": 0}
            continue

        yt = y_true[:, idx]
        yp = y_prob[:, idx]
        ns_labels = [labels[i] for i in idx]

        fmax, f_t = fmax_score(yt, yp)
        smin, s_t = smin_score(yt, yp, ns_labels, ontology, ic)

        results[ns] = {
            "num_terms": len(idx),
            "fmax": fmax,
            "fmax_threshold": f_t,
            "aupr": aupr_score(yt, yp),
            "smin": smin,
            "smin_threshold": s_t,
        }

    return results


class ProteinDataset(torch.utils.data.Dataset):
    #Loads and aligns sequence, text, structure, and label data by UniProt ID.
    def __init__(self, csv_path, seq_path, text_path, text_index_path, residue_path, structures_dir, label_to_idx):
        self.df = pd.read_csv(csv_path)
        self.df["uniprot_id"] = self.df["uniprot_id"].astype(str)

        with open(seq_path, "rb") as f:
            self.seq = {str(k): v for k, v in pickle.load(f).items()}

        self.text = np.load(text_path)
        self.text_index = pd.read_csv(text_index_path)
        self.text_index["uniprot_id"] = self.text_index["uniprot_id"].astype(str)

        with open(residue_path, "rb") as f:
            residue = {str(k): v for k, v in pickle.load(f).items()}

        self.struct_ds = StructureDataset(csv_path, structures_dir, residue)
        self.struct_ds.df["uniprot_id"] = self.struct_ds.df["uniprot_id"].astype(str)

        pdb_ids = {x.replace(".pdb", "") for x in os.listdir(structures_dir) if x.endswith(".pdb")}
        valid_ids = (
            set(self.seq)
            & set(self.text_index["uniprot_id"])
            & set(residue)
            & set(self.struct_ds.df["uniprot_id"])
            & pdb_ids
        )

        self.df = self.df[self.df["uniprot_id"].isin(valid_ids)].reset_index(drop=True)

        self.text_lookup = {
            row["uniprot_id"]: int(row["embedding_idx"])
            for _, row in self.text_index.iterrows()
        }

        self.struct_lookup = {
            row["uniprot_id"]: i
            for i, row in self.struct_ds.df.reset_index(drop=True).iterrows()
        }

        self.label_to_idx = label_to_idx

    #Returns the number of proteins with all required modalities.
    def __len__(self):
        return len(self.df)

    #Returns one protein's sequence/text/structure inputs and GO label vector.
    def __getitem__(self, i):
        row = self.df.iloc[i]
        uid = row["uniprot_id"]

        struct_item = self.struct_ds[self.struct_lookup[uid]]

        return {
            "uid": uid,
            "z_seq": torch.tensor(self.seq[uid], dtype=torch.float32),
            "z_text": torch.tensor(self.text[self.text_lookup[uid]], dtype=torch.float32),
            "x": struct_item["x"],
            "adj": struct_item["adj"],
            "y": torch.tensor(multi_hot(parse_go(row["go_terms"]), self.label_to_idx), dtype=torch.float32),
        }


class CombinedModel(nn.Module):
    #Defines the final gated fusion model for sequence, text, and structure.
    def __init__(self, seq_dim, text_dim, struct_dim, num_labels, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)

        self.gate = nn.Linear(hidden_dim * 3, 3)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    #Combines the three modality embeddings and predicts GO-term logits.
    def forward(self, z_seq, z_text, z_struct):
        zs = self.seq_proj(z_seq)
        zt = self.text_proj(z_text)
        zg = self.struct_proj(z_struct)

        alpha = torch.softmax(self.gate(torch.cat([zs, zt, zg], dim=-1)), dim=-1)
        fused = alpha[:, 0:1] * zs + alpha[:, 1:2] * zt + alpha[:, 2:3] * zg

        return self.head(fused), alpha


#Runs one training epoch over the aligned multimodal dataset.
def train_epoch(model, struct_encoder, ds, optimizer, pos_weight, device, batch_size):
    model.train()
    struct_encoder.train()

    order = np.random.permutation(len(ds))
    total_loss = 0.0

    for start in range(0, len(order), batch_size):
        batch_idx = order[start:start + batch_size]

        z_seq, z_text, z_struct, y = [], [], [], []

        for idx in batch_idx:
            item = ds[int(idx)]

            seq = item["z_seq"].to(device)
            text = item["z_text"].to(device)
            x = item["x"].to(device)
            adj = item["adj"].to(device)
            labels = item["y"].to(device)

            z_seq.append(seq)
            z_text.append(text)
            z_struct.append(struct_encoder(x, adj))
            y.append(labels)

        z_seq = torch.stack(z_seq)
        z_text = torch.stack(z_text)
        z_struct = torch.stack(z_struct)
        y = torch.stack(y)

        logits, _ = model(z_seq, z_text, z_struct)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(struct_encoder.parameters()), 5.0)
        optimizer.step()

        total_loss += loss.item() * len(batch_idx)

    return total_loss / max(len(ds), 1)


#Runs inference and stores probabilities, labels, gates, and protein IDs.
def predict(model, struct_encoder, ds, device):
    model.eval()
    struct_encoder.eval()

    y_true, y_prob, gates, protein_ids = [], [], [], []

    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]

            z_seq = item["z_seq"].unsqueeze(0).to(device)
            z_text = item["z_text"].unsqueeze(0).to(device)
            z_struct = struct_encoder(item["x"].to(device), item["adj"].to(device)).unsqueeze(0)

            logits, alpha = model(z_seq, z_text, z_struct)

            y_true.append(item["y"].numpy())
            y_prob.append(torch.sigmoid(logits).squeeze(0).cpu().numpy())
            gates.append(alpha.squeeze(0).cpu().numpy())
            protein_ids.append(item["uid"])

    return {
        "protein_ids": protein_ids,
        "y_true": np.stack(y_true),
        "y_prob": np.stack(y_prob),
        "gates": np.stack(gates),
    }


#Collects overall and MF/BP/CC evaluation metrics into one dictionary.
def summarize_metrics(pred, labels, ontology, ic):
    y_true = pred["y_true"]
    y_prob = pred["y_prob"]
    gates = pred["gates"]

    overall_fmax, overall_t = fmax_score(y_true, y_prob)

    return {
        "n": int(y_true.shape[0]),
        "overall": {
            "fmax": overall_fmax,
            "fmax_threshold": overall_t,
            "aupr": aupr_score(y_true, y_prob),
            "mean_labels_per_protein": float(y_true.sum(axis=1).mean()),
        },
        "MF_BP_CC": evaluate_namespace(y_true, y_prob, labels, ontology, ic),
        "mean_gate_sequence": float(gates[:, 0].mean()),
        "mean_gate_text": float(gates[:, 1].mean()),
        "mean_gate_structure": float(gates[:, 2].mean()),
    }


#Parses arguments, trains the model, evaluates the test set, and saves outputs.
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--train-csv", default="data/processed/train_dataset_propagated.csv")
    p.add_argument("--val-csv", default="data/processed/val_dataset_propagated.csv")
    p.add_argument("--test-csv", default="data/processed/test_dataset_propagated.csv")

    p.add_argument("--train-seq", default="data/processed/train_embeddings_esm2_embeddings.pkl")
    p.add_argument("--val-seq", default="data/processed/val_embeddings_esm2_embeddings.pkl")
    p.add_argument("--test-seq", default="data/processed/test_embeddings_esm2_embeddings.pkl")

    p.add_argument("--train-text", default="data/text_encodings/train_embeddings.npy")
    p.add_argument("--val-text", default="data/text_encodings/val_embeddings.npy")
    p.add_argument("--test-text", default="data/text_encodings/test_embeddings.npy")

    p.add_argument("--train-text-index", default="data/text_encodings/train_index.csv")
    p.add_argument("--val-text-index", default="data/text_encodings/val_index.csv")
    p.add_argument("--test-text-index", default="data/text_encodings/test_index.csv")

    p.add_argument("--train-residue", default="data/processed/train_residue_embeddings.pkl")
    p.add_argument("--val-residue", default="data/processed/val_residue_embeddings.pkl")
    p.add_argument("--test-residue", default="data/processed/test_residue_embeddings.pkl")

    p.add_argument("--structures", default="data/structures")
    p.add_argument("--go-obo", default="data/go-basic.obo")

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--struct-hidden-dim", type=int, default=512)
    p.add_argument("--struct-out-dim", type=int, default=256)
    p.add_argument("--min-go-freq", type=int, default=3)
    p.add_argument("--pos-weight-cap", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="final_combined_model_results.json")

    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ontology = GOOntology(args.go_obo)
    ic = information_content(args.train_csv, ontology)

    labels = build_label_space(args.train_csv, args.min_go_freq)
    label_to_idx = {t: i for i, t in enumerate(labels)}
    pos_weight = class_pos_weight(args.train_csv, label_to_idx, args.pos_weight_cap).to(device)

    train_ds = ProteinDataset(args.train_csv, args.train_seq, args.train_text, args.train_text_index, args.train_residue, args.structures, label_to_idx)
    val_ds = ProteinDataset(args.val_csv, args.val_seq, args.val_text, args.val_text_index, args.val_residue, args.structures, label_to_idx)
    test_ds = ProteinDataset(args.test_csv, args.test_seq, args.test_text, args.test_text_index, args.test_residue, args.structures, label_to_idx)

    sample = train_ds[0]

    struct_encoder = StructureEncoder(
        in_dim=sample["x"].shape[1],
        hidden_dim=args.struct_hidden_dim,
        out_dim=args.struct_out_dim,
        dropout=args.dropout,
        pool="mean",
    ).to(device)

    model = CombinedModel(
        seq_dim=sample["z_seq"].shape[0],
        text_dim=sample["z_text"].shape[0],
        struct_dim=args.struct_out_dim,
        num_labels=len(labels),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(struct_encoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val_fmax = -1.0

    for _ in range(args.epochs):
        train_epoch(model, struct_encoder, train_ds, optimizer, pos_weight, device, args.batch_size)

        val_pred = predict(model, struct_encoder, val_ds, device)
        val_fmax, _ = fmax_score(val_pred["y_true"], val_pred["y_prob"])

        if val_fmax > best_val_fmax:
            best_val_fmax = val_fmax
            best_state = {
                "model": model.state_dict(),
                "structure_encoder": struct_encoder.state_dict(),
                "labels": labels,
                "args": vars(args),
            }

    model.load_state_dict(best_state["model"])
    struct_encoder.load_state_dict(best_state["structure_encoder"])

    test_pred = predict(model, struct_encoder, test_ds, device)
    test_metrics = summarize_metrics(test_pred, labels, ontology, ic)

    results = {
        "model": "sequence_text_structure_fusion",
        "best_val_fmax": best_val_fmax,
        "test_remote_homology_subset": test_metrics,
    }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    torch.save(best_state, "best_final_combined_model.pt")

    print(f"saved {args.out}")
    print("saved best_final_combined_model.pt")


if __name__ == "__main__":
    main()