"""Sequence-only baseline (proposal §3.3 baseline #1): ESM-2 mean-pool + MLP.

Mirrors eval_structure_only.py shape so all baselines share the same metric
harness (GO ontology, Fmax, AUPR, Smin, MF/BP/CC). Output JSON matches the
shape of final_structure_only_15ep.json.
"""

import argparse
import ast
import json
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from combined_model import (
    GOOntology,
    build_label_space,
    class_pos_weight,
    fmax_score,
    information_content,
    multi_hot,
    parse_go,
    summarize_metrics,
)


class SequenceMLP(nn.Module):
    """ESM-2 pooled embedding -> MLP head matching combined_model head shape."""

    def __init__(self, in_dim: int, num_labels: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, z_seq):
        return self.head(z_seq)


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, seq_path: str, label_to_idx):
        self.df = pd.read_csv(csv_path)
        self.df["uniprot_id"] = self.df["uniprot_id"].astype(str)
        with open(seq_path, "rb") as f:
            self.seq = {str(k): v for k, v in pickle.load(f).items()}
        self.df = self.df[self.df["uniprot_id"].isin(self.seq)].reset_index(drop=True)
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        uid = row["uniprot_id"]
        return {
            "uid": uid,
            "z_seq": torch.tensor(self.seq[uid], dtype=torch.float32),
            "y": torch.tensor(multi_hot(parse_go(row["go_terms"]), self.label_to_idx), dtype=torch.float32),
        }


def predict(model, ds, device):
    model.eval()
    y_true, y_prob, ids = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            logits = model(s["z_seq"].unsqueeze(0).to(device))
            y_true.append(s["y"].numpy())
            y_prob.append(torch.sigmoid(logits).squeeze(0).cpu().numpy())
            ids.append(s["uid"])
    return {
        "protein_ids": ids,
        "y_true": np.stack(y_true),
        "y_prob": np.stack(y_prob),
        "gates": np.zeros((len(ids), 3), dtype=np.float32),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", default="data/processed/train_dataset_propagated.csv")
    p.add_argument("--val-csv", default="data/processed/val_dataset_propagated.csv")
    p.add_argument("--test-csv", default="data/processed/test_dataset_propagated.csv")
    p.add_argument("--train-seq", default="data/processed/train_embeddings_esm2_embeddings.pkl")
    p.add_argument("--val-seq", default="data/processed/val_embeddings_esm2_embeddings.pkl")
    p.add_argument("--test-seq", default="data/processed/test_embeddings_esm2_embeddings.pkl")
    p.add_argument("--go-obo", default="data/go-basic.obo")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--min-go-freq", type=int, default=3)
    p.add_argument("--pos-weight-cap", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="final_seq_only_15ep.json")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")

    ontology = GOOntology(args.go_obo)
    ic = information_content(args.train_csv, ontology)
    labels = build_label_space(args.train_csv, args.min_go_freq)
    label_to_idx = {t: i for i, t in enumerate(labels)}
    pos_weight = class_pos_weight(args.train_csv, label_to_idx, args.pos_weight_cap).to(device)
    print(f"[info] label space (min_freq={args.min_go_freq}): {len(labels)}")

    train_ds = SeqDataset(args.train_csv, args.train_seq, label_to_idx)
    val_ds = SeqDataset(args.val_csv, args.val_seq, label_to_idx)
    test_ds = SeqDataset(args.test_csv, args.test_seq, label_to_idx)
    print(f"[info] splits — train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    in_dim = train_ds[0]["z_seq"].shape[0]
    model = SequenceMLP(in_dim=in_dim, num_labels=len(labels), hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_fmax = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_ds))
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            idxs = order[start:start + args.batch_size]
            zs = torch.stack([train_ds[int(i)]["z_seq"] for i in idxs]).to(device)
            y = torch.stack([train_ds[int(i)]["y"] for i in idxs]).to(device)
            logits = model(zs)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(idxs)
        val_pred = predict(model, val_ds, device)
        val_fmax, val_t = fmax_score(val_pred["y_true"], val_pred["y_prob"])
        print(f"[ep {ep:02d}] loss={total/len(train_ds):.4f}  val_fmax={val_fmax:.4f}@t={val_t:.2f}")
        if val_fmax > best_val_fmax:
            best_val_fmax = val_fmax
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred = predict(model, test_ds, device)
    main_metrics = summarize_metrics(test_pred, labels, ontology, ic)
    main_metrics.pop("mean_gate_sequence", None)
    main_metrics.pop("mean_gate_text", None)
    main_metrics.pop("mean_gate_structure", None)

    results = {
        "model": "sequence_only",
        "best_val_fmax": best_val_fmax,
        "test_remote_homology_subset": main_metrics,
        "_notes": {
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "epochs": args.epochs,
            "comment": "ESM-2 mean-pooled embedding -> MLP. Test set is already the CD-HIT 40%-identity remote-homology split.",
        },
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    torch.save(best_state, "best_seq_only.pt")
    print(f"[done] saved {args.out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
