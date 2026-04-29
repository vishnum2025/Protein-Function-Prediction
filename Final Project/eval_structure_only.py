"""Structure-only GO-prediction eval, mirroring combined_model.py's metric harness.

Uses the SAME evaluation functions (GOOntology, fmax, aupr, smin, MF/BP/CC
breakdown) as combined_model.py so numbers are directly comparable to the
fusion model. Hyperparameters mirror combined_model.py defaults where
applicable. Output JSON has the same shape as final_combined_model_results.json.
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from structure_encoder import StructureDataset, StructureEncoder
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


class StructureClassifier(nn.Module):
    """StructureEncoder -> MLP head; mirrors combined_model.py head shape."""

    def __init__(self, encoder: StructureEncoder, num_labels: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.gcn3.lin.out_features
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x, adj):
        z = self.encoder(x, adj)
        return self.head(z)


def filter_split(csv_path: str, residue_path: str, structures_dir: str):
    """Build (filtered_df, residue_dict) keeping only proteins that have both
    a residue embedding and a PDB on disk. Mirrors the intersection logic
    from combined_model.ProteinDataset for fairness."""
    df = pd.read_csv(csv_path)
    df["uniprot_id"] = df["uniprot_id"].astype(str)
    with open(residue_path, "rb") as f:
        residue = {str(k): v for k, v in pickle.load(f).items()}
    pdb_ids = {x.replace(".pdb", "") for x in os.listdir(structures_dir) if x.endswith(".pdb")}
    keep = set(residue) & pdb_ids
    df = df[df["uniprot_id"].isin(keep)].reset_index(drop=True)
    return df, residue


def predict(model: StructureClassifier, ds: StructureDataset, label_to_idx, device):
    model.eval()
    y_true, y_prob, ids = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            y = multi_hot(parse_go(s["go_terms"]), label_to_idx)
            logits = model(s["x"].to(device), s["adj"].to(device))
            p = torch.sigmoid(logits).cpu().numpy()
            y_true.append(y)
            y_prob.append(p)
            ids.append(s["uniprot_id"])
    return {
        "protein_ids": ids,
        "y_true": np.stack(y_true),
        "y_prob": np.stack(y_prob),
        # No gates for structure-only; summarize_metrics expects this key.
        "gates": np.zeros((len(ids), 3), dtype=np.float32),
    }


def train_epoch(model, ds, label_to_idx, optimizer, pos_weight, device, batch_size):
    model.train()
    n = len(ds)
    order = np.random.permutation(n)
    total_loss = 0.0
    seen = 0
    for start in range(0, n, batch_size):
        idxs = order[start:start + batch_size]
        if len(idxs) == 0:
            continue
        logits_b, y_b = [], []
        for i in idxs:
            s = ds[int(i)]
            y = torch.from_numpy(multi_hot(parse_go(s["go_terms"]), label_to_idx)).to(device)
            logits = model(s["x"].to(device), s["adj"].to(device))
            logits_b.append(logits)
            y_b.append(y)
        logits_t = torch.stack(logits_b, dim=0)
        y_t = torch.stack(y_b, dim=0)
        loss = F.binary_cross_entropy_with_logits(logits_t, y_t, pos_weight=pos_weight)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * len(idxs)
        seen += len(idxs)
    return total_loss / max(seen, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", default="data/processed/train_dataset_propagated.csv")
    p.add_argument("--val-csv", default="data/processed/val_dataset_propagated.csv")
    p.add_argument("--test-csv", default="data/processed/test_dataset_propagated.csv")
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
    p.add_argument("--out", default="final_structure_only_results.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")

    ontology = GOOntology(args.go_obo)
    print(f"[info] ontology loaded: {len(ontology.namespace)} GO terms")

    ic = information_content(args.train_csv, ontology)

    labels = build_label_space(args.train_csv, args.min_go_freq)
    label_to_idx = {t: i for i, t in enumerate(labels)}
    print(f"[info] label space (min_freq={args.min_go_freq}): {len(labels)} GO terms")
    pos_weight = class_pos_weight(args.train_csv, label_to_idx, args.pos_weight_cap).to(device)

    # Filter splits to (residue ∩ pdb), then build StructureDataset.
    train_df, train_res = filter_split(args.train_csv, args.train_residue, args.structures)
    val_df, val_res = filter_split(args.val_csv, args.val_residue, args.structures)
    test_df, test_res = filter_split(args.test_csv, args.test_residue, args.structures)

    # Persist filtered CSVs to a temp location so StructureDataset reads them.
    tmp = ".eval_structure_only_tmp"
    os.makedirs(tmp, exist_ok=True)
    train_csv = os.path.join(tmp, "train.csv"); train_df.to_csv(train_csv, index=False)
    val_csv = os.path.join(tmp, "val.csv");     val_df.to_csv(val_csv, index=False)
    test_csv = os.path.join(tmp, "test.csv");   test_df.to_csv(test_csv, index=False)

    train_ds = StructureDataset(train_csv, args.structures, train_res)
    val_ds = StructureDataset(val_csv, args.structures, val_res)
    test_ds = StructureDataset(test_csv, args.structures, test_res)
    print(f"[info] splits — train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    in_dim = train_ds[0]["x"].shape[1]
    encoder = StructureEncoder(
        in_dim=in_dim,
        hidden_dim=args.struct_hidden_dim,
        out_dim=args.struct_out_dim,
        dropout=args.dropout,
        pool="mean",
    )
    model = StructureClassifier(encoder, num_labels=len(labels), hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_fmax = -1.0
    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, train_ds, label_to_idx, optimizer, pos_weight, device, args.batch_size)
        val_pred = predict(model, val_ds, label_to_idx, device)
        val_fmax, val_t = fmax_score(val_pred["y_true"], val_pred["y_prob"])
        print(f"[ep {ep:02d}] train_loss={loss:.4f}  val_fmax={val_fmax:.4f} @ t={val_t:.2f}")
        if val_fmax > best_val_fmax:
            best_val_fmax = val_fmax
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred = predict(model, test_ds, label_to_idx, device)
    test_metrics = summarize_metrics(test_pred, labels, ontology, ic)
    # Gates are meaningless for structure-only — strip them.
    test_metrics.pop("mean_gate_sequence", None)
    test_metrics.pop("mean_gate_text", None)
    test_metrics.pop("mean_gate_structure", None)

    results = {
        "model": "structure_only",
        "best_val_fmax": best_val_fmax,
        "test_remote_homology_subset": test_metrics,
        "_notes": {
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "comment": (
                "Test n differs from final_combined_model_results.json (n=10) "
                "because structure-only does not require text-encoding overlap."
            ),
        },
    }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    torch.save(best_state, "best_structure_only.pt")
    print(f"[done] saved {args.out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
