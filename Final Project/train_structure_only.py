"""Structure-only GO term prediction — the proposal's baseline #2 (DeepFRI-style).

Trains StructureEncoder + an MLP head on multi-label GO prediction with
weighted BCE. Reports CAFA-style protein-centric Fmax and micro-AUPR on
val/test. This is what tells you whether the structure encoder is actually
producing functionally informative embeddings.
"""

import argparse
import ast
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from structure_encoder import StructureDataset, StructureEncoder


def parse_go(cell):
    if isinstance(cell, list):
        return cell
    if pd.isna(cell):
        return []
    return ast.literal_eval(cell)


def build_label_space(train_csv: str, min_freq: int):
    df = pd.read_csv(train_csv)
    counts = Counter()
    for terms in df["go_terms"].map(parse_go):
        counts.update(terms)
    return sorted([t for t, c in counts.items() if c >= min_freq])


def multi_hot(go_terms, label_to_idx):
    y = np.zeros(len(label_to_idx), dtype=np.float32)
    for t in go_terms:
        if t in label_to_idx:
            y[label_to_idx[t]] = 1.0
    return y


def class_pos_weight(train_csv: str, label_to_idx):
    df = pd.read_csv(train_csv)
    n = len(df)
    pos = np.zeros(len(label_to_idx))
    for terms in df["go_terms"].map(parse_go):
        for t in terms:
            if t in label_to_idx:
                pos[label_to_idx[t]] += 1
    pos = np.clip(pos, 1, None)
    pw = (n - pos) / pos
    pw = np.clip(pw, 1.0, 50.0)  # cap to avoid extreme tail terms dominating
    return torch.tensor(pw, dtype=torch.float32)


class StructureClassifier(nn.Module):
    def __init__(self, encoder: StructureEncoder, num_labels: int, mlp_hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        out_dim = encoder.gcn3.lin.out_features
        self.head = nn.Sequential(
            nn.Linear(out_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_labels),
        )

    def forward(self, x, adj):
        z = self.encoder(x, adj)
        return self.head(z)


def protein_centric_fmax(y_true: np.ndarray, y_prob: np.ndarray):
    """CAFA-style Fmax: protein-centric precision/recall over thresholds."""
    best_f, best_t = 0.0, 0.0
    eps = 1e-9
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= t).astype(np.int32)
        tp = (y_pred * y_true).sum(axis=1)
        fp = (y_pred * (1 - y_true)).sum(axis=1)
        fn = ((1 - y_pred) * y_true).sum(axis=1)
        has_pred = y_pred.sum(axis=1) > 0
        if not has_pred.any():
            continue
        prec = (tp[has_pred] / (tp[has_pred] + fp[has_pred] + eps)).mean()
        rec = (tp / (tp + fn + eps)).mean()
        f = 2 * prec * rec / (prec + rec + eps)
        if f > best_f:
            best_f, best_t = float(f), float(t)
    return best_f, best_t


def evaluate(model, ds, label_to_idx, device):
    model.eval()
    Y, P = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            y = multi_hot(parse_go(s["go_terms"]), label_to_idx)
            logits = model(s["x"].to(device), s["adj"].to(device))
            P.append(torch.sigmoid(logits).cpu().numpy())
            Y.append(y)
    Y = np.stack(Y); P = np.stack(P)
    fm, th = protein_centric_fmax(Y, P)
    aupr = average_precision_score(Y, P, average="micro")
    return {
        "n": len(ds),
        "fmax": fm,
        "fmax_threshold": th,
        "aupr_micro": float(aupr),
        "mean_labels_per_protein": float(Y.sum(axis=1).mean()),
    }


def load_split(csv_path, emb_path, structures_dir):
    with open(emb_path, "rb") as f:
        embeds = pickle.load(f)
    return StructureDataset(csv_path, structures_dir, embeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data/processed/train_dataset.csv")
    ap.add_argument("--val-csv", default="data/processed/val_dataset.csv")
    ap.add_argument("--test-csv", default="data/processed/test_dataset.csv")
    ap.add_argument("--train-emb", default="data/processed/train_residue_embeddings.pkl")
    ap.add_argument("--val-emb", default="data/processed/val_residue_embeddings.pkl")
    ap.add_argument("--test-emb", default="data/processed/test_residue_embeddings.pkl")
    ap.add_argument("--structures", default="data/structures")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--min-go-freq", type=int, default=3)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    labels = build_label_space(args.train_csv, args.min_go_freq)
    label_to_idx = {t: i for i, t in enumerate(labels)}
    print(f"Label space: {len(labels)} GO terms (min_freq={args.min_go_freq})")
    if not labels:
        raise SystemExit("No GO terms passed the frequency filter; lower --min-go-freq.")

    pos_weight = class_pos_weight(args.train_csv, label_to_idx).to(device)

    train_ds = load_split(args.train_csv, args.train_emb, args.structures)
    val_ds = load_split(args.val_csv, args.val_emb, args.structures)
    test_ds = load_split(args.test_csv, args.test_emb, args.structures)
    print(f"Splits — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    in_dim = train_ds[0]["x"].shape[1]
    encoder = StructureEncoder(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim, dropout=args.dropout)
    model = StructureClassifier(encoder, num_labels=len(labels), mlp_hidden=args.hidden_dim, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_fmax = -1.0
    best_state = None
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_ds))
        ep_loss = 0.0
        for i in order:
            s = train_ds[int(i)]
            y = torch.from_numpy(multi_hot(parse_go(s["go_terms"]), label_to_idx)).to(device)
            logits = model(s["x"].to(device), s["adj"].to(device))
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += loss.item()
        ep_loss /= len(train_ds)
        val = evaluate(model, val_ds, label_to_idx, device)
        history.append({"epoch": ep, "train_loss": ep_loss, **{f"val_{k}": v for k, v in val.items()}})
        print(f"epoch {ep:3d}  loss={ep_loss:.4f}  val_fmax={val['fmax']:.4f}@t={val['fmax_threshold']:.2f}  val_aupr={val['aupr_micro']:.4f}")
        if val["fmax"] > best_fmax:
            best_fmax = val["fmax"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, test_ds, label_to_idx, device)
    print("\n=== TEST (best-by-val checkpoint) ===")
    print(json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
