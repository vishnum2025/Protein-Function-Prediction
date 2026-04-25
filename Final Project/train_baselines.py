"""Unified training/evaluation for the proposal's baselines.

Models:
  - seq                : ESM-2 mean-pool + MLP                  (proposal #1)
  - struct             : GCN with ESM-2 residue node features   (proposal #2, DeepFRI-style)
  - struct_contrastive : struct + auxiliary InfoNCE between
                         sequence pool and structure pool       (proposal #3, PenLight-style)

Reports protein-centric Fmax and micro-AUPR (CAFA convention).
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


# ---------- Label space + multi-hot ----------

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


def class_pos_weight(train_csv: str, label_to_idx, cap: float = 10.0):
    """sqrt(neg/pos) is more stable than (N-pos)/pos for tail-heavy GO terms."""
    df = pd.read_csv(train_csv)
    n = len(df)
    pos = np.zeros(len(label_to_idx))
    for terms in df["go_terms"].map(parse_go):
        for t in terms:
            if t in label_to_idx:
                pos[label_to_idx[t]] += 1
    pos = np.clip(pos, 1, None)
    pw = np.sqrt(np.clip((n - pos) / pos, 1.0, None))
    pw = np.clip(pw, 1.0, cap)
    return torch.tensor(pw, dtype=torch.float32)


# ---------- Models ----------

class SequenceMLP(nn.Module):
    """Baseline #1: ESM-2 mean-pooled embedding -> MLP."""

    def __init__(self, in_dim: int, num_labels: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        return self.head(z_seq)


class StructureClassifier(nn.Module):
    """Baseline #2: GCN over the ESM-2 residue contact graph -> MLP."""

    def __init__(self, encoder: StructureEncoder, num_labels: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        out_dim = encoder.gcn3.lin.out_features
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x, adj):
        z = self.encoder(x, adj)
        return self.head(z)

    def encode(self, x, adj):
        return self.encoder(x, adj)


class ContrastiveStructureModel(nn.Module):
    """Baseline #3: structure GCN + sequence projection, InfoNCE in shared space."""

    def __init__(self, encoder: StructureEncoder, seq_in_dim: int, num_labels: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        struct_dim = encoder.gcn3.lin.out_features
        self.seq_proj = nn.Sequential(
            nn.LayerNorm(seq_in_dim),
            nn.Linear(seq_in_dim, struct_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(struct_dim * 2),
            nn.Linear(struct_dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x, adj, z_seq):
        z_struct = self.encoder(x, adj).unsqueeze(0)  # (1, struct_dim)
        z_seq_proj = self.seq_proj(z_seq)             # (1, struct_dim)
        logits = self.head(torch.cat([z_struct, z_seq_proj], dim=-1))
        return logits, z_struct, z_seq_proj


def info_nce(z_a: torch.Tensor, z_b: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = z_a @ z_b.t() / temp
    labels = torch.arange(z_a.size(0), device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# ---------- Metrics ----------

def protein_centric_fmax(y_true: np.ndarray, y_prob: np.ndarray):
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


def evaluate(model_kind: str, model, ds, label_to_idx, pooled_seq, device):
    model.eval()
    Y, P = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            uid = s["uniprot_id"]
            y = multi_hot(parse_go(s["go_terms"]), label_to_idx)

            if model_kind == "seq":
                z_seq = torch.from_numpy(pooled_seq[uid]).to(device)
                logits = model(z_seq.unsqueeze(0)).squeeze(0)
            elif model_kind == "struct":
                logits = model(s["x"].to(device), s["adj"].to(device))
            elif model_kind == "struct_contrastive":
                z_seq = torch.from_numpy(pooled_seq[uid]).to(device)
                logits, _, _ = model(s["x"].to(device), s["adj"].to(device), z_seq.unsqueeze(0))
                logits = logits.squeeze(0)
            else:
                raise ValueError(model_kind)
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


# ---------- Data ----------

def load_split(csv_path, residue_emb_path, structures_dir):
    with open(residue_emb_path, "rb") as f:
        residue = pickle.load(f)
    return StructureDataset(csv_path, structures_dir, residue)


def load_pooled(emb_path):
    with open(emb_path, "rb") as f:
        return pickle.load(f)


# ---------- Training ----------

def train_one_epoch(model_kind, model, opt, train_ds, label_to_idx, pooled_seq, pos_weight, device, batch_size, contrastive_lambda):
    model.train()
    n = len(train_ds)
    order = np.random.permutation(n)
    ep_loss = 0.0
    ep_contrast = 0.0
    seen = 0

    for start in range(0, n, batch_size):
        idxs = order[start:start + batch_size]
        if len(idxs) == 0:
            continue

        # Always accumulate in-loop because protein lengths vary.
        logits_batch, y_batch = [], []
        z_struct_batch, z_seq_batch = [], []

        for i in idxs:
            s = train_ds[int(i)]
            uid = s["uniprot_id"]
            y = torch.from_numpy(multi_hot(parse_go(s["go_terms"]), label_to_idx)).to(device)

            if model_kind == "seq":
                z_seq = torch.from_numpy(pooled_seq[uid]).to(device)
                logits = model(z_seq.unsqueeze(0)).squeeze(0)
            elif model_kind == "struct":
                logits = model(s["x"].to(device), s["adj"].to(device))
            elif model_kind == "struct_contrastive":
                z_seq = torch.from_numpy(pooled_seq[uid]).to(device)
                logits, z_struct, z_seq_proj = model(s["x"].to(device), s["adj"].to(device), z_seq.unsqueeze(0))
                logits = logits.squeeze(0)
                z_struct_batch.append(z_struct)
                z_seq_batch.append(z_seq_proj.squeeze(0))
            logits_batch.append(logits)
            y_batch.append(y)

        logits_t = torch.stack(logits_batch, dim=0)
        y_t = torch.stack(y_batch, dim=0)
        loss = F.binary_cross_entropy_with_logits(logits_t, y_t, pos_weight=pos_weight)

        contrast_term = torch.tensor(0.0, device=device)
        if model_kind == "struct_contrastive" and len(z_struct_batch) >= 2:
            z_s = torch.stack([z.squeeze() if z.dim() > 1 else z for z in z_struct_batch], dim=0)
            z_q = torch.stack(z_seq_batch, dim=0)
            contrast_term = info_nce(z_s, z_q)
            loss = loss + contrastive_lambda * contrast_term

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        ep_loss += loss.item() * len(idxs)
        ep_contrast += float(contrast_term) * len(idxs)
        seen += len(idxs)

    return ep_loss / max(seen, 1), ep_contrast / max(seen, 1)


def build_model(model_kind, in_dim_seq, in_dim_residue, num_labels, args):
    if model_kind == "seq":
        return SequenceMLP(in_dim_seq, num_labels, hidden=args.hidden_dim, dropout=args.dropout)
    encoder = StructureEncoder(
        in_dim=in_dim_residue,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        dropout=args.dropout,
        pool=args.pool,
    )
    if model_kind == "struct":
        return StructureClassifier(encoder, num_labels=num_labels, hidden=args.hidden_dim, dropout=args.dropout)
    if model_kind == "struct_contrastive":
        return ContrastiveStructureModel(encoder, seq_in_dim=in_dim_seq, num_labels=num_labels, hidden=args.hidden_dim, dropout=args.dropout)
    raise ValueError(model_kind)


def run(args, model_kind):
    print(f"\n{'='*60}\nTraining: {model_kind}\n{'='*60}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = build_label_space(args.train_csv, args.min_go_freq)
    label_to_idx = {t: i for i, t in enumerate(labels)}
    print(f"Label space: {len(labels)} GO terms (min_freq={args.min_go_freq})")
    if not labels:
        raise SystemExit("No GO terms passed the frequency filter")

    pos_weight = class_pos_weight(args.train_csv, label_to_idx, cap=args.pos_weight_cap).to(device)

    train_ds = load_split(args.train_csv, args.train_emb, args.structures)
    val_ds = load_split(args.val_csv, args.val_emb, args.structures)
    test_ds = load_split(args.test_csv, args.test_emb, args.structures)
    print(f"Splits — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    pooled_train = load_pooled(args.train_pooled)
    pooled_val = load_pooled(args.val_pooled)
    pooled_test = load_pooled(args.test_pooled)
    pooled_all = {**pooled_train, **pooled_val, **pooled_test}

    in_dim_seq = next(iter(pooled_train.values())).shape[0]
    in_dim_residue = train_ds[0]["x"].shape[1]
    print(f"Pooled seq dim: {in_dim_seq}, residue dim: {in_dim_residue}")

    model = build_model(model_kind, in_dim_seq, in_dim_residue, len(labels), args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_fmax = -1.0
    best_state = None
    for ep in range(1, args.epochs + 1):
        train_loss, contrast = train_one_epoch(
            model_kind, model, opt, train_ds, label_to_idx, pooled_all,
            pos_weight, device, args.batch_size, args.contrastive_lambda,
        )
        val = evaluate(model_kind, model, val_ds, label_to_idx, pooled_all, device)
        extra = f"  contrast={contrast:.4f}" if model_kind == "struct_contrastive" else ""
        print(f"  ep {ep:3d}  loss={train_loss:.4f}{extra}  val_fmax={val['fmax']:.4f}@t={val['fmax_threshold']:.2f}  val_aupr={val['aupr_micro']:.4f}")
        if val["fmax"] > best_fmax:
            best_fmax = val["fmax"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model_kind, model, test_ds, label_to_idx, pooled_all, device)
    return {"model": model_kind, "best_val_fmax": best_fmax, "test": test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["seq", "struct", "struct_contrastive"], choices=["seq", "struct", "struct_contrastive"])
    ap.add_argument("--train-csv", default="data/processed/train_dataset.csv")
    ap.add_argument("--val-csv", default="data/processed/val_dataset.csv")
    ap.add_argument("--test-csv", default="data/processed/test_dataset.csv")
    ap.add_argument("--train-emb", default="data/processed/train_residue_embeddings.pkl")
    ap.add_argument("--val-emb", default="data/processed/val_residue_embeddings.pkl")
    ap.add_argument("--test-emb", default="data/processed/test_residue_embeddings.pkl")
    ap.add_argument("--train-pooled", default="data/processed/train_embeddings_esm2_embeddings.pkl")
    ap.add_argument("--val-pooled", default="data/processed/val_embeddings_esm2_embeddings.pkl")
    ap.add_argument("--test-pooled", default="data/processed/test_embeddings_esm2_embeddings.pkl")
    ap.add_argument("--structures", default="data/structures")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--min-go-freq", type=int, default=3)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--pool", choices=["mean", "sum"], default="mean")
    ap.add_argument("--pos-weight-cap", type=float, default=10.0)
    ap.add_argument("--contrastive-lambda", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results = []
    for m in args.models:
        results.append(run(args, m))

    print("\n\n" + "=" * 70)
    print(f"{'Model':<22} {'Val Fmax':>10} {'Test Fmax':>10} {'Test AUPR':>12} {'thr':>6}")
    print("-" * 70)
    for r in results:
        t = r["test"]
        print(f"{r['model']:<22} {r['best_val_fmax']:>10.4f} {t['fmax']:>10.4f} {t['aupr_micro']:>12.4f} {t['fmax_threshold']:>6.2f}")
    print("=" * 70)
    print(f"\n(n_test={results[0]['test']['n']}, mean_labels/protein={results[0]['test']['mean_labels_per_protein']:.2f})")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
