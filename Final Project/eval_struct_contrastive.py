"""PenLight-style baseline (proposal §3.3 baseline #3): GCN over contact graph
+ sequence projection, with InfoNCE contrastive alignment between z_seq and z_struct.

Reuses the model class (ContrastiveStructureModel + info_nce) from
train_baselines.py and the GO-ontology metric harness from combined_model.py.
Output JSON shape mirrors final_structure_only_15ep.json.
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
from train_baselines import ContrastiveStructureModel, info_nce
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


def filter_split(csv_path, residue_path, structures_dir):
    df = pd.read_csv(csv_path)
    df["uniprot_id"] = df["uniprot_id"].astype(str)
    with open(residue_path, "rb") as f:
        residue = {str(k): v for k, v in pickle.load(f).items()}
    pdb_ids = {x.replace(".pdb", "") for x in os.listdir(structures_dir) if x.endswith(".pdb")}
    keep = set(residue) & pdb_ids
    df = df[df["uniprot_id"].isin(keep)].reset_index(drop=True)
    return df, residue


def predict(model, ds, pooled_seq, label_to_idx, device):
    model.eval()
    y_true, y_prob, ids = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            uid = s["uniprot_id"]
            z_seq = torch.from_numpy(pooled_seq[uid]).to(device).unsqueeze(0)
            logits, _, _ = model(s["x"].to(device), s["adj"].to(device), z_seq)
            y = multi_hot(parse_go(s["go_terms"]), label_to_idx)
            y_true.append(y)
            y_prob.append(torch.sigmoid(logits).squeeze(0).cpu().numpy())
            ids.append(uid)
    return {
        "protein_ids": ids,
        "y_true": np.stack(y_true),
        "y_prob": np.stack(y_prob),
        "gates": np.zeros((len(ids), 3), dtype=np.float32),
    }


def train_epoch(model, ds, pooled_seq, label_to_idx, optimizer, pos_weight, device, batch_size, contrastive_lambda):
    model.train()
    n = len(ds)
    order = np.random.permutation(n)
    total = 0.0
    contrast_total = 0.0
    seen = 0
    for start in range(0, n, batch_size):
        idxs = order[start:start + batch_size]
        if len(idxs) == 0:
            continue
        logits_b, y_b, zs_b, zg_b = [], [], [], []
        for i in idxs:
            s = ds[int(i)]
            uid = s["uniprot_id"]
            z_seq = torch.from_numpy(pooled_seq[uid]).to(device).unsqueeze(0)
            logits, z_struct, z_seq_proj = model(s["x"].to(device), s["adj"].to(device), z_seq)
            logits_b.append(logits.squeeze(0))
            y_b.append(torch.from_numpy(multi_hot(parse_go(s["go_terms"]), label_to_idx)).to(device))
            zs_b.append(z_seq_proj.squeeze(0))
            zg_b.append(z_struct.squeeze(0))

        logits_t = torch.stack(logits_b, dim=0)
        y_t = torch.stack(y_b, dim=0)
        loss = F.binary_cross_entropy_with_logits(logits_t, y_t, pos_weight=pos_weight)

        contrast = torch.tensor(0.0, device=device)
        if len(zs_b) >= 2:
            contrast = info_nce(torch.stack(zg_b, 0), torch.stack(zs_b, 0))
            loss = loss + contrastive_lambda * contrast

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item() * len(idxs)
        contrast_total += float(contrast) * len(idxs)
        seen += len(idxs)
    return total / max(seen, 1), contrast_total / max(seen, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", default="data/processed/train_dataset_propagated.csv")
    p.add_argument("--val-csv", default="data/processed/val_dataset_propagated.csv")
    p.add_argument("--test-csv", default="data/processed/test_dataset_propagated.csv")
    p.add_argument("--train-residue", default="data/processed/train_residue_embeddings.pkl")
    p.add_argument("--val-residue", default="data/processed/val_residue_embeddings.pkl")
    p.add_argument("--test-residue", default="data/processed/test_residue_embeddings.pkl")
    p.add_argument("--train-pooled", default="data/processed/train_embeddings_esm2_embeddings.pkl")
    p.add_argument("--val-pooled", default="data/processed/val_embeddings_esm2_embeddings.pkl")
    p.add_argument("--test-pooled", default="data/processed/test_embeddings_esm2_embeddings.pkl")
    p.add_argument("--structures", default="data/structures")
    p.add_argument("--go-obo", default="data/go-basic.obo")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--struct-hidden-dim", type=int, default=512)
    p.add_argument("--struct-out-dim", type=int, default=256)
    p.add_argument("--head-hidden", type=int, default=512)
    p.add_argument("--min-go-freq", type=int, default=3)
    p.add_argument("--pos-weight-cap", type=float, default=10.0)
    p.add_argument("--contrastive-lambda", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="final_struct_contrastive_15ep.json")
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

    train_df, train_res = filter_split(args.train_csv, args.train_residue, args.structures)
    val_df, val_res = filter_split(args.val_csv, args.val_residue, args.structures)
    test_df, test_res = filter_split(args.test_csv, args.test_residue, args.structures)

    tmp = ".eval_struct_contrastive_tmp"
    os.makedirs(tmp, exist_ok=True)
    train_csv = os.path.join(tmp, "train.csv"); train_df.to_csv(train_csv, index=False)
    val_csv = os.path.join(tmp, "val.csv");     val_df.to_csv(val_csv, index=False)
    test_csv = os.path.join(tmp, "test.csv");   test_df.to_csv(test_csv, index=False)
    train_ds = StructureDataset(train_csv, args.structures, train_res)
    val_ds = StructureDataset(val_csv, args.structures, val_res)
    test_ds = StructureDataset(test_csv, args.structures, test_res)
    print(f"[info] splits — train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    with open(args.train_pooled, "rb") as f: pooled_train = {str(k): v for k, v in pickle.load(f).items()}
    with open(args.val_pooled, "rb") as f:   pooled_val = {str(k): v for k, v in pickle.load(f).items()}
    with open(args.test_pooled, "rb") as f:  pooled_test = {str(k): v for k, v in pickle.load(f).items()}
    pooled_all = {**pooled_train, **pooled_val, **pooled_test}

    in_dim_residue = train_ds[0]["x"].shape[1]
    in_dim_pooled = next(iter(pooled_train.values())).shape[0]
    encoder = StructureEncoder(
        in_dim=in_dim_residue,
        hidden_dim=args.struct_hidden_dim,
        out_dim=args.struct_out_dim,
        dropout=args.dropout,
        pool="mean",
    )
    model = ContrastiveStructureModel(
        encoder, seq_in_dim=in_dim_pooled, num_labels=len(labels),
        hidden=args.head_hidden, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_fmax = -1.0
    for ep in range(1, args.epochs + 1):
        loss, contrast = train_epoch(model, train_ds, pooled_all, label_to_idx, optimizer, pos_weight, device, args.batch_size, args.contrastive_lambda)
        val_pred = predict(model, val_ds, pooled_all, label_to_idx, device)
        val_fmax, val_t = fmax_score(val_pred["y_true"], val_pred["y_prob"])
        print(f"[ep {ep:02d}] loss={loss:.4f}  contrast={contrast:.4f}  val_fmax={val_fmax:.4f}@t={val_t:.2f}")
        if val_fmax > best_val_fmax:
            best_val_fmax = val_fmax
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred = predict(model, test_ds, pooled_all, label_to_idx, device)
    main_metrics = summarize_metrics(test_pred, labels, ontology, ic)
    main_metrics.pop("mean_gate_sequence", None)
    main_metrics.pop("mean_gate_text", None)
    main_metrics.pop("mean_gate_structure", None)

    results = {
        "model": "structure_contrastive_penlight",
        "best_val_fmax": best_val_fmax,
        "test_remote_homology_subset": main_metrics,
        "_notes": {
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "epochs": args.epochs,
            "contrastive_lambda": args.contrastive_lambda,
            "comment": "PenLight-style: GCN over residue contact graph + InfoNCE alignment between z_struct and pooled-ESM2 z_seq. Test set is the CD-HIT 40%-identity remote-homology split.",
        },
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    torch.save(best_state, "best_struct_contrastive.pt")
    print(f"[done] saved {args.out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
