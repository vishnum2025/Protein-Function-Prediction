"""Combined model v3: concat fusion (no softmax gate) + aux heads + modality dropout.

Removes the softmax gating from v2 because v2 still collapsed onto a single
modality (text, with mean_gate_text ≈ 1.0). Concat fusion lets the main head
learn its own per-feature mixing without a winner-take-all bottleneck.

Keeps v2's two interventions:
  Fix 1 — per-modality auxiliary BCE heads
  Fix 2 — modality dropout (zero one modality at fusion stage during training)

Reuses metric harness from combined_model.py for direct comparability.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from structure_encoder import StructureDataset, StructureEncoder
from combined_model import (
    GOOntology,
    ProteinDataset,
    build_label_space,
    class_pos_weight,
    fmax_score,
    information_content,
    multi_hot,
    parse_go,
    summarize_metrics,
)
def make_head(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class CombinedModelV3(nn.Module):
    """Concat fusion + per-modality aux heads + training-time modality dropout."""

    def __init__(
        self,
        seq_dim: int,
        text_dim: int,
        struct_dim: int,
        num_labels: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        modality_dropout_prob: float = 0.3,
    ):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)

        # Auxiliary heads (one per modality)
        self.aux_seq_head = make_head(hidden_dim, hidden_dim, num_labels, dropout)
        self.aux_text_head = make_head(hidden_dim, hidden_dim, num_labels, dropout)
        self.aux_struct_head = make_head(hidden_dim, hidden_dim, num_labels, dropout)

        # Concat fusion: head input is 3 * hidden_dim, no gate.
        self.main_head = make_head(hidden_dim * 3, hidden_dim, num_labels, dropout)

        self.modality_dropout_prob = modality_dropout_prob

    def _modality_dropout(self, zs, zt, zg):
        if not self.training or self.modality_dropout_prob <= 0:
            return zs, zt, zg
        if torch.rand((), device=zs.device).item() >= self.modality_dropout_prob:
            return zs, zt, zg
        which = int(torch.randint(0, 3, (), device=zs.device).item())
        if which == 0:
            zs = torch.zeros_like(zs)
        elif which == 1:
            zt = torch.zeros_like(zt)
        else:
            zg = torch.zeros_like(zg)
        return zs, zt, zg

    def forward(self, z_seq, z_text, z_struct):
        zs = self.seq_proj(z_seq)
        zt = self.text_proj(z_text)
        zg = self.struct_proj(z_struct)

        aux_seq = self.aux_seq_head(zs)
        aux_text = self.aux_text_head(zt)
        aux_struct = self.aux_struct_head(zg)

        zs_f, zt_f, zg_f = self._modality_dropout(zs, zt, zg)
        fused = torch.cat([zs_f, zt_f, zg_f], dim=-1)
        main = self.main_head(fused)

        # No gate; report per-modality L2 norms as a contribution proxy.
        contrib = torch.stack([
            zs_f.norm(dim=-1),
            zt_f.norm(dim=-1),
            zg_f.norm(dim=-1),
        ], dim=-1)  # (B, 3)

        return {
            "main": main,
            "aux_seq": aux_seq,
            "aux_text": aux_text,
            "aux_struct": aux_struct,
            "contrib": contrib,
        }


def train_epoch(model, struct_encoder, ds, optimizer, pos_weight, device, batch_size, aux_weight):
    model.train()
    struct_encoder.train()
    order = np.random.permutation(len(ds))
    total = 0.0
    aux_total = 0.0
    seen = 0
    for start in range(0, len(order), batch_size):
        idxs = order[start:start + batch_size]
        z_seq, z_text, z_struct, y = [], [], [], []
        for idx in idxs:
            item = ds[int(idx)]
            z_seq.append(item["z_seq"].to(device))
            z_text.append(item["z_text"].to(device))
            z_struct.append(struct_encoder(item["x"].to(device), item["adj"].to(device)))
            y.append(item["y"].to(device))
        z_seq = torch.stack(z_seq); z_text = torch.stack(z_text)
        z_struct = torch.stack(z_struct); y = torch.stack(y)

        out = model(z_seq, z_text, z_struct)
        loss_main = F.binary_cross_entropy_with_logits(out["main"], y, pos_weight=pos_weight)
        loss_aux = (
            F.binary_cross_entropy_with_logits(out["aux_seq"], y, pos_weight=pos_weight)
            + F.binary_cross_entropy_with_logits(out["aux_text"], y, pos_weight=pos_weight)
            + F.binary_cross_entropy_with_logits(out["aux_struct"], y, pos_weight=pos_weight)
        )
        loss = loss_main + aux_weight * loss_aux

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(struct_encoder.parameters()), 5.0)
        optimizer.step()

        total += loss.item() * len(idxs)
        aux_total += loss_aux.item() * len(idxs)
        seen += len(idxs)
    return total / max(seen, 1), aux_total / max(seen, 1)


def predict(model, struct_encoder, ds, device):
    model.eval()
    struct_encoder.eval()
    y_true, y_prob, contrib, ids = [], [], [], []
    aux_seq_p, aux_text_p, aux_struct_p = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            zs = item["z_seq"].unsqueeze(0).to(device)
            zt = item["z_text"].unsqueeze(0).to(device)
            zg = struct_encoder(item["x"].to(device), item["adj"].to(device)).unsqueeze(0)
            out = model(zs, zt, zg)
            y_true.append(item["y"].numpy())
            y_prob.append(torch.sigmoid(out["main"]).squeeze(0).cpu().numpy())
            aux_seq_p.append(torch.sigmoid(out["aux_seq"]).squeeze(0).cpu().numpy())
            aux_text_p.append(torch.sigmoid(out["aux_text"]).squeeze(0).cpu().numpy())
            aux_struct_p.append(torch.sigmoid(out["aux_struct"]).squeeze(0).cpu().numpy())
            contrib.append(out["contrib"].squeeze(0).cpu().numpy())
            ids.append(item["uid"])
    contrib = np.stack(contrib)
    return {
        "protein_ids": ids,
        "y_true": np.stack(y_true),
        "y_prob": np.stack(y_prob),
        "y_prob_aux_seq": np.stack(aux_seq_p),
        "y_prob_aux_text": np.stack(aux_text_p),
        "y_prob_aux_struct": np.stack(aux_struct_p),
        # Provide a normalized contribution vector in the gates slot so that
        # summarize_metrics' mean_gate_* fields stay populated with something
        # meaningful (relative L2 contribution per modality).
        "gates": contrib / np.clip(contrib.sum(axis=1, keepdims=True), 1e-9, None),
    }


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
    p.add_argument("--aux-weight", type=float, default=0.3)
    p.add_argument("--modality-dropout-prob", type=float, default=0.3)
    p.add_argument("--out", default="final_combined_model_v3_results.json")
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

    train_ds = ProteinDataset(args.train_csv, args.train_seq, args.train_text, args.train_text_index, args.train_residue, args.structures, label_to_idx)
    val_ds = ProteinDataset(args.val_csv, args.val_seq, args.val_text, args.val_text_index, args.val_residue, args.structures, label_to_idx)
    test_ds = ProteinDataset(args.test_csv, args.test_seq, args.test_text, args.test_text_index, args.test_residue, args.structures, label_to_idx)
    print(f"[info] splits — train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    sample = train_ds[0]
    struct_encoder = StructureEncoder(
        in_dim=sample["x"].shape[1],
        hidden_dim=args.struct_hidden_dim,
        out_dim=args.struct_out_dim,
        dropout=args.dropout,
        pool="mean",
    ).to(device)

    model = CombinedModelV3(
        seq_dim=sample["z_seq"].shape[0],
        text_dim=sample["z_text"].shape[0],
        struct_dim=args.struct_out_dim,
        num_labels=len(labels),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        modality_dropout_prob=args.modality_dropout_prob,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(struct_encoder.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    best_state = None
    best_val_fmax = -1.0
    for ep in range(1, args.epochs + 1):
        loss, aux_loss = train_epoch(model, struct_encoder, train_ds, optimizer, pos_weight, device, args.batch_size, args.aux_weight)
        val_pred = predict(model, struct_encoder, val_ds, device)
        val_fmax, val_t = fmax_score(val_pred["y_true"], val_pred["y_prob"])
        gmean = val_pred["gates"].mean(axis=0)
        print(f"[ep {ep:02d}] loss={loss:.4f}  aux_loss={aux_loss:.4f}  "
              f"val_fmax(main)={val_fmax:.4f}@t={val_t:.2f}  "
              f"contrib(s,t,g)=({gmean[0]:.3f},{gmean[1]:.3f},{gmean[2]:.3f})")
        if val_fmax > best_val_fmax:
            best_val_fmax = val_fmax
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "struct": {k: v.detach().cpu().clone() for k, v in struct_encoder.state_dict().items()},
            }

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        struct_encoder.load_state_dict(best_state["struct"])

    test_pred = predict(model, struct_encoder, test_ds, device)
    main_metrics = summarize_metrics(test_pred, labels, ontology, ic)

    def aux_overall(prob_key):
        fm, t = fmax_score(test_pred["y_true"], test_pred[prob_key])
        return {"fmax": fm, "fmax_threshold": t}

    results = {
        "model": "sequence_text_structure_concat_v3",
        "best_val_fmax": best_val_fmax,
        "test_remote_homology_subset": main_metrics,
        "aux_overall": {
            "seq": aux_overall("y_prob_aux_seq"),
            "text": aux_overall("y_prob_aux_text"),
            "struct": aux_overall("y_prob_aux_struct"),
        },
        "_v3_settings": {
            "fusion": "concat",
            "aux_weight": args.aux_weight,
            "modality_dropout_prob": args.modality_dropout_prob,
            "epochs": args.epochs,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "note": (
                "mean_gate_* fields hold relative L2 contribution per modality "
                "(not softmax gates — there is no gate in v3)."
            ),
        },
    }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    torch.save(best_state, "best_combined_model_v3.pt")
    print(f"[done] saved {args.out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
