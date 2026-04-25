"""Per-residue ESM-2 features for use as GCN node features.

Mirrors `sequence_encoder.SequenceEncoder` but skips the mean-pool so the
structure encoder can use residue-level representations. The output sequence
length matches the *input* UniProt sequence — the structure encoder is
responsible for keeping its CA-only graph aligned to the same order (which
the data pipeline already guarantees via `validate_sequence`).
"""

import os
import pickle

import esm
import pandas as pd
import torch

from sequence_encoder import clean_sequence


class ResidueEmbedder:
    def __init__(self, model_size: str = "small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_size == "small":
            self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        else:
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval().to(self.device)

    @torch.no_grad()
    def encode_batch(self, sequences):
        if not sequences:
            return {}
        labels, strs, tokens = self.batch_converter([(pid, seq) for pid, seq in sequences])
        tokens = tokens.to(self.device)
        out = self.model(tokens, repr_layers=[self.model.num_layers])
        reps = out["representations"][self.model.num_layers]

        per_residue = {}
        for i, pid in enumerate(labels):
            seq_len = len(strs[i])
            per_residue[pid] = reps[i, 1:seq_len + 1].cpu().numpy()
        return per_residue

    def encode(self, sequences, batch_size: int = 2):
        all_embeds = {}
        for i in range(0, len(sequences), batch_size):
            chunk = sequences[i:i + batch_size]
            all_embeds.update(self.encode_batch(chunk))
            print(f"Processed {min(i + batch_size, len(sequences))}/{len(sequences)} sequences")
        return all_embeds


def run_residue_embedder(csv_path: str, output_path: str, batch_size: int = 2, model_size: str = "small"):
    df = pd.read_csv(csv_path)
    if "uniprot_id" not in df.columns or "sequence" not in df.columns:
        raise ValueError("CSV must contain 'uniprot_id' and 'sequence' columns")

    df = df[["uniprot_id", "sequence"]].copy()
    df["sequence"] = df["sequence"].apply(clean_sequence)
    df = df.dropna(subset=["uniprot_id", "sequence"])

    sequences = list(zip(df["uniprot_id"], df["sequence"]))
    embedder = ResidueEmbedder(model_size=model_size)
    embeddings = embedder.encode(sequences, batch_size=batch_size)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saved residue-level embeddings to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model-size", choices=["small", "large"], default="small")
    args = parser.parse_args()
    run_residue_embedder(args.csv, args.out, args.batch_size, args.model_size)
