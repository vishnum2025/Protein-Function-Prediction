import os
import pickle
import pandas as pd
import torch
import esm


class SequenceEncoder:
    def __init__(self, model_size="small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_size == "small":
            self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        else:
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def encode_batch(self, sequences):
        if not sequences:
            return {}

        batch = [(pid, seq) for pid, seq in sequences]
        labels, strs, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)

        outputs = self.model(tokens, repr_layers=[self.model.num_layers])
        reps = outputs["representations"][self.model.num_layers]

        embeddings = {}
        for i, pid in enumerate(labels):
            seq_len = len(strs[i])
            residue_embeds = reps[i, 1:seq_len + 1]
            pooled = residue_embeds.mean(dim=0)
            embeddings[pid] = pooled.cpu().numpy()

        return embeddings

    def encode(self, sequences, batch_size=4):
        all_embeddings = {}

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            batch_embeddings = self.encode_batch(batch)
            all_embeddings.update(batch_embeddings)

            print(f"Processed {min(i + batch_size, len(sequences))}/{len(sequences)} sequences")

        return all_embeddings


def clean_sequence(seq):
    if pd.isna(seq):
        return None
    seq = str(seq).strip().upper().replace(" ", "")
    return seq if seq else None


def run_encoder(csv_path, output_path, batch_size=3, model_size="small"):
    df = pd.read_csv(csv_path)

    if "uniprot_id" not in df.columns or "sequence" not in df.columns:
        raise ValueError("CSV must contain 'uniprot_id' and 'sequence' columns")

    df = df[["uniprot_id", "sequence"]].copy()
    df["sequence"] = df["sequence"].apply(clean_sequence)
    df = df.dropna(subset=["uniprot_id", "sequence"])

    sequences = list(zip(df["uniprot_id"], df["sequence"]))

    encoder = SequenceEncoder(model_size=model_size)
    embeddings = encoder.encode(sequences, batch_size=batch_size)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Saved embeddings to {output_path}")