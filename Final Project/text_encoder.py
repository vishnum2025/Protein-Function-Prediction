import ast
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import os


class PubMedBERTEmbedder:
    MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

    def __init__(self, batch_size=16, max_length=512):
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()

    def _prepare_inputs(self, df: pd.DataFrame) -> list[str]:
        df = df.copy()
        # Filter out rows with missing or empty clean_text
        df = df[df["clean_text"].notna() & (df["clean_text"].str.strip() != "")]
        df["go_terms_str"] = df["go_terms"].apply(
            lambda x: " ".join(ast.literal_eval(x)) if pd.notna(x) else ""
        )
        return (df["clean_text"] + " [SEP] " + df["go_terms_str"]).tolist()

    def _mean_pool(self, token_embeddings, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**encoded)

            embeddings = self._mean_pool(
                output.last_hidden_state, encoded["attention_mask"]
            ).cpu().numpy()

            all_embeddings.append(embeddings)
            print(f"Processed {min(i + self.batch_size, len(texts))}/{len(texts)}")

        return np.vstack(all_embeddings)

    def embed_dataframe(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        texts = self._prepare_inputs(df)
        texts = [str(text) for text in texts]  # Ensure all texts are strings
        filtered_df = df[df["clean_text"].notna() & (df["clean_text"].str.strip() != "")]
        return self.embed_texts(texts), filtered_df

    def save(self, embeddings: np.ndarray, df: pd.DataFrame,
             emb_path="pubmedbert_embeddings.npy", idx_path="embedding_index.csv"):
        np.save(emb_path, embeddings)
        df[["uniprot_id"]].assign(embedding_idx=range(len(df))).to_csv(idx_path, index=False)
        print(f"Saved {embeddings.shape} embeddings to {emb_path}")


if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), "data/processed")
    output_dir = os.path.join(os.getcwd(), "data/text_encodings")

    # Initialize embedder
    embedder = PubMedBERTEmbedder(batch_size=16)

    # Process train, val, and test splits
    splits = ["train", "val", "test"]

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")

        # Load dataset
        file_name = f"{split}_dataset_propagated.csv"
        df = pd.read_csv(os.path.join(data_dir, file_name))
        print(f"Loaded {len(df)} samples from {split}_dataset_propagated.csv")

        # Generate embeddings
        print(f"Generating embeddings for {split} split...")
        embeddings, df_filtered = embedder.embed_dataframe(df)
        print(f"Filtered to {len(df_filtered)} valid samples")
        print(f"Generated embeddings shape: {embeddings.shape}")

        # Save embeddings and index
        emb_path = os.path.join(output_dir, f"{split}_embeddings.npy")
        idx_path = os.path.join(output_dir, f"{split}_index.csv")

        embedder.save(embeddings, df_filtered, emb_path, idx_path)
        print(f"Saved {split} embeddings to {emb_path}")

    print(f"\n{'='*60}")
    print("All splits processed successfully!")
    print(f"Embeddings saved to {output_dir}")
    print(f"{'='*60}")