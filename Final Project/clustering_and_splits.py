import subprocess
import os
import pandas as pd
import re
import numpy as np

class SequenceClustering:
    def __init__(self, df: pd.DataFrame, threshold: float):
        self.df = df
        self.threshold = threshold

    def export_fasta(self, filepath: str):
        with open(filepath, 'w') as f:
            for _, row in self.df.iterrows():
                f.write(f">{row['uniprot_id']}\n{row['sequence']}\n")

    def generate_remote_homology_splits(self, fasta_file: str, output_dir: str):
        output_cluster_file = os.path.join(output_dir, "clusters")
        cmd = [
            "cd-hit",
            "-i", fasta_file,
            "-o", output_cluster_file,
            "-c", str(self.threshold),
            "-n", "2"
        ]
        try:
            subprocess.run(cmd, check=True)
            self._parse_clusters_and_split(f"{output_cluster_file}.clstr", output_dir)
        except FileNotFoundError:
            print("CD-HIT not installed. Please install it to execute clustering.")

    def _parse_clusters_and_split(self, clstr_file: str, output_dir: str):
        cluster_map = {}
        current_cluster = -1
        
        with open(clstr_file, 'r') as f:
            for line in f:
                if line.startswith(">Cluster"):
                    current_cluster += 1
                else:
                    match = re.search(r'>([A-Z0-9]+)', line)
                    if match:
                        cluster_map[match.group(1)] = current_cluster
                        
        self.df['cluster_id'] = self.df['uniprot_id'].map(cluster_map)
        
        clusters = self.df['cluster_id'].dropna().unique()
        np.random.seed(42)
        np.random.shuffle(clusters)
        
        train_idx = int(len(clusters) * 0.8)
        val_idx = int(len(clusters) * 0.9)
        
        train_clusters = clusters[:train_idx]
        val_clusters = clusters[train_idx:val_idx]
        test_clusters = clusters[val_idx:]
        
        train_df = self.df[self.df['cluster_id'].isin(train_clusters)]
        val_df = self.df[self.df['cluster_id'].isin(val_clusters)]
        test_df = self.df[self.df['cluster_id'].isin(test_clusters)]
        
        train_df.to_csv(os.path.join(output_dir, "train_dataset.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val_dataset.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test_dataset.csv"), index=False)
        print(f"Splits saved to {output_dir}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test.")