import subprocess
import os
import pandas as pd

class SequenceClustering:
    def __init__(self, df: pd.DataFrame, threshold: float):
        self.df = df
        self.threshold = threshold

    def export_fasta(self, filepath: str):
        with open(filepath, 'w') as f:
            for _, row in self.df.iterrows():
                f.write(f">{row['uniprot_id']}\n{row['sequence']}\n")

    def generate_remote_homology_splits(self, fasta_file: str, output_dir: str):
        """Uses CD-HIT to cluster sequences ensuring test/train lack high similarity[cite: 1, 129]."""
        output_cluster_file = os.path.join(output_dir, "clusters")
        
        # Construct CD-HIT command (requires cd-hit installed on system)
        cmd = [
            "cd-hit",
            "-i", fasta_file,
            "-o", output_cluster_file,
            "-c", str(self.threshold),
            "-n", "2" # Word length for threshold 0.3
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("Clustering completed. Parse the .clstr file to create train/val/test dataframes.")
        except FileNotFoundError:
            print("CD-HIT not installed. Please install it to execute clustering.")