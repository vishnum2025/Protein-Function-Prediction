import os

# API Endpoints
UNIPROT_API_BASE = "https://rest.uniprot.org/uniprotkb"
ALPHAFOLD_API_BASE = "https://alphafold.ebi.ac.uk/api/prediction"

# Directory Structure
DATA_DIR = "./data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
STRUCTURES_DIR = os.path.join(DATA_DIR, "structures")

# Preprocessing Parameters
HOMOLOGY_THRESHOLD = 0.30  # 30% sequence identity for remote-homology splits
MAX_RESOLUTION = 2.5  # Angstroms, for PDB fallback

for d in [RAW_DIR, PROCESSED_DIR, STRUCTURES_DIR]:
    os.makedirs(d, exist_ok=True)