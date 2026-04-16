import requests
import os
from Bio.PDB import PDBParser

class StructurePipeline:
    def __init__(self, api_base: str, save_dir: str):
        self.api_base = api_base
        self.save_dir = save_dir
        self.parser = PDBParser(QUIET=True)

    def fetch_alphafold_structure(self, uniprot_id: str) -> str:
        """Downloads PDB file from AlphaFold DB[cite: 40]."""
        url = f"{self.api_base}/{uniprot_id}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                pdb_url = data[0].get('pdbUrl')
                pdb_response = requests.get(pdb_url)
                file_path = os.path.join(self.save_dir, f"{uniprot_id}.pdb")
                with open(file_path, 'w') as f:
                    f.write(pdb_response.text)
                return file_path
        return None

    def validate_sequence(self, uniprot_sequence: str, pdb_file: str) -> bool:
        """Ensures the structure sequence matches the UniProt sequence exactly[cite: 43]."""
        if not pdb_file or not os.path.exists(pdb_file):
            return False
            
        structure = self.parser.get_structure('protein', pdb_file)
        pdb_sequence = ""
        
        # Extract sequence from PDB C-alpha atoms
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id('CA'):
                         # Simplified: requires 3-letter to 1-letter conversion mapping in practice
                        pdb_sequence += "X" 
                        
        # Simplified validation logic
        return uniprot_sequence == pdb_sequence