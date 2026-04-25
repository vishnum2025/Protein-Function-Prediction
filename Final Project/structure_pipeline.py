import requests
import os
from Bio.PDB import PDBParser
from Bio.Data.IUPACData import protein_letters_3to1

class StructurePipeline:
    def __init__(self, api_base: str, save_dir: str):
        self.api_base = api_base
        self.save_dir = save_dir
        self.parser = PDBParser(QUIET=True)
        # Create an all-caps mapping dictionary (e.g., 'ALA': 'A')
        self.d3to1 = {k.upper(): v for k, v in protein_letters_3to1.items()}

    def fetch_alphafold_structure(self, uniprot_id: str) -> str:
        """Downloads PDB file from AlphaFold DB (cached on disk)."""
        file_path = os.path.join(self.save_dir, f"{uniprot_id}.pdb")
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return file_path
        url = f"{self.api_base}/{uniprot_id}"
        try:
            response = requests.get(url, timeout=30)
        except requests.exceptions.RequestException:
            return None
        if response.status_code == 200:
            data = response.json()
            if data:
                pdb_url = data[0].get('pdbUrl')
                try:
                    pdb_response = requests.get(pdb_url, timeout=60)
                except requests.exceptions.RequestException:
                    return None
                with open(file_path, 'w') as f:
                    f.write(pdb_response.text)
                return file_path
        return None

    def validate_sequence(self, uniprot_sequence: str, pdb_file: str) -> bool:
        """Ensures the structure sequence matches the UniProt sequence exactly."""
        if not pdb_file or not os.path.exists(pdb_file):
            return False
        try:
            structure = self.parser.get_structure('protein', pdb_file)
            pdb_sequence = ""
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        # Only look at standard amino acids (ignore water, ligands)
                        resname = residue.get_resname().upper()
                        if residue.has_id('CA') and resname in self.d3to1:
                            pdb_sequence += self.d3to1[resname]
                            
            return uniprot_sequence == pdb_sequence
        except Exception as e:
            print(f"Error parsing {pdb_file}: {e}")
            return False