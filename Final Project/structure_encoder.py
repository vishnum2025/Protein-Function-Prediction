"""Structure encoder: PDB -> contact graph -> 3-layer GCN -> z_G.

Mirrors the residue order of the validated UniProt sequence (only residues
with a CA atom and a standard amino-acid code are kept) so per-residue ESM-2
features line up 1:1 with the graph nodes.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser
from torch.utils.data import Dataset

_D3TO1 = {k.upper(): v for k, v in protein_letters_3to1.items()}
_PARSER = PDBParser(QUIET=True)


@dataclass
class ProteinGraph:
    uniprot_id: str
    sequence: str
    ca_coords: torch.Tensor  # (L, 3)
    plddt: torch.Tensor      # (L,) per-residue pLDDT from B-factor
    adj: torch.Tensor        # (L, L) symmetric 0/1 contact adjacency (no self-loops)


def parse_pdb_to_graph(pdb_path: str, uniprot_id: str, cutoff: float = 10.0) -> ProteinGraph:
    """Parse an AlphaFold PDB into a residue-level contact graph.

    Edges connect residues whose Cα atoms are within `cutoff` Å. pLDDT is read
    from the B-factor column of the Cα atom (AlphaFold convention).
    """
    structure = _PARSER.get_structure(uniprot_id, pdb_path)

    seq_chars, coords, plddt = [], [], []
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().upper()
                if not residue.has_id("CA") or resname not in _D3TO1:
                    continue
                ca = residue["CA"]
                seq_chars.append(_D3TO1[resname])
                coords.append(ca.get_coord())
                plddt.append(ca.get_bfactor())

    ca_coords = torch.tensor(np.asarray(coords), dtype=torch.float32)
    plddt_t = torch.tensor(plddt, dtype=torch.float32)
    diff = ca_coords.unsqueeze(0) - ca_coords.unsqueeze(1)
    dist = diff.norm(dim=-1)
    adj = (dist <= cutoff).float()
    adj.fill_diagonal_(0.0)

    return ProteinGraph(
        uniprot_id=uniprot_id,
        sequence="".join(seq_chars),
        ca_coords=ca_coords,
        plddt=plddt_t,
        adj=adj,
    )


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization with self-loops: D^-1/2 (A+I) D^-1/2."""
    n = adj.size(0)
    a_hat = adj + torch.eye(n, device=adj.device, dtype=adj.dtype)
    deg = a_hat.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt = torch.where(torch.isinf(deg_inv_sqrt), torch.zeros_like(deg_inv_sqrt), deg_inv_sqrt)
    return deg_inv_sqrt.unsqueeze(1) * a_hat * deg_inv_sqrt.unsqueeze(0)


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm @ self.lin(x)


class StructureEncoder(nn.Module):
    """3-layer GCN with LayerNorm + dropout, global sum pooling.

    Input:  x (L, in_dim) residue features, adj (L, L) 0/1 contact graph.
    Output: z_G (out_dim,) protein-level structural representation.
    """

    def __init__(
        self,
        in_dim: int = 640,
        hidden_dim: int = 512,
        out_dim: int = 256,
        dropout: float = 0.3,
        pool: str = "mean",
    ):
        super().__init__()
        if pool not in {"mean", "sum"}:
            raise ValueError(f"pool must be 'mean' or 'sum', got {pool}")
        self.pool = pool
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = GCNLayer(hidden_dim, out_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        a_norm = normalize_adj(adj)
        h = self.dropout(F.relu(self.ln1(self.gcn1(x, a_norm))))
        h = self.dropout(F.relu(self.ln2(self.gcn2(h, a_norm))))
        h = self.ln3(self.gcn3(h, a_norm))
        return h.mean(dim=0) if self.pool == "mean" else h.sum(dim=0)

    def forward_batch(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Sequential forward over a list of (x, adj) pairs.

        Variable-length proteins make true batching awkward without PyG; this
        is fine for ~200 proteins. Swap to block-diagonal batching if needed.
        """
        return torch.stack([self.forward(x, adj) for x, adj in batch], dim=0)


class StructureDataset(Dataset):
    """Yields (uniprot_id, x_residue_features, adj, sequence, go_terms).

    `residue_embeddings` is a dict[uniprot_id] -> np.ndarray of shape (L, in_dim)
    matching the validated CA-only sequence order produced by parse_pdb_to_graph.
    """

    def __init__(
        self,
        csv_path: str,
        structures_dir: str,
        residue_embeddings: dict,
        cutoff: float = 10.0,
    ):
        self.df = pd.read_csv(csv_path)
        self.structures_dir = structures_dir
        self.residue_embeddings = residue_embeddings
        self.cutoff = cutoff
        self.df = self.df[self.df["uniprot_id"].isin(residue_embeddings.keys())].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        uniprot_id = row["uniprot_id"]
        pdb_path = os.path.join(self.structures_dir, f"{uniprot_id}.pdb")
        graph = parse_pdb_to_graph(pdb_path, uniprot_id, cutoff=self.cutoff)

        feats = self.residue_embeddings[uniprot_id]
        if feats.shape[0] != graph.ca_coords.shape[0]:
            raise ValueError(
                f"{uniprot_id}: residue embedding length {feats.shape[0]} "
                f"!= graph length {graph.ca_coords.shape[0]}"
            )
        x = torch.as_tensor(feats, dtype=torch.float32)

        return {
            "uniprot_id": uniprot_id,
            "x": x,
            "adj": graph.adj,
            "plddt": graph.plddt,
            "sequence": graph.sequence,
            "go_terms": row.get("go_terms"),
        }
