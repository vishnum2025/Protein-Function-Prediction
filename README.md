# MLCB Hackathon & Final Project

Coursework from Georgia Tech CS 4803/8803 MLCB (Machine Learning in Computational Biology), Spring 2025. Two projects on machine-learning approaches to protein modeling:

1. **Final Project** — Multimodal protein function prediction (Gene Ontology terms) from sequence, 3D structure, and free-text annotations.
2. **Hackathon** — Active-learning-guided prediction of protein fitness from single-site mutations (Deep Mutational Scanning).


## Team

Nawal Reza · Vishnu Mothukuri · Saanvi Bhumpalle · Akshay Shivashankar · Prajna Dhinakaran

---

## `Final Project/` — Multimodal Protein Function Prediction

Predicting Gene Ontology terms by fusing three biological modalities: 1D amino-acid sequence, 3D residue contact graph, and free-text UniProt annotations. Evaluated under a remote-homology split (CD-HIT ≤ 40% sequence identity) to test true functional generalization rather than sequence memorization.

### Data

- **Sequences + GO labels + text**: UniProtKB/Swiss-Prot
- **3D structures**: AlphaFold Protein Structure Database
- **Split**: 80/10/10 train/val/test at the CD-HIT cluster level
- Text sanitization scrubs exact GO terms and EC numbers to prevent label leakage; missing annotations backfilled via nearest sequence-neighbor.

### Architecture

| Modality | Encoder | Dim |
| --- | --- | --- |
| Sequence | ESM-2 (`esm2_t30_150M_UR50D`), mean-pooled | 640 |
| Structure | 3-layer GCN on Cα contact graph (10 Å threshold) with per-residue ESM-2 node features | 256 |
| Text | PubMedBERT (`BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`), mean-pooled | 768 |

Each modality is linearly projected to a shared hidden dimension, then concatenated and passed to a classifier head. Training uses:
- **Auxiliary heads** for each modality (loss weight λ = 0.3) so each encoder stays independently predictive.
- **Modality dropout** (p = 0.3) during training to prevent over-reliance on any single input.
- Weighted binary cross-entropy over GO terms to handle class imbalance.

### Results (remote-homology test set)

| Model | Fmax | AUPR |
| --- | --- | --- |
| Sequence only | 0.440 | 0.378 |
| Sequence + Structure (contact graph) | 0.438 | 0.378 |
| Sequence + Structure (contrastive) | 0.441 | 0.381 |
| Sequence + Text | 0.423 | 0.377 |
| **Sequence + Structure + Text (combined)** | **0.478** | **0.455** |

Per-sub-ontology, the combined model wins across Molecular Function (Fmax 0.535), Biological Process (0.431), and Cellular Component (0.648), and achieves the lowest Smin in every category.

---

## `Hackathon/` — Protein Fitness Prediction with Active Learning

Predicting DMS fitness scores for single-site mutations on a 656-residue protein, starting from only ~10% labeled coverage and using three rounds of active learning to expand the labeled set.

### Approach

Iterative progression from physicochemical baselines to structure- and evolution-aware models:

| Model | Spearman ρ |
| --- | --- |
| Ridge Regression (ESM-2 zero-shot only) | 0.393 |
| SaProt Embeddings + MLP head | 0.402 |
| XGBoost (BLOSUM62 + physicochemical) | 0.415 |
| Random Forest + amino-acid embeddings | 0.418 |
| GP, Matern-5/2 kernel, ESM-2 features | 0.438 |
| GP, Matern-5/2 kernel, 16 features (ESM-2 + physicochemical + VHSE) | 0.476 |
| **GP, composite Matern + RBF kernel, 16 features, 1500 iters + ReduceLROnPlateau** | **0.502** |

The final model is a Gaussian Process regressor over a 16-dimensional feature vector combining:
- 4 ESM-2 features (masked-marginal zero-shot score, positional Shannon entropy, wild-type log-prob, mutant log-prob)
- 4 physicochemical substitution metrics (BLOSUM62, Δhydrophobicity, Δcharge, Δvolume)
- 8 VHSE principal-component descriptors

### Active-learning strategy

Three query rounds, progressing from exploitation to exploration to a principled trade-off:

1. **Round 1** — Greedy selection by Ridge-predicted fitness (positional coverage 60 → 160).
2. **Round 2** — Maximum-entropy exploration over the ESM-2 amino-acid distribution (coverage 160 → 260).
3. **Round 3** — GP-UCB acquisition using the GP's own posterior uncertainty (coverage 260 → 360).

Position-split 5-fold cross-validation prevents position memorization; a caching layer for masked-marginal log-probabilities across all 656 wild-type positions kept the active-learning loop tractable.


---

## References

Key references are listed in each folder's project report PDF. Core dependencies include ESM-2 (Lin et al., 2023), DeepFRI-style GCN structure encoders (Gligorijević et al., 2021), PubMedBERT, AlphaFold (Jumper et al., 2021), and Gaussian Process regression on protein fitness landscapes (Romero, Krause, Arnold, 2013).

## Note

This repo was mirrored from an internal Georgia Tech Enterprise GitHub instance for archival on my personal account.
