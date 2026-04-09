# ML-Guided Protein Engineering with Active Learning

## Overview

This project focuses on predicting the functional impact of single-site protein mutations using machine learning under limited labeled data (~10%). We leverage pretrained protein language models (ESM-2), regression models, Gaussian Processes, and active learning strategies to explore the protein fitness landscape.

Our goal is to:

* Predict fitness scores for unseen mutations
* Maximize Spearman correlation with ground truth (experimentally measured fitness scores)
* Identify the top 10 beneficial mutations

---

## Repository Structure

```
Hackathon/
│
├── Data/
│   ├── sequence.fasta
│   ├── train.csv
│   ├── test.csv
│
├── GP/
│   ├── Hackathon_v2.ipynb        # Main GP + feature engineering pipeline
│   ├── predictions.csv           # Final predictions
│   ├── submission_round4_opt.csv # Final optimized submission
│   ├── top10.txt                 # Top 10 predicted mutations
│
├── Ridge/
│   ├── ridge.py                  # Zero-shot + Ridge calibration model
│   ├── predictions_round1_ridge.csv
│
├── xgboost_model/
│   ├── xgboost_model.ipynb       # XGBoost baseline model
│   ├── predictions_xgb.csv
│   ├── query_xgb.txt
│   ├── top10_xgb.txt
|
│── randomforest_regressor/       # Baseline RF model with engineered features
|   ├── pos_aa_embedding.ipynb
|   ├── aa_predictions.csv
|
├── early_checkpoint/             # Checkpoint submissions
│
├── APIKey.txt
├── GroupName.txt
├── Hackathon.md                  # Report / notes
```

---

## Setup Instructions

### 1. Create Environment

```bash
conda create -n protein_ml python=3.10
conda activate protein_ml
pip install torch numpy pandas scikit-learn scipy esm xgboost
```

---

### 2. Data Setup

Ensure the following files are placed in the `Data/` directory:

* `sequence.fasta`
* `train.csv`
* `test.csv`

---

### 3. Running Models

#### Ridge Model (Zero-shot + Calibration)

```bash
cd Ridge
python ridge.py
```

This:

* Loads ESM-2 zero-shot scores
* Trains Ridge regression
* Outputs predictions

---

#### Random Forest Baseline

```bash
cd randomforest_regressor
jupyter notebook pos_aa_embedding.ipynb
```

* Uses engineered features (physicochemical + positional)
* Serves as an interpretable baseline

---

#### XGBoost Model

```bash
cd xgboost_model
jupyter notebook xgboost_model.ipynb
```

Used for:

* Baseline modeling
* Feature-based regression (BLOSUM + physicochemical)

---

#### Gaussian Process Model

```bash
cd GP
jupyter notebook Hackathon_v2.ipynb
```

This notebook:

* Combines ESM-2 + engineered features
* Trains GP with Matern + RBF kernels
* Performs active learning integration
* Generates final predictions and top 10 mutations

---
## Feature Engineering Pipeline

Raw mutation inputs (e.g., `M59L`) are transformed into numerical features through the following pipeline:

1. **Mutation Parsing**

   * Extract wild-type residue, position, and mutant residue

2. **ESM-2 Zero-Shot Scores**

   * Log-likelihood difference between mutant and wild-type amino acids

3. **Uncertainty Features**

   * Shannon entropy from amino acid probability distribution

4. **Physicochemical Features**

   * Δhydrophobicity, Δcharge, Δvolume

5. **Evolutionary Features**

   * BLOSUM62 substitution scores

6. **VHSE Descriptors**

   * Vector encoding of amino acid properties

7. **Normalization**

   * Standard scaling for stable model training

---

## Model Development Progression

Our modeling approach evolved iteratively:

1. **Random Forest (Baseline)**

   * Feature-based model using physicochemical descriptors
   * Limited ability to capture long-range dependencies

2. **XGBoost**

   * Improved nonlinear modeling of engineered features

3. **Ridge Regression with ESM-2**

   * Introduced pretrained embeddings
   * Strong improvement in generalization

4. **Gaussian Process (Final Model)**

   * Combines ESM features + engineered features
   * Provides uncertainty estimates for active learning
   * Achieves best performance

---

## Validation Strategy

* **Position-split cross-validation**

  * Ensures model generalizes to unseen mutation positions
* **Evaluation Metric**

  * Spearman correlation (ranking accuracy)

---

## Active Learning Strategy

We used three query rounds:

1. **Round 1 — Greedy Selection**

   * Selected highest predicted fitness mutations

2. **Round 2 — Entropy-Based Exploration**

   * Selected mutations from high-uncertainty positions

3. **Round 3 — GP-UCB**

   * Balanced exploration and exploitation using:

     * Prediction mean
     * Model uncertainty

---

## Outputs

* `predictions.csv` → predicted fitness for all test mutations
* `submission_round4_opt.csv` → final leaderboard submission
* `top10.txt` → top 10 predicted mutations

---

## Reproducibility Notes

* ESM-2 outputs are cached to reduce recomputation time
* Random seeds are fixed where applicable
* Query data from all rounds is merged before training

---

## Key Takeaways

* Pretrained protein language models significantly improve performance under low-data settings
* Feature engineering plays a critical role in model effectiveness
* Active learning enables efficient exploration of large mutation spaces
* Gaussian Processes provide both strong predictions and uncertainty estimates

---
