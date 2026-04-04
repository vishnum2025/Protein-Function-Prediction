"""
Gaussian Process

Input features :
  1. zero_shot_score   = log P(mut) - log P(wt)
  2. position_entropy  = Shannon entropy of ESM distribution at position
  3. wt_log_prob       = log P(wt | context)
  4. normalized_pos    = position / seq_len  (structural context proxy)

Setup:  pip install gpytorch torch scikit-learn pandas scipy numpy

"""

import os
import torch
import gpytorch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.expanduser('~/v_files/GT/sem2/MLB/hackathon/Hackathon_data')
TRAIN_PATH = f'{DATA_DIR}/train.csv'
TEST_PATH  = f'{DATA_DIR}/test.csv'
FASTA_PATH = f'{DATA_DIR}/sequence.fasta'
CACHE_PATH = f'{DATA_DIR}/esm2_150M_zeroshot_scores.npz'

QUERY_PATHS = [f'{DATA_DIR}/query1_labeled.csv']  # ← update each round
ROUND       = 2                                    # ← update each round
GP_ITERS    = 300

# ── Load data ──────────────────────────────────────────────────────────────────
with open(FASTA_PATH) as f:
    sequence_wt = f.readlines()[1].strip()
SEQ_LEN = len(sequence_wt)

df_train = pd.read_csv(TRAIN_PATH)
df_test  = pd.read_csv(TEST_PATH)
for qp in QUERY_PATHS:
    df_train = pd.concat([df_train, pd.read_csv(qp)], ignore_index=True)
df_train = df_train.drop_duplicates(subset='mutant').reset_index(drop=True)

print(f"Train: {len(df_train)} | Test: {len(df_test)} | Seq: {SEQ_LEN}")

# ── Load ESM-2 cache ───────────────────────────────────────────────────────────
AAs = list('ACDEFGHIKLMNPQRSTVWY')
print("Loading ESM-2 cache ...")
pos_logprobs = np.load(CACHE_PATH, allow_pickle=True)['pos_logprobs'].item()
print(f"  {len(pos_logprobs)} positions loaded")

# ── Feature extraction ─────────────────────────────────────────────────────────
def build_features(df):
    feats = []
    for _, row in df.iterrows():
        m      = row['mutant']
        wt, mt = m[0], m[-1]
        pos    = int(m[1:-1])
        lp     = pos_logprobs[pos]

        zs       = lp[mt] - lp[wt]
        logprobs = np.array([lp[aa] for aa in AAs])
        probs    = np.exp(logprobs); probs /= probs.sum()
        entropy  = float(-np.sum(probs * np.log(probs + 1e-10)))
        wt_logp  = lp[wt]
        norm_pos = pos / SEQ_LEN

        feats.append([zs, entropy, wt_logp, norm_pos])
    return np.array(feats, dtype=np.float32)

print("Extracting features ...")
X_train_raw = build_features(df_train)
X_test_raw  = build_features(df_test)
y_train     = df_train['DMS_score'].values.astype(np.float32)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_test  = scaler.transform(X_test_raw).astype(np.float32)

# ── GP Model ───────────────────────────────────────────────────────────────────
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=train_x.shape[1]
            )
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x)
        )


def train_gp(X, y, n_iter=GP_ITERS, lr=0.1):
    train_x = torch.tensor(X)
    train_y = torch.tensor(y)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model      = ExactGPModel(train_x, train_y, likelihood)
    model.train(); likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll       = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(n_iter):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            ls = model.covar_module.base_kernel.lengthscale.detach().numpy().flatten()
            print(f"  Iter {i+1}/{n_iter} | Loss: {loss.item():.4f} | "
                  f"Lengthscales: {np.round(ls, 3)}")

    return model, likelihood


def gp_predict(model, likelihood, X):
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(torch.tensor(X)))
    return pred.mean.numpy(), pred.stddev.numpy()


# ── Position-split CV ──────────────────────────────────────────────────────────
print("\nRunning position-split 5-fold CV ...")
train_pos  = df_train['mutant'].apply(lambda x: int(x[1:-1])).values
unique_pos = np.unique(train_pos)
np.random.seed(42)
np.random.shuffle(unique_pos)
fold_size  = len(unique_pos) // 5
cv_scores  = []

for fold in range(5):
    val_pos  = set(unique_pos[fold*fold_size:(fold+1)*fold_size])
    val_mask = np.array([p in val_pos for p in train_pos])
    tr_mask  = ~val_mask

    gp_model, gp_lik = train_gp(X_train[tr_mask], y_train[tr_mask])
    mu, _            = gp_predict(gp_model, gp_lik, X_train[val_mask])
    r, _             = spearmanr(y_train[val_mask], mu)
    cv_scores.append(r)
    print(f"  Fold {fold+1}: {r:.4f}")

print(f"\nGP CV Spearman: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# ── Train final GP on all data ─────────────────────────────────────────────────
print("\nTraining final GP on all data ...")
final_gp, final_lik   = train_gp(X_train, y_train)
mu_test, std_test     = gp_predict(final_gp, final_lik, X_test)
y_pred                = np.clip(mu_test, 0, 1)

# ── Save submission ────────────────────────────────────────────────────────────
submission = pd.DataFrame({'id': range(len(df_test)), 'DMS_score': y_pred})
submission.to_csv(f'{DATA_DIR}/predictions.csv', index=False)
submission.to_csv(f'{DATA_DIR}/submission_round{ROUND}.csv', index=False)
print(f"\nSaved predictions.csv - upload to Kaggle")

# ── Top 10 predicted mutations ─────────────────────────────────────────────────
df_test_out               = df_test.copy()
df_test_out['fitness']    = y_pred
df_test_out['uncertainty'] = std_test
top10 = df_test_out.nlargest(10, 'fitness')[['mutant', 'fitness', 'uncertainty']]
print("\nTop 10 predicted high-fitness mutations:")
print(top10.to_string(index=False))