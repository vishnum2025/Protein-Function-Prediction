
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

# ── Config — only section you need to edit ─────────────────────────────────────
DATA_DIR   = os.path.expanduser('~/v_files/GT/sem2/MLB/hackathon/Hackathon_data')
FASTA_PATH = f'{DATA_DIR}/sequence.fasta'
TRAIN_PATH = f'{DATA_DIR}/train.csv'
TEST_PATH  = f'{DATA_DIR}/test.csv'
CACHE_PATH = f'{DATA_DIR}/esm2_150M_zeroshot_scores.npz'

QUERY_PATHS = [f'{DATA_DIR}/query1_labeled.csv']   # ← update each round
N_ENSEMBLE  = 5    
BOOTSTRAP   = 0.9
N_QUERY     = 100
ROUND       = 2    # ← update each round


device = torch.device('cpu')
print("Device: CPU")

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

# ── ESM-2 zero-shot: load cache ────────────────────────────────────────────────
# Run compute_esm2_cache.py first if cache doesn't exist.
all_df        = pd.concat([df_train, df_test], ignore_index=True)
all_positions = all_df['mutant'].apply(lambda x: int(x[1:-1])).unique()
AAs           = list('ACDEFGHIKLMNPQRSTVWY')

if os.path.exists(CACHE_PATH):
    print(f"Loading ESM-2 cache ...")
    pos_logprobs = np.load(CACHE_PATH, allow_pickle=True)['pos_logprobs'].item()
    print(f"  {len(pos_logprobs)} positions loaded")
else:
    print("Cache not found — run compute_esm2_cache.py first.")
    print("Computing with 150M on CPU as fallback ...")
    import esm
    model_esm, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model_esm.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    aa_tokens       = [alphabet.get_idx(aa) for aa in AAs]

    pos_logprobs = {}
    positions    = list(all_positions)
    for i in range(0, len(positions), 8):
        batch_pos  = positions[i:i+8]
        batch_data = [(f"p{p}", sequence_wt[:p]+'<mask>'+sequence_wt[p+1:]) for p in batch_pos]
        _, _, tokens = batch_converter(batch_data)
        with torch.no_grad():
            logits = model_esm(tokens.to(device), repr_layers=[], return_contacts=False)['logits']
        lp = torch.log_softmax(logits, dim=-1).cpu().float().numpy()
        for j, pos in enumerate(batch_pos):
            pos_logprobs[pos] = {aa: float(lp[j, pos+1, tok]) for aa, tok in zip(AAs, aa_tokens)}
        if i % 80 == 0:
            print(f"  {i+len(batch_pos)}/{len(positions)} positions")

    del model_esm
    np.savez(CACHE_PATH, pos_logprobs=pos_logprobs)
    print(f"Saved cache → {CACHE_PATH}")


def zero_shot_scores(df):
    """log P(mutant | context) - log P(wt | context)"""
    scores = []
    for _, row in df.iterrows():
        m   = row['mutant']
        pos = int(m[1:-1])
        scores.append(pos_logprobs[pos][m[-1]] - pos_logprobs[pos][m[0]])
    return np.array(scores, dtype=np.float32)

zs_train = zero_shot_scores(df_train)
zs_test  = zero_shot_scores(df_test)
y_train  = df_train['DMS_score'].values.astype(np.float32)

r_zs, _ = spearmanr(y_train, zs_train)
print(f"Zero-shot Spearman on train: {r_zs:.4f}")


# ── Ridge calibration ──────────────────────────────────────────────────────────
cal   = Ridge(alpha=1.0)
cal.fit(zs_train.reshape(-1, 1), y_train)
y_pred = np.clip(cal.predict(zs_test.reshape(-1, 1)), 0, 1)

# Position-split CV — honest leaderboard estimate
train_pos  = df_train['mutant'].apply(lambda x: int(x[1:-1])).values
unique_pos = np.unique(train_pos)
np.random.seed(42)
np.random.shuffle(unique_pos)
fold_size  = len(unique_pos) // 5
cv_scores  = []
for fold in range(5):
    val_pos  = set(unique_pos[fold*fold_size:(fold+1)*fold_size])
    val_mask = np.array([p in val_pos for p in train_pos])
    c = Ridge(alpha=1.0)
    c.fit(zs_train[~val_mask].reshape(-1, 1), y_train[~val_mask])
    r, _ = spearmanr(y_train[val_mask], c.predict(zs_train[val_mask].reshape(-1, 1)))
    cv_scores.append(r)
print(f"Position-split CV Spearman: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# ── Save submission ────────────────────────────────────────────────────────────
submission = pd.DataFrame({'id': range(len(df_test)), 'DMS_score': y_pred})
submission.to_csv(f'{DATA_DIR}/predictions.csv', index=False)
submission.to_csv(f'{DATA_DIR}/submission_round{ROUND}.csv', index=False)
print(f"Saved predictions.csv -> upload to Kaggle")
