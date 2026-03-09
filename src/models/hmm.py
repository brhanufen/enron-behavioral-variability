#!/usr/bin/env python3
"""hmm.py - M4: HMM on hourly binned counts with Poisson + NegBin emissions."""
import pandas as pd
import numpy as np
from hmmlearn.hmm import PoissonHMM, GaussianHMM
from scipy.special import gammaln
import pickle, os, json, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
PROC_DIR = "/home/claude/enron-behavioral-variability/data/processed"
RESULTS_DIR = "/home/claude/enron-behavioral-variability/data/results"

sent = pd.read_parquet(os.path.join(PROC_DIR, 'cohort_sent.parquet'))
with open(os.path.join(RESULTS_DIR, 'split_indices.pkl'), 'rb') as f:
    splits = pickle.load(f)

def bin_to_hourly(timestamps):
    """Convert timestamps to hourly count series."""
    ts = pd.Series(timestamps)
    t_min = ts.min().floor('h')
    t_max = ts.max().ceil('h')
    bins = pd.date_range(t_min, t_max, freq='h', tz='UTC')
    counts = pd.Series(0, index=bins[:-1])
    for t in ts:
        b = t.floor('h')
        if b in counts.index:
            counts[b] += 1
    return counts.values

def fit_poisson_hmm(train_counts, K):
    """Fit PoissonHMM, return model or None."""
    X = train_counts.reshape(-1, 1)
    best = None
    for _ in range(3):
        try:
            model = PoissonHMM(n_components=K, n_iter=200, tol=1e-4, random_state=np.random.randint(10000))
            model.fit(X)
            if best is None or model.score(X) > best.score(X):
                best = model
        except:
            pass
    return best

def poisson_hmm_test_ll(model, test_counts):
    """Held-out log-likelihood."""
    X = test_counts.reshape(-1, 1)
    try:
        return float(model.score(X))
    except:
        return None

# Also fit K=1 baseline for BIC comparison
results = []
h2_tests = []
m1_lls = pd.read_csv(os.path.join(RESULTS_DIR, 'results_m1.csv'))
m1_map = dict(zip(m1_lls['user'], m1_lls['loglik_per_event']))

for i, (user, sp) in enumerate(splits.items()):
    ud = sent[sent['sender'] == user].sort_values('timestamp_utc')
    train = ud.loc[sp['train_idx']]
    test = ud.loc[sp['test_idx']]
    
    train_counts = bin_to_hourly(train['timestamp_utc'])
    test_counts = bin_to_hourly(test['timestamp_utc'])
    
    if len(train_counts) < 48 or len(test_counts) < 24:
        continue
    
    # Fit K=1 (single state baseline)
    m_k1 = fit_poisson_hmm(train_counts, K=1)
    # Fit K=2
    m_k2 = fit_poisson_hmm(train_counts, K=2)
    # Fit K=3
    m_k3 = fit_poisson_hmm(train_counts, K=3)
    
    best_model = None
    best_K = None
    best_bic = np.inf
    
    for K, m in [(1, m_k1), (2, m_k2), (3, m_k3)]:
        if m is None:
            continue
        X_tr = train_counts.reshape(-1, 1)
        n_params = K * K + K - 1 + K  # transitions + priors + rates
        bic = -2 * m.score(X_tr) + n_params * np.log(len(train_counts))
        if bic < best_bic:
            best_bic = bic
            best_model = m
            best_K = K
    
    if best_model is None:
        continue
    
    test_ll = poisson_hmm_test_ll(best_model, test_counts)
    if test_ll is None:
        continue
    
    ll_per_event = test_ll / max(sp['n_test'], 1)
    
    # Extract state rates
    rates = best_model.lambdas_.flatten()
    sorted_rates = np.sort(rates)
    
    # H2: compare K=2 vs best non-switching baseline (M1)
    m1_ll = m1_map.get(user, None)
    improvement = None
    if m1_ll is not None and m1_ll != 0 and m_k2 is not None:
        k2_ll = poisson_hmm_test_ll(m_k2, test_counts)
        if k2_ll is not None:
            k2_ll_pe = k2_ll / max(sp['n_test'], 1)
            if m1_ll < 0:
                improvement = ((k2_ll_pe - m1_ll) / abs(m1_ll)) * 100
            else:
                improvement = None
    
    results.append({
        'user': user, 'model': 'M4_HMM',
        'best_K': best_K,
        'n_train': sp['n_train'], 'n_test': sp['n_test'],
        'state_rates': sorted_rates.tolist(),
        'loglik': round(float(test_ll), 2),
        'loglik_per_event': round(float(ll_per_event), 4),
        'bic': round(float(best_bic), 2),
        'n_params': int(best_K * best_K + best_K - 1 + best_K),
        'h2_improvement_pct': round(float(improvement), 2) if improvement is not None else None,
    })
    
    if (i+1) % 15 == 0:
        print(f"  {i+1}/{len(splits)} done")

rdf = pd.DataFrame(results)
rdf.to_csv(os.path.join(RESULTS_DIR, 'results_m4.csv'), index=False)

# H2 test
valid_h2 = rdf.dropna(subset=['h2_improvement_pct'])
h2_pass_pct = (valid_h2['h2_improvement_pct'] > 10).mean() * 100 if len(valid_h2) > 0 else 0

print(f"\nM4: {len(rdf)} users fitted")
print(f"Best K distribution: {rdf['best_K'].value_counts().to_dict()}")
print(f"Mean LL/event: {rdf['loglik_per_event'].mean():.4f}")
print(f"H2: >=10% improvement for {h2_pass_pct:.0f}% of users")

with open(os.path.join(RESULTS_DIR, 'h2_test.json'), 'w') as f:
    json.dump({'n_fitted': int(len(rdf)), 'best_K_dist': rdf['best_K'].value_counts().to_dict(),
               'h2_pct_improving_10pct': round(float(h2_pass_pct), 1),
               'H2_pass': bool(h2_pass_pct > 50)}, f, indent=2, default=str)
