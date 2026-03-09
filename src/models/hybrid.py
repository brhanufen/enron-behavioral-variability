#!/usr/bin/env python3
"""hybrid.py - M5: HMM on Hawkes residuals (tests H3)."""
import pandas as pd
import numpy as np
from numba import njit
from hmmlearn.hmm import GaussianHMM
import pickle, os, json, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

PROC_DIR = "/home/claude/enron-behavioral-variability/data/processed"
RESULTS_DIR = "/home/claude/enron-behavioral-variability/data/results"

sent = pd.read_parquet(os.path.join(PROC_DIR, 'cohort_sent.parquet'))
with open(os.path.join(RESULTS_DIR, 'split_indices.pkl'), 'rb') as f:
    splits = pickle.load(f)
m3 = pd.read_csv(os.path.join(RESULTS_DIR, 'results_m3.csv'))
m3_map = {r['user']: (r['mu'], r['alpha'], r['beta']) for _, r in m3.iterrows()}

@njit
def hawkes_residuals(events, mu, alpha, beta):
    """Time-rescaled residuals under fitted Hawkes intensity."""
    n = len(events)
    residuals = np.empty(n - 1)
    for i in range(1, n):
        # Integral of lambda from events[i-1] to events[i]
        dt = events[i] - events[i - 1]
        integral = mu * dt
        for j in range(i):
            t1 = events[i - 1] - events[j]
            t2 = events[i] - events[j]
            if t1 >= 0 and t2 >= 0:
                integral += (alpha / beta) * (np.exp(-beta * t1) - np.exp(-beta * t2))
        residuals[i - 1] = integral
    return residuals

# Warmup
_ = hawkes_residuals(np.array([0.0, 1.0, 2.0]), 0.5, 0.3, 1.0)

results = []
h3_tests = []

for i, (user, sp) in enumerate(splits.items()):
    if user not in m3_map:
        continue
    mu, alpha, beta = m3_map[user]
    
    ud = sent[sent['sender'] == user].sort_values('timestamp_utc')
    test = ud.loc[sp['test_idx']]
    
    t0 = test['timestamp_utc'].min()
    events = (test['timestamp_utc'] - t0).dt.total_seconds().values.astype(np.float64)
    
    if len(events) < 30:
        continue
    
    # Compute residuals (cap at 500 events for speed)
    ev = events[:500] if len(events) > 500 else events
    resid = hawkes_residuals(ev, mu, alpha, beta)
    resid = resid[np.isfinite(resid) & (resid > 0)]
    
    if len(resid) < 20:
        continue
    
    # Log-transform for Gaussian HMM
    log_resid = np.log(resid).reshape(-1, 1)
    
    # Fit K=1 vs K=2 HMM on residuals
    try:
        m_k1 = GaussianHMM(n_components=1, n_iter=100, random_state=42)
        m_k1.fit(log_resid)
        bic_k1 = -2 * m_k1.score(log_resid) + 3 * np.log(len(log_resid))
        
        m_k2 = GaussianHMM(n_components=2, n_iter=100, random_state=42)
        m_k2.fit(log_resid)
        bic_k2 = -2 * m_k2.score(log_resid) + 8 * np.log(len(log_resid))
        
        k2_preferred = bic_k2 < bic_k1
        
        results.append({
            'user': user, 'model': 'M5_Hybrid',
            'n_residuals': len(resid),
            'bic_k1': round(float(bic_k1), 2),
            'bic_k2': round(float(bic_k2), 2),
            'delta_bic': round(float(bic_k1 - bic_k2), 2),
            'k2_preferred': bool(k2_preferred),
        })
        h3_tests.append(k2_preferred)
    except:
        continue
    
    if (i+1) % 15 == 0:
        print(f"  {i+1}/{len(splits)} done")

rdf = pd.DataFrame(results)
rdf.to_csv(os.path.join(RESULTS_DIR, 'results_m5.csv'), index=False)

h3_pct = np.mean(h3_tests) * 100 if h3_tests else 0
print(f"\nM5: {len(rdf)} users analyzed")
print(f"K=2 preferred (residuals show state-switching): {sum(h3_tests)}/{len(h3_tests)} ({h3_pct:.0f}%)")
print(f"H3: {'PASS' if h3_pct > 50 else 'FAIL'}")

with open(os.path.join(RESULTS_DIR, 'h3_test.json'), 'w') as f:
    json.dump({'n_users': len(rdf), 'k2_preferred_pct': round(float(h3_pct), 1),
               'H3_pass': bool(h3_pct > 50)}, f, indent=2)
