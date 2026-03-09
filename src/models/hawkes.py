#!/usr/bin/env python3
"""hawkes.py - M3: Hawkes process with numba-accelerated log-likelihood."""
import pandas as pd
import numpy as np
from numba import njit
from scipy.optimize import minimize
import pickle, os, json, time

PROC_DIR = "/home/claude/enron-behavioral-variability/data/processed"
RESULTS_DIR = "/home/claude/enron-behavioral-variability/data/results"

@njit
def hawkes_negll(log_mu, log_alpha, log_beta, events, T):
    mu = np.exp(log_mu)
    alpha = np.exp(log_alpha)
    beta = np.exp(log_beta)
    if alpha >= beta:
        return 1e10
    n = len(events)
    if n == 0:
        return mu * T
    ll = 0.0
    A = 0.0
    for i in range(n):
        if i > 0:
            A = np.exp(-beta * (events[i] - events[i-1])) * (1.0 + A)
        lam_i = mu + alpha * A
        if lam_i <= 1e-15:
            return 1e10
        ll += np.log(lam_i)
    integral = mu * T
    for i in range(n):
        integral += (alpha / beta) * (1.0 - np.exp(-beta * (T - events[i])))
    return -(ll - integral)

def wrapper(params, events, T):
    return hawkes_negll(params[0], params[1], params[2], events, T)

# Warmup numba
_ = hawkes_negll(0.0, -1.0, 0.0, np.array([1.0, 2.0, 3.0]), 4.0)
print("Numba warmed up.")

sent = pd.read_parquet(os.path.join(PROC_DIR, 'cohort_sent.parquet'))
with open(os.path.join(RESULTS_DIR, 'split_indices.pkl'), 'rb') as f:
    splits = pickle.load(f)

results = []
t_start = time.time()

for i, (user, sp) in enumerate(splits.items()):
    ud = sent[sent['sender'] == user].sort_values('timestamp_utc')
    train = ud.loc[sp['train_idx']]
    test = ud.loc[sp['test_idx']]
    
    t0_tr = train['timestamp_utc'].min()
    tr_ev = (train['timestamp_utc'] - t0_tr).dt.total_seconds().values.astype(np.float64)
    T_tr = tr_ev[-1] + 1.0

    t0_te = test['timestamp_utc'].min()
    te_ev = (test['timestamp_utc'] - t0_te).dt.total_seconds().values.astype(np.float64)
    T_te = te_ev[-1] + 1.0
    
    mu0 = np.log(len(tr_ev) / T_tr)
    best = None
    for la, lb in [(np.log(0.3), np.log(2.0)), (np.log(0.8), np.log(5.0))]:
        try:
            res = minimize(wrapper, [mu0, la, lb], args=(tr_ev, T_tr),
                         method='Nelder-Mead', options={'maxiter': 3000})
            if best is None or res.fun < best.fun:
                best = res
        except:
            pass
    
    if best is None or best.fun > 1e9:
        continue
    
    mu, alpha, beta = np.exp(best.x)
    if alpha >= beta or mu <= 0:
        continue
    
    test_ll = -hawkes_negll(best.x[0], best.x[1], best.x[2], te_ev, T_te)
    
    results.append({
        'user': user, 'model': 'M3_Hawkes',
        'n_train': sp['n_train'], 'n_test': sp['n_test'],
        'mu': round(float(mu), 6), 'alpha': round(float(alpha), 6),
        'beta': round(float(beta), 6),
        'branching_ratio': round(float(alpha / beta), 4),
        'loglik': round(float(test_ll), 2),
        'loglik_per_event': round(float(test_ll / max(sp['n_test'], 1)), 4),
        'n_params': 3,
    })
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(splits)} done ({time.time()-t_start:.0f}s)")

rdf = pd.DataFrame(results)
rdf.to_csv(os.path.join(RESULTS_DIR, 'results_m3.csv'), index=False)

h1_pass = (rdf['alpha'] > 0.001).mean() * 100
print(f"\nM3: {len(rdf)} users fitted in {time.time()-t_start:.0f}s")
print(f"Mean LL/event: {rdf['loglik_per_event'].mean():.4f}")
print(f"Mean branching ratio: {rdf['branching_ratio'].mean():.4f}")
print(f"H1: alpha>0 for {h1_pass:.0f}% -> {'PASS' if h1_pass>=80 else 'FAIL'}")

with open(os.path.join(RESULTS_DIR, 'h1_test.json'), 'w') as f:
    json.dump({'n_fitted': len(rdf), 'pct_alpha_gt_zero': round(h1_pass,1), 'H1_pass': h1_pass>=80}, f, indent=2)
