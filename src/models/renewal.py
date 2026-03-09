#!/usr/bin/env python3
"""renewal.py - M2: Lognormal renewal process on inter-event times."""
import pandas as pd
import numpy as np
from scipy import stats
import pickle, os

PROC_DIR = "/home/claude/enron-behavioral-variability/data/processed"
RESULTS_DIR = "/home/claude/enron-behavioral-variability/data/results"

sent = pd.read_parquet(os.path.join(PROC_DIR, 'cohort_sent.parquet'))
with open(os.path.join(RESULTS_DIR, 'split_indices.pkl'), 'rb') as f:
    splits = pickle.load(f)

results = []
for user, sp in splits.items():
    ud = sent[sent['sender'] == user].sort_values('timestamp_utc')
    train = ud.loc[sp['train_idx']]
    test = ud.loc[sp['test_idx']]
    
    # Compute inter-event times (seconds)
    train_iets = np.diff(train['timestamp_utc'].values).astype('timedelta64[s]').astype(float)
    test_iets = np.diff(test['timestamp_utc'].values).astype('timedelta64[s]').astype(float)
    
    train_iets = train_iets[train_iets > 0]
    test_iets = test_iets[test_iets > 0]
    
    if len(train_iets) < 10 or len(test_iets) < 5:
        continue
    
    # Fit lognormal to training IETs
    log_iets = np.log(train_iets)
    mu_log = np.mean(log_iets)
    sigma_log = np.std(log_iets, ddof=1)
    
    # Test log-likelihood
    ll = np.sum(stats.lognorm.logpdf(test_iets, s=sigma_log, scale=np.exp(mu_log)))
    ll_per_event = ll / len(test_iets)
    
    aic = -2 * ll + 2 * 2  # 2 params: mu, sigma
    
    results.append({
        'user': user, 'model': 'M2_Renewal',
        'n_train': sp['n_train'], 'n_test': sp['n_test'],
        'mu_log': round(mu_log, 4), 'sigma_log': round(sigma_log, 4),
        'loglik': round(float(ll), 2),
        'loglik_per_event': round(float(ll_per_event), 4),
        'aic': round(float(aic), 2),
        'n_params': 2,
    })

rdf = pd.DataFrame(results)
rdf.to_csv(os.path.join(RESULTS_DIR, 'results_m2.csv'), index=False)
print(f"M2 fitted for {len(rdf)} users")
print(f"Mean log-lik/event: {rdf['loglik_per_event'].mean():.4f}")
