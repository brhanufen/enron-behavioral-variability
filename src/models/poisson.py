#!/usr/bin/env python3
"""poisson.py - M1: Inhomogeneous Poisson with circadian/weekly covariates."""
import pandas as pd
import numpy as np
import pickle, json, os
from scipy.special import gammaln

PROC_DIR = "/home/claude/enron-behavioral-variability/data/processed"
RESULTS_DIR = "/home/claude/enron-behavioral-variability/data/results"

sent = pd.read_parquet(os.path.join(PROC_DIR, 'cohort_sent.parquet'))
with open(os.path.join(RESULTS_DIR, 'split_indices.pkl'), 'rb') as f:
    splits = pickle.load(f)

BIN_MINUTES = 60  # 1-hour bins

results = []
for user, sp in splits.items():
    ud = sent[sent['sender'] == user].sort_values('timestamp_utc')
    train = ud.loc[sp['train_idx']]
    test = ud.loc[sp['test_idx']]
    
    # Bin into hourly counts for train period
    train_ts = train['timestamp_utc']
    t_start = train_ts.min().floor('h')
    t_end_train = train_ts.max().ceil('h')
    t_end_test = test['timestamp_utc'].max().ceil('h')
    
    # Create full hourly grid for train
    train_bins = pd.date_range(t_start, t_end_train, freq='h', tz='UTC')
    train_counts = pd.Series(0, index=train_bins[:-1])
    for t in train_ts:
        b = t.floor('h')
        if b in train_counts.index:
            train_counts[b] += 1
    
    # Learn rate per (hour_of_day, day_of_week) = 24*7 = 168 bins
    train_df = pd.DataFrame({'count': train_counts})
    train_df['hour'] = train_df.index.hour
    train_df['dow'] = train_df.index.dayofweek
    
    rate_table = train_df.groupby(['hour', 'dow'])['count'].mean()
    # Smoothing: add small constant to avoid zero rates
    rate_table = rate_table + 0.001
    
    # Evaluate on test set
    test_ts = test['timestamp_utc']
    test_start = test_ts.min().floor('h')
    test_bins = pd.date_range(test_start, t_end_test, freq='h', tz='UTC')
    test_counts = pd.Series(0, index=test_bins[:-1])
    for t in test_ts:
        b = t.floor('h')
        if b in test_counts.index:
            test_counts[b] += 1
    
    test_df = pd.DataFrame({'count': test_counts})
    test_df['hour'] = test_df.index.hour
    test_df['dow'] = test_df.index.dayofweek
    test_df['lambda'] = test_df.apply(
        lambda r: rate_table.get((r['hour'], r['dow']), 0.001), axis=1)
    
    # Log-likelihood: sum of log P(c_t | lambda_t) for Poisson
    c = test_df['count'].values
    lam = test_df['lambda'].values
    ll = np.sum(c * np.log(lam) - lam - gammaln(c + 1))
    ll_per_event = ll / max(sp['n_test'], 1)
    
    # Brier score: P(at least 1 event in 30min) - approximate as 1 - exp(-lambda/2)
    p_event = 1 - np.exp(-lam / 2)
    y_event = (c > 0).astype(float)
    brier = np.mean((p_event - y_event) ** 2)
    
    n_params = len(rate_table)
    aic = -2 * ll + 2 * n_params
    
    results.append({
        'user': user, 'model': 'M1_Poisson',
        'n_train': sp['n_train'], 'n_test': sp['n_test'],
        'loglik': round(float(ll), 2),
        'loglik_per_event': round(float(ll_per_event), 4),
        'brier': round(float(brier), 4),
        'aic': round(float(aic), 2),
        'n_params': int(n_params),
    })

rdf = pd.DataFrame(results)
rdf.to_csv(os.path.join(RESULTS_DIR, 'results_m1.csv'), index=False)
print(f"M1 fitted for {len(rdf)} users")
print(f"Mean log-lik/event: {rdf['loglik_per_event'].mean():.4f}")
print(f"Mean Brier: {rdf['brier'].mean():.4f}")
