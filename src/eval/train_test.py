#!/usr/bin/env python3
"""train_test.py - Time-based 70/30 split per user."""
import pandas as pd
import numpy as np
import pickle, os

PROC_DIR = "/home/claude/enron-behavioral-variability/data/processed"
RESULTS_DIR = "/home/claude/enron-behavioral-variability/data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

sent = pd.read_parquet(os.path.join(PROC_DIR, 'cohort_sent.parquet'))
cohort = pd.read_csv(os.path.join(PROC_DIR, 'cohort_users.csv'))

splits = {}
for _, row in cohort.iterrows():
    user = row['sender']
    ud = sent[sent['sender'] == user].sort_values('timestamp_utc')
    n = len(ud)
    si = int(n * 0.70)
    splits[user] = {
        'n_train': si, 'n_test': n - si,
        'train_end': str(ud.iloc[si]['timestamp_utc']),
        'train_idx': ud.index[:si].tolist(),
        'test_idx': ud.index[si:].tolist(),
    }

# Verify no leakage
for u, s in splits.items():
    ud = sent[sent['sender'] == u]
    assert ud.loc[s['train_idx']]['timestamp_utc'].max() <= ud.loc[s['test_idx']]['timestamp_utc'].min()

with open(os.path.join(RESULTS_DIR, 'split_indices.pkl'), 'wb') as f:
    pickle.dump(splits, f)

print(f"Split {len(splits)} users | Train: {sum(s['n_train'] for s in splits.values())} | Test: {sum(s['n_test'] for s in splits.values())} | No leakage.")
