#!/usr/bin/env python3
"""burstiness_stats.py - Compute B and M for all cohort users."""
import pandas as pd
import numpy as np
import json, os

PROC_DIR = "/home/claude/enron-behavioral-variability/data/processed"

sent = pd.read_parquet(os.path.join(PROC_DIR, 'cohort_sent.parquet'))
cohort = pd.read_csv(os.path.join(PROC_DIR, 'cohort_users.csv'))

results = []
for _, row in cohort.iterrows():
    user = row['sender']
    ud = sent[sent['sender'] == user].sort_values('timestamp_utc')
    ts = ud['timestamp_utc'].values
    
    if len(ts) < 10:
        continue
    
    # Inter-event times in seconds
    iets = np.diff(ts).astype('timedelta64[s]').astype(float)
    iets = iets[iets > 0]  # remove zero gaps
    
    if len(iets) < 5:
        continue
    
    mu = np.mean(iets)
    sigma = np.std(iets)
    
    # Burstiness B = (sigma - mu) / (sigma + mu)
    B = (sigma - mu) / (sigma + mu) if (sigma + mu) > 0 else 0
    
    # Memory M = autocorrelation of consecutive IETs
    if len(iets) > 2:
        M = np.corrcoef(iets[:-1], iets[1:])[0, 1]
        if np.isnan(M):
            M = 0.0
    else:
        M = 0.0
    
    results.append({
        'sender': user,
        'n_emails': int(row['n_sent']),
        'n_iets': len(iets),
        'mean_iet_hours': round(mu / 3600, 2),
        'std_iet_hours': round(sigma / 3600, 2),
        'burstiness_B': round(B, 4),
        'memory_M': round(M, 4),
    })

bdf = pd.DataFrame(results)
bdf.to_csv(os.path.join(PROC_DIR, 'burstiness_stats.csv'), index=False)

print(f"Computed B and M for {len(bdf)} users")
print(f"\nBurstiness B: mean={bdf['burstiness_B'].mean():.3f}, "
      f"median={bdf['burstiness_B'].median():.3f}, "
      f"range=[{bdf['burstiness_B'].min():.3f}, {bdf['burstiness_B'].max():.3f}]")
print(f"Memory M:     mean={bdf['memory_M'].mean():.3f}, "
      f"median={bdf['memory_M'].median():.3f}, "
      f"range=[{bdf['memory_M'].min():.3f}, {bdf['memory_M'].max():.3f}]")
print(f"\nUsers with B > 0 (bursty): {(bdf['burstiness_B'] > 0).sum()}/{len(bdf)}")
print(f"Users with M > 0 (memory): {(bdf['memory_M'] > 0).sum()}/{len(bdf)}")
