#!/usr/bin/env python3
"""reply_pairs.py - Heuristic reply pairing via Subject normalization + time windows."""
import pandas as pd
import numpy as np
import re, json, os

PROC_DIR = "/home/claude/enron-behavioral-variability/data/processed"

# Load cohort data
sent = pd.read_parquet(os.path.join(PROC_DIR, 'cohort_sent.parquet'))
incoming = pd.read_parquet(os.path.join(PROC_DIR, 'cohort_incoming.parquet'))
cohort = pd.read_csv(os.path.join(PROC_DIR, 'cohort_users.csv'))
cohort_senders = set(cohort['sender'])

print(f"Cohort sent emails: {len(sent)}")
print(f"Cohort incoming emails: {len(incoming)}")

def normalize_subject(subj):
    """Strip Re:/Fwd: prefixes and normalize whitespace."""
    if not isinstance(subj, str):
        return ''
    s = re.sub(r'^(re|fw|fwd)\s*:\s*', '', subj.strip(), flags=re.IGNORECASE)
    s = re.sub(r'^(re|fw|fwd)\s*:\s*', '', s.strip(), flags=re.IGNORECASE)  # double strip
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s

# Normalize subjects
sent['subj_norm'] = sent['subject'].apply(normalize_subject)
incoming['subj_norm'] = incoming['subject'].apply(normalize_subject)

# Filter out empty/generic subjects
generic = {'', 're', 'fwd', 'fw', 'hello', 'hi', 'thanks', 'thank you', 'meeting', 'update'}
sent_valid = sent[~sent['subj_norm'].isin(generic) & (sent['subj_norm'].str.len() > 3)].copy()
incoming_valid = incoming[~incoming['subj_norm'].isin(generic) & (incoming['subj_norm'].str.len() > 3)].copy()

print(f"\nAfter subject filtering: {len(sent_valid)} sent, {len(incoming_valid)} incoming")

# For each time window, find reply pairs
windows_days = [1, 3, 7]
all_results = {}

for window in windows_days:
    print(f"\n=== Window: {window} day(s) ===")
    pairs = []
    window_td = pd.Timedelta(days=window)
    
    # Process per cohort user
    for user in sorted(cohort_senders):
        user_sent = sent_valid[sent_valid['sender'] == user].sort_values('timestamp_utc')
        user_incoming = incoming_valid[incoming_valid['receiver'] == user].sort_values('timestamp_utc')
        
        if len(user_sent) == 0 or len(user_incoming) == 0:
            continue
        
        # Group incoming by normalized subject for fast lookup
        inc_by_subj = user_incoming.groupby('subj_norm')
        
        for _, s_row in user_sent.iterrows():
            subj = s_row['subj_norm']
            if subj not in inc_by_subj.groups:
                continue
            
            candidates = inc_by_subj.get_group(subj)
            # Find incoming emails BEFORE this sent email within window
            t_sent = s_row['timestamp_utc']
            mask = (candidates['timestamp_utc'] < t_sent) & \
                   (candidates['timestamp_utc'] >= t_sent - window_td)
            matches = candidates[mask]
            
            if len(matches) > 0:
                # Take the most recent incoming as the "trigger"
                best = matches.iloc[-1]
                tau = (t_sent - best['timestamp_utc']).total_seconds()
                pairs.append({
                    'user': user,
                    'sent_time': t_sent,
                    'incoming_time': best['timestamp_utc'],
                    'incoming_sender': best['sender'],
                    'subject_norm': subj,
                    'tau_seconds': tau,
                    'tau_hours': tau / 3600,
                    'msg_id_sent': s_row['msg_id'],
                })
    
    pairs_df = pd.DataFrame(pairs)
    print(f"  Reply pairs found: {len(pairs_df)}")
    if len(pairs_df) > 0:
        print(f"  Users with pairs: {pairs_df['user'].nunique()}")
        print(f"  Median tau: {pairs_df['tau_hours'].median():.2f} hours")
        print(f"  Mean tau: {pairs_df['tau_hours'].mean():.2f} hours")
        print(f"  90th pctl tau: {pairs_df['tau_hours'].quantile(0.9):.2f} hours")
    
    fname = f'reply_pairs_heuristic_{window}d.parquet'
    pairs_df.to_parquet(os.path.join(PROC_DIR, fname), index=False)
    all_results[f'{window}d'] = {
        'n_pairs': len(pairs_df),
        'n_users': int(pairs_df['user'].nunique()) if len(pairs_df) > 0 else 0,
        'median_tau_hours': round(float(pairs_df['tau_hours'].median()), 2) if len(pairs_df) > 0 else None,
        'mean_tau_hours': round(float(pairs_df['tau_hours'].mean()), 2) if len(pairs_df) > 0 else None,
        'p90_tau_hours': round(float(pairs_df['tau_hours'].quantile(0.9)), 2) if len(pairs_df) > 0 else None,
    }

# Save summary
report = {
    'method': 'heuristic_subject_time_window',
    'subject_filtering': 'Removed empty, generic, and short (<=3 char) subjects',
    'windows': all_results,
}
with open(os.path.join(PROC_DIR, 'reply_pairs_report.json'), 'w') as f:
    json.dump(report, f, indent=2)

print(f"\nSaved: reply_pairs_heuristic_{{1,3,7}}d.parquet + reply_pairs_report.json")
print(json.dumps(report, indent=2))
