#!/usr/bin/env python3
"""cohort.py - Filter to core cohort based on activity thresholds."""
import pandas as pd
import numpy as np
import json, os

PROC_DIR = "/home/claude/enron-behavioral-variability/data/processed"
df = pd.read_parquet(os.path.join(PROC_DIR, 'events_tz.parquet'))

# Focus on SENT folder emails only (outgoing activity)
sent = df[df['is_sent_folder']].copy()
print(f"Total sent-folder emails: {len(sent)}")

# Per-user stats
user_stats = []
for sender, grp in sent.groupby('sender'):
    ts = grp['timestamp_utc'].sort_values()
    n = len(ts)
    first = ts.iloc[0]
    last = ts.iloc[-1]
    span_days = (last - first).total_seconds() / 86400
    density = n / max(span_days, 1)
    
    user_stats.append({
        'sender': sender,
        'n_sent': n,
        'first_email': first,
        'last_email': last,
        'active_span_days': round(span_days, 1),
        'emails_per_day': round(density, 2),
    })

stats = pd.DataFrame(user_stats).sort_values('n_sent', ascending=False)
print(f"Total unique senders in sent folders: {len(stats)}")

# Apply cohort filters
MIN_EMAILS = 500
MIN_DENSITY = 1.0

cohort = stats[(stats['n_sent'] >= MIN_EMAILS) & (stats['emails_per_day'] >= MIN_DENSITY)].copy()
print(f"\n=== COHORT SELECTION ===")
print(f"Filter: n_sent >= {MIN_EMAILS} AND density >= {MIN_DENSITY}/day")
print(f"Users passing: {len(cohort)}")

# Also check with lower threshold (contingency)
cohort_300 = stats[(stats['n_sent'] >= 300) & (stats['emails_per_day'] >= MIN_DENSITY)]
print(f"Users at >=300 threshold: {len(cohort_300)}")

print(f"\n=== COHORT MEMBERS ===")
print(cohort[['sender', 'n_sent', 'active_span_days', 'emails_per_day']].to_string(index=False))

# Save cohort
cohort.to_csv(os.path.join(PROC_DIR, 'cohort_users.csv'), index=False)

# Build per-user event tables for cohort (sent emails only)
cohort_senders = set(cohort['sender'])
cohort_sent = sent[sent['sender'].isin(cohort_senders)].copy()

# Also build incoming events for cohort users (emails received by them)
# A user "receives" an email if their address appears in recipients_to or recipients_cc
print(f"\nBuilding incoming event streams for cohort users...")
incoming_records = []
for _, row in df.iterrows():
    all_recip = (row.get('recipients_to','') or '') + ';' + (row.get('recipients_cc','') or '')
    for addr in all_recip.split(';'):
        addr = addr.strip().lower()
        if addr in cohort_senders:
            incoming_records.append({
                'receiver': addr,
                'sender': row['sender'],
                'timestamp_utc': row['timestamp_utc'],
                'msg_id': row['msg_id'],
                'subject': row['subject'],
            })

incoming_df = pd.DataFrame(incoming_records)
print(f"Incoming events for cohort users: {len(incoming_df)}")

# Save
cohort_sent.to_parquet(os.path.join(PROC_DIR, 'cohort_sent.parquet'), index=False)
incoming_df.to_parquet(os.path.join(PROC_DIR, 'cohort_incoming.parquet'), index=False)

# Summary report
report = {
    'filter_min_emails': MIN_EMAILS,
    'filter_min_density': MIN_DENSITY,
    'total_senders_in_sent': len(stats),
    'cohort_size': len(cohort),
    'cohort_size_at_300': len(cohort_300),
    'cohort_total_sent_emails': len(cohort_sent),
    'cohort_total_incoming_emails': len(incoming_df),
    'cohort_date_range': f"{cohort_sent['timestamp_utc'].min().date()} to {cohort_sent['timestamp_utc'].max().date()}",
    'go_no_go': 'PASS' if len(cohort) >= 20 else 'FAIL - activate contingency',
    'members': cohort[['sender','n_sent','emails_per_day']].to_dict('records'),
}
with open(os.path.join(PROC_DIR, 'cohort_summary.json'), 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"\nSaved: cohort_users.csv, cohort_sent.parquet, cohort_incoming.parquet, cohort_summary.json")
print(f"\n=== GO/NO-GO: {'PASS (>= 20 users)' if len(cohort) >= 20 else 'FAIL'} ===")
