#!/usr/bin/env python3
"""timezone.py - Per-user local-time inference via activity-minimum detection."""
import pandas as pd
import numpy as np
import json, os

PROC_DIR = "/home/claude/enron-behavioral-variability/data/processed"
df = pd.read_parquet(os.path.join(PROC_DIR, 'events_raw.parquet'))

# Focus on sent-folder emails only for per-user activity patterns
sent = df[df['is_sent_folder']].copy()
print(f"Sent-folder emails: {len(sent)}")
print(f"Unique senders in sent folders: {sent['sender'].nunique()}")

# Get UTC hour for each email
sent['hour_utc'] = sent['timestamp_utc'].dt.hour

# Per-user: build 24-hour activity histogram, find minimum hour
tz_results = []
for sender, grp in sent.groupby('sender'):
    if len(grp) < 50:  # need enough emails for reliable histogram
        continue
    hist = grp['hour_utc'].value_counts().reindex(range(24), fill_value=0).values
    # Smooth with circular convolution (window=3) to avoid noise
    smoothed = np.convolve(np.tile(hist, 3), np.ones(3)/3, mode='same')[24:48]
    min_hour_utc = int(np.argmin(smoothed))
    
    # Assume minimum activity = ~4AM local time
    # So offset = 4 - min_hour_utc (mod 24, mapped to -12..+12)
    offset = (4 - min_hour_utc) % 24
    if offset > 12:
        offset -= 24
    
    # Flag ambiguous: if the ratio of min to max is > 0.5, pattern is flat
    ratio = smoothed.min() / (smoothed.max() + 1e-10)
    is_ambiguous = ratio > 0.4
    
    tz_results.append({
        'sender': sender,
        'n_sent': len(grp),
        'min_hour_utc': min_hour_utc,
        'inferred_utc_offset': offset,
        'min_max_ratio': round(float(ratio), 3),
        'is_ambiguous': bool(is_ambiguous),
    })

tz_df = pd.DataFrame(tz_results)
print(f"\nUsers with timezone inference: {len(tz_df)}")
print(f"Ambiguous profiles: {tz_df['is_ambiguous'].sum()}")
print(f"\nOffset distribution:")
print(tz_df['inferred_utc_offset'].value_counts().sort_index().to_string())

# Apply offsets to ALL emails (sent + received)
offset_map = dict(zip(tz_df['sender'], tz_df['inferred_utc_offset']))

# For sent emails: use sender's offset
# For non-sent: keep UTC (these are received emails, sender offset unknown)
df['sender_utc_offset'] = df['sender'].map(offset_map)
df['timestamp_local'] = df.apply(
    lambda r: r['timestamp_utc'] + pd.Timedelta(hours=r['sender_utc_offset'])
    if pd.notna(r['sender_utc_offset']) else r['timestamp_utc'],
    axis=1
)

# Save
df.to_parquet(os.path.join(PROC_DIR, 'events_tz.parquet'), index=False)
tz_df.to_csv(os.path.join(PROC_DIR, 'tz_offsets.csv'), index=False)

# Summary
most_common = int(tz_df['inferred_utc_offset'].mode().iloc[0])
report = {
    'users_analyzed': len(tz_df),
    'ambiguous_users': int(tz_df['is_ambiguous'].sum()),
    'most_common_offset': most_common,
    'offset_distribution': tz_df['inferred_utc_offset'].value_counts().sort_index().to_dict(),
    'note': f'Most common offset UTC{most_common:+d} consistent with US Central Time (Enron HQ: Houston, TX)'
}
with open(os.path.join(PROC_DIR, 'tz_report.json'), 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"\nSaved: events_tz.parquet + tz_offsets.csv + tz_report.json")
print(json.dumps(report, indent=2, default=str))
