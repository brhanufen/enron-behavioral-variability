#!/usr/bin/env python3
"""parse_headers.py - Extract email headers and deduplicate."""
import os, re, email, json, warnings
import pandas as pd
from email.utils import parsedate_to_datetime
from datetime import timezone
warnings.filterwarnings('ignore')

RAW_DIR = "/home/claude/enron-behavioral-variability/data/raw/maildir"
OUT_DIR = "/home/claude/enron-behavioral-variability/data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

def extract_emails(raw):
    return [e.strip().lower() for e in re.findall(r'[\w.+-]+@[\w.-]+', raw)]

records = []
errors = 0
total = 0

print("Parsing all email headers...")
for user in sorted(os.listdir(RAW_DIR)):
    user_path = os.path.join(RAW_DIR, user)
    if not os.path.isdir(user_path):
        continue
    for root, dirs, files in os.walk(user_path):
        for fname in files:
            total += 1
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r', errors='ignore') as f:
                    msg = email.message_from_file(f)
                msg_id = msg.get('Message-ID', '').strip()
                sender = msg.get('From', '').strip().lower()
                date_str = msg.get('Date', '')
                if not msg_id or not sender or not date_str:
                    errors += 1; continue
                ts = None
                try:
                    ts = parsedate_to_datetime(date_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    ts = ts.astimezone(timezone.utc)
                except:
                    errors += 1; continue
                sender_match = extract_emails(sender)
                sender_email = sender_match[0] if sender_match else sender
                to_list = extract_emails(msg.get('To','') or '')
                cc_list = extract_emails(msg.get('Cc','') or '')
                folder_rel = root.replace(user_path, '').lower()
                is_sent = any(s in folder_rel for s in ['sent', '_sent_mail'])
                records.append({
                    'mailbox_user': user,
                    'msg_id': msg_id,
                    'sender': sender_email,
                    'recipients_to': ';'.join(to_list),
                    'recipients_cc': ';'.join(cc_list),
                    'timestamp_utc': ts.isoformat(),
                    'subject': (msg.get('Subject','') or '').strip(),
                    'in_reply_to': (msg.get('In-Reply-To','') or '').strip(),
                    'is_sent_folder': is_sent,
                    'x_folder': (msg.get('X-Folder','') or '').strip(),
                })
            except:
                errors += 1
            if total % 100000 == 0:
                print(f"  ...processed {total} files, {len(records)} valid")

print(f"\nDone: {total} files, {len(records)} records, {errors} errors")
df = pd.DataFrame(records)
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
before = len(df)
df = df[(df['timestamp_utc'].dt.year >= 1998) & (df['timestamp_utc'].dt.year <= 2003)]
date_filt = before - len(df)
before_dd = len(df)
df = df.drop_duplicates(subset=['msg_id','sender','timestamp_utc'], keep='first')
dupes = before_dd - len(df)
print(f"Date filter removed: {date_filt} | Dedup removed: {dupes} | Final: {len(df)}")
df.to_parquet(os.path.join(OUT_DIR, 'events_raw.parquet'), index=False)
report = {'total_scanned': total, 'valid_extracted': before,
          'date_filtered': int(date_filt), 'dupes_removed': int(dupes),
          'final_records': len(df), 'errors': errors,
          'unique_senders': int(df['sender'].nunique()),
          'date_range': f"{df['timestamp_utc'].min().date()} to {df['timestamp_utc'].max().date()}",
          'sent_folder_pct': round(df['is_sent_folder'].mean()*100, 1)}
with open(os.path.join(OUT_DIR, 'dedup_report.json'), 'w') as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
