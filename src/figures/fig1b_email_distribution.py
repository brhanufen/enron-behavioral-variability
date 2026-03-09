#!/usr/bin/env python3
"""fig1b_email_distribution.py - Distribution of outgoing emails per user (log scale)."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os

# Publication style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

PROC_DIR = "/home/claude/enron-behavioral-variability/data/processed"
FIG_DIR = "/home/claude/enron-behavioral-variability/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Load sent emails
df = pd.read_parquet(os.path.join(PROC_DIR, 'events_tz.parquet'))
sent = df[df['is_sent_folder']]

# Per-user counts
user_counts = sent.groupby('sender').size().sort_values(ascending=False)
# Filter to users with at least 10 emails (exclude noise)
user_counts = user_counts[user_counts >= 10]

cohort = pd.read_csv(os.path.join(PROC_DIR, 'cohort_users.csv'))
cohort_senders = set(cohort['sender'])

# Separate cohort vs non-cohort
in_cohort = user_counts[user_counts.index.isin(cohort_senders)]
not_cohort = user_counts[~user_counts.index.isin(cohort_senders)]

# Create figure
fig, ax = plt.subplots(figsize=(5.5, 4.0))

# Histogram on log scale
bins = np.logspace(np.log10(10), np.log10(user_counts.max() * 1.2), 30)
ax.hist(not_cohort.values, bins=bins, color='#b0bec5', edgecolor='#78909c',
        linewidth=0.5, alpha=0.85, label=f'Below threshold (n={len(not_cohort)})')
ax.hist(in_cohort.values, bins=bins, color='#2c5f8a', edgecolor='#1a3a5c',
        linewidth=0.5, alpha=0.9, label=f'Cohort (n={len(in_cohort)})')

# Threshold line
ax.axvline(x=500, color='#c62828', linewidth=1.5, linestyle='--', zorder=5)
ax.text(520, ax.get_ylim()[1] * 0.85, 'Threshold\n(500 emails)',
        fontsize=8.5, color='#c62828', fontweight='bold', va='top')

ax.set_xscale('log')
ax.set_xlabel('Number of Outgoing Emails')
ax.set_ylabel('Number of Users')
ax.set_title('')

# Panel label
ax.text(-0.12, 1.05, 'B', transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', ha='left', color='#1a2744')

ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
ax.set_xlim(8, user_counts.max() * 1.5)

# Clean grid
ax.grid(axis='y', alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)

# Remove top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig1b_email_distribution.pdf'), format='pdf')
fig.savefig(os.path.join(FIG_DIR, 'fig1b_email_distribution.png'), format='png', dpi=300)
plt.close()
print("Saved: fig1b_email_distribution.pdf + .png")
