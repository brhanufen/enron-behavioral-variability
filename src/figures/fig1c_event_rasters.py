#!/usr/bin/env python3
"""fig1c_event_rasters.py - Event raster for 3 representative users (best week, relative time)."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

PROC_DIR = "data/processed"
FIG_DIR = "figures"

sent = pd.read_parquet(os.path.join(PROC_DIR, 'cohort_sent.parquet'))
cohort = pd.read_csv(os.path.join(PROC_DIR, 'cohort_users.csv'))
cs = cohort.sort_values('n_sent', ascending=False)

users = [cs.iloc[0]['sender'], cs.iloc[len(cs)//2]['sender'], cs.iloc[-1]['sender']]
labels = ['High activity', 'Medium activity', 'Low activity']
colors = ['#1a5276', '#2c5f8a', '#5499c7']

fig, axes = plt.subplots(3, 1, figsize=(7.0, 4.2), sharex=True)

for i, (user, label, color) in enumerate(zip(users, labels, colors)):
    ax = axes[i]
    ud = sent[sent['sender'] == user].copy()
    ud = ud.sort_values('timestamp_utc')
    ud['date'] = ud['timestamp_utc'].dt.date
    daily = ud.groupby('date').size()

    # Find best 7-day window
    if len(daily) >= 7:
        roll = daily.rolling(7).sum().dropna()
        best_end_date = roll.idxmax()
        best_start_date = best_end_date - pd.Timedelta(days=6)
        ws = pd.Timestamp(best_start_date, tz='UTC')
        we = pd.Timestamp(best_end_date, tz='UTC') + pd.Timedelta(days=1)
        week = ud[(ud['timestamp_utc'] >= ws) & (ud['timestamp_utc'] < we)]
    else:
        week = ud.head(30)
        ws = week['timestamp_utc'].min()

    # Convert to hours from start of week
    hours = (week['timestamp_utc'] - ws).dt.total_seconds() / 3600.0

    # Plot tick marks
    for h in hours:
        ax.axvline(x=h, color=color, linewidth=0.7, alpha=0.7, ymin=0.15, ymax=0.85)

    short_name = user.split('@')[0].replace('.', ' ').title()
    n_week = len(hours)
    ax.text(0.01, 0.85, short_name, transform=ax.transAxes,
            fontsize=9.5, fontweight='bold', color=color, va='top')
    ax.text(0.01, 0.42, f'{label} ({n_week} emails/week)',
            transform=ax.transAxes, fontsize=8.5, color='#555555', va='top')

    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Day separators
    for d in range(8):
        ax.axvline(x=d*24, color='#d0d0d0', linewidth=0.6, linestyle='-', zorder=0)

ax.set_xlim(0, 168)
ax.set_xticks([12, 36, 60, 84, 108, 132, 156])
ax.set_xticklabels(['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'])
axes[-1].set_xlabel('Relative Time (Best 7-Day Window Per User)')

axes[0].text(-0.08, 1.2, 'C', transform=axes[0].transAxes, fontsize=16,
             fontweight='bold', va='top', ha='left', color='#1a2744')

plt.tight_layout(h_pad=0.3)
fig.savefig(os.path.join(FIG_DIR, 'fig1c_event_rasters.pdf'), format='pdf')
fig.savefig(os.path.join(FIG_DIR, 'fig1c_event_rasters.png'), format='png', dpi=300)
plt.close()
print("Saved: fig1c_event_rasters.pdf + .png")
