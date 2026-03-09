#!/usr/bin/env python3
"""fig1_dataset.py - Figure 1 composite (4 panels). 1B and 1C match originals exactly."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

PROC_DIR = "/home/claude/enron-behavioral-variability/data/processed"
FIG_DIR = "/home/claude/enron-behavioral-variability/figures"

fig = plt.figure(figsize=(14, 11))
# Top row: 1A (left), 1B (right)
# Bottom row: 1C (left, 3 sub-rows), 1D (right)
outer = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.28, height_ratios=[1, 1.1])

# ════════════════════════════════════════════
# PANEL 1A: Pipeline / cohort flow diagram
# ════════════════════════════════════════════
ax = fig.add_subplot(outer[0, 0])
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

def draw_box(ax, x, y, w, h, text, color='#e3f2fd', edge='#1565c0', fontsize=8):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.15", facecolor=color,
                         edgecolor=edge, linewidth=1.2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='#1a2744')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

draw_box(ax, 5, 9.2, 4.5, 0.9, 'Enron Corpus (CMU)\n517,401 emails', '#fff3e0', '#e65100')
draw_arrow(ax, 5, 8.75, 5, 8.15)
draw_box(ax, 5, 7.7, 4.5, 0.9, 'Header Parsing + Dedup\n516,359 records', '#e3f2fd', '#1565c0')
draw_arrow(ax, 5, 7.25, 5, 6.65)
draw_box(ax, 5, 6.2, 4.5, 0.9, 'Timezone Inference\n(activity-minimum method)', '#e3f2fd', '#1565c0')
draw_arrow(ax, 5, 5.75, 5, 5.15)
draw_box(ax, 5, 4.7, 4.5, 0.9, 'Cohort Selection\n≥500 sent, ≥1/day → 58 users', '#e8f5e9', '#2e7d32')
draw_arrow(ax, 3.5, 4.25, 3.5, 3.65)
draw_arrow(ax, 6.5, 4.25, 6.5, 3.65)
draw_box(ax, 3.5, 3.1, 3.8, 1.0, 'Sending Activity\n126,846 outgoing\n(Primary analysis)', '#e8f5e9', '#2e7d32')
draw_box(ax, 6.8, 3.1, 3.2, 1.0, 'Reply Pairs\n16,600 pairs\n(Supplementary)', '#fff9c4', '#f9a825')
draw_arrow(ax, 3.5, 2.55, 3.5, 1.95)
draw_box(ax, 5, 1.5, 7.5, 0.9, 'Models: M1 Poisson → M2 Renewal → M3 Hawkes → M4 HMM → M5 Hybrid',
         '#fce4ec', '#c62828', fontsize=7.5)

ax.text(9.5, 7.7, '0 dupes\nremoved', fontsize=6.5, color='#888888', ha='center', style='italic')
ax.text(9.5, 6.2, '172 users\nanalyzed', fontsize=6.5, color='#888888', ha='center', style='italic')
ax.text(-0.05, 1.02, 'A', transform=ax.transAxes, fontsize=16, fontweight='bold', color='#1a2744')

# ════════════════════════════════════════════
# PANEL 1B: Email distribution — MATCHES ORIGINAL
# ════════════════════════════════════════════
ax = fig.add_subplot(outer[0, 1])

df = pd.read_parquet(os.path.join(PROC_DIR, 'events_tz.parquet'))
sent_all = df[df['is_sent_folder']]
user_counts = sent_all.groupby('sender').size().sort_values(ascending=False)
user_counts = user_counts[user_counts >= 10]  # same filter as original

cohort = pd.read_csv(os.path.join(PROC_DIR, 'cohort_users.csv'))
cohort_senders = set(cohort['sender'])
in_cohort = user_counts[user_counts.index.isin(cohort_senders)]
not_cohort = user_counts[~user_counts.index.isin(cohort_senders)]

bins = np.logspace(np.log10(10), np.log10(user_counts.max() * 1.2), 30)
ax.hist(not_cohort.values, bins=bins, color='#b0bec5', edgecolor='#78909c',
        linewidth=0.5, alpha=0.85, label=f'Below threshold (n={len(not_cohort)})')
ax.hist(in_cohort.values, bins=bins, color='#2c5f8a', edgecolor='#1a3a5c',
        linewidth=0.5, alpha=0.9, label=f'Cohort (n={len(in_cohort)})')

ax.axvline(x=500, color='#c62828', linewidth=1.5, linestyle='--', zorder=5)
ax.text(520, ax.get_ylim()[1] * 0.85, 'Threshold\n(500 emails)',
        fontsize=8, color='#c62828', fontweight='bold', va='top')

ax.set_xscale('log')
ax.set_xlabel('Number of Outgoing Emails')
ax.set_ylabel('Number of Users')
ax.set_xlim(8, user_counts.max() * 1.5)
ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='#cccccc', fontsize=7.5)
ax.grid(axis='y', alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.12, 1.02, 'B', transform=ax.transAxes, fontsize=16, fontweight='bold', color='#1a2744')

# ════════════════════════════════════════════
# PANEL 1C: Event rasters — MATCHES ORIGINAL (3 sub-rows)
# ════════════════════════════════════════════
inner_c = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1, 0], hspace=0.08)

sent = pd.read_parquet(os.path.join(PROC_DIR, 'cohort_sent.parquet'))
cs = cohort.sort_values('n_sent', ascending=False)
users_c = [cs.iloc[0]['sender'], cs.iloc[len(cs)//2]['sender'], cs.iloc[-1]['sender']]
labels_c = ['High activity', 'Medium activity', 'Low activity']
colors_c = ['#1a5276', '#2c5f8a', '#5499c7']

raster_axes = []
for idx, (user, label, color) in enumerate(zip(users_c, labels_c, colors_c)):
    ax = fig.add_subplot(inner_c[idx])
    raster_axes.append(ax)
    
    ud = sent[sent['sender'] == user].copy().sort_values('timestamp_utc')
    ud['date'] = ud['timestamp_utc'].dt.date
    daily = ud.groupby('date').size()
    
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
    
    hours = (week['timestamp_utc'] - ws).dt.total_seconds() / 3600.0
    
    for h in hours:
        ax.axvline(x=h, color=color, linewidth=0.7, alpha=0.7, ymin=0.15, ymax=0.85)
    
    short_name = user.split('@')[0].replace('.', ' ').title()
    n_week = len(hours)
    ax.text(0.01, 0.85, short_name, transform=ax.transAxes,
            fontsize=9, fontweight='bold', color=color, va='top')
    ax.text(0.01, 0.42, f'{label} ({n_week} emails/week)',
            transform=ax.transAxes, fontsize=8, color='#555555', va='top')
    
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    for d in range(8):
        ax.axvline(x=d*24, color='#d0d0d0', linewidth=0.6, linestyle='-', zorder=0)
    
    if idx < 2:
        ax.set_xticks([])
    ax.set_xlim(0, 168)

# Bottom raster axis labels
raster_axes[-1].set_xticks([12, 36, 60, 84, 108, 132, 156])
raster_axes[-1].set_xticklabels(['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'])
raster_axes[-1].set_xlabel('Relative Time (Best 7-Day Window Per User)')
raster_axes[0].text(-0.08, 1.15, 'C', transform=raster_axes[0].transAxes,
                     fontsize=16, fontweight='bold', va='top', color='#1a2744')

# ════════════════════════════════════════════
# PANEL 1D: Reply-pair construction schematic
# ════════════════════════════════════════════
ax = fig.add_subplot(outer[1, 1])
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.7, 'Heuristic Reply-Pair Construction', fontsize=10, fontweight='bold',
        ha='center', color='#1a2744')

draw_box(ax, 2.5, 8.3, 4, 0.8, '1. Subject Normalization\nstrip Re:/Fwd:, lowercase', '#e3f2fd', '#1565c0', 7.5)
draw_box(ax, 7.5, 8.3, 3.5, 0.8, '2. Time Window\nΔt ≤ 1 day (primary)', '#e3f2fd', '#1565c0', 7.5)
draw_arrow(ax, 2.5, 7.85, 2.5, 7.25)
draw_arrow(ax, 7.5, 7.85, 7.5, 7.25)
draw_box(ax, 5, 6.8, 6, 0.8, '3. Match: same subject + sender↔recipient within Δt',
         '#e8f5e9', '#2e7d32', 7.5)
draw_arrow(ax, 5, 6.35, 5, 5.75)
draw_box(ax, 5, 5.3, 5.5, 0.8, '16,600 reply pairs (1-day window)\nτ = t_reply − t_received',
         '#e8f5e9', '#2e7d32', 7.5)
draw_arrow(ax, 5, 4.85, 5, 4.45)
draw_box(ax, 5, 3.8, 6, 1.2,
         'Precision Audit (n = 200)\n'
         '✓ Subject > 8 chars: 91.5%\n'
         '✓ Not self-reply: 93.0%\n'
         '✓ All checks pass: 81.0%',
         '#fff9c4', '#f9a825', 7)

ax.text(5, 2.2, 'Sensitivity Analysis', fontsize=8.5, fontweight='bold', ha='center', color='#1a2744')
col_x = [3, 5, 7]
for x, h in zip(col_x, ['Window', 'Pairs', 'Median τ']):
    ax.text(x, 1.7, h, fontsize=7, ha='center', fontweight='bold', color='#555555')
for r, row in enumerate([['1 day', '16,600', '1.3 hr'],
                          ['3 days', '18,955', '2.0 hr'],
                          ['7 days', '20,876', '2.6 hr']]):
    y = 1.25 - r * 0.4
    fw = 'bold' if r == 0 else 'normal'
    for x, val in zip(col_x, row):
        ax.text(x, y, val, fontsize=7, ha='center', color='#333333', fontweight=fw)

ax.text(-0.05, 1.02, 'D', transform=ax.transAxes, fontsize=16, fontweight='bold', color='#1a2744')

fig.savefig(os.path.join(FIG_DIR, 'fig1_dataset.pdf'), format='pdf')
fig.savefig(os.path.join(FIG_DIR, 'fig1_dataset.png'), format='png', dpi=300)
plt.close()
print("Saved: fig1_dataset.pdf + .png")
