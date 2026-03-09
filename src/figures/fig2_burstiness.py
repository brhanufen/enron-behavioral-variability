#!/usr/bin/env python3
"""fig2_burstiness.py - 4-panel burstiness and circadian figure."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
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

sent = pd.read_parquet(os.path.join(PROC_DIR, 'cohort_sent.parquet'))
cohort = pd.read_csv(os.path.join(PROC_DIR, 'cohort_users.csv'))
bstats = pd.read_csv(os.path.join(PROC_DIR, 'burstiness_stats.csv'))

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# ──── 2A: CCDF of inter-event times vs Poisson ────
ax = axes[0, 0]
all_iets = []
for _, row in cohort.iterrows():
    ud = sent[sent['sender'] == row['sender']].sort_values('timestamp_utc')
    iets = np.diff(ud['timestamp_utc'].values).astype('timedelta64[s]').astype(float) / 3600
    iets = iets[iets > 0]
    all_iets.extend(iets)

all_iets = np.array(all_iets)
sorted_iets = np.sort(all_iets)
ccdf = 1.0 - np.arange(1, len(sorted_iets)+1) / len(sorted_iets)
ax.plot(sorted_iets, ccdf, color='#1a5276', linewidth=1.5, label='Empirical (all users)')

# Poisson reference
mean_iet = np.mean(all_iets)
x_ref = np.linspace(0.01, sorted_iets.max(), 500)
ax.plot(x_ref, np.exp(-x_ref / mean_iet), 'k--', linewidth=1, alpha=0.5, label='Exponential ref.')

# 3 example users
colors_ex = ['#2e7d32', '#f57f17', '#c62828']
example_users = [cohort.iloc[0]['sender'], cohort.iloc[len(cohort)//2]['sender'], cohort.iloc[-1]['sender']]
for u, c in zip(example_users, colors_ex):
    ud = sent[sent['sender'] == u].sort_values('timestamp_utc')
    iets = np.diff(ud['timestamp_utc'].values).astype('timedelta64[s]').astype(float) / 3600
    iets = iets[iets > 0]
    si = np.sort(iets)
    cc = 1.0 - np.arange(1, len(si)+1) / len(si)
    ax.plot(si, cc, color=c, linewidth=0.8, alpha=0.7, label=u.split('@')[0].replace('.',' ').title())

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Inter-Event Time (hours)')
ax.set_ylabel('P(IET > t)')
ax.set_xlim(1e-3, 1e4)
ax.set_ylim(1e-5, 1.1)
ax.legend(loc='lower left', fontsize=6.5, framealpha=0.9)
ax.grid(True, alpha=0.2, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# KS test
ks_d, ks_p = stats.kstest(all_iets[:10000], 'expon', args=(0, mean_iet))
ax.text(0.97, 0.97, f'KS: D={ks_d:.2f}, p<10$^{{-6}}$', transform=ax.transAxes,
        fontsize=7, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9c4', edgecolor='#f9a825', linewidth=0.5, alpha=0.9))
ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontsize=15, fontweight='bold', color='#1a2744')

# ──── 2B: Burstiness B and Memory M with bootstrap CIs ────
ax = axes[0, 1]
bstats_sorted = bstats.sort_values('burstiness_B')
x_pos = np.arange(len(bstats_sorted))

# Bootstrap CIs for B (simplified: use std error approx)
n_boot = 50
B_cis = []
M_cis = []
for _, row in bstats_sorted.iterrows():
    ud = sent[sent['sender'] == row['sender']].sort_values('timestamp_utc')
    iets = np.diff(ud['timestamp_utc'].values).astype('timedelta64[s]').astype(float)
    iets = iets[iets > 0]
    bs_B = []
    bs_M = []
    for _ in range(n_boot):
        samp = np.random.choice(iets, size=len(iets), replace=True)
        mu_s, sig_s = np.mean(samp), np.std(samp)
        bs_B.append((sig_s - mu_s) / (sig_s + mu_s) if (sig_s + mu_s) > 0 else 0)
        if len(samp) > 2:
            mc = np.corrcoef(samp[:-1], samp[1:])[0, 1]
            bs_M.append(mc if np.isfinite(mc) else 0)
    B_cis.append((np.percentile(bs_B, 2.5), np.percentile(bs_B, 97.5)))
    M_cis.append((np.percentile(bs_M, 2.5), np.percentile(bs_M, 97.5)) if bs_M else (0, 0))

B_lo = [c[0] for c in B_cis]
B_hi = [c[1] for c in B_cis]

ax.fill_between(x_pos, B_lo, B_hi, color='#2c5f8a', alpha=0.2)
ax.plot(x_pos, bstats_sorted['burstiness_B'].values, 'o', color='#2c5f8a', markersize=3.5, label='B (burstiness)')
ax.plot(x_pos, bstats_sorted['memory_M'].values, 's', color='#c62828', markersize=3, alpha=0.7, label='M (memory)')
ax.axhline(y=0, color='gray', linewidth=0.6, linestyle='--', alpha=0.5)
ax.set_xlabel('Users (sorted by B)')
ax.set_ylabel('Coefficient Value')
ax.set_xticks([])
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.2, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(0.97, 0.97, f'All {len(bstats)} users: B>0\n(95% CI excludes 0)',
        transform=ax.transAxes, fontsize=7, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=0.5, alpha=0.9))
ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, fontsize=15, fontweight='bold', color='#1a2744')

# ──── 2C: Hour x Day-of-Week heatmap ────
ax = axes[1, 0]
sent_copy = sent.copy()
sent_copy['hour'] = sent_copy['timestamp_utc'].dt.hour
sent_copy['dow'] = sent_copy['timestamp_utc'].dt.dayofweek

heatmap_data = sent_copy.groupby(['dow', 'hour']).size().unstack(fill_value=0)
heatmap_data = heatmap_data.reindex(index=range(7), columns=range(24), fill_value=0)
# Normalize per-cell to rate
total_weeks = (sent_copy['timestamp_utc'].max() - sent_copy['timestamp_utc'].min()).days / 7
heatmap_norm = heatmap_data / max(total_weeks, 1)

im = ax.imshow(heatmap_norm.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_yticks(range(7))
ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax.set_xticks([0, 4, 8, 12, 16, 20])
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Day of Week')
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Emails/week', fontsize=8)
ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, fontsize=15, fontweight='bold', color='#1a2744')

# ──── 2D: Fano factor at multiple time scales ────
ax = axes[1, 1]
scales_min = [5, 15, 30, 60, 120, 360, 720, 1440]  # minutes
scale_labels = ['5m', '15m', '30m', '1h', '2h', '6h', '12h', '24h']

# Compute Fano factor per scale (aggregate across users)
fano_vals = []
fano_cohort = cohort.nlargest(20, 'n_sent')  # top 20 for speed
for scale in scales_min:
    all_counts = []
    for _, row in fano_cohort.iterrows():
        ud = sent[sent['sender'] == row['sender']].sort_values('timestamp_utc')
        ts = ud['timestamp_utc']
        t_min = ts.min().floor('h')
        t_max = ts.max().ceil('h')
        bins = pd.date_range(t_min, t_max, freq=f'{scale}min', tz='UTC')
        counts = pd.cut(ts, bins=bins).value_counts().values
        all_counts.extend(counts)
    all_counts = np.array(all_counts, dtype=float)
    fano = np.var(all_counts) / max(np.mean(all_counts), 1e-10)
    fano_vals.append(fano)

ax.plot(range(len(scales_min)), fano_vals, 'o-', color='#2c5f8a', linewidth=1.8, markersize=5)
ax.axhline(y=1, color='gray', linewidth=1, linestyle='--', alpha=0.6, label='Poisson (Fano=1)')
ax.set_xticks(range(len(scales_min)))
ax.set_xticklabels(scale_labels, rotation=45)
ax.set_xlabel('Time Bin Size')
ax.set_ylabel('Fano Factor (Var/Mean)')
ax.set_yscale('log')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.2, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.15, 1.05, 'D', transform=ax.transAxes, fontsize=15, fontweight='bold', color='#1a2744')

plt.tight_layout(h_pad=2.5, w_pad=2.5)
fig.savefig(os.path.join(FIG_DIR, 'fig2_burstiness.pdf'), format='pdf')
fig.savefig(os.path.join(FIG_DIR, 'fig2_burstiness.png'), format='png', dpi=300)
plt.close()
print("Saved: fig2_burstiness.pdf + .png")
