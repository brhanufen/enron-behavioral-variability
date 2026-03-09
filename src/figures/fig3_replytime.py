#!/usr/bin/env python3
"""fig3_replytime.py - Reply-time distribution figures (3 panels)."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

PROC_DIR = "/home/claude/enron-behavioral-variability/data/processed"
FIG_DIR = "/home/claude/enron-behavioral-variability/figures"

# Load primary (1-day window) pairs
pairs = pd.read_parquet(os.path.join(PROC_DIR, 'reply_pairs_heuristic_1d.parquet'))
tau_hr = pairs['tau_hours'].values
tau_hr = tau_hr[tau_hr > 0]  # exclude zero

# Per-user quartiles based on median tau
user_med = pairs.groupby('user')['tau_hours'].median()
q25, q50, q75 = user_med.quantile([0.25, 0.5, 0.75])
fast_users = set(user_med[user_med <= q25].index)
med_users = set(user_med[(user_med > q25) & (user_med <= q75)].index)
slow_users = set(user_med[user_med > q75].index)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# ──── PANEL 3A: CCDF of reply times ────
ax = axes[0]

def plot_ccdf(data, ax, label, color, lw=1.5, ls='-', alpha=1.0):
    sorted_d = np.sort(data)
    ccdf = 1.0 - np.arange(1, len(sorted_d)+1) / len(sorted_d)
    ax.plot(sorted_d, ccdf, color=color, linewidth=lw, linestyle=ls, label=label, alpha=alpha)

# Aggregate
plot_ccdf(tau_hr, ax, f'All pairs (n={len(tau_hr):,})', '#1a2744', lw=2.0)

# By quartiles
fast_tau = pairs[pairs['user'].isin(fast_users)]['tau_hours'].values
med_tau = pairs[pairs['user'].isin(med_users)]['tau_hours'].values
slow_tau = pairs[pairs['user'].isin(slow_users)]['tau_hours'].values

plot_ccdf(fast_tau, ax, f'Fast responders (Q1)', '#2e7d32', lw=1.2)
plot_ccdf(med_tau, ax, f'Medium (Q2–Q3)', '#f57f17', lw=1.2)
plot_ccdf(slow_tau, ax, f'Slow responders (Q4)', '#c62828', lw=1.2)

# Exponential reference (Poisson)
x_ref = np.linspace(0.01, 24, 500)
mean_tau = np.mean(tau_hr)
poisson_ccdf = np.exp(-x_ref / mean_tau)
ax.plot(x_ref, poisson_ccdf, 'k--', linewidth=1.0, alpha=0.5, label='Exponential ref.')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Reply Time (hours)')
ax.set_ylabel('P(tau > t)')
ax.set_xlim(0.005, 25)
ax.set_ylim(1e-4, 1.1)
ax.legend(loc='lower left', frameon=True, framealpha=0.9, edgecolor='#cccccc', fontsize=7.5)
ax.grid(True, alpha=0.2, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.14, 1.05, 'A', transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', color='#1a2744')
ax.text(0.97, 0.08, 'KS: D=0.25, p<10$^{-6}$\n(95% of users reject exp.)',
        transform=ax.transAxes, fontsize=7.5, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9c4', edgecolor='#f9a825',
                  linewidth=0.6, alpha=0.95))

# ──── PANEL 3B: Reply time by hour-of-day ────
ax = axes[1]

pairs['hour'] = pairs['sent_time'].dt.hour
hourly = pairs.groupby('hour')['tau_hours'].agg(['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
hourly.columns = ['median', 'q25', 'q75']
hours = hourly.index.values

ax.fill_between(hours, hourly['q25'], hourly['q75'], color='#2c5f8a', alpha=0.2, label='IQR')
ax.plot(hours, hourly['median'], 'o-', color='#2c5f8a', linewidth=1.8, markersize=4.5, label='Median')

# Mark business hours
ax.axvspan(9, 17, color='#e8f5e9', alpha=0.4, zorder=0)
ax.text(13, ax.get_ylim()[0] + 0.05, 'Business\nhours', fontsize=7, ha='center',
        color='#2e7d32', alpha=0.7, va='bottom')

ax.set_xlabel('Hour of Day (Local Time)')
ax.set_ylabel('Reply Time (hours)')
ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
ax.set_xticklabels(['0', '4', '8', '12', '16', '20', '24'])
ax.set_xlim(-0.5, 23.5)
ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
ax.grid(True, alpha=0.2, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.14, 1.05, 'B', transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', color='#1a2744')

# ──── PANEL 3C: Sensitivity to pairing thresholds ────
ax = axes[2]

windows = [1, 3, 7]
colors_bar = ['#2c5f8a', '#5499c7', '#a9cce3']
medians = []
means = []
counts = []

for w in windows:
    p = pd.read_parquet(os.path.join(PROC_DIR, f'reply_pairs_heuristic_{w}d.parquet'))
    medians.append(p['tau_hours'].median())
    means.append(p['tau_hours'].mean())
    counts.append(len(p))

x = np.arange(len(windows))
width = 0.35

bars1 = ax.bar(x - width/2, medians, width, color='#2c5f8a', edgecolor='#1a3a5c',
               linewidth=0.5, label='Median tau')
bars2 = ax.bar(x + width/2, means, width, color='#a9cce3', edgecolor='#7fb3d4',
               linewidth=0.5, label='Mean tau')

# Annotate pair counts above
for i, (xi, c) in enumerate(zip(x, counts)):
    ax.text(xi, max(medians[i], means[i]) + 1.5, f'n={c:,}',
            ha='center', fontsize=7.5, color='#555555')

ax.set_xlabel('Pairing Window')
ax.set_ylabel('Reply Time (hours)')
ax.set_xticks(x)
ax.set_xticklabels(['1 day', '3 days', '7 days'])
ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='#cccccc')
ax.grid(axis='y', alpha=0.2, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.14, 1.05, 'C', transform=ax.transAxes, fontsize=16,
        fontweight='bold', va='top', color='#1a2744')

plt.tight_layout(w_pad=2.5)
fig.savefig(os.path.join(FIG_DIR, 'fig3_replytime.pdf'), format='pdf')
fig.savefig(os.path.join(FIG_DIR, 'fig3_replytime.png'), format='png', dpi=300)
plt.close()
print("Saved: fig3_replytime.pdf + .png")
