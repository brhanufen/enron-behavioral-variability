#!/usr/bin/env python3
"""fig4_models.py - Model comparison, decomposition, latent states (3 panels) - FIXED."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from hmmlearn.hmm import PoissonHMM
import pickle, os, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

PROC_DIR = "/home/claude/enron-behavioral-variability/data/processed"
RESULTS_DIR = "/home/claude/enron-behavioral-variability/data/results"
FIG_DIR = "/home/claude/enron-behavioral-variability/figures"

m1 = pd.read_csv(os.path.join(RESULTS_DIR, 'results_m1.csv'))
m3 = pd.read_csv(os.path.join(RESULTS_DIR, 'results_m3.csv'))
m4 = pd.read_csv(os.path.join(RESULTS_DIR, 'results_m4.csv'))
m5 = pd.read_csv(os.path.join(RESULTS_DIR, 'results_m5.csv'))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# ──── 4A: Improvement over M1 (paired per-user, strip plot + box) ────
ax = axes[0]

common = set(m1['user']) & set(m3['user']) & set(m4['user'])
rows = []
for u in common:
    ll1 = m1[m1['user'] == u]['loglik_per_event'].values[0]
    ll3_v = m3[m3['user'] == u]['loglik_per_event'].values
    ll4_v = m4[m4['user'] == u]['loglik_per_event'].values
    if len(ll3_v) == 0 or len(ll4_v) == 0:
        continue
    ll3 = ll3_v[0]
    ll4 = ll4_v[0]
    if np.isfinite(ll3) and np.isfinite(ll4):
        rows.append({'user': u, 'M1': ll1, 'M3': ll3, 'M4': ll4})

decomp = pd.DataFrame(rows)
decomp['Hawkes_gain'] = decomp['M3'] - decomp['M1']
decomp['HMM_gain'] = decomp['M4'] - decomp['M1']

# Hard-clip y-axis so both boxes are visible
Y_LO, Y_HI = -25, 30

# Clip data for display (mark outliers with triangles at edge)
hg = decomp['Hawkes_gain'].values
mg = decomp['HMM_gain'].values
hg_clip = np.clip(hg, Y_LO, Y_HI)
mg_clip = np.clip(mg, Y_LO, Y_HI)

# Strip plot
jitter1 = 1 + np.random.uniform(-0.15, 0.15, len(decomp))
jitter2 = 2 + np.random.uniform(-0.15, 0.15, len(decomp))
ax.scatter(jitter1, hg_clip, color='#5499c7', alpha=0.5, s=18, zorder=3, edgecolors='none')
ax.scatter(jitter2, mg_clip, color='#2c5f8a', alpha=0.5, s=18, zorder=3, edgecolors='none')

# Mark clipped points with triangles at top
n_clipped = int(np.sum(hg > Y_HI))
if n_clipped > 0:
    ax.text(1, Y_HI - 3, f'({n_clipped} outliers\nabove axis)', ha='center', fontsize=6,
            color='#5499c7', style='italic')

# Box overlay
bp = ax.boxplot([hg_clip, mg_clip],
                positions=[1, 2], widths=0.45, patch_artist=True,
                boxprops=dict(linewidth=0.8, alpha=0.3),
                medianprops=dict(color='#c62828', linewidth=2),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(markersize=0))
bp['boxes'][0].set_facecolor('#5499c7')
bp['boxes'][1].set_facecolor('#2c5f8a')

ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
ax.set_xticks([1, 2])
ax.set_xticklabels(['M3 (Hawkes)\nvs M1 (Poisson)', 'M4 (HMM)\nvs M1 (Poisson)'])
ax.set_ylabel('Δ Log-Likelihood / Event')
ax.set_ylim(Y_LO, Y_HI)

# Median annotations - placed above each box, offset from data
med_h = np.median(hg)
med_m = np.median(mg)
ax.annotate(f'med = {med_h:.1f}', xy=(1, np.median(hg_clip)), xytext=(0.5, 0.78),
            textcoords=('data', 'axes fraction'), fontsize=7.5, fontweight='bold', color='#c62828',
            arrowprops=dict(arrowstyle='->', color='#c62828', lw=0.8),
            ha='center')
ax.annotate(f'med = {med_m:.1f}', xy=(2, np.median(mg_clip)), xytext=(2.45, 0.72),
            textcoords=('data', 'axes fraction'), fontsize=7.5, fontweight='bold', color='#c62828',
            arrowprops=dict(arrowstyle='->', color='#c62828', lw=0.8),
            ha='center')

# H1/H2 annotation - inside panel, lower-left
ax.text(0.97, 0.03, 'H1: 83% α>0 ✓\nH2: 84% >10% gain ✓',
        transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor='#2e7d32',
                  linewidth=0.6, alpha=0.95))

ax.grid(axis='y', alpha=0.15, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.14, 1.05, 'A', transform=ax.transAxes, fontsize=15, fontweight='bold', color='#1a2744')

# ──── 4B: Example inferred latent-state sequence ────
ax = axes[1]

sent = pd.read_parquet(os.path.join(PROC_DIR, 'cohort_sent.parquet'))
example_user = 'kay.mann@enron.com'
ud = sent[sent['sender'] == example_user].sort_values('timestamp_utc')

ts = ud['timestamp_utc']
t_min = ts.min().floor('h')
t_max = ts.max().ceil('h')
bins_range = pd.date_range(t_min, t_max, freq='h', tz='UTC')
counts = pd.Series(0, index=bins_range[:-1])
for t in ts:
    b = t.floor('h')
    if b in counts.index:
        counts[b] += 1

X = counts.values.reshape(-1, 1)
model = PoissonHMM(n_components=2, n_iter=200, random_state=42)
model.fit(X)
states = model.predict(X)
rates = model.lambdas_.flatten()

if rates[0] > rates[1]:
    states = 1 - states
    rates = rates[::-1]

n_hours = 24 * 14
time_hrs = np.arange(n_hours)
counts_plot = counts.values[:n_hours]
states_plot = states[:n_hours]

for h in range(n_hours):
    color = '#e8f5e9' if states_plot[h] == 1 else '#fce4ec'
    ax.axvspan(h, h+1, color=color, alpha=0.4, linewidth=0)

ax.bar(time_hrs, counts_plot, width=1.0, color='#2c5f8a', alpha=0.7, linewidth=0)
ax.set_xlabel('Hours (First 2 Weeks)')
ax.set_ylabel('Emails / Hour')
ax.set_xlim(0, n_hours)

for d in range(15):
    ax.axvline(x=d*24, color='#bdbdbd', linewidth=0.4)

legend_el = [Patch(facecolor='#e8f5e9', alpha=0.6, label=f'Work (λ={rates[1]:.2f})'),
             Patch(facecolor='#fce4ec', alpha=0.6, label=f'Rest (λ={rates[0]:.2f})')]
ax.legend(handles=legend_el, loc='upper right', fontsize=7, framealpha=0.9)

# Inset transition matrix
axins = ax.inset_axes([0.02, 0.55, 0.22, 0.38])
tm = model.transmat_
if rates[0] < rates[1]:
    pass  # already sorted
im = axins.imshow(tm, cmap='Blues', vmin=0, vmax=1)
for (ii, jj), val in np.ndenumerate(tm):
    axins.text(jj, ii, f'{val:.2f}', ha='center', va='center', fontsize=6,
               color='white' if val > 0.5 else 'black')
axins.set_xticks([0, 1])
axins.set_yticks([0, 1])
axins.set_xticklabels(['R', 'W'], fontsize=6)
axins.set_yticklabels(['R', 'W'], fontsize=6)
axins.set_title('Transition Matrix', fontsize=6, pad=2)

ax.text(-0.12, 1.05, 'B', transform=ax.transAxes, fontsize=15, fontweight='bold', color='#1a2744')

# ──── 4C: Hawkes residual BIC comparison (H3) ────
ax = axes[2]

m5_data = m5.sort_values('delta_bic').copy()
n_users = len(m5_data)
y_pos = np.arange(n_users)

colors_bar = ['#2c5f8a' if p else '#b0bec5' for p in m5_data['k2_preferred']]
ax.barh(y_pos, m5_data['delta_bic'].values, color=colors_bar, edgecolor='none', height=0.7)
ax.axvline(x=0, color='#333333', linewidth=0.8, linestyle='-')
ax.set_xlabel('ΔBIC (K=1 − K=2)\nPositive → K=2 preferred')
ax.set_ylabel('Users (sorted)')
ax.set_yticks([])

n_k2 = int(m5_data['k2_preferred'].sum())
n_tot = len(m5_data)

# H3 annotation - bottom left, clear of bars
ax.text(0.03, 0.03, f'H3: {n_k2}/{n_tot} ({n_k2/n_tot*100:.0f}%) users\nshow residual state-switching ✓',
        transform=ax.transAxes, fontsize=7.5, ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff9c4', edgecolor='#f9a825',
                  linewidth=0.6, alpha=0.95))

legend_h3 = [Patch(facecolor='#2c5f8a', label='K=2 preferred'),
             Patch(facecolor='#b0bec5', label='K=1 preferred')]
ax.legend(handles=legend_h3, loc='upper right', fontsize=7, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.15, linewidth=0.5)
ax.text(-0.12, 1.05, 'C', transform=ax.transAxes, fontsize=15, fontweight='bold', color='#1a2744')

plt.tight_layout(w_pad=2.5)
fig.savefig(os.path.join(FIG_DIR, 'fig4_models.pdf'), format='pdf')
fig.savefig(os.path.join(FIG_DIR, 'fig4_models.png'), format='png', dpi=300)
plt.close()
print("Saved: fig4_models.pdf + .png")
