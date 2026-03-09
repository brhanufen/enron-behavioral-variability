# Enron Behavioral Variability — Reproducibility Package

## Paper
**"Decomposing Behavioral Variability in Email Communication: Self-Excitation, Latent State-Switching, and Their Interaction in the Enron Corpus"**

Author: Brhanu Fentaw Znabu,pradeep devote, Zohiaib Atif

---

## Quick Start

### Requirements
```
Python >= 3.10
pandas >= 2.0
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.7
seaborn >= 0.12
statsmodels >= 0.14
hmmlearn >= 0.3
numba >= 0.58
pyarrow >= 12.0
```

### Reproduce Figures (from included data)
```bash
cd src/figures/
python fig1_dataset.py        # Figure 1: Dataset overview (4 panels)
python fig2_burstiness.py     # Figure 2: Burstiness & circadian (4 panels)
python fig3_replytime.py      # Figure 3: Reply-time distributions (3 panels)
python fig4_models.py         # Figure 4: Model comparison (3 panels)
```

### Reproduce Models (from included data)
```bash
cd src/eval/
python train_test.py          # Generate train/test split

cd ../models/
python poisson.py             # M1: Inhomogeneous Poisson
python renewal.py             # M2: Lognormal Renewal
python hawkes.py              # M3: Hawkes process (+ H1 test)
python hmm.py                 # M4: Poisson HMM (+ H2 test)
python hybrid.py              # M5: Hawkes residuals → HMM (+ H3 test)
```

### Reproduce From Raw Corpus (full pipeline)
```bash
# 1. Download Enron corpus from CMU
python src/data/parse_headers.py      # Parse headers, dedup
python src/data/timezone.py           # Per-user timezone inference
python src/data/cohort.py             # Cohort selection (≥500 sent, ≥1/day)
python src/data/reply_pairs.py        # Heuristic reply pairing
python src/data/reply_pair_audit.py   # Precision audit
python src/data/burstiness_stats.py   # Burstiness B and Memory M

# 2. Then run models and figures as above
```

---

## Directory Structure
```
data/
  processed/
    events_tz.parquet          # All 516K emails with timezone offsets
    cohort_sent.parquet        # 58-user cohort sent emails
    cohort_incoming.parquet    # 58-user cohort incoming emails
    cohort_users.csv           # Cohort member list with stats
    burstiness_stats.csv       # B and M per user
    reply_pairs_heuristic_*.parquet  # Reply pairs (1d/3d/7d windows)
    *.json                     # QC and summary reports
  results/
    results_m1–m5.csv          # Per-user model fit results
    h1_test.json               # H1: self-excitation test
    h2_test.json               # H2: state-switching test
    h3_test.json               # H3: complementarity test
    split_indices.pkl          # Train/test split indices
    ks_test_replytime.json     # KS test on reply times

src/
  data/                        # Data processing scripts
  models/                      # M1–M5 model fitting
  eval/                        # Train/test splitting
  figures/                     # All figure generation scripts

figures/                       # Publication-ready PDF + PNG (300 DPI)
```

---

## Data Source
Enron email corpus: https://www.cs.cmu.edu/~enron/
Only email headers used (no message bodies). All user identifiers are email addresses as distributed in the public corpus.

## Random Seed
All stochastic operations use seed=42 for full deterministic reproducibility.

## License
Code: MIT. Data: Enron corpus is public domain for research.
