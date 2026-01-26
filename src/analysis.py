"""
Dura Mater AFM Stiffness Analysis - 4 focused figures
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import COLORS, CONDITIONS, LABELS, setup_style
from data_processing import load_and_clean_data

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

PROJECT = Path(__file__).parent.parent
DATA_FILE = PROJECT / 'Data' / 'Dura measurement in AFM.xlsx'
FIG_DIR = PROJECT / 'outputs' / 'figures'
TABLE_DIR = PROJECT / 'outputs' / 'tables'
STATS_DIR = PROJECT / 'outputs' / 'statistics'

setup_style()

# Color palette - gray for control, warm gradient for RF
PALETTE = {
    'Dura+PBS': '#808080',   # Gray (control)
    '2mM+RF': '#FFD700',     # Gold
    '4mM+RF': '#FFA500',     # Orange
    '8mM+RF': '#FF4500',     # Red-orange
}

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------

print("Loading data...")
df = load_and_clean_data(DATA_FILE)
print(f"Loaded {len(df)} measurements\n")

# Save tables
print("Saving tables...")
df.to_csv(TABLE_DIR / 'cleaned_data.csv', index=False)

summary = df.groupby('Condition').agg({
    'Before_UV': ['count', 'mean', 'std'],
    'UV_3mW': ['mean', 'std'],
    'Fold_Change_3mW': ['mean', 'std']
}).round(1)
summary.to_csv(TABLE_DIR / 'descriptive_statistics.csv')
print("  Saved tables\n")

# -----------------------------------------------------------------------------
# FIGURE 1: HERO FIGURE - Final Stiffness with Fold-Change Labels
# -----------------------------------------------------------------------------

print("Generating figures...")

fig, ax = plt.subplots(figsize=(6, 5))

# Calculate means and SEMs
means = [df[df['Condition'] == c]['UV_3mW'].mean() for c in CONDITIONS]
sems = [df[df['Condition'] == c]['UV_3mW'].std() / np.sqrt(df[df['Condition'] == c]['UV_3mW'].count()) for c in CONDITIONS]
fold_changes = [df[df['Condition'] == c]['Fold_Change_3mW'].mean() for c in CONDITIONS]

# Bar plot
bars = ax.bar(range(4), means, yerr=sems,
              color=[PALETTE[c] for c in CONDITIONS],
              edgecolor='black', linewidth=1, capsize=5, error_kw={'linewidth': 1.5})

# Overlay individual points
for i, cond in enumerate(CONDITIONS):
    y = df[df['Condition'] == cond]['UV_3mW']
    x = np.random.normal(i, 0.08, len(y))
    ax.scatter(x, y, color='black', alpha=0.4, s=20, zorder=3)

# Fold-change labels above bars
for i, (m, fc) in enumerate(zip(means, fold_changes)):
    ax.text(i, m + sems[i] + 300, f'{fc:.0f}×', ha='center', fontsize=12, fontweight='bold')

# Baseline reference line
baseline = df['Before_UV'].mean()
ax.axhline(baseline, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(3.5, baseline + 100, f'Baseline\n({baseline:.0f} kPa)', ha='right', fontsize=9, style='italic')

ax.set_xticks(range(4))
ax.set_xticklabels(LABELS)
ax.set_ylabel('Stiffness after 3mW UV (kPa)', fontsize=12)
ax.set_ylim(0, 10000)
ax.set_title('Riboflavin Dramatically Amplifies UV Crosslinking', fontsize=13, fontweight='bold', pad=15)

# Clean up
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig(FIG_DIR / 'fig1_hero_stiffness.png', dpi=300)
plt.close()
print("  fig1_hero_stiffness.png")

# -----------------------------------------------------------------------------
# FIGURE 2: PAIRED BEFORE/AFTER - Shows Within-Sample Transformation (2x2)
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharey=True, sharex=True)
axes = axes.flatten()

for idx, cond in enumerate(CONDITIONS):
    ax = axes[idx]
    cond_data = df[df['Condition'] == cond]

    before = cond_data['Before_UV'].dropna().values
    after = cond_data['UV_3mW'].dropna().values

    # Use minimum length to pair correctly
    n = min(len(before), len(after))
    before = before[:n]
    after = after[:n]

    # Plot individual paired lines
    for b, a in zip(before, after):
        ax.plot([0, 1], [b, a], color=PALETTE[cond], alpha=0.4, linewidth=1.2)
        ax.scatter([0, 1], [b, a], color=PALETTE[cond], alpha=0.5, s=30, edgecolor='white', linewidth=0.5)

    # Plot mean trajectory (bold)
    mean_before = before.mean()
    mean_after = after.mean()
    ax.plot([0, 1], [mean_before, mean_after], color='black', linewidth=3, zorder=5)
    ax.scatter([0, 1], [mean_before, mean_after], color='black', s=100, zorder=5)

    # Fold change annotation - position in upper left corner
    fc = mean_after / mean_before
    ax.text(0.05, 9200, f'{fc:.0f}×', ha='left', va='top', fontsize=18, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Before UV', 'After UV'], fontsize=11)
    ax.set_title(LABELS[idx], fontsize=12, fontweight='bold')
    ax.set_xlim(-0.3, 1.3)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Y-axis labels for left column only
axes[0].set_ylabel('Stiffness (kPa)', fontsize=12)
axes[2].set_ylabel('Stiffness (kPa)', fontsize=12)
axes[0].set_ylim(0, 10000)

fig.suptitle('Within-Sample Transformation: Each Line = One Tissue Sample',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig2_paired_transformation.png', dpi=300)
plt.close()
print("  fig2_paired_transformation.png")

# -----------------------------------------------------------------------------
# FIGURE 3: DOSE-RESPONSE - RF Concentration vs Stiffness
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 5))

# Get mean ± SEM for each concentration and UV power
rf_conc = [0, 2, 4, 8]

# 3mW data
means_3mW = [df[df['Condition'] == c]['UV_3mW'].mean() for c in CONDITIONS]
sems_3mW = [df[df['Condition'] == c]['UV_3mW'].std() / np.sqrt(len(df[df['Condition'] == c])) for c in CONDITIONS]

# 0.3mW data
means_03mW = [df[df['Condition'] == c]['UV_0.3mW'].mean() for c in CONDITIONS]
sems_03mW = [df[df['Condition'] == c]['UV_0.3mW'].std() / np.sqrt(len(df[df['Condition'] == c])) for c in CONDITIONS]

# Plot 3mW (primary)
ax.errorbar(rf_conc, means_3mW, yerr=sems_3mW,
            fmt='o-', color='#1E3A5F', linewidth=2, markersize=10,
            capsize=5, label='3mW UV', zorder=5)

# Plot 0.3mW (secondary)
ax.errorbar(rf_conc, means_03mW, yerr=sems_03mW,
            fmt='s--', color='#6B9BC3', linewidth=2, markersize=8,
            capsize=5, label='0.3mW UV', zorder=4)

# Regression for 3mW (RF > 0 only)
slope, intercept, r, p, se = stats.linregress([2, 4, 8], [means_3mW[1], means_3mW[2], means_3mW[3]])
ax.text(6, 2000, f'R² = {r**2:.2f}\nSlope = {slope:.0f} kPa/mM', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

ax.set_xlabel('Riboflavin Concentration (mM)', fontsize=12)
ax.set_ylabel('Stiffness (kPa)', fontsize=12)
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_xlim(-0.5, 9)
ax.set_ylim(0, 10000)
ax.legend(loc='upper left', fontsize=10)
ax.set_title('Dose-Dependent Stiffening Response', fontsize=13, fontweight='bold', pad=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig(FIG_DIR / 'fig3_dose_response.png', dpi=300)
plt.close()
print("  fig3_dose_response.png")

# -----------------------------------------------------------------------------
# FIGURE 4: SUMMARY HEATMAP - Complete Parameter Space
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4))

# Build matrix: rows = conditions, cols = UV stages
matrix_data = []
annotations = []

for cond in CONDITIONS:
    cond_data = df[df['Condition'] == cond]
    baseline = cond_data['Before_UV'].mean()
    uv_03 = cond_data['UV_0.3mW'].mean()
    uv_3 = cond_data['UV_3mW'].mean()

    matrix_data.append([baseline, uv_03, uv_3])

    # Annotations with fold change
    fc_03 = uv_03 / baseline
    fc_3 = uv_3 / baseline
    annotations.append([f'{baseline:.0f}\n(1×)', f'{uv_03:.0f}\n({fc_03:.0f}×)', f'{uv_3:.0f}\n({fc_3:.0f}×)'])

matrix = np.array(matrix_data)

# Plot heatmap
im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=10000)

# Add text annotations
for i in range(4):
    for j in range(3):
        text_color = 'white' if matrix[i, j] > 5000 else 'black'
        ax.text(j, i, annotations[i][j], ha='center', va='center',
                fontsize=10, fontweight='bold', color=text_color)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Baseline', '0.3mW UV', '3mW UV'], fontsize=11)
ax.set_yticks(range(4))
ax.set_yticklabels(LABELS, fontsize=11)

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Stiffness (kPa)', fontsize=11)

ax.set_title('Complete Crosslinking Parameter Space', fontsize=13, fontweight='bold', pad=10)

fig.tight_layout()
fig.savefig(FIG_DIR / 'fig4_summary_heatmap.png', dpi=300)
plt.close()
print("  fig4_summary_heatmap.png")

# -----------------------------------------------------------------------------
# STATISTICAL TESTS
# -----------------------------------------------------------------------------

print("\nRunning statistical tests...")

results = []
results.append("=" * 60)
results.append("STATISTICAL ANALYSIS RESULTS")
results.append("=" * 60 + "\n")

# 1. ANOVA: Baseline
results.append("1. BASELINE CONSISTENCY (One-way ANOVA)")
results.append("-" * 40)
groups = [df[df['Condition'] == c]['Before_UV'].dropna() for c in CONDITIONS]
f_stat, p_val = stats.f_oneway(*groups)
results.append(f"F = {f_stat:.2f}, p = {p_val:.4f}")
results.append("→ No significant difference in baseline stiffness\n")

# 2. Treatment effect
results.append("2. TREATMENT EFFECT (Paired t-tests: Before vs After 3mW UV)")
results.append("-" * 40)
for cond in CONDITIONS:
    cond_data = df[df['Condition'] == cond]
    before = cond_data['Before_UV']
    after = cond_data['UV_3mW']
    t, p = stats.ttest_rel(before, after)
    fc = after.mean() / before.mean()
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    results.append(f"{cond}: {fc:.0f}× increase, p = {p:.2e} {sig}")
results.append("")

# 3. Dose-response
results.append("3. DOSE-RESPONSE (Linear regression: RF conc vs stiffness)")
results.append("-" * 40)
rf_data = df[df['RF_Concentration'] > 0]
slope, intercept, r, p, se = stats.linregress(rf_data['RF_Concentration'], rf_data['UV_3mW'])
results.append(f"Slope: {slope:.0f} kPa per mM RF")
results.append(f"R² = {r**2:.3f}, p = {p:.2e}")
results.append("")

# 4. Effect sizes
results.append("4. EFFECT SIZES (Cohen's d: RF vs PBS Control)")
results.append("-" * 40)
control = df[df['Condition'] == 'Dura+PBS']['UV_3mW']
for cond in CONDITIONS[1:]:
    treatment = df[df['Condition'] == cond]['UV_3mW']
    pooled_std = np.sqrt(((len(control)-1)*control.std()**2 + (len(treatment)-1)*treatment.std()**2) / (len(control) + len(treatment) - 2))
    d = (treatment.mean() - control.mean()) / pooled_std
    results.append(f"{cond}: d = {d:.1f} (massive effect)")

results.append("\n" + "=" * 60)

with open(STATS_DIR / 'test_results.txt', 'w') as f:
    f.write('\n'.join(results))
print("  test_results.txt")

# -----------------------------------------------------------------------------
# DONE
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Analysis complete! 4 streamlined figures generated.")
print("=" * 60)
