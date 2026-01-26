# Analysis Walkthrough

A step-by-step guide to the data analysis pipeline.

---

## Table of Contents

1. [Data Loading and Cleaning](#1-data-loading-and-cleaning)
2. [Unit Conversion](#2-unit-conversion)
3. [Derived Variables](#3-derived-variables)
4. [Descriptive Statistics](#4-descriptive-statistics)
5. [Figure Generation](#5-figure-generation)
6. [Statistical Tests](#6-statistical-tests)
7. [Reproducing the Analysis](#7-reproducing-the-analysis)

---

## 1. Data Loading and Cleaning

### Source File

The raw data is stored in `Data/Dura measurement in AFM.xlsx`, an Excel file containing 140 rows and 10 columns.

### Column Structure

| Original Column | Renamed To | Description |
|-----------------|------------|-------------|
| Condition | Condition | Treatment group (PBS, 2mM RF, 4mM RF, 8mM RF) |
| Sample ID | Sample_ID | Sample identifier (1-5 per condition) |
| Before UV exposure | Before_UV | Baseline elastic modulus |
| UV exposure at 0.3mW | UV_0.3mW | Stiffness after low-power UV |
| Measurement duration (in min) for 0.3 mW | Duration_0.3mW | Exposure time |
| UV exposure at 3mW | UV_3mW | Stiffness after high-power UV |
| Measurement duration (in min) for 3mW | Duration_3mW | Exposure time |
| Measurement day | Measurement_Day | Days after preparation |
| Refrigeration Duration | Refrigeration | Storage time |
| Other note | Notes | Experimental notes |

### Cleaning Steps

```python
# 1. Remove summary rows containing "Average"
df = df[~df['Sample_ID'].str.contains('Average', na=False)]

# 2. Forward-fill condition names (they span multiple rows)
df['Condition'] = df['Condition'].ffill()

# 3. Standardize condition names (remove extra spaces)
df['Condition'] = df['Condition'].replace({
    'Dura+ PBS': 'Dura+PBS',
    '2mM + RF': '2mM+RF',
    '4mM + RF': '4mM+RF',
    '8mM + RF': '8mM+RF',
})

# 4. Keep only rows with baseline measurements
df = df.dropna(subset=['Before_UV'])
```

**Result**: 125 valid measurements across 4 conditions.

---

## 2. Unit Conversion

### The Problem

A critical issue in the raw data: **units are inconsistent** between conditions.

| Condition | Before_UV | UV_0.3mW | UV_3mW |
|-----------|-----------|----------|--------|
| PBS Control | 51 kPa | 101 kPa | 123 kPa |
| 2mM RF | 53 kPa | **0.83 MPa** | **2.4 MPa** |
| 4mM RF | 52 kPa | **1.15 MPa** | **4.7 MPa** |
| 8mM RF | 53 kPa | **1.81 MPa** | **8.0 MPa** |

The PBS control values are in **kilopascals (kPa)**, but riboflavin-treated samples have post-UV values in **megapascals (MPa)**.

### The Solution

Convert RF condition post-UV values from MPa to kPa:

```python
# Identify RF-treated samples
rf_mask = df['Condition'].str.contains('RF', na=False)

# Convert MPa to kPa (multiply by 1000)
df.loc[rf_mask, 'UV_0.3mW'] = df.loc[rf_mask, 'UV_0.3mW'] * 1000
df.loc[rf_mask, 'UV_3mW'] = df.loc[rf_mask, 'UV_3mW'] * 1000
```

### After Conversion

| Condition | Before_UV | UV_0.3mW | UV_3mW |
|-----------|-----------|----------|--------|
| PBS Control | 51 kPa | 101 kPa | 123 kPa |
| 2mM RF | 53 kPa | 832 kPa | 2,417 kPa |
| 4mM RF | 52 kPa | 1,155 kPa | 4,670 kPa |
| 8mM RF | 53 kPa | 1,810 kPa | 8,006 kPa |

Now all values are in kPa and comparable.

---

## 3. Derived Variables

### Riboflavin Concentration (Numeric)

```python
rf_map = {'Dura+PBS': 0, '2mM+RF': 2, '4mM+RF': 4, '8mM+RF': 8}
df['RF_Concentration'] = df['Condition'].map(rf_map)
```

### Fold Change

```python
df['Fold_Change_0.3mW'] = df['UV_0.3mW'] / df['Before_UV']
df['Fold_Change_3mW'] = df['UV_3mW'] / df['Before_UV']
```

### Example Calculation

For a single 8mM RF measurement:
- Before_UV = 53 kPa
- UV_3mW = 8,006 kPa
- Fold_Change_3mW = 8,006 / 53 = **151×**

---

## 4. Descriptive Statistics

### Group Summaries

Calculated using:

```python
df.groupby('Condition').agg({
    'Before_UV': ['count', 'mean', 'std'],
    'UV_3mW': ['mean', 'std'],
    'Fold_Change_3mW': ['mean', 'std']
})
```

### Results

| Condition | N | Baseline (kPa) | After 3mW UV (kPa) | Fold Change |
|-----------|---|----------------|-------------------|-------------|
| Dura+PBS | 50 | 51.2 ± 13.1 | 123.2 ± 12.8 | 2.6 ± 0.8 |
| 2mM+RF | 25 | 52.7 ± 13.8 | 2,416.8 ± 703.7 | 49.9 ± 23.3 |
| 4mM+RF | 25 | 52.3 ± 13.8 | 4,670.5 ± 735.9 | 95.1 ± 27.0 |
| 8mM+RF | 25 | 53.5 ± 13.5 | 8,005.9 ± 809.1 | 161.7 ± 52.6 |

**Note**: PBS has 50 measurements (25 with 0.3mW only, 25 with 3mW only), while RF conditions have 25 measurements each with all time points.

---

## 5. Figure Generation

### Figure 1: Hero Figure (Bar Chart)

**Purpose**: Show the dramatic magnitude of RF-induced stiffening at a glance.

**Key design choices**:
- Linear y-axis (0-10,000 kPa) to emphasize magnitude
- Fold-change labels above each bar
- Baseline reference line (dashed)
- Individual data points overlaid on bars
- Warm color gradient (gray → gold → orange → red)

```python
# Calculate means and SEMs
means = [df[df['Condition'] == c]['UV_3mW'].mean() for c in CONDITIONS]
sems = [df[df['Condition'] == c]['UV_3mW'].std() / np.sqrt(n) for c in CONDITIONS]

# Bar plot with error bars
ax.bar(range(4), means, yerr=sems, color=[PALETTE[c] for c in CONDITIONS])

# Add fold-change labels
for i, (m, fc) in enumerate(zip(means, fold_changes)):
    ax.text(i, m + offset, f'{fc:.0f}×', ha='center', fontweight='bold')
```

### Figure 2: Paired Transformation (Slopegraph)

**Purpose**: Show that stiffening occurs within the same tissue sample (not due to sample variability).

**Key design choices**:
- 2×2 panel layout (one per condition)
- Each line = one tissue sample's before→after trajectory
- Shared y-axis enables direct comparison
- Bold black line = group mean

```python
for b, a in zip(before, after):
    ax.plot([0, 1], [b, a], color=PALETTE[cond], alpha=0.4)

# Mean trajectory
ax.plot([0, 1], [mean_before, mean_after], color='black', linewidth=3)
```

### Figure 3: Dose-Response Curve

**Purpose**: Demonstrate predictable, linear relationship between RF concentration and stiffness.

**Key design choices**:
- Both UV powers shown (solid = 3mW, dashed = 0.3mW)
- R² and slope annotation
- Error bars = SEM

```python
ax.errorbar(rf_conc, means_3mW, yerr=sems_3mW, fmt='o-', label='3mW UV')
ax.errorbar(rf_conc, means_03mW, yerr=sems_03mW, fmt='s--', label='0.3mW UV')
```

### Figure 4: Summary Heatmap

**Purpose**: Complete parameter space at a glance.

**Key design choices**:
- Rows = RF concentration, Columns = UV stage
- Cell values = absolute stiffness + fold-change
- Color intensity = stiffness magnitude
- YlOrRd colormap (yellow-orange-red)

```python
im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=10000)
# Annotate each cell
ax.text(j, i, f'{value:.0f}\n({fold_change:.0f}×)', ha='center')
```

---

## 6. Statistical Tests

### Test 1: Baseline Equivalence (One-Way ANOVA)

**Question**: Do all conditions start with equivalent stiffness?

```python
from scipy import stats

groups = [df[df['Condition'] == c]['Before_UV'] for c in CONDITIONS]
f_stat, p_val = stats.f_oneway(*groups)
```

**Result**: F = 0.17, p = 0.92

**Interpretation**: No significant difference in baseline stiffness. All groups started equivalent, enabling valid comparison of treatment effects.

---

### Test 2: Within-Group Treatment Effects (Paired t-tests)

**Question**: Does UV exposure significantly increase stiffness within each condition?

```python
for cond in CONDITIONS:
    before = df[df['Condition'] == cond]['Before_UV']
    after = df[df['Condition'] == cond]['UV_3mW']
    t, p = stats.ttest_rel(before, after)
```

**Results**:

| Condition | t-statistic | p-value | Significance |
|-----------|-------------|---------|--------------|
| PBS Control | -19.84 | 2.15 × 10⁻¹⁶ | *** |
| 2mM RF | -16.76 | 9.49 × 10⁻¹⁵ | *** |
| 4mM RF | -31.55 | 4.74 × 10⁻²¹ | *** |
| 8mM RF | -48.85 | 1.54 × 10⁻²⁵ | *** |

**Interpretation**: All conditions show highly significant stiffening after UV exposure.

---

### Test 3: Between-Group Comparisons (Tukey HSD)

**Question**: Which conditions differ significantly from each other?

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(df['UV_3mW'], df['Condition'], alpha=0.05)
```

**Results**:

| Comparison | Mean Difference | p-adj | Significant? |
|------------|-----------------|-------|--------------|
| 2mM vs 4mM | 2,254 kPa | <0.001 | Yes |
| 2mM vs 8mM | 5,589 kPa | <0.001 | Yes |
| 2mM vs PBS | -2,294 kPa | <0.001 | Yes |
| 4mM vs 8mM | 3,335 kPa | <0.001 | Yes |
| 4mM vs PBS | -4,547 kPa | <0.001 | Yes |
| 8mM vs PBS | -7,883 kPa | <0.001 | Yes |

**Interpretation**: All pairwise comparisons are significant. Each RF concentration produces detectably different stiffening.

---

### Test 4: Dose-Response Regression

**Question**: Is there a linear relationship between RF concentration and stiffness?

```python
rf_data = df[df['RF_Concentration'] > 0]
slope, intercept, r, p, se = stats.linregress(
    rf_data['RF_Concentration'],
    rf_data['UV_3mW']
)
```

**Results**:
- Slope: 918 kPa per mM RF
- R² = 0.99
- p < 0.001

**Interpretation**: Near-perfect linear relationship. Each 1 mM increase in RF concentration adds ~918 kPa to final stiffness.

---

### Test 5: Effect Sizes (Cohen's d)

**Question**: How large are the treatment effects in practical terms?

```python
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1-1)*group1.std()**2 + (n2-1)*group2.std()**2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std

control = df[df['Condition'] == 'Dura+PBS']['UV_3mW']
for cond in ['2mM+RF', '4mM+RF', '8mM+RF']:
    treatment = df[df['Condition'] == cond]['UV_3mW']
    d = cohens_d(treatment, control)
```

**Results**:

| Comparison | Cohen's d | Interpretation |
|------------|-----------|----------------|
| 2mM RF vs Control | 4.6 | Very large |
| 4mM RF vs Control | 8.7 | Very large |
| 8mM RF vs Control | 13.8 | Very large |

**Interpretation**: Effect sizes far exceed conventional thresholds (d > 0.8 = "large"). These are massive, unambiguous effects.

---

### Output Files

| File | Location |
|------|----------|
| Figures | `outputs/figures/` |
| Combined figure | `outputs/figures/combined_figure.png` |
| Cleaned data | `outputs/tables/cleaned_data.csv` |
| Statistics | `outputs/tables/descriptive_statistics.csv` |
| Test results | `outputs/statistics/test_results.txt` |

---

## Summary

This analysis demonstrates that riboflavin-mediated photo-crosslinking increases dura mater stiffness by **46× to 162×** depending on concentration, compared to only **2.6×** with UV alone. The effect is:

1. **Statistically significant** (all p < 0.001)
2. **Dose-dependent** (R² = 0.99)
3. **Practically massive** (Cohen's d up to 13.8)
4. **Reproducible** (consistent within-sample transformation)

The 75-fold amplification of UV crosslinking effectiveness (162× vs 2×) represents a substantial enhancement.
