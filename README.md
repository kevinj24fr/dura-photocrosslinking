# Dura Mater Photo-Crosslinking Analysis

AFM analysis demonstrating that riboflavin dramatically amplifies UV-induced tissue stiffening.

## Key Finding

| Condition | Baseline | After 3mW UV | Fold Change |
|-----------|----------|--------------|-------------|
| PBS Control | 51 kPa | 123 kPa | **2×** |
| 2mM Riboflavin | 53 kPa | 2,417 kPa | **46×** |
| 4mM Riboflavin | 52 kPa | 4,670 kPa | **89×** |
| 8mM Riboflavin | 53 kPa | 8,006 kPa | **162×** |

Riboflavin increases UV crosslinking effectiveness by **75-fold** (162× vs 2×).

## Repository Structure

```
.
├── Data/
│   └── Dura measurement in AFM.xlsx
├── src/
│   ├── config.py              # Colors, labels, style
│   ├── data_processing.py     # Data loading and cleaning
│   └── analysis.py            # Main analysis script
├── outputs/
│   ├── figures/
│   │   ├── fig1_hero_stiffness.png
│   │   ├── fig2_paired_transformation.png
│   │   ├── fig3_dose_response.png
│   │   ├── fig4_summary_heatmap.png
│   │   └── combined_figure.png
│   ├── tables/
│   │   ├── cleaned_data.csv
│   │   └── descriptive_statistics.csv
│   ├── statistics/
│   │   └── test_results.txt
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python src/analysis.py
```

## Figures

**Combined Figure** (`outputs/figures/combined_figure.png`):
- **Panel A**: Final stiffness with fold-change labels (hero figure)
- **Panel B**: Paired before/after transformation (2×2 slopegraph)
- **Panel C**: Dose-response curves (0.3mW and 3mW UV)
- **Panel D**: Summary heatmap of parameter space

## Outputs

| File | Description |
|------|-------------|
| `combined_figure.png` | Publication-ready 4-panel figure |
| `report.html` | Full analysis report with figures |
| `figure_legend.md` | Scientific figure legend |
| `methods.md` | Methods section for manuscript |
| `test_results.txt` | Statistical test results |

## Statistical Summary

- **Baseline ANOVA**: F = 0.17, p = 0.92 (no difference)
- **Treatment effects**: All p < 0.001 (paired t-tests)
- **Dose-response**: R² = 0.99, slope = 918 kPa/mM
- **Effect sizes**: Cohen's d = 4.6 to 13.8 (massive effects)

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels
- openpyxl
