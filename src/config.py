"""
Style configuration for figures.
"""

import matplotlib.pyplot as plt

# Colors for each condition
COLORS = {
    'Dura+PBS': '#E41A1C',   # Red (Control)
    '2mM+RF': '#4DAF4A',     # Green
    '4mM+RF': '#984EA3',     # Purple
    '8mM+RF': '#377EB8',     # Blue
}

# Order of conditions for plotting
CONDITIONS = ['Dura+PBS', '2mM+RF', '4mM+RF', '8mM+RF']
LABELS = ['PBS (Control)', '2mM RF', '4mM RF', '8mM RF']


def setup_style():
    """Apply publication-ready matplotlib style."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })
