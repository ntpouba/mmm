"""
Lag correlation utilities, extracted by Claude from lag_corr_wiki.ipynb.
"""
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def lag_cor(vec1, vec2, max_lag=20):
    """
    Compute Pearson correlations between vec1 and vec2 at lags from -max_lag to +max_lag.
    Returns a list of length (2*max_lag + 1), with np.nan where windows are too short.
    """
    correlations = []
    n = min(len(vec1), len(vec2))
    lags = range(-max_lag, max_lag + 1)

    for lag in lags:
        if lag < 0:
            v1_window = vec1[-lag:n] # bugfix, original script missing n
            v2_window = vec2[:n + lag]
        elif lag > 0:
            v1_window = vec1[:n - lag]
            v2_window = vec2[lag:n]
        else:
            v1_window = vec1[:n]
            v2_window = vec2[:n]

        if len(v1_window) < 2 or len(v2_window) < 2:
            correlations.append(np.nan)
            continue

        try:
            r, _ = pearsonr(v1_window, v2_window)
            correlations.append(r)
        except Exception:
            correlations.append(np.nan)

    return correlations


# Distinct, colour-blind-safe hues so four overlaid curves stay separable.
_CURVE_COLORS = ['#0072b2', '#d55e00', '#009e73', '#cc79a7',
                 '#56b4e9', '#e69f00', '#000000']


def plot_lag_result(curves, max_lag=20, title="", ylim=None):
    """Plot one or more lag correlation curves on shared axes, peaks marked.

    `curves` maps a label to that curve's lag correlations, each of length
    2 * max_lag + 1. The legend carries each curve's peak lag and r, so the
    figure is readable without the printed summary.

    `ylim` fixes the correlation axis to a caller-supplied (lo, hi). Pass the
    same range to every figure in a set so they can be compared side by side;
    leave it None to autoscale each figure independently.
    """
    lags = np.arange(-max_lag, max_lag + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for i, (label, lag_corrs) in enumerate(curves.items()):
        lag_corrs = np.array(lag_corrs, dtype=float)
        color = _CURVE_COLORS[i % len(_CURVE_COLORS)]

        if np.all(np.isnan(lag_corrs)):
            ax.plot([], [], color=color, label=f"{label}: all NaN")
            continue

        peak_idx = np.nanargmax(lag_corrs)
        peak_lag = lags[peak_idx]
        peak_corr = lag_corrs[peak_idx]
        ax.plot(lags, lag_corrs, "-o", color=color, markersize=3, linewidth=1.4,
                label=f"{label}: {peak_lag:+d} TR (r={peak_corr:.2f})")
        ax.plot(peak_lag, peak_corr, "o", color=color, markersize=9,
                markeredgecolor="k", markeredgewidth=0.8, zorder=5)

    ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title or "Lag Correlation", fontsize=10)
    ax.set_xlabel("Lag (TR)")
    ax.set_ylabel("Correlation")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
