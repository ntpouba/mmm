"""
Audio-loudness lag correlation for the two clips shown in every session (The
Bench, From Dad To Son), for EVERY showing rather than only the first.

One figure per clip, one panel per subject, one dBA curve per showing coloured
light (ses-19) to dark (ses-28), so a change across repeats reads as a colour
gradient. Cropping, envelope alignment and the lag convention are exactly as in
audio_envelope_corr.py (negative lag = audio leads BOLD).

Usage:
    uv run python -m mmm.analyses.lag_correlation.audio_repeated_corr
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd

from mmm.utils.atlas import ORDER_TXT
from mmm.utils.layout import ORDER_CSV
from mmm.analyses.lag_correlation.audio_envelope import get_audio_envelopes
from mmm.analyses.lag_correlation.audio_envelope_corr import (
    MAX_LAG, OUT_DIR, REPEATED_MOVIES, RUNS, SUBJECTS, VARIANT_TAGS, compute_one,
    get_roi_rows, movie_mask, scan_start_in_clip, slugify, subject_preflight)

KIND = 'dBA'
CMAP = plt.get_cmap('viridis_r')


def showings(df, movie_name):
    """subject -> [csv rows of every showing, in session order]."""
    rows = df[movie_mask(df, movie_name) & df.run.isin(RUNS) & df.subject.isin(SUBJECTS)]
    return {sub: [r for _, r in g.sort_values(['session', 'run']).iterrows()]
            for sub, g in rows.groupby('subject')}


def compute_movie(movie_name, df, roi_rows, variant_tag, skips,
                  get_features=get_audio_envelopes, kind=KIND):
    """subject -> [(session, run, curve)] for every usable showing."""
    curves = {}
    for subject, rows in showings(df, movie_name).items():
        for row in rows:
            ok, reason = subject_preflight(subject, row.session, row)
            if not ok:
                skips.append((movie_name, subject, reason))
                continue
            features = get_features(movie_name, scan_start_in_clip(subject, row.session, row))
            res = compute_one(subject, row.session, variant_tag, row, features, roi_rows)
            curves.setdefault(subject, []).append((row.session, row.run, res['curves'][kind]))
    return curves


def plot_movie(movie_name, curves, variant_tag, sessions, ylim,
               kind=KIND, out_dir=OUT_DIR / 'repeated', stimulus='audio'):
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    colour = {ses: CMAP(0.1 + 0.85 * i / max(len(sessions) - 1, 1))
              for i, ses in enumerate(sessions)}

    fig, axes = plt.subplots(1, len(SUBJECTS), figsize=(5 * len(SUBJECTS), 4.2),
                             sharey=True, constrained_layout=True)
    for ax, subject in zip(np.atleast_1d(axes), SUBJECTS):
        for session, _, curve in curves.get(subject, []):
            ax.plot(lags, curve, color=colour[session], linewidth=1.2)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.axhline(0, color='0.6', linewidth=0.6)
        ax.set_title(f'{subject}  (n = {len(curves.get(subject, []))})', fontsize=10)
        ax.set_xlabel(f'Lag (TR)   negative = {stimulus} leads BOLD')
        ax.grid(True, alpha=0.4)
    np.atleast_1d(axes)[0].set_ylabel(f'Correlation ({kind})')
    np.atleast_1d(axes)[0].set_ylim(*ylim)

    handles = [Line2D([], [], color=colour[s], linewidth=2) for s in sessions]
    fig.legend(handles, sessions, loc='outside right center', fontsize=8, frameon=False,
               title='session')
    fig.suptitle(f'{movie_name}: every showing, {kind}, {variant_tag}', fontsize=11)

    out = out_dir / f'{slugify(movie_name)}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def print_summary(movie_name, curves, sessions, kind=KIND):
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    print(f'\n=== {movie_name}: {kind} peak lag (TR) / r per showing ===')
    print(f'{"session":10s}' + ''.join(f'{s:>16s}' for s in SUBJECTS))
    by_session = {s: {} for s in sessions}
    for subject, items in curves.items():
        for session, run, curve in items:
            i = int(np.nanargmax(curve))
            by_session[session][subject] = f'{run[-2:]} {lags[i]:+d} r={curve[i]:.2f}'
    for session in sessions:
        print(f'{session:10s}' + ''.join(f'{by_session[session].get(s, "-"):>16s}'
                                         for s in SUBJECTS))


def main():
    df = pd.read_csv(ORDER_CSV)
    aud_rows = get_roi_rows(ORDER_TXT, 'Aud')
    skips = []

    computed = {}
    for movie_name in REPEATED_MOVIES:
        for variant_tag in VARIANT_TAGS:
            print(f'{movie_name} ({variant_tag})...')
            computed[movie_name, variant_tag] = compute_movie(movie_name, df, aud_rows,
                                                              variant_tag, skips)

    # one correlation axis for every figure, so the two clips are comparable
    values = np.concatenate([c for curves in computed.values()
                             for items in curves.values() for _, _, c in items])
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    ylim = (lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))

    for (movie_name, variant_tag), curves in computed.items():
        sessions = sorted({s for items in curves.values() for s, _, _ in items})
        out = plot_movie(movie_name, curves, variant_tag, sessions, ylim)
        print_summary(movie_name, curves, sessions)
        print(f'figure -> {out}')

    for movie_name, subject, reason in skips:
        print(f'[skip] {movie_name} / {subject}: {reason}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
