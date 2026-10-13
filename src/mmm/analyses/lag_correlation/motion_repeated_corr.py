"""
Visual counterpart of audio_repeated_corr.py: visual-cortex motion-energy lag correlation
for every showing of The Bench and From Dad To Son.

One figure per clip, one panel per subject, one curve per showing coloured
light (ses-19) to dark (ses-28). Everything but the ROI and the stimulus
feature is as in audio_repeated_corr.py (negative lag = video leads BOLD).

Usage:
    uv run python -m mmm.analyses.lag_correlation.motion_repeated_corr
"""
import sys

import numpy as np
import pandas as pd

from mmm.utils.atlas import ORDER_TXT
from mmm.utils.layout import ORDER_CSV
from mmm.analyses.lag_correlation.motion_energy import get_motion_energy
from mmm.analyses.lag_correlation.audio_envelope_corr import REPEATED_MOVIES, VARIANT_TAGS, get_roi_rows
from mmm.analyses.lag_correlation.motion_energy_corr import OUT_DIR, ROI_TAG
from mmm.analyses.lag_correlation.audio_repeated_corr import compute_movie, plot_movie, print_summary

KIND = 'motion'


def main():
    df = pd.read_csv(ORDER_CSV)
    roi_rows = get_roi_rows(ORDER_TXT, ROI_TAG)
    skips = []

    computed = {}
    for movie_name in REPEATED_MOVIES:
        for variant_tag in VARIANT_TAGS:
            print(f'{movie_name} ({variant_tag})...')
            computed[movie_name, variant_tag] = compute_movie(
                movie_name, df, roi_rows, variant_tag, skips,
                get_features=get_motion_energy, kind=KIND)

    values = np.concatenate([c for curves in computed.values()
                             for items in curves.values() for _, _, c in items])
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    ylim = (lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))

    for (movie_name, variant_tag), curves in computed.items():
        sessions = sorted({s for items in curves.values() for s, _, _ in items})
        out = plot_movie(movie_name, curves, variant_tag, sessions, ylim, kind=KIND,
                         out_dir=OUT_DIR / 'repeated', stimulus='video')
        print_summary(movie_name, curves, sessions, kind=KIND)
        print(f'figure -> {out}')

    for movie_name, subject, reason in skips:
        print(f'[skip] {movie_name} / {subject}: {reason}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
