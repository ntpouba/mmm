"""
Visual counterpart of audio_envelope_corr.py: lag-correlate each subject's visual
timecourse (the Schaefer VisCent parcels) against the clip's motion energy
(motion_energy.py), for the first showing of each clip.

Clip selection, cropping, per-subject window alignment, the lag convention
(negative lag = video leads BOLD) and the plausibility checks are all
audio_envelope_corr.py's; only the ROI and the stimulus feature differ.

Usage:
    uv run python -m mmm.analyses.lag_correlation.motion_energy_corr          # every runnable clip
    uv run python -m mmm.analyses.lag_correlation.motion_energy_corr "Fool" "Big Take"
    uv run python -m mmm.analyses.lag_correlation.motion_energy_corr --exclude "Snack Attack"

REPEATED_MOVIES are excluded unless named explicitly; every showing of them is
plotted by motion_repeated_corr.py.
"""
import argparse
import sys

import pandas as pd

from mmm.utils.atlas import ORDER_TXT
from mmm.utils.layout import FIGURES, ORDER_CSV
from mmm.analyses.lag_correlation.motion_energy import get_motion_energy
from mmm.analyses.lag_correlation.audio_envelope_corr import (
    REPEATED_MOVIES, OVERVIEW_MAX_COLS, VARIANT_TAGS, compute_movie, discover_movies,
    get_roi_rows, print_skip_report, render_movie, save_overview, shared_ylim, split_evenly)

ROI_TAG = 'VisCent'
OUT_DIR = FIGURES / 'lag_correlation' / 'motion_energy_corr'
INDIVIDUAL_DIR = OUT_DIR / 'individual'
OVERVIEW_DIR = OUT_DIR / 'overview'


def render_overviews(computed, ylim):
    """All clips spread evenly over grids of at most OVERVIEW_MAX_COLS columns."""
    OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for old in OVERVIEW_DIR.glob('*.png'):
        old.unlink()

    for variant_tag in VARIANT_TAGS:
        movies = [(m, payload[variant_tag]) for m, payload, _ in computed if payload is not None]
        chunks = split_evenly(movies, OVERVIEW_MAX_COLS)
        for k, chunk in enumerate(chunks, 1):
            out = save_overview(f'clips-{k:02d}', f'Visual motion energy ({k}/{len(chunks)})',
                                chunk, variant_tag, ylim, out_dir=OVERVIEW_DIR, stimulus='video')
            print(f'overview -> {out}  ({len(chunk)} clips)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    parser.add_argument('movies', nargs='*',
                        help='clip names as they appear in movie_order.csv; '
                             'default is every clip that is currently runnable, '
                             'except REPEATED_MOVIES')
    parser.add_argument('--exclude', nargs='+', default=[], metavar='MOVIE',
                        help='clip names to leave out (matched case-insensitively)')
    args = parser.parse_args(argv)

    roi_rows = get_roi_rows(ORDER_TXT, ROI_TAG)
    print(f'Visual ROI: {len(roi_rows)} parcels, rows={roi_rows}')

    df = pd.read_csv(ORDER_CSV)

    excluded = {m.strip().casefold() for m in args.exclude}
    if not args.movies:
        excluded |= {m.casefold() for m in REPEATED_MOVIES}
    candidates = args.movies or discover_movies(df)
    movies = [m for m in candidates if m.strip().casefold() not in excluded]
    if not movies:
        print('\nno clips to run')
        return 1
    print(f'Clips to check ({len(movies)}): {", ".join(movies)}')

    skips = []
    computed = [(m,) + compute_movie(m, df, roi_rows, skips, get_features=get_motion_energy)
                for m in movies]

    payloads = [payload for _, payload, _ in computed if payload is not None]
    if not payloads:
        print('\nno clips could be run')
        print_skip_report(skips)
        return 1
    ylim = shared_ylim(payloads)
    print(f'\nShared correlation axis for all figures: {ylim[0]:.3f} to {ylim[1]:.3f}')

    for movie_name, payload, notes in computed:
        render_movie(movie_name, payload, notes, ylim, out_dir=INDIVIDUAL_DIR)
    print()
    render_overviews(computed, ylim)

    print_skip_report(skips)
    return 0


if __name__ == '__main__':
    sys.exit(main())
