import argparse
import json
import sys

import matplotlib
matplotlib.use('Agg')  # must precede any pyplot import, including inside lag_corr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd
from pathlib import Path

from mmm.analyses.lag_correlation.lag_corr import _CURVE_COLORS, lag_cor, plot_lag_result
from mmm.analyses.lag_correlation.audio_envelope import TR, get_audio_envelopes, stimulus_status

from mmm.utils.atlas import ORDER_TXT
from mmm.utils.layout import FIGURES, ORDER_CSV, Run, iter_runs, mni_bold, parcels

VARIANT_TAGS = ['basecsfwm']
# Shown in every session; every showing is plotted by audio_repeated_corr.py,
# so they are left out of this script's default set of clips.
REPEATED_MOVIES = ['The Bench', 'From Dad To Son']
RUNS = ['run-01', 'run-02']  # searched per subject; whichever run holds the clip is used
SUBJECTS = sorted({r.sub for r in iter_runs(task='NATencoding')})

MAX_LAG = 20     # TRs (+-30s)

# lag_cor(vec1, vec2, lag) convention (see lag_corr.py): for lag > 0 it pairs
# vec1[i] with vec2[i+lag], i.e. vec1 LEADS. We call lag_cor(aud_tc, envelope, ...),
# so a NEGATIVE lag means the envelope (audio, vec2) leads aud_tc (BOLD, vec1) --
# that's the physiologically expected direction (stimulus first, hemodynamic
# response ~3-7.5s / 2-5 TR later).
PLAUSIBLE_LAG_RANGE = (-5, -2)   # TRs: audio leads BOLD by 3-7.5s
R_THRESHOLD = 0.1

# This script writes OUT_DIR/individual/<movie>/<subject>.png and
# OUT_DIR/overview/{no-speech,speech-NN}.png (subjects x clips);
# audio_repeated_corr.py writes OUT_DIR/repeated/.
# The variant is not in the path (only one is run, see VARIANT_TAGS); it is in
# each figure's title.
OUT_DIR = FIGURES / 'lag_correlation' / 'audio_envelope_corr'
INDIVIDUAL_DIR = OUT_DIR / 'individual'
OVERVIEW_DIR = OUT_DIR / 'overview'
OVERVIEW_MAX_COLS = 7  # speech clips are spread evenly over as few grids as this allows


def check_csv_internal_consistency(csv_path):
    print('=== CSV internal consistency ===')
    df = pd.read_csv(csv_path)
    failures = 0
    for (sub, ses, run), g in df.groupby(['subject', 'session', 'run']):
        g = g.sort_values('order')
        if not bool(g['title_onset_s'].is_monotonic_increasing):
            failures += 1
            print(f'[FAIL] {sub} {ses} {run}: clip onsets are not monotonic')
    print(f'{"all runs OK" if not failures else f"{failures} run(s) FAILED"} '
          f'({df.groupby(["subject", "session", "run"]).ngroups} runs checked)')
    print()


def get_roi_rows(order_txt_path, tag):
    rows = []
    for line in order_txt_path.read_text().splitlines():
        idx_str, name = line.split('\t')[:2]
        if tag in name:
            rows.append(int(idx_str) - 1)
    return sorted(rows)


def slugify(name):
    return name.strip().lower().replace(' ', '-')


def parcel_path(subject, session, run, variant_tag):
    return parcels(Run(subject, session, 'NATencoding', run), variant_tag)


def movie_mask(df, movie_name):
    return df.movie_name.str.strip().str.casefold() == movie_name.strip().casefold()


def canonical_names(df):
    """key -> display name, one entry per clip, matched case-insensitively.
    Where the CSV disagrees with itself, the most common spelling wins."""
    names = df.movie_name.str.strip()
    return {key: group.mode().iloc[0]
            for key, group in names.groupby(names.str.casefold())}


def first_session(df, movie_name):
    """The earliest session in which this clip was shown, or None if unknown.

    Repeated clips are always taken from their first showing; see the module
    docstring for why a later showing is never substituted.
    """
    shown = df[movie_mask(df, movie_name)]
    return None if shown.empty else sorted(shown.session.unique())[0]


def get_movie_rows(df, movie_name, session):
    """subject -> its movie_order.csv row for this clip in `session`, for
    whichever of SUBJECTS watched it (may be a subset). Each row carries its own
    run, onset and duration, so subjects who saw the clip in different runs of
    that session are handled transparently."""
    rows = {}
    for subject in SUBJECTS:
        matches = df[
            (df.subject == subject) & (df.session == session)
            & (df.run.isin(RUNS)) & movie_mask(df, movie_name)
        ]
        if not matches.empty:
            # A subject sees a clip at most once per session; if that ever
            # breaks, take the earlier run and let the skip report show it.
            rows[subject] = matches.sort_values('run').iloc[0]
    return rows


def subject_preflight(subject, session, row):
    """(ok, reason): is this subject's parcellated data usable for this clip?

    Checked before any correlating so that a missing or mis-shaped file drops
    one subject with an explanation instead of aborting the whole run. Reads
    only the .npy header, not the array.
    """
    missing = [v for v in VARIANT_TAGS
               if not parcel_path(subject, session, row.run, v).exists()]
    if missing:
        return False, f'no parcellated data for {session} {row.run} ({", ".join(missing)})'

    shape = np.load(parcel_path(subject, session, row.run, VARIANT_TAGS[0]),
                    mmap_mode='r').shape
    if shape[1] != int(row.run_n_volumes):
        return False, (f'{session} {row.run} TR-count mismatch vs CSV '
                       f'({shape[1]} != {int(row.run_n_volumes)})')
    return True, None


def crop_onset_idx(onset_tr):
    return max(round(onset_tr), 0)


def crop_window(combined, onset_tr, duration_tr):
    onset_idx = crop_onset_idx(onset_tr)
    n_trs = round(duration_tr)
    end_idx = min(onset_idx + n_trs, combined.shape[1])
    return combined[:, onset_idx:end_idx]


def slice_time_reference(subject, session, run):
    """Seconds after its trigger that a volume represents after fMRIPrep's
    slice-timing correction (StartTime in the BOLD sidecar). The sidecar is
    shared by every output space, so the MNI one stands in for fsaverage6."""
    sidecar = Path(str(mni_bold(Run(subject, session, 'NATencoding', run)))
                   .replace('.nii.gz', '.json'))
    meta = json.loads(sidecar.read_text())
    if not meta.get('SliceTimingCorrected'):
        raise ValueError(f'{sidecar.name}: not slice-timing corrected')
    return float(meta['StartTime'])


def scan_start_in_clip(subject, session, row):
    """Where the first cropped volume sits on the clip's audio clock, in seconds."""
    volume_time = crop_onset_idx(row.movie_onset_TR) * TR
    return (volume_time + slice_time_reference(subject, session, row.run)
            - row.movie_onset_s)


def compute_one(subject, session, variant_tag, row, features, roi_rows):
    combined = np.load(parcel_path(subject, session, row.run, variant_tag))

    window = crop_window(combined, row.movie_onset_TR, row.movie_duration_TR)
    roi_tc = window[roi_rows, :].mean(axis=0)

    curves, peaks = {}, {}
    for kind, feature in features.items():
        n = min(len(roi_tc), len(feature))

        lc = np.array(lag_cor(roi_tc[:n], feature[:n], max_lag=MAX_LAG), dtype=float)
        curves[kind] = lc

        if np.all(np.isnan(lc)):
            peaks[kind] = {'peak_lag': None, 'peak_r': np.nan}
        else:
            peak_idx = np.nanargmax(lc)
            peaks[kind] = {
                'peak_lag': int(np.arange(-MAX_LAG, MAX_LAG + 1)[peak_idx]),
                'peak_r': float(lc[peak_idx]),
            }

    return {'curves': curves, 'peaks': peaks, 'run': row.run, 'session': session}


def save_figure(subject, variant_tag, res, movie_name, ylim, out_dir=INDIVIDUAL_DIR):
    plot_lag_result(
        res['curves'], max_lag=MAX_LAG,
        title=f'{subject} {res["session"]} {res["run"]} {variant_tag} ({movie_name})',
        ylim=ylim,
    )
    fig = plt.gcf()
    out_dir = out_dir / slugify(movie_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f'{subject}.png', dpi=150)
    plt.close(fig)


def shared_ylim(payloads, margin=0.05):
    """One correlation range covering every curve in every figure, so the whole
    set is comparable side by side."""
    values = np.concatenate([
        curve
        for payload in payloads
        for results in payload.values()
        for res in results.values()
        for curve in res['curves'].values()
    ])
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    pad = (hi - lo) * margin
    return lo - pad, hi + pad


def is_plausible(peak):
    """A peak counts as sane when the audio leads the BOLD by a hemodynamically
    believable amount and the correlation clears the noise threshold."""
    lag, r = peak['peak_lag'], peak['peak_r']
    return (
        lag is not None
        and PLAUSIBLE_LAG_RANGE[0] <= lag <= PLAUSIBLE_LAG_RANGE[1]
        and r > R_THRESHOLD
    )


def format_peak(peak):
    lag, r = peak['peak_lag'], peak['peak_r']
    if lag is None:
        return 'n/a'
    # trailing * marks a cell that satisfies both plausibility criteria
    return f'{lag:+d}TR r={r:.3f}{"*" if is_plausible(peak) else " "}'


def print_summary(variant_tag, results, movie_name, out_dir=INDIVIDUAL_DIR):
    kinds = list(next(iter(results.values()))['curves'])
    print(f'--- Variant: {variant_tag} ---')

    print('=== Check 1: pipeline sanity (* = plausible lag and r above threshold) ===')
    print(f'{"subject":10s}{"session":10s}{"run":9s}'
          + ''.join(f'{k:>16s}' for k in kinds))
    for sub, res in results.items():
        cells = ''.join(f'{format_peak(res["peaks"][k]):>16s}' for k in kinds)
        print(f'{sub:10s}{res["session"]:10s}{res["run"]:9s}{cells}')
    print()

    print('=== Check 2: cross-subject coherence ===')
    if len(results) < 2:
        print('only one subject has usable data for this clip -> skipped')
    else:
        for kind in kinds:
            lags = [res['peaks'][kind]['peak_lag'] for res in results.values()]
            if any(lag is None for lag in lags):
                print(f'{kind:>10s}: a subject has no usable peak -> skipped')
                continue
            spread = max(lags) - min(lags)
            print(f'{kind:>10s}: peak lags {lags} (spread={spread} TR) '
                  f'-> {"PASS" if spread <= 2 else "CHECK"} (coherent)')
    print()
    print(f'Figures saved under {out_dir / slugify(movie_name)}/<subject>.png')


def resolve_movie(df, movie_name):
    """(session, rows, reason): where to read this clip from, per subject.

    rows is {subject: csv_row} restricted to subjects whose parcellated data
    passes preflight. reason is set instead when the clip cannot be run at all.
    Also returns per-subject drop reasons for the skip report.
    """
    session = first_session(df, movie_name)
    if session is None:
        return None, {}, f'not listed in {ORDER_CSV}', []

    _, stim_reason = stimulus_status(movie_name)
    if stim_reason is not None:
        return session, {}, stim_reason, []

    rows = get_movie_rows(df, movie_name, session)
    if not rows:
        return session, {}, f'none of {SUBJECTS} watched it in {session}', []

    usable, dropped = {}, []
    for subject, row in rows.items():
        ok, reason = subject_preflight(subject, session, row)
        if ok:
            usable[subject] = row
        else:
            dropped.append((subject, reason))

    if not usable:
        return session, {}, f'no subject has parcellated data for {session}', dropped
    return session, usable, None, dropped


def discover_movies(df):
    """Every clip that is runnable right now, ordered by first session.

    Used when no clips are named on the command line, so the script tracks
    whatever has been cleaned without needing to be edited.
    """
    names = sorted(canonical_names(df).values(),
                   key=lambda m: (first_session(df, m), m))
    return [m for m in names if resolve_movie(df, m)[2] is None]


def compute_movie(movie_name, df, roi_rows, skips, get_features=get_audio_envelopes):
    """Every lag curve for one clip: {variant_tag: {subject: result}}.

    Returns (payload, notes). payload is None when the clip cannot be run;
    anything skipped is appended to `skips` for the end-of-run report.
    """
    session, rows, reason, dropped = resolve_movie(df, movie_name)
    for subject, drop_reason in dropped:
        skips.append((movie_name, subject, drop_reason))

    if reason is not None:
        skips.append((movie_name, None, reason))
        return None, [f'[skip] {reason}']

    try:
        features = {
            subject: get_features(movie_name, scan_start_in_clip(subject, session, row))
            for subject, row in rows.items()
        }
    except (FileNotFoundError, ValueError) as e:
        skips.append((movie_name, None, str(e)))
        return None, [f'[skip] {e}']

    notes = [f'  first shown {session}; '
             + ', '.join(f'{s}: {r.run}' for s, r in rows.items())]
    missing = [s for s in SUBJECTS if s not in rows]
    if missing:
        notes.append(f'  (running without {missing})')

    payload = {
        variant_tag: {
            subject: compute_one(subject, session, variant_tag, row, features[subject], roi_rows)
            for subject, row in rows.items()
        }
        for variant_tag in VARIANT_TAGS
    }
    return payload, notes


def render_movie(movie_name, payload, notes, ylim, out_dir=INDIVIDUAL_DIR):
    print(f'\n{"=" * 60}\nMovie: {movie_name}\n{"=" * 60}')
    for note in notes:
        print(note)
    if payload is None:
        return

    for variant_tag, results in payload.items():
        for subject, res in results.items():
            save_figure(subject, variant_tag, res, movie_name, ylim, out_dir)
        print_summary(variant_tag, results, movie_name, out_dir)


def save_overview(name, label, movies, variant_tag, ylim, out_dir=OVERVIEW_DIR, stimulus='audio'):
    """One grid: rows = SUBJECTS, columns = clips, each panel the same curves
    as the individual figure. `movies` is [(movie_name, {subject: result})]."""
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    kinds = list(next(r for _, results in movies for r in results.values())['curves'])
    nrow, ncol = len(SUBJECTS), len(movies)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.3 * nrow),
                             sharex=True, sharey=True, squeeze=False,
                             constrained_layout=True)

    for j, (movie_name, results) in enumerate(movies):
        axes[0, j].set_title(movie_name, fontsize=9)
        for i, subject in enumerate(SUBJECTS):
            ax = axes[i, j]
            ax.axvline(0, color='k', linestyle='--', linewidth=0.6)
            ax.grid(True, alpha=0.4)
            ax.tick_params(labelsize=7)
            res = results.get(subject)
            if res is None:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, color='0.5')
                continue
            labels = []
            for k, kind in enumerate(kinds):
                color = _CURVE_COLORS[k % len(_CURVE_COLORS)]
                ax.plot(lags, res['curves'][kind], color=color, linewidth=1)
                peak = res['peaks'][kind]
                if peak['peak_lag'] is not None:
                    ax.plot(peak['peak_lag'], peak['peak_r'], 'o', color=color,
                            markersize=4, markeredgecolor='k', markeredgewidth=0.5)
                    labels.append((f'{peak["peak_lag"]:+d} r={peak["peak_r"]:.2f}', color))
            for k, (text, color) in enumerate(labels):
                ax.text(0.98, 0.97 - 0.11 * k, text, transform=ax.transAxes,
                        ha='right', va='top', fontsize=7, color=color)

    for i, subject in enumerate(SUBJECTS):
        axes[i, 0].set_ylabel(subject, fontsize=9)
    for j in range(ncol):
        axes[-1, j].set_xlabel('Lag (TR)', fontsize=8)
    axes[0, 0].set_ylim(*ylim)

    handles = [Line2D([], [], color=_CURVE_COLORS[k % len(_CURVE_COLORS)], linewidth=1.5)
               for k in range(len(kinds))]
    fig.legend(handles, kinds, loc='outside lower center',
               ncol=len(kinds), fontsize=8, frameon=False)
    fig.suptitle(f'{label}: first showing of each clip, {variant_tag} '
                 f'(negative lag = {stimulus} leads BOLD)', fontsize=10)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'{name}.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def has_speech(df, movie_name):
    """False for clips movie_order.csv labels as having no, almost no or very
    little speech. Blank style cells are ignored (one The Bench row has one)."""
    styles = df.loc[movie_mask(df, movie_name), 'style'].dropna().str.casefold()
    return not styles.str.contains(r'no speech|very little').any()


def split_evenly(items, max_cols):
    """Fewest chunks of at most max_cols, sizes differing by at most one."""
    n_chunks = -(-len(items) // max_cols)
    base, extra = divmod(len(items), n_chunks)
    chunks, i = [], 0
    for k in range(n_chunks):
        size = base + (k < extra)
        chunks.append(items[i:i + size])
        i += size
    return chunks


def render_overviews(computed, ylim, df):
    """One grid of every no-speech clip, then the speech clips spread evenly
    over grids of at most OVERVIEW_MAX_COLS columns. Clips keep the order they
    were computed in (first session, then name)."""
    OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for old in OVERVIEW_DIR.glob('*.png'):  # grid count changes as clips are added
        old.unlink()

    for variant_tag in VARIANT_TAGS:
        movies = [(m, payload[variant_tag]) for m, payload, _ in computed if payload is not None]
        silent = [mv for mv in movies if not has_speech(df, mv[0])]
        speech = [mv for mv in movies if has_speech(df, mv[0])]

        grids = []
        if silent:
            grids.append(('no-speech', 'No / minimal speech', silent))
        chunks = split_evenly(speech, OVERVIEW_MAX_COLS) if speech else []
        for k, chunk in enumerate(chunks, 1):
            grids.append((f'speech-{k:02d}', f'Speech ({k}/{len(chunks)})', chunk))

        for name, label, grid_movies in grids:
            out = save_overview(name, label, grid_movies, variant_tag, ylim)
            print(f'overview -> {out}  ({len(grid_movies)} clips)')


def print_skip_report(skips):
    if not skips:
        return
    print(f'\n{"=" * 60}\nSkipped\n{"=" * 60}')
    for movie_name, subject, reason in skips:
        who = f'{movie_name} / {subject}' if subject else movie_name
        print(f'  {who}: {reason}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    parser.add_argument('movies', nargs='*',
                        help='clip names as they appear in movie_order.csv; '
                             'default is every clip that is currently runnable, '
                             'except REPEATED_MOVIES')
    parser.add_argument('--exclude', nargs='+', default=[], metavar='MOVIE',
                        help='clip names to leave out (matched case-insensitively)')
    args = parser.parse_args(argv)

    check_csv_internal_consistency(ORDER_CSV)

    aud_rows = get_roi_rows(ORDER_TXT, 'Aud')
    print(f'Auditory ROI: {len(aud_rows)} parcels, rows={aud_rows}')

    df = pd.read_csv(ORDER_CSV)

    excluded = {m.strip().casefold() for m in args.exclude}
    if not args.movies:
        excluded |= {m.casefold() for m in REPEATED_MOVIES}
    candidates = args.movies or discover_movies(df)
    movies = [m for m in candidates if m.strip().casefold() not in excluded]
    left_out = [m for m in candidates if m.strip().casefold() in excluded]
    if left_out:
        print(f'Excluded ({len(left_out)}): {", ".join(left_out)}')
    if not movies and left_out:
        print('\nevery clip was excluded')
        return 1
    if not movies:
        print('\nno runnable clips found -- need a stimulus file in data/stimuli/movies/ '
              'and parcellated data for the clip\'s first session')
        return 1
    print(f'Clips to check ({len(movies)}): {", ".join(movies)}')

    # Everything is correlated up front so that one correlation range can be
    # shared by every figure; only then is anything drawn.
    skips = []
    computed = [(m,) + compute_movie(m, df, aud_rows, skips) for m in movies]

    payloads = [payload for _, payload, _ in computed if payload is not None]
    if not payloads:
        print('\nno clips could be run')
        print_skip_report(skips)
        return 1
    ylim = shared_ylim(payloads)
    print(f'\nShared correlation axis for all figures: '
          f'{ylim[0]:.3f} to {ylim[1]:.3f}')

    for movie_name, payload, notes in computed:
        render_movie(movie_name, payload, notes, ylim)
    print()
    render_overviews(computed, ylim, df)

    print_skip_report(skips)
    return 0


if __name__ == '__main__':
    sys.exit(main())
