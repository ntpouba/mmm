"""
Sanity check using individual movie clips cropped out of the encoding runs
(data/stimuli/<slug>*): for each requested movie, does the audio envelope ->
lag-correlation pipeline work against real stimulus audio and real
auditory-cortex BOLD?

Crop each subject's auditory-ROI timecourse to the onset/duration window
movie_order.csv declares for the clip and lag-correlate it against the clip's
own audio envelope. The CSV's clip mapping is taken as correct.

Four envelope definitions are compared at once (audio_envelope.ENVELOPE_KINDS),
the 2x2 of frequency weighting x amplitude compression:
  linear   Hilbert magnitude, as-is
  log      log10 of the above -- loudness is perceived on a ratio scale
  aweight  Hilbert magnitude of the A-weighted waveform (IEC 61672), which
           discards the low/high frequency energy the cochlea barely responds to
  dbA      both, i.e. the standard psychoacoustic loudness quantity
All four appear on every figure so that a difference can be attributed to the
weighting, to the compression, or to their combination. Weighting matters most
for clips whose energy is dominated by something other than speech.

Which session a clip is taken from is resolved from movie_order.csv rather than
hardcoded: a clip is always read from the FIRST session in which it was shown.
Two clips (The Bench, From Dad To Son) are shown ~10 times each across
ses-19..ses-28 while the other 59 are one-shot, so pinning to the first showing
keeps every number an exposure-matched first viewing. All three subjects happen
to first see each clip in the same session, so this also keeps the
cross-subject check comparing like with like. A clip whose first session has
not been confound-regressed + parcellated yet is skipped rather than falling
back to a later showing, which would silently compare a first viewing against a
seventh.

Within that session, the run is looked up per subject -- the same clip can sit
in run-01 for one subject and run-02 for another.

Usage:
    python audio_envelope_corr.py                    # every runnable clip
    python audio_envelope_corr.py "The Bench" "Big Take"

With no arguments the script discovers every clip that has both a stimulus file
and a parcellated first session. Named clips that cannot be run are reported in
a skip summary at the end, together with any subject dropped for missing or
malformed parcellated data.
"""
import argparse
import sys

import matplotlib
matplotlib.use('Agg')  # must precede any pyplot import, including inside lag_corr
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pathlib import Path

from lag_corr import lag_cor, plot_lag_result
from audio_envelope import ENVELOPE_KINDS, get_audio_envelopes, stimulus_status

from confounds import VARIANT_TAGS
# RUNS are searched per subject; whichever run holds the clip is used
from layout import RUNS, SUBJECTS, Run, parcels

MAX_LAG = 20     # TRs (+-30s)

# lag_cor(vec1, vec2, lag) convention (see lag_corr.py): for lag > 0 it pairs
# vec1[i] with vec2[i+lag], i.e. vec1 LEADS. We call lag_cor(aud_tc, envelope, ...),
# so a NEGATIVE lag means the envelope (audio, vec2) leads aud_tc (BOLD, vec1) --
# that's the physiologically expected direction (stimulus first, hemodynamic
# response ~3-7.5s / 2-5 TR later).
PLAUSIBLE_LAG_RANGE = (-5, -2)   # TRs: audio leads BOLD by 3-7.5s
R_THRESHOLD = 0.1

ORDER_CSV = Path('movie_order.csv')
ORDER_TXT = Path('standard/Schaefer2018_400Parcels_17Networks_order.txt')
# Nested <movie>/<subject>/<variant>.png -- movie and subject are the axes you
# browse by; variant is the leaf file.
OUT_DIR = Path('scratch/audio_envelope_corr')


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


def get_auditory_rows(order_txt_path):
    """0-indexed combined-matrix rows for every parcel whose name contains 'Aud'."""
    rows = []
    for line in order_txt_path.read_text().splitlines():
        idx_str, name = line.split('\t')[:2]
        if 'Aud' in name:
            rows.append(int(idx_str) - 1)
    return sorted(rows)


def slugify(name):
    return name.strip().lower().replace(' ', '-')


def parcel_path(subject, session, run, variant_tag):
    return parcels(Run(subject, session, run), variant_tag)


def movie_mask(df, movie_name):
    """Case-insensitive match against movie_order.csv's movie_name.

    The CSV is not self-consistent about capitalisation: 'From Dad To Son' and
    'From Dad to Son' are the same clip, the latter being how its ses-22 rows
    happen to be spelled. An exact match would split one clip into two and
    resolve the wrong first session -- ses-22 instead of ses-19, i.e. a fourth
    viewing presented as a first.
    """
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


def crop_window(combined, onset_tr, duration_tr):
    onset_idx = max(round(onset_tr), 0)
    n_trs = round(duration_tr)
    end_idx = min(onset_idx + n_trs, combined.shape[1])
    return combined[:, onset_idx:end_idx]


def compute_one(subject, session, variant_tag, row, envelopes, aud_rows):
    """Lag-correlate one subject's auditory timecourse against every envelope
    kind. Returns the curves themselves plus each kind's peak; plotting is
    deferred so that every figure can share one correlation axis. The data is
    assumed to have passed subject_preflight."""
    combined = np.load(parcel_path(subject, session, row.run, variant_tag))

    window = crop_window(combined, row.movie_onset_TR, row.movie_duration_TR)
    aud_tc = window[aud_rows, :].mean(axis=0)

    curves, peaks = {}, {}
    for kind in ENVELOPE_KINDS:
        envelope = envelopes[kind]
        n = min(len(aud_tc), len(envelope))

        lc = np.array(lag_cor(aud_tc[:n], envelope[:n], max_lag=MAX_LAG), dtype=float)
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


def save_figure(subject, variant_tag, res, movie_name, ylim):
    plot_lag_result(
        res['curves'], max_lag=MAX_LAG,
        title=f'{subject} {res["session"]} {res["run"]} {variant_tag} ({movie_name})',
        ylim=ylim,
    )
    fig = plt.gcf()
    out_subdir = OUT_DIR / slugify(movie_name) / subject
    out_subdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_subdir / f'{variant_tag}.png', dpi=150)
    plt.close(fig)


def shared_ylim(payloads, margin=0.05):
    """One correlation range covering every curve in every figure, so the whole
    set is comparable side by side."""
    values = np.concatenate([
        res['curves'][kind]
        for payload in payloads
        for results in payload.values()
        for res in results.values()
        for kind in ENVELOPE_KINDS
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


def print_summary(variant_tag, results, movie_name):
    print(f'--- Variant: {variant_tag} ---')

    print('=== Check 1: pipeline sanity (* = plausible lag and r above threshold) ===')
    print(f'{"subject":10s}{"session":10s}{"run":9s}'
          + ''.join(f'{k:>16s}' for k in ENVELOPE_KINDS))
    for sub, res in results.items():
        cells = ''.join(f'{format_peak(res["peaks"][k]):>16s}' for k in ENVELOPE_KINDS)
        print(f'{sub:10s}{res["session"]:10s}{res["run"]:9s}{cells}')
    print()

    print('=== Check 2: cross-subject coherence ===')
    if len(results) < 2:
        print('only one subject has usable data for this clip -> skipped')
    else:
        for kind in ENVELOPE_KINDS:
            lags = [res['peaks'][kind]['peak_lag'] for res in results.values()]
            if any(lag is None for lag in lags):
                print(f'{kind:>10s}: a subject has no usable peak -> skipped')
                continue
            spread = max(lags) - min(lags)
            print(f'{kind:>10s}: peak lags {lags} (spread={spread} TR) '
                  f'-> {"PASS" if spread <= 2 else "CHECK"} (coherent)')
    print()
    print(f'Figures saved under {OUT_DIR / slugify(movie_name)}/<subject>/{variant_tag}.png')


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


def compute_movie(movie_name, df, aud_rows, skips):
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
        envelopes = get_audio_envelopes(movie_name)
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
            subject: compute_one(subject, session, variant_tag, row, envelopes, aud_rows)
            for subject, row in rows.items()
        }
        for variant_tag in VARIANT_TAGS
    }
    return payload, notes


def render_movie(movie_name, payload, notes, ylim):
    print(f'\n{"=" * 60}\nMovie: {movie_name}\n{"=" * 60}')
    for note in notes:
        print(note)
    if payload is None:
        return

    for variant_tag, results in payload.items():
        for subject, res in results.items():
            save_figure(subject, variant_tag, res, movie_name, ylim)
        print_summary(variant_tag, results, movie_name)


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
                             'default is every clip that is currently runnable')
    args = parser.parse_args(argv)

    check_csv_internal_consistency(ORDER_CSV)

    aud_rows = get_auditory_rows(ORDER_TXT)
    print(f'Auditory ROI: {len(aud_rows)} parcels, rows={aud_rows}')

    df = pd.read_csv(ORDER_CSV)

    movies = args.movies or discover_movies(df)
    if not movies:
        print('\nno runnable clips found -- need a stimulus file in data/stimuli/ '
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

    print_skip_report(skips)
    return 0


if __name__ == '__main__':
    sys.exit(main())
