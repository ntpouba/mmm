import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import zscore
from audio_envelope import get_audio_envelope
from subj_info import get_subj_info

ROICORRPATH = Path(__file__).parent / 'intersubj' / 'roicorr'
TR = 1.5


def _lagcorr(x, y, lags):
    out = np.full(len(lags), np.nan)
    for i, s in enumerate(lags):
        a = abs(s)
        xa, ya = (x[:-a], y[a:]) if s > 0 else (x[a:], y[:-a]) if s < 0 else (x, y)
        mask = ~(np.isnan(xa) | np.isnan(ya))
        if mask.sum() > 1:
            out[i] = np.corrcoef(xa[mask], ya[mask])[0, 1]
    return out


def plot_audio_lag(roiname, condname, names, w=14, tmin=None, tmax=None):
    r = np.load(ROICORRPATH / f'{roiname}_{condname}_roicorr.npz')
    all_names = get_subj_info('names', condname)
    if names is None:
        names = all_names
    idx = [all_names.index(n) for n in names]
    meantc = zscore(np.nanmean(r['roitc'][:, idx], axis=1))

    envelope = get_audio_envelope(condname)
    envelope = zscore(envelope.astype(float))

    n = min(len(meantc), len(envelope))
    lo = tmin if tmin is not None else 0
    hi = min(tmax, n) if tmax is not None else n
    meantc = meantc[lo:hi]
    envelope = envelope[lo:hi]

    lags = np.arange(-w, w + 1)
    lc = _lagcorr(envelope, meantc, lags)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags * TR, lc, 'b-o', linewidth=1.5, markersize=4)
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    peak_lag = lags[np.nanargmax(lc)] * TR
    ax.set_xlabel('Lag (s)')
    ax.set_ylabel('Correlation')
    n_subs = len(names)
    window_str = f'TRs {lo}–{hi}'
    ax.set_title(
        f'Audio envelope × {roiname.replace("_", "-")} mean TC '
        f'({condname.replace("_", "-")}, n={n_subs}, {window_str})  peak={peak_lag:.1f}s'
    )
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_audio_timecourse(roiname, condname, names):
    r = np.load(ROICORRPATH / f'{roiname}_{condname}_roicorr.npz')
    all_names = get_subj_info('names', condname)
    if names is None:
        names = all_names
    idx = [all_names.index(n) for n in names]
    meantc = zscore(np.nanmean(r['roitc'][:, idx], axis=1))

    envelope = get_audio_envelope(condname)
    envelope = zscore(envelope.astype(float))

    n = min(len(meantc), len(envelope))
    trs = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(trs, envelope[:n], color='orange', linewidth=1, label='audio envelope')
    ax.plot(trs, meantc[:n], color='steelblue', linewidth=1, label=f'{roiname.replace("_", "-")} mean TC')
    ax.set_xlabel('TR')
    ax.set_ylabel('Z')
    ax.set_title(f'{roiname.replace("_", "-")} mean TC vs audio envelope  ({condname.replace("_", "-")}, n={len(names)})')
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_audio_lag_subject(roiname, condname, subj_name, w=14, tmin=None, tmax=None):
    r = np.load(ROICORRPATH / f'{roiname}_{condname}_roicorr.npz')
    all_names = get_subj_info('names', condname)
    idx = all_names.index(subj_name)
    tc = zscore(r['roitc'][:, idx])

    envelope = get_audio_envelope(condname)
    envelope = zscore(envelope.astype(float))

    n = min(len(tc), len(envelope))
    lo = tmin if tmin is not None else 0
    hi = min(tmax, n) if tmax is not None else n
    tc = tc[lo:hi]
    envelope = envelope[lo:hi]

    lags = np.arange(-w, w + 1)
    lc = _lagcorr(envelope, tc, lags)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags * TR, lc, 'b-o', linewidth=1.5, markersize=4)
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    peak_lag = lags[np.nanargmax(lc)] * TR
    ax.set_xlabel('Lag (s)')
    ax.set_ylabel('Correlation')
    window_str = f'TRs {lo}–{hi}'
    ax.set_title(
        f'Audio envelope × {roiname.replace("_", "-")} '
        f'({condname.replace("_", "-")}, {subj_name}, {window_str})  peak={peak_lag:.1f}s'
    )
    ax.grid(True)
    plt.tight_layout()
    plt.show()
