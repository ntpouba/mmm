from functools import lru_cache
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import av

from mmm.utils.layout import ROOT

STIMPATH = ROOT / 'data' / 'stimuli' / 'movies'
CACHEPATH = ROOT / 'data' / 'stimuli' / 'envelopes'
TR = 1.5

_AUDIO_EXTS = {'.wav', '.mov', '.mp4', '.m4a', '.mp3'}

# The envelope definitions available from get_audio_envelopes():
#   linear  mean Hilbert magnitude of the raw waveform per window
#   dBA     A-weighted equivalent sound level (dBA) of each window, computed in
#           the frequency domain (from a_weighted_loudness.py)
ENVELOPE_KINDS = ('linear', 'dBA')

# dBA settings, as in a_weighted_loudness.py
DBA_FREQ_MIN_HZ = 50
DBA_FREQ_MAX_HZ = 8000
DBA_WELCH_SEGMENT = 2048

# Explicit overrides, only needed when auto-discovery by slug is ambiguous
# or the filename doesn't start with the movie name's slug.
_CONDNAME_TO_FILE = {}


def _slugify(name):
    return name.strip().lower().replace(' ', '-')


def _resolve_stim_path(condname):
    slug = _slugify(condname)
    if slug in _CONDNAME_TO_FILE:
        return STIMPATH / _CONDNAME_TO_FILE[slug]

    # Case-insensitive: the movie files are inconsistently capitalised
    # (Body-Double_..., the-bench_...).
    candidates = [
        p for p in STIMPATH.iterdir()
        if p.suffix.lower() in _AUDIO_EXTS
        and (p.stem.lower() == slug or p.name.lower().startswith(slug + '_'))
    ] if STIMPATH.is_dir() else []
    if not candidates:
        raise FileNotFoundError(
            f"No stimulus file for condname={condname!r} (slug={slug!r}) in {STIMPATH}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous stimulus files for slug={slug!r}: {candidates} "
            f"-- add an explicit _CONDNAME_TO_FILE entry"
        )
    return candidates[0]


def stimulus_status(condname):
    """(path, reason) for a clip's stimulus file, without decoding anything.

    path is None when no usable file exists, and reason then says why. Lets a
    caller decide up front whether a clip is runnable, rather than discovering
    it part-way through an expensive decode.
    """
    try:
        return _resolve_stim_path(condname), None
    except (FileNotFoundError, ValueError) as e:
        return None, str(e)


def _load_wav(path):
    fs, signal = wav.read(str(path))
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    signal = signal.astype(float) / 32767.0
    return fs, signal


def _load_audio_via_pyav(path):
    """Decode the first audio stream of a video/audio container via PyAV."""
    container = av.open(str(path))
    if not container.streams.audio:
        raise ValueError(f"{path} has no audio stream")
    astream = container.streams.audio[0]
    fs = astream.rate
    resampler = av.AudioResampler(format='s16', layout='mono', rate=fs)

    chunks = []
    for frame in container.decode(astream):
        for rframe in resampler.resample(frame):
            chunks.append(rframe.to_ndarray())
    for rframe in resampler.resample(None):  # flush
        chunks.append(rframe.to_ndarray())
    container.close()

    signal = np.concatenate(chunks, axis=-1).flatten().astype(float) / 32768.0
    return fs, signal


def _load_audio(path):
    if path.suffix.lower() == '.wav':
        return _load_wav(path)
    return _load_audio_via_pyav(path)


def _a_weighting_db(freqs):
    """IEC 61672 A-weighting magnitude in dB at each frequency, 0 dB at 1 kHz."""
    f2 = np.asarray(freqs, dtype=float) ** 2
    response = (12194 ** 2 * f2 ** 2
                / ((f2 + 20.6 ** 2)
                   * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
                   * (f2 + 12194 ** 2)))
    return 20 * np.log10(response) + 2.0


def _volume_windows(n_samples, fs, scan_start):
    """[start, end) sample bounds of each volume's audio window.

    Volume k represents time t_k = scan_start + k*TR on the clip's audio clock,
    and its window is centred on it: [t_k - TR/2, t_k + TR/2). Volumes are
    generated while t_k is inside the audio; windows are clipped to the audio,
    so the first and last can be partial.
    """
    n = max(int(np.ceil((n_samples / fs - scan_start) / TR)), 0)
    times = scan_start + np.arange(n) * TR
    start = np.clip(np.round((times - TR / 2) * fs), 0, n_samples).astype(int)
    end = np.clip(np.round((times + TR / 2) * fs), 0, n_samples).astype(int)
    return start, end


def _window_means(values, start, end):
    """Mean of `values` over each [start, end) window; NaN for an empty window."""
    csum = np.concatenate([[0.0], np.cumsum(values)])
    n = end - start
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, (csum[end] - csum[start]) / n, np.nan)


def _dba(fs, signal, start, end):
    """A-weighted equivalent level (dBA re full scale) of each window.

    Welch power spectral density, restricted to DBA_FREQ_MIN_HZ..MAX, weighted
    by the A-curve in power (dB/10), integrated over frequency, then 10*log10.
    The weighting is applied before the sum and the log. NaN where a window is
    too short to resolve any frequency in range.
    """
    out = np.full(len(start), np.nan)
    for k, (s, e) in enumerate(zip(start, end)):
        window = signal[s:e]
        if len(window) < 2:
            continue
        freqs, psd = scipy.signal.welch(window, fs=fs,
                                        nperseg=min(DBA_WELCH_SEGMENT, len(window)))
        keep = (freqs >= DBA_FREQ_MIN_HZ) & (freqs < DBA_FREQ_MAX_HZ)
        if not keep.any():
            continue
        weight = 10 ** (_a_weighting_db(freqs[keep]) / 10)
        out[k] = 10 * np.log10((psd[keep] * weight).sum() * (freqs[1] - freqs[0]))
    return out


@lru_cache(maxsize=1)
def _decoded(condname):
    """(fs, signal, |hilbert(signal)|) for one clip.

    Kept in memory for the most recent clip, since a clip is usually windowed
    at several offsets (one per subject) in a row.
    """
    stim_path = _resolve_stim_path(condname)
    print(f'  decoding {condname} ({stim_path.name})...')
    fs, signal = _load_audio(stim_path)
    return fs, signal, np.abs(scipy.signal.hilbert(signal))


def get_audio_envelopes(condname, scan_start):
    """All of ENVELOPE_KINDS for one clip, as a dict of (n_volumes,) arrays.

    scan_start is where the first volume of the BOLD it will be compared with
    sits on the clip's audio clock, in seconds (negative if before the clip
    began), including any slice-timing reference. Every kind uses the same
    centred windows (see _volume_windows); linear is the mean magnitude per
    window.

    Results are cached per (clip, scan_start to the ms).
    """
    slug = _slugify(condname)
    offset_ms = int(round(scan_start * 1000))
    cache = CACHEPATH / f'{slug}_{offset_ms:+d}ms.npz'

    if cache.exists():
        cached = np.load(cache)
        linear, dba = cached['linear'], cached['dBA']
    else:
        fs, signal, linear_env = _decoded(condname)
        start, end = _volume_windows(len(signal), fs, offset_ms / 1000)
        linear = _window_means(linear_env, start, end)
        dba = _dba(fs, signal, start, end)

        CACHEPATH.mkdir(parents=True, exist_ok=True)
        np.savez(cache, linear=linear, dBA=dba)
        print(f'  saved → {cache}  ({len(linear)} volumes)')

    return {'linear': linear, 'dBA': dba}


def get_audio_envelope(condname, scan_start, kind='linear'):
    """One named envelope for a clip. See ENVELOPE_KINDS."""
    if kind not in ENVELOPE_KINDS:
        raise ValueError(f'unknown envelope kind {kind!r}; expected one of {ENVELOPE_KINDS}')
    return get_audio_envelopes(condname, scan_start)[kind]
