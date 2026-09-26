from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import av

BASEPATH = Path(__file__).parent
STIMPATH = BASEPATH / 'data' / 'stimuli'
TR = 1.5

_AUDIO_EXTS = {'.wav', '.mov', '.mp4', '.m4a', '.mp3'}

# The envelope definitions available from get_audio_envelopes(). They are the
# 2x2 of frequency weighting (flat / A-weighted) x amplitude compression
# (linear / log), so any difference between them is attributable to one factor.
ENVELOPE_KINDS = ('linear', 'log', 'aweight', 'dbA')

# Explicit overrides, only needed when auto-discovery by slug is ambiguous
# or the filename doesn't start with the movie name's slug.
_CONDNAME_TO_FILE = {}


def _slugify(name):
    return name.strip().lower().replace(' ', '-')


def _resolve_stim_path(condname):
    slug = _slugify(condname)
    if slug in _CONDNAME_TO_FILE:
        return STIMPATH / _CONDNAME_TO_FILE[slug]

    candidates = [
        p for p in STIMPATH.glob(f'{slug}*')
        if p.suffix.lower() in _AUDIO_EXTS
    ]
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


def _a_weighting_sos(fs):
    """IEC 61672 A-weighting as second-order sections at sample rate `fs`.

    The analog prototype has four zeros at DC and poles at 20.6/107.7/737.9/
    12194.2 Hz (the first and last doubled), normalised to 0 dB at 1 kHz. It
    approximates the 40-phon equal-loudness contour: strong attenuation below
    ~500 Hz and above ~10 kHz, i.e. the energy the cochlea is least sensitive
    to. Bilinear transform warps the response near Nyquist, which is the usual
    and harmless caveat at audio rates.
    """
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    a1000 = 1.9997  # dB of gain the prototype has at 1 kHz; divided back out
    pi = np.pi

    num = [(2 * pi * f4) ** 2 * 10 ** (a1000 / 20.0), 0, 0, 0, 0]
    den = np.polymul([1, 4 * pi * f4, (2 * pi * f4) ** 2],
                     [1, 4 * pi * f1, (2 * pi * f1) ** 2])
    den = np.polymul(np.polymul(den, [1, 2 * pi * f3]), [1, 2 * pi * f2])

    return scipy.signal.tf2sos(*scipy.signal.bilinear(num, den, fs))


def _a_weight(fs, signal):
    """A-weight a waveform with zero phase distortion.

    filtfilt rather than a one-pass filter is essential here: everything
    downstream is a *lag* measurement, and a causal IIR would contribute its
    own group delay that is indistinguishable from a hemodynamic shift.
    """
    return scipy.signal.sosfiltfilt(_a_weighting_sos(fs), signal)


def _envelope_from_samples(fs, signal):
    """Hilbert magnitude, averaged into one value per TR.

    The per-TR mean is taken as a plain block average rather than with
    resample_poly. Decimating by int(fs * TR) -- 66150 at 44.1 kHz -- in a
    single stage builds a ~1.3M-tap FIR that rings and overshoots below zero,
    which is impossible for a magnitude and used to leave a couple of negative
    samples per clip. Those artificial near-zero samples are what previously
    forced a floor on the log envelopes. A block mean is exactly "mean
    amplitude per TR", cannot ring, and stays strictly positive.

    Any trailing partial TR is dropped; callers already truncate the envelope
    and the BOLD timecourse to their common length.
    """
    envelope = np.abs(scipy.signal.hilbert(signal))
    q = int(fs * TR)
    n = len(envelope) // q
    return envelope[:n * q].reshape(n, q).mean(axis=1)


def get_audio_envelopes(condname):
    """All of ENVELOPE_KINDS for one clip, as a dict of (n_TRs,) arrays.

    Only the two Hilbert envelopes are cached, since decoding the audio and
    running two Hilbert transforms is the entire cost. The log pair is derived
    on each call, so changing how compression is done never invalidates a cache.
    """
    slug = _slugify(condname)
    cache = STIMPATH / f'{slug}_envelopes.npz'

    if cache.exists():
        cached = np.load(cache)
        linear, aweight = cached['linear'], cached['aweight']
    else:
        stim_path = _resolve_stim_path(condname)
        print(f'  computing envelopes for {condname} ({stim_path.name})...')

        fs, signal = _load_audio(stim_path)
        linear = _envelope_from_samples(fs, signal)
        aweight = _envelope_from_samples(fs, _a_weight(fs, signal))

        np.savez(cache, linear=linear, aweight=aweight)
        print(f'  saved → {cache}  ({len(linear)} TRs)')

    # A block-averaged magnitude cannot be <= 0 unless the clip contains a
    # genuinely silent TR, which would make the log envelopes meaningless.
    # Fail loudly rather than quietly clamping to an arbitrary floor.
    for name, env in (('linear', linear), ('aweight', aweight)):
        if env.min() <= 0:
            raise ValueError(
                f'{condname}: {name} envelope has {(env <= 0).sum()} non-positive '
                f'sample(s) (min={env.min():.3g}); the log envelopes need a floor. '
                f'Delete {cache} and check the clip for digital silence.'
            )

    return {
        'linear': linear,
        'log': np.log10(linear),
        'aweight': aweight,
        'dbA': np.log10(aweight),
    }


def get_audio_envelope(condname, kind='linear'):
    """One named envelope for a clip. See ENVELOPE_KINDS."""
    if kind not in ENVELOPE_KINDS:
        raise ValueError(f'unknown envelope kind {kind!r}; expected one of {ENVELOPE_KINDS}')
    return get_audio_envelopes(condname)[kind]
