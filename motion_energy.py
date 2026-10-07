from functools import lru_cache
import numpy as np
import av

from audio_envelope import BASEPATH, TR, _resolve_stim_path, _slugify, _window_means

CACHEPATH = BASEPATH / 'data' / 'stimuli' / 'motion'

# Frames are converted to grayscale at this width (aspect kept) before differencing.
FRAME_WIDTH = 160

# motion  mean absolute grayscale difference between consecutive frames,
#         averaged over the frames in each volume's window
MOTION_KINDS = ('motion',)


def _decode_motion(path):
    """(times, motion, duration): per-frame motion energy for one clip.

    motion[i] is the mean |frame - previous frame| over pixels, timestamped at
    the later frame, so the first frame contributes no value.
    """
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    height = 2 * round(FRAME_WIDTH * stream.height / stream.width / 2)

    times, motion, prev = [], [], None
    for frame in container.decode(stream):
        img = frame.reformat(width=FRAME_WIDTH, height=height,
                             format='gray').to_ndarray().astype(np.float32)
        if prev is not None:
            times.append(frame.time)
            motion.append(np.abs(img - prev).mean())
        prev = img
    duration = float(stream.duration * stream.time_base)
    container.close()
    return np.array(times), np.array(motion), duration


@lru_cache(maxsize=1)
def _per_frame(condname):
    """Per-frame motion energy, decoded once per clip and cached to disk."""
    cache = CACHEPATH / f'{_slugify(condname)}.npz'
    if not cache.exists():
        stim_path = _resolve_stim_path(condname)
        print(f'  decoding video of {condname} ({stim_path.name})...')
        times, motion, duration = _decode_motion(stim_path)
        CACHEPATH.mkdir(parents=True, exist_ok=True)
        np.savez(cache, times=times, motion=motion, duration=duration)
        print(f'  saved → {cache}  ({len(motion)} frames)')
    cached = np.load(cache)
    return cached['times'], cached['motion'], float(cached['duration'])


def get_motion_energy(condname, scan_start):
    """{'motion': (n_volumes,) array} for one clip.

    Windows are the same as audio_envelope's: volume k sits at
    t_k = scan_start + k*TR on the clip's clock (scan_start rounded to the ms,
    as there) and averages the frames timed in [t_k - TR/2, t_k + TR/2).
    Audio and video share the clip's clock, so the two line up volume for volume.
    """
    times, motion, duration = _per_frame(condname)
    scan_start = round(scan_start * 1000) / 1000
    n = max(int(np.ceil((duration - scan_start) / TR)), 0)
    centres = scan_start + np.arange(n) * TR
    start = np.searchsorted(times, centres - TR / 2)
    end = np.searchsorted(times, centres + TR / 2)
    return {'motion': _window_means(motion, start, end)}
