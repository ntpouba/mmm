"""
Leave-one-out intersubject functional connectivity (LOO-ISFC) for a single
movie clip ("Negative Space") shared by all 3 subjects during ses-19, across
the same Schaefer-400 parcel timecourses and confound-regression variants
used by compute_fc_matrix.py.

For each subject, correlates their own parcel timecourses during the clip
against the mean of the other subjects' timecourses for the same clip,
isolating stimulus-locked connectivity from subject-specific noise (Simony
et al., 2016). The resulting matrix is deliberately NOT symmetric, and that
asymmetry is the point: cell (p, q) is "my parcel p vs others' parcel q"
while cell (q, p) is "my parcel q vs others' parcel p". Those are different
calculations on different signal pairs, so they carry distinct directional
information and are kept separate rather than averaged together.

Orientation matters when reading these matrices:
    row p    = the held-out subject's OWN parcel p
    column q = the other subjects' MEAN parcel q

The diagonal (p, p) is the one place direction collapses -- it is the same
cell either way, and is the standard leave-one-out ISC for that parcel.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from confounds import VARIANT_TAGS
from layout import Run, parcels

NPARCEL = 400
LH_RH_BOUNDARY = 199.5

MOVIE_TAG = "negspace"

# Clip location (run, onset TR, duration TR) per subject for "Negative Space"
# during ses-19, from movie_order.csv.
CLIP_SESSION = "ses-19"
CLIP_LOCATIONS = {
    "sub-03": {"run": "run-02", "onset_tr": 10.3221, "duration_tr": 192.666},
    "sub-04": {"run": "run-01", "onset_tr": 360.9829, "duration_tr": 192.6557},
    "sub-05": {"run": "run-02", "onset_tr": 463.6497, "duration_tr": 192.666},
}
SUBJECTS = list(CLIP_LOCATIONS)
N_TR = int(np.floor(min(c["duration_tr"] for c in CLIP_LOCATIONS.values())))

OUT_DIR = Path("scratch/isfc_matrices")
OUT_DIR_NETWORK = Path("scratch/isfc_matrices_bynetwork")
ORDER_TXT = Path("standard/Schaefer2018_400Parcels_17Networks_order.txt")
NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default", "TempPar"]


def load_movie_segment(subject, variant_tag):
    loc = CLIP_LOCATIONS[subject]
    combined = np.load(parcels(Run(subject, CLIP_SESSION, loc["run"]), variant_tag))  # (NPARCEL, TR_total)
    start = round(loc["onset_tr"])
    end = start + N_TR
    if end > combined.shape[1]:
        raise ValueError(
            f"{subject}/{variant_tag}: clip window [{start}:{end}] exceeds "
            f"available TRs ({combined.shape[1]})"
        )
    return combined[:, start:end]  # (NPARCEL, N_TR)


def zscore_rows(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / std


def cross_corr(a, b):
    """a, b: (NPARCEL, N_TR). Returns (NPARCEL, NPARCEL) with entry [p, q] =
    corr(a's parcel p, b's parcel q). Not symmetric when a is not b."""
    az = zscore_rows(a)
    bz = zscore_rows(b)
    return (az @ bz.T) / az.shape[1]


def loo_isfc(segments, subject):
    """segments: dict subject -> (NPARCEL, N_TR) for the shared clip. Returns
    the LOO-ISFC matrix for `subject` vs the mean of the others, left
    asymmetric: row p is the subject's own parcel p, column q is the others'
    mean parcel q."""
    others = np.mean([seg for s, seg in segments.items() if s != subject], axis=0)
    m = cross_corr(segments[subject], others)
    return m


def load_network_labels(order_txt=ORDER_TXT):
    labels = []
    for line in order_txt.read_text().splitlines():
        _, name = line.split("\t")[:2]
        match = next((net for net in NETWORK_ORDER if net in name), None)
        if match is None:
            raise ValueError(f"Could not map parcel name to a network group: {name!r}")
        labels.append(match)
    if len(labels) != NPARCEL:
        raise ValueError(f"Expected {NPARCEL} parcel labels, got {len(labels)}")
    return labels


def network_sort_order(labels):
    return sorted(range(len(labels)), key=lambda i: NETWORK_ORDER.index(labels[i]))


def network_block_centers(sorted_labels):
    boundaries, centers = [], []
    start = 0
    for i in range(1, len(sorted_labels) + 1):
        if i == len(sorted_labels) or sorted_labels[i] != sorted_labels[start]:
            centers.append((start + i - 1) / 2)
            if i != len(sorted_labels):
                boundaries.append(i - 0.5)
            start = i
    return boundaries, centers


def plot_isfc(fc, title, out_path, vlim):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(fc, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax.set_title(title)
    ax.set_xlabel("Others' mean parcel (0-199 LH, 200-399 RH)")
    ax.set_ylabel("Own parcel (0-199 LH, 200-399 RH)")
    ax.axvline(LH_RH_BOUNDARY, color="k", linewidth=0.5)
    ax.axhline(LH_RH_BOUNDARY, color="k", linewidth=0.5)
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_isfc_by_network(fc, labels, title, out_path, vlim):
    order = network_sort_order(labels)
    fc_sorted = fc[np.ix_(order, order)]
    sorted_labels = [labels[i] for i in order]
    boundaries, centers = network_block_centers(sorted_labels)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(fc_sorted, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax.set_title(title)
    for b in boundaries:
        ax.axvline(b, color="k", linewidth=0.6)
        ax.axhline(b, color="k", linewidth=0.6)
    ax.set_xticks(centers)
    ax.set_xticklabels(NETWORK_ORDER, rotation=45, ha="right")
    ax.set_yticks(centers)
    ax.set_yticklabels(NETWORK_ORDER)
    ax.set_xlabel("Others' mean")
    ax.set_ylabel("Own")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    print(f"Clip: {MOVIE_TAG}, N_TR={N_TR}")
    labels = load_network_labels()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for variant_tag in VARIANT_TAGS:
        print(f"\n=== Variant: {variant_tag} ===")

        segments = {s: load_movie_segment(s, variant_tag) for s in SUBJECTS}

        subject_matrices = {s: loo_isfc(segments, s) for s in SUBJECTS}
        group_matrix = np.mean(list(subject_matrices.values()), axis=0)
        all_matrices = {**subject_matrices, "group": group_matrix}

        vlim = 0.25

        for name, m in all_matrices.items():
            out_npy = OUT_DIR / f"{name}_{MOVIE_TAG}_{variant_tag}_isfc400.npy"
            np.save(out_npy, m)

            out_png = OUT_DIR / f"{name}_{MOVIE_TAG}_{variant_tag}_isfc400.png"
            plot_isfc(m, f"{name} LOO-ISFC ({MOVIE_TAG}, {variant_tag})", out_png, vlim)

            net_out_dir = OUT_DIR_NETWORK / name / MOVIE_TAG
            net_out_dir.mkdir(parents=True, exist_ok=True)
            net_out = net_out_dir / f"{variant_tag}_isfc400_bynetwork.png"
            plot_isfc_by_network(
                m, labels,
                f"{name} LOO-ISFC ({MOVIE_TAG}, {variant_tag}, by network)",
                net_out, vlim,
            )

            print(f"  [{name}] saved {out_npy.name}, {out_png.name}, {net_out}")

    print("\ndone")


if __name__ == "__main__":
    main()
