"""
Hippocampal FC profile across confound-regression variants, self-contained.

Question: how much does the choice of cleaning regime reshape hippocampal
connectivity? For each variant we build the hippocampus's FC profile -- each
hippocampal ROI's timecourse correlated against all 400 Schaefer cortical
parcels -- then correlate those profiles pairwise across variants, giving a
variant x variant matrix per hippocampal ROI.

Deliberately NOT smoothed, and deliberately independent of the mask_and_smooth
-> regress_out_confounds_volume -> extract_parcel_timecourses_volume chain:

  - The hippocampus is ~1cm across and borders the ventricles and white
    matter, so a 4mm kernel would pull CSF/WM signal straight into it.
    Averaging 86-341 voxels per ROI is already substantial spatial averaging;
    pre-smoothing mostly buys leakage across ROI boundaries. The reference
    code this is modeled on (mentor's `_load_bold_data`) likewise defaults to
    `smooth=None`.
  - With no smoothing, no voxel is influenced by its neighbors, so there is
    nothing for skull/scalp signal to leak into -- the separate skull-strip
    step is unnecessary here. Atlas voxels are still intersected with each
    subject's brain mask so that atlas voxels falling outside that subject's
    brain are dropped.
  - Nothing needs the full 3D grid, so voxels are pulled out once per run and
    everything downstream runs on small 2D arrays.

"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from nilearn.signal import clean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from confounds import CONFOUND_VARIANTS, VARIANT_TAGS, build_covariates  # noqa: E402

SUBJECTS = ["sub-03", "sub-04", "sub-05"]
SESSION = "ses-19"
RUNS = ["run-01", "run-02"]
TR = 1.5

NPARCEL = 400

DATADIR = ROOT / "data"
SCHAEFER_PATH = ROOT / "standard/Schaefer2018_400Parcels_MNI152NLin2009cAsym_2mm.nii.gz"

# Parcel -> network grouping, used to order the columns of the profile
# heatmaps. Same atlas order file and same grouping as plot_fc_networks.py;
# the volume atlas's labels 1-400 match its row order (verified against
# world-space X coordinates).
ORDER_TXT = ROOT / "standard/Schaefer2018_400Parcels_17Networks_order.txt"
NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default", "TempPar"]

# Melbourne Subcortex Atlas (Tian et al. 2020), scale II, in our own space.
# Both files come as a pair: the label text file has one ROI name per line,
# where line i (0-based) corresponds to atlas value i + 1 (0 = background).
TIAN_PATH = ROOT / "standard/Tian_Subcortex_S2_3T_2009cAsym.nii.gz"
TIAN_LABELS_PATH = ROOT / "standard/Tian_Subcortex_S2_3T_label.txt"

# Hippocampal ROIs are picked out by name rather than by hardcoded index, so
# this stays correct regardless of the atlas scale's numbering.
HIPPOCAMPUS_PATTERN = "HIP"

# The atlas's hippocampal subregions (anterior/posterior x left/right) are
# pooled into one whole-hippocampus timecourse by averaging over every
# hippocampal voxel at once, so each voxel counts equally -- the conventional
# way a whole-hippocampus ROI is defined, and what a single hippocampus mask
# would give. This weights the larger anterior subregions (395/380 voxels)
# slightly above the posterior ones (304/310); averaging the four subregion
# means instead would have given all four equal weight.
HIPPOCAMPUS_ROI_NAME = "hippocampus"

OUT_DIR = ROOT / "scratch/hippocampal_fc"
# Output is split by plot type: variant x variant similarity matrices under
# SIMILARITY_DIR, profile heatmaps under HEATMAP_DIR, each nesting by subject.
SIMILARITY_DIR = OUT_DIR / "similarity"
HEATMAP_DIR = OUT_DIR / "heatmaps"

def bold_path(subject, run):
    return (
        DATADIR / subject / "func"
        / f"{subject}_{SESSION}_task-NATencoding_{run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    )


def brain_mask_path(subject, run):
    return (
        DATADIR / subject / "func"
        / f"{subject}_{SESSION}_task-NATencoding_{run}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
    )


def confound_path(subject, run):
    return (
        DATADIR / subject / "func"
        / f"{subject}_{SESSION}_task-NATencoding_{run}_desc-confounds_timeseries.tsv"
    )


def load_hippocampus_atlas(reference_img):
    """Returns (hippocampus label volume, ordered ROI names, ordered label values).

    The Tian atlas is resampled onto the reference grid if it doesn't already
    match it (nearest-neighbour, since these are integer labels).
    """
    for path in (TIAN_PATH, TIAN_LABELS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Tian atlas file: {path}\n"
                "Both the .nii.gz and its label .txt are needed (Melbourne "
                "Subcortex Atlas, Tian et al. 2020, scale II, 2009cAsym variant)."
            )

    names = [line.strip() for line in TIAN_LABELS_PATH.read_text().splitlines() if line.strip()]

    atlas_img = nb.load(TIAN_PATH)
    if atlas_img.shape[:3] != reference_img.shape[:3] or not np.allclose(
        atlas_img.affine, reference_img.affine
    ):
        print("  Tian atlas grid differs from BOLD grid; resampling (nearest)")
        atlas_img = resample_to_img(
            source_img=atlas_img,
            target_img=reference_img,
            interpolation="nearest",
            copy_header=True,
            force_resample=True,
        )

    atlas = np.round(atlas_img.get_fdata()).astype(int)

    # line i (0-based) in the label file <-> atlas value i + 1
    hipp = [
        (value, name)
        for value, name in enumerate(names, start=1)
        if HIPPOCAMPUS_PATTERN.lower() in name.lower()
    ]
    if not hipp:
        raise ValueError(
            f"No label in {TIAN_LABELS_PATH.name} matched {HIPPOCAMPUS_PATTERN!r}. "
            f"Labels found: {names}"
        )

    values = [v for v, _ in hipp]
    hipp_names = [n for _, n in hipp]
    return atlas, hipp_names, values


def roi_means(voxel_timeseries, labels, values):
    """voxel_timeseries: (n_voxels, T); labels: (n_voxels,) -> (len(values), T)."""
    return np.vstack([voxel_timeseries[labels == v].mean(axis=0) for v in values])


def correlate_profiles(a, b):
    """Pearson correlation between two 1D FC profiles."""
    return float(np.corrcoef(a, b)[0, 1])


def fisher_mean_stack(arrays):
    """Elementwise Fisher-z mean over a stack of correlation arrays."""
    stack = np.asarray(arrays, dtype=float)
    return np.tanh(np.nanmean(np.arctanh(np.clip(stack, -0.999999, 0.999999)), axis=0))


def load_network_labels():
    """Length-400 list of network group names, in parcel order."""
    labels = []
    for line in ORDER_TXT.read_text().splitlines():
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


def compute_profiles():
    """Returns (profiles, roi_names).

    profiles[(subject, run, variant)] = (1, NPARCEL) array holding the whole
    hippocampus's correlation against every cortical parcel. All hippocampal
    voxels are pooled into one timecourse before correlating.
    """
    schaefer_img = nb.load(SCHAEFER_PATH)
    schaefer = np.round(schaefer_img.get_fdata()).astype(int)

    reference_img = nb.load(bold_path(SUBJECTS[0], RUNS[0]))
    if schaefer.shape != reference_img.shape[:3] or not np.allclose(
        schaefer_img.affine, reference_img.affine
    ):
        raise ValueError("Schaefer atlas grid does not match the BOLD grid")

    tian, hipp_names, hipp_values = load_hippocampus_atlas(reference_img)
    roi_names = [HIPPOCAMPUS_ROI_NAME]
    print(
        f"Pooling voxels from {len(hipp_names)} hippocampal subregions "
        f"into one timecourse: {hipp_names}"
    )

    overlap = int(((schaefer > 0) & np.isin(tian, hipp_values)).sum())
    if overlap:
        print(f"  note: {overlap} voxels labelled both cortex and hippocampus; hippocampus takes precedence")

    profiles = {}

    for subject in SUBJECTS:
        for run in RUNS:
            print(f"\nProcessing {subject} {run}")

            bold_img = nb.load(bold_path(subject, run))
            brain = nb.load(brain_mask_path(subject, run)).get_fdata().astype(bool)

            hipp_vol = np.where(np.isin(tian, hipp_values), tian, 0)
            # hippocampus takes precedence where the two atlases disagree
            cortex_vol = np.where(hipp_vol > 0, 0, schaefer)

            roi_mask = ((cortex_vol > 0) | (hipp_vol > 0)) & brain

            data = bold_img.get_fdata(dtype=np.float32)
            voxels = data[roi_mask]  # (n_roi_voxels, T)
            del data

            cortex_labels = cortex_vol[roi_mask]
            hipp_labels = hipp_vol[roi_mask]
            print(f"  {voxels.shape[0]} ROI voxels, {voxels.shape[1]} TRs")

            confounds = pd.read_csv(confound_path(subject, run), sep="\t")
            if voxels.shape[1] != len(confounds):
                raise ValueError(
                    f"{subject} {run}: {voxels.shape[1]} TRs in volume but "
                    f"{len(confounds)} rows in confounds"
                )

            for variant_tag, variant_spec in CONFOUND_VARIANTS.items():
                covariates = build_covariates(confounds, variant_spec)

                cleaned = clean(
                    voxels.T,  # nilearn wants (n_timepoints, n_features)
                    confounds=covariates.values,
                    detrend=False,
                    filter=False,
                    standardize="zscore_sample",
                    t_r=TR,
                ).T

                # z-score per voxel (done by clean) then pool into ROIs
                cortex_ts = roi_means(cleaned, cortex_labels, range(1, NPARCEL + 1))
                hipp_ts = cleaned[hipp_labels > 0].mean(axis=0, keepdims=True)  # (1, T)

                # whole hippocampus vs every cortical parcel
                profile = np.vstack(
                    [
                        [correlate_profiles(h, c) for c in cortex_ts]
                        for h in hipp_ts
                    ]
                )
                profiles[(subject, run, variant_tag)] = profile
                print(f"  [{variant_tag}] profile shape={profile.shape}")

    return profiles, roi_names


def variant_similarity(stack):
    """Variant x variant similarity computed directly from one already-averaged
    (n_variants, NPARCEL) profile stack.

    Note this is average-then-correlate: runs are averaged into a single
    profile per subject first, and the correlation is taken on that. Averaging
    cancels noise, so these values run higher than correlating each run
    separately and averaging the correlations would give -- the two are not
    comparable to one another."""
    n = len(VARIANT_TAGS)
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = matrix[j, i] = correlate_profiles(stack[i], stack[j])
    return matrix



def plot_variant_matrix(matrix, title, out_path, vmin):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=1.0)
    ax.set_xticks(range(len(VARIANT_TAGS)))
    ax.set_xticklabels(VARIANT_TAGS, rotation=45, ha="right")
    ax.set_yticks(range(len(VARIANT_TAGS)))
    ax.set_yticklabels(VARIANT_TAGS)
    for i in range(len(VARIANT_TAGS)):
        for j in range(len(VARIANT_TAGS)):
            ax.text(
                j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                color="w" if matrix[i, j] < (vmin + 1.0) / 2 else "k",
                fontsize=8,
            )
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="FC-profile similarity (r)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def profile_stack(profiles, roi_idx, pairs):
    """(n_variants, NPARCEL) FC profile for one hippocampal ROI, rows in
    VARIANT_TAGS order, Fisher-z averaged over the given (subject, run) pairs.
    A single pair returns that run's own profile."""
    return np.vstack(
        [
            fisher_mean_stack([profiles[(s, r, v)][roi_idx] for s, r in pairs])
            for v in VARIANT_TAGS
        ]
    )


def plot_profile_heatmap(stack, labels, title, out_path, vlim, cbar_label):
    """Rows = cleaning variants, columns = the 400 cortical parcels grouped
    into network blocks. Shows how the hippocampal FC profile itself changes
    across variants, rather than just how similar the profiles are."""
    order = network_sort_order(labels)
    sorted_stack = stack[:, order]
    sorted_labels = [labels[i] for i in order]
    boundaries, centers = network_block_centers(sorted_labels)

    fig, ax = plt.subplots(figsize=(11, 3.4))
    im = ax.imshow(sorted_stack, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    for b in boundaries:
        ax.axvline(b, color="k", linewidth=0.6)
    # row separators, so each variant's band is clearly delimited
    for y in np.arange(len(VARIANT_TAGS) - 1) + 0.5:
        ax.axhline(y, color="k", linewidth=0.8)
    ax.set_xticks(centers)
    ax.set_xticklabels(NETWORK_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(len(VARIANT_TAGS)))
    ax.set_yticklabels(VARIANT_TAGS)
    ax.set_xlabel("Cortical parcel, grouped by network")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def safe_filename(name):
    return name.replace("/", "_").replace(" ", "_")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    profiles, roi_names = compute_profiles()
    roi_name = roi_names[0]
    safe = safe_filename(roi_name)

    np.save(
        OUT_DIR / "profiles.npy",
        {str(k): v for k, v in profiles.items()},
        allow_pickle=True,
    )

    net_labels = load_network_labels()

    # Runs are collapsed within subject: one profile per subject, built by
    # Fisher-z averaging that subject's two runs parcel by parcel.
    subject_stacks = {
        s: profile_stack(profiles, 0, [(s, r) for r in RUNS]) for s in SUBJECTS
    }
    subject_matrices = {s: variant_similarity(st) for s, st in subject_stacks.items()}

    # One scale shared across subjects so their panels stay comparable.
    vlim_subject = float(max(np.abs(v).max() for v in subject_stacks.values()))
    vmin_subject = float(min(m.min() for m in subject_matrices.values()))
    print(
        f"\nProfile scale: +-{vlim_subject:.2f}"
        f"  |  similarity scale: {vmin_subject:.2f}-1.0"
    )

    print("\n=== Per subject (runs averaged) ===")
    for subject in SUBJECTS:
        sim_dir = SIMILARITY_DIR / subject
        heat_dir = HEATMAP_DIR / subject
        sim_dir.mkdir(parents=True, exist_ok=True)
        heat_dir.mkdir(parents=True, exist_ok=True)

        matrix = subject_matrices[subject]
        np.save(sim_dir / f"{safe}_variant_similarity.npy", matrix)
        plot_variant_matrix(
            matrix,
            f"{roi_name}: FC-profile similarity across cleaning regimes -- {subject}",
            sim_dir / f"{safe}_variant_similarity.png",
            vmin_subject,
        )
        plot_profile_heatmap(
            subject_stacks[subject], net_labels,
            f"{roi_name}: FC profile by cleaning regime -- {subject} (runs averaged)",
            heat_dir / f"{safe}_profile_bynetwork.png",
            vlim_subject, "Correlation with hippocampus (r)",
        )

        off_diag = matrix[~np.eye(len(VARIANT_TAGS), dtype=bool)]
        print(
            f"  {subject}: cross-variant r min={off_diag.min():.3f} "
            f"mean={off_diag.mean():.3f} max={off_diag.max():.3f}"
        )

    print(f"\ndone -- outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
