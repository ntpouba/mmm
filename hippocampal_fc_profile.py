import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from nilearn.signal import clean
from tqdm import tqdm

from atlas import NETWORK_ORDER, NPARCEL, STANDARD_DIR, load_network_labels, network_block_centers, network_sort_order
from confounds import CONFOUND_VARIANTS, VARIANT_TAGS, build_covariates
from layout import (
    TASKS, confounds_tsv, hippo_avg, hippo_overview_png, hippo_png, hippo_profile,
    iter_runs, mni_bold, mni_brain_mask,
)

TR = 1.5

# Profile heatmap columns are grouped by network via atlas.py, which reads the
# surface atlas's order file; this volume atlas's labels 1-400 match that row
# order (verified against world-space X coordinates).
SCHAEFER_PATH = STANDARD_DIR / "Schaefer2018_400Parcels_MNI152NLin2009cAsym_2mm.nii.gz"

# Melbourne Subcortex Atlas (Tian et al. 2020), scale II, in our own space.
# Both files come as a pair: the label text file has one ROI name per line,
# where line i (0-based) corresponds to atlas value i + 1 (0 = background).
TIAN_PATH = STANDARD_DIR / "Tian_Subcortex_S2_3T_2009cAsym.nii.gz"
TIAN_LABELS_PATH = STANDARD_DIR / "Tian_Subcortex_S2_3T_label.txt"

# Hippocampal subregions are picked out by name rather than by hardcoded index, so
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


# ------------------------------------------------------------------
# Stage 1: per-run profiles
# ------------------------------------------------------------------

_ATLASES = None  # loaded once per process, on its first run


def atlas_volumes(reference_img):
    """(schaefer, hippocampus) label volumes on the BOLD grid. Hippocampus
    takes precedence where the two atlases disagree."""
    global _ATLASES
    if _ATLASES is None:
        schaefer_img = nb.load(SCHAEFER_PATH)
        schaefer = np.round(schaefer_img.get_fdata()).astype(int)
        if schaefer.shape != reference_img.shape[:3] or not np.allclose(
            schaefer_img.affine, reference_img.affine
        ):
            raise ValueError("Schaefer atlas grid does not match the BOLD grid")

        tian, _, hipp_values = load_hippocampus_atlas(reference_img)
        hipp_vol = np.where(np.isin(tian, hipp_values), tian, 0)
        cortex_vol = np.where(hipp_vol > 0, 0, schaefer)
        _ATLASES = (cortex_vol, hipp_vol)
    return _ATLASES


def compute_run_profiles(r):
    """{variant: (NPARCEL,) profile} for one run: the whole hippocampus's
    correlation against every cortical parcel. All hippocampal voxels are
    pooled into one timecourse before correlating."""
    bold_img = nb.load(mni_bold(r))
    brain = nb.load(mni_brain_mask(r)).get_fdata().astype(bool)
    cortex_vol, hipp_vol = atlas_volumes(bold_img)

    roi_mask = ((cortex_vol > 0) | (hipp_vol > 0)) & brain

    data = bold_img.get_fdata(dtype=np.float32)
    voxels = data[roi_mask]  # (n_roi_voxels, T)
    del data

    cortex_labels = cortex_vol[roi_mask]
    hipp_labels = hipp_vol[roi_mask]

    confounds = pd.read_csv(confounds_tsv(r), sep="\t")
    if voxels.shape[1] != len(confounds):
        raise ValueError(
            f"{voxels.shape[1]} TRs in volume but {len(confounds)} rows in confounds"
        )

    profiles = {}
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
        hipp_ts = cleaned[hipp_labels > 0].mean(axis=0)  # (T,)

        # whole hippocampus vs every cortical parcel
        profiles[variant_tag] = np.array([correlate_profiles(hipp_ts, c) for c in cortex_ts])
    return profiles


def process_run(r, force):
    """Returns (status, message) with status one of "done", "skipped", "failed"."""
    outputs = {v: hippo_profile(r, v) for v in VARIANT_TAGS}
    if not force and all(p.exists() for p in outputs.values()):
        return "skipped", "profiles exist"

    missing = [p.name for p in (mni_bold(r), mni_brain_mask(r), confounds_tsv(r)) if not p.exists()]
    if missing:
        return "failed", f"missing raw input {', '.join(missing)}"

    try:
        profiles = compute_run_profiles(r)
        for variant_tag, profile in profiles.items():
            out = outputs[variant_tag]
            out.parent.mkdir(parents=True, exist_ok=True)
            # write then rename, so a crash never leaves a truncated file a rerun would skip
            staged = out.with_name(out.stem + ".partial.npy")
            np.save(staged, profile)
            os.replace(staged, out)
        return "done", f"{len(profiles)} variants"
    except Exception as e:
        return "failed", f"{type(e).__name__}: {e}"


def compute_all_profiles(runs, force, jobs):
    counts = {"done": 0, "skipped": 0, "failed": 0}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(process_run, r, force): r for r in runs}
        with tqdm(total=len(runs), unit="run") as bar:
            for future in as_completed(futures):
                status, message = future.result()
                counts[status] += 1
                tqdm.write(f"[{status:7s}] {futures[future]}: {message}")
                bar.set_postfix(counts)
                bar.update()
    print(f"stage 1: done {counts['done']}, skipped {counts['skipped']}, failed {counts['failed']}")


# ------------------------------------------------------------------
# Stage 2: per task x subject averages, similarity, plots
# ------------------------------------------------------------------

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


def mean_profile_stack(runs):
    """(n_variants, NPARCEL) hippocampal FC profile, rows in VARIANT_TAGS
    order, plain mean over the given runs (every run weighted equally)."""
    return np.vstack(
        [np.mean([np.load(hippo_profile(r, v)) for r in runs], axis=0) for v in VARIANT_TAGS]
    )


def draw_variant_matrix(ax, matrix, title, vmin):
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
    return im


def draw_profile_heatmap(ax, stack, labels, title, vlim, xlabel=True):
    """Rows = cleaning variants, columns = the 400 cortical parcels grouped
    into network blocks. Shows how the hippocampal FC profile itself changes
    across variants, rather than just how similar the profiles are."""
    order = network_sort_order(labels)
    sorted_stack = stack[:, order]
    sorted_labels = [labels[i] for i in order]
    boundaries, centers = network_block_centers(sorted_labels)

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
    if xlabel:
        ax.set_xlabel("Cortical parcel, grouped by network")
    ax.set_title(title, fontsize=10)
    return im


SIM_CBAR = "FC-profile similarity (r)"
PROFILE_CBAR = "Correlation with hippocampus (r)"


def plot_variant_matrix(matrix, title, out_path, vmin):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = draw_variant_matrix(ax, matrix, title, vmin)
    fig.colorbar(im, ax=ax, label=SIM_CBAR)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_profile_heatmap(stack, labels, title, out_path, vlim):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    im = draw_profile_heatmap(ax, stack, labels, title, vlim)
    fig.colorbar(im, ax=ax, label=PROFILE_CBAR)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overview(task, results, labels, out_path, vmin, vlim):
    """One row per subject: similarity matrix, then profile heatmap."""
    subjects = sorted(results)
    fig, axes = plt.subplots(
        len(subjects), 2, figsize=(17, 3.9 * len(subjects) + 0.6), squeeze=False,
        gridspec_kw={"width_ratios": [1, 2.6]}, layout="constrained",
    )
    for i, sub in enumerate(subjects):
        stack, matrix, n = results[sub]
        im_sim = draw_variant_matrix(axes[i, 0], matrix, f"similarity (n={n} runs)", vmin)
        im_prof = draw_profile_heatmap(
            axes[i, 1], stack, labels, f"profile (mean of {n} runs)", vlim,
            xlabel=(i == len(subjects) - 1),
        )
        axes[i, 0].set_ylabel(sub, fontsize=13, fontweight="bold")
    fig.colorbar(im_sim, ax=axes[:, 0], shrink=0.6, label=SIM_CBAR, location="left")
    fig.colorbar(im_prof, ax=axes[:, 1], shrink=0.6, label=PROFILE_CBAR)
    fig.suptitle(f"{task}: {HIPPOCAMPUS_ROI_NAME} FC profile by cleaning regime, per-subject mean over runs", fontsize=14)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def average_and_plot(runs, labels):
    # results[task][sub] = (mean profile stack, similarity matrix, n runs)
    results = {}
    for task in TASKS:
        for sub in sorted({r.sub for r in runs}):
            group = [
                r for r in runs
                if r.task == task and r.sub == sub
                and all(hippo_profile(r, v).exists() for v in VARIANT_TAGS)
            ]
            if not group:
                print(f"[{sub} {task}] no runs with profiles, skipping")
                continue
            stack = mean_profile_stack(group)
            matrix = variant_similarity(stack)
            results.setdefault(task, {})[sub] = (stack, matrix, len(group))

            for kind, arr in (("profiles", stack), ("similarity", matrix)):
                out = hippo_avg(sub, task, kind)
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, arr)

            off_diag = matrix[~np.eye(len(VARIANT_TAGS), dtype=bool)]
            print(
                f"  {sub} {task} (n={len(group)} runs): cross-variant r "
                f"min={off_diag.min():.3f} mean={off_diag.mean():.3f} max={off_diag.max():.3f}"
            )

    if not results:
        return

    # One scale per plot kind, shared by every subject and both tasks.
    everything = [v for per_sub in results.values() for v in per_sub.values()]
    vlim = float(max(np.abs(stack).max() for stack, _, _ in everything))
    vmin = float(min(matrix.min() for _, matrix, _ in everything))
    print(f"\nProfile scale: +-{vlim:.2f}  |  similarity scale: {vmin:.2f}-1.0")

    roi = HIPPOCAMPUS_ROI_NAME
    for task, per_sub in results.items():
        for sub, (stack, matrix, n) in per_sub.items():
            sim_png, prof_png = hippo_png(sub, task, "similarity"), hippo_png(sub, task, "profile")
            sim_png.parent.mkdir(parents=True, exist_ok=True)
            plot_variant_matrix(
                matrix, f"{roi}: FC-profile similarity across cleaning regimes -- {sub} {task} (n={n} runs)",
                sim_png, vmin,
            )
            plot_profile_heatmap(
                stack, labels, f"{roi}: FC profile by cleaning regime -- {sub} {task} (mean of {n} runs)",
                prof_png, vlim,
            )
        overview = hippo_overview_png(task)
        plot_overview(task, per_sub, labels, overview, vmin, vlim)
        print(f"[{task}] overview -> {overview}")


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--force", action="store_true",
                        help="recompute runs whose profiles already exist")
    parser.add_argument("--jobs", type=int, default=1, help="runs processed in parallel")
    args = parser.parse_args()

    runs = list(iter_runs())
    print(f"{len(runs)} runs discovered")

    print("=== Stage 1: per-run profiles ===")
    compute_all_profiles(runs, args.force, args.jobs)

    print("\n=== Stage 2: per-subject averages ===")
    average_and_plot(runs, load_network_labels())

    print("\ndone")


if __name__ == "__main__":
    main()
