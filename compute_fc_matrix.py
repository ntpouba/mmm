"""
Compute full parcel-by-parcel functional connectivity (FC) matrices from the
Schaefer-400 parcellated timecourses (see extract_parcel_timecourses.py),
one 400x400 matrix per run per confound-regression variant, saved as .npy.

The per-run matrices are then averaged (plain mean, every run weighted
equally) within each task x subject x variant, and only those averages are
plotted, grouped by network membership, mirroring the layout in "Filmfest FC
matrices.pdf" (parcel-level heatmap with network grid lines). See atlas.py
for how parcels map to networks.
"""
import numpy as np
import matplotlib.pyplot as plt

from atlas import NETWORK_ORDER, NPARCEL, load_network_labels, network_block_centers, network_sort_order
from confounds import VARIANT_TAGS
from layout import TASKS, fc_avg, fc_avg_grid_png, fc_avg_png, fc_matrix, iter_runs, parcels


def compute_fc(combined):
    """combined: (400, TR) parcel timecourses -> (400, 400) Pearson FC matrix."""
    return np.corrcoef(combined)


def draw_fc_by_network(ax, fc, labels, title, xticklabels=True, yticklabels=True, fontsize=None):
    """Draw one FC matrix onto `ax`, parcels grouped into network blocks.
    Returns the image, for attaching a colorbar."""
    order = network_sort_order(labels)
    fc_sorted = fc[np.ix_(order, order)]
    sorted_labels = [labels[i] for i in order]
    boundaries, centers = network_block_centers(sorted_labels)

    im = ax.imshow(fc_sorted, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_title(title)
    for b in boundaries:
        ax.axvline(b, color="k", linewidth=0.6)
        ax.axhline(b, color="k", linewidth=0.6)
    ax.set_xticks(centers)
    ax.set_xticklabels(NETWORK_ORDER if xticklabels else [], rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticks(centers)
    ax.set_yticklabels(NETWORK_ORDER if yticklabels else [], fontsize=fontsize)
    return im


def plot_fc_by_network(fc, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = draw_fc_by_network(ax, fc, labels, title)
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fc_grid(means, subjects, task, labels, out_path):
    """3x3-style overview for one task: rows = subjects (top to bottom),
    columns = variants in VARIANT_TAGS order (left to right).
    means: {(sub, variant): (mean_fc, n_runs)}; missing cells are left blank."""
    nrow, ncol = len(subjects), len(VARIANT_TAGS)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.2 * ncol + 1.2, 4.2 * nrow + 0.6),
        squeeze=False, layout="constrained",
    )
    im = None
    for i, sub in enumerate(subjects):
        for j, variant_tag in enumerate(VARIANT_TAGS):
            ax = axes[i, j]
            if (sub, variant_tag) not in means:
                ax.axis("off")
                continue
            mean_fc, n = means[(sub, variant_tag)]
            im = draw_fc_by_network(
                ax, mean_fc, labels, f"{variant_tag} (n={n} runs)",
                xticklabels=(i == nrow - 1), yticklabels=(j == 0), fontsize=8,
            )
            if j == 0:
                ax.set_ylabel(sub, fontsize=13, fontweight="bold")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label="Pearson r")
    fig.suptitle(f"{task} FC, per-subject mean over runs", fontsize=15)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_run_fcs(runs):
    """Stage 1: one FC matrix per run and variant, saved as .npy (no plots).
    Returns {(run, variant): fc} for every matrix computed."""
    fcs = {}
    for r in runs:
        for variant_tag in VARIANT_TAGS:
            pf = parcels(r, variant_tag)
            if not pf.exists():
                print(f"[{r}/{variant_tag}] missing {pf}, skipping")
                continue

            combined = np.load(pf)  # (NPARCEL, TR)
            if combined.shape[0] != NPARCEL:
                print(f"[{r}/{variant_tag}] unexpected shape {combined.shape}, skipping")
                continue

            fc = compute_fc(combined)

            out_npy = fc_matrix(r, variant_tag)
            out_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_npy, fc)
            fcs[(r, variant_tag)] = fc

            off_diag = fc[~np.eye(NPARCEL, dtype=bool)]
            print(
                f"[{r}/{variant_tag}] saved {out_npy.name}  "
                f"TR={combined.shape[1]}  mean off-diag r={off_diag.mean():.3f}"
            )
    return fcs


def average_and_plot(runs, fcs, labels):
    """Stage 2: per (task, subject, variant), plain mean of that subject's
    per-run FC matrices (every run weighted equally), saved and plotted, plus
    one subjects x variants grid per task."""
    subjects = sorted({r.sub for r in runs})
    for task in TASKS:
        means = {}
        for sub in subjects:
            for variant_tag in VARIANT_TAGS:
                group = [
                    fcs[(r, variant_tag)] for r in runs
                    if r.task == task and r.sub == sub and (r, variant_tag) in fcs
                ]
                if not group:
                    print(f"[{sub} {task} {variant_tag}] no runs, skipping")
                    continue

                mean_fc = np.mean(group, axis=0)
                means[(sub, variant_tag)] = (mean_fc, len(group))

                out_npy = fc_avg(sub, task, variant_tag)
                out_npy.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_npy, mean_fc)

                out_png = fc_avg_png(sub, task, variant_tag)
                out_png.parent.mkdir(parents=True, exist_ok=True)
                title = f"{sub} {task} FC ({variant_tag}, mean of {len(group)} runs)"
                plot_fc_by_network(mean_fc, labels, title, out_png)

                print(f"[{sub} {task} {variant_tag}] n={len(group)} runs -> {out_npy.name}, {out_png}")

        if means:
            grid_png = fc_avg_grid_png(task)
            grid_png.parent.mkdir(parents=True, exist_ok=True)
            plot_fc_grid(means, subjects, task, labels, grid_png)
            print(f"[{task}] grid -> {grid_png}")


def main():
    labels = load_network_labels()
    runs = list(iter_runs())

    print("=== Stage 1: per-run FC ===")
    fcs = compute_run_fcs(runs)

    print("\n=== Stage 2: per-subject averages ===")
    average_and_plot(runs, fcs, labels)

    print("\ndone")


if __name__ == "__main__":
    main()
