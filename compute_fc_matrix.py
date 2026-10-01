"""
Compute full parcel-by-parcel functional connectivity (FC) matrices from the
Schaefer-400 parcellated timecourses (see extract_parcel_timecourses.py),
one 400x400 matrix per run per confound-regression variant.

Each matrix is saved as .npy and plotted grouped by network membership,
mirroring the layout in "Filmfest FC matrices.pdf" (parcel-level heatmap with
network grid lines). See atlas.py for how parcels map to networks.
"""
import numpy as np
import matplotlib.pyplot as plt

from atlas import NETWORK_ORDER, NPARCEL, load_network_labels, network_block_centers, network_sort_order
from confounds import VARIANT_TAGS
from layout import fc_bynetwork_png, fc_matrix, iter_runs, parcels


def compute_fc(combined):
    """combined: (400, TR) parcel timecourses -> (400, 400) Pearson FC matrix."""
    return np.corrcoef(combined)


def plot_fc_by_network(fc, labels, title, out_path):
    order = network_sort_order(labels)
    fc_sorted = fc[np.ix_(order, order)]
    sorted_labels = [labels[i] for i in order]
    boundaries, centers = network_block_centers(sorted_labels)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(fc_sorted, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_title(title)
    for b in boundaries:
        ax.axvline(b, color="k", linewidth=0.6)
        ax.axhline(b, color="k", linewidth=0.6)
    ax.set_xticks(centers)
    ax.set_xticklabels(NETWORK_ORDER, rotation=45, ha="right")
    ax.set_yticks(centers)
    ax.set_yticklabels(NETWORK_ORDER)
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    labels = load_network_labels()

    for r in iter_runs():
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

            out_png = fc_bynetwork_png(r, variant_tag)
            out_png.parent.mkdir(parents=True, exist_ok=True)
            plot_fc_by_network(fc, labels, f"{r} FC ({variant_tag}, by network)", out_png)

            off_diag = fc[~np.eye(NPARCEL, dtype=bool)]
            print(
                f"[{r}/{variant_tag}] saved {out_npy.name}, {out_png}  "
                f"TR={combined.shape[1]}  mean off-diag r={off_diag.mean():.3f}"
            )

    print("\ndone")


if __name__ == "__main__":
    main()
