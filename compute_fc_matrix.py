"""
Compute full parcel-by-parcel functional connectivity (FC) matrices from the
Schaefer-400 parcellated timecourses (see extract_parcel_timecourses.py),
one 400x400 matrix per run per confound-regression variant.
"""
import numpy as np
import matplotlib.pyplot as plt

from confounds import VARIANT_TAGS
from layout import fc_matrix, iter_runs, parcels

NPARCEL = 400
LH_RH_BOUNDARY = 199.5  # rows/cols 0-199 = LH, 200-399 = RH


def compute_fc(combined):
    """combined: (400, TR) parcel timecourses -> (400, 400) Pearson FC matrix."""
    return np.corrcoef(combined)


def plot_fc(fc, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(fc, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Parcel (0-199 LH, 200-399 RH)")
    ax.set_ylabel("Parcel (0-199 LH, 200-399 RH)")
    ax.axvline(LH_RH_BOUNDARY, color="k", linewidth=0.5)
    ax.axhline(LH_RH_BOUNDARY, color="k", linewidth=0.5)
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
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

            out_npy = fc_matrix(r, variant_tag, ".npy")
            out_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_npy, fc)

            out_png = fc_matrix(r, variant_tag, ".png")
            plot_fc(fc, f"{r} FC ({NPARCEL}x{NPARCEL}, {variant_tag})", out_png)

            off_diag = fc[~np.eye(NPARCEL, dtype=bool)]
            print(
                f"[{r}/{variant_tag}] saved {out_npy.name}, {out_png.name}  "
                f"TR={combined.shape[1]}  mean off-diag r={off_diag.mean():.3f}"
            )

    print("\ndone")


if __name__ == "__main__":
    main()
