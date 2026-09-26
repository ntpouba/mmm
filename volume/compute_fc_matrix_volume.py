"""
Volume-space equivalent of compute_fc_matrix.py: full parcel-by-parcel FC
matrices from the volume-pipeline Schaefer-400 parcellated timecourses (see
extract_parcel_timecourses_volume.py), one 400x400 matrix per subject per
run per confound-regression variant.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SUBJECTS = ["sub-03", "sub-04", "sub-05"]
SESSION = "ses-19"
RUNS = ["run-01", "run-02"]
VARIANT_TAGS = ["base", "basecsfwm", "baseacc6", "baseacc20", "gsr"]

NPARCEL = 400
LH_RH_BOUNDARY = 199.5

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "scratch/fc_matrices_volume"


def parcel_file(subject, run, variant_tag):
    return (
        ROOT / "data" / subject / "func" / "parcellated"
        / f"{subject}_{SESSION}_task-NATencoding_{run}_space-MNI152NLin2009cAsym_res-2_desc-schaefer400{variant_tag}.npy"
    )


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for subject in SUBJECTS:
        for run in RUNS:
            for variant_tag in VARIANT_TAGS:
                pf = parcel_file(subject, run, variant_tag)
                if not pf.exists():
                    print(f"[{subject}/{run}/{variant_tag}] missing {pf}, skipping")
                    continue

                combined = np.load(pf)  # (NPARCEL, TR)
                if combined.shape[0] != NPARCEL:
                    print(f"[{subject}/{run}/{variant_tag}] unexpected shape {combined.shape}, skipping")
                    continue

                fc = compute_fc(combined)

                out_npy = OUT_DIR / f"{subject}_{run}_{variant_tag}_fc400.npy"
                np.save(out_npy, fc)

                out_png = OUT_DIR / f"{subject}_{run}_{variant_tag}_fc400.png"
                plot_fc(fc, f"{subject} {run} FC ({NPARCEL}x{NPARCEL}, {variant_tag}, volume)", out_png)

                off_diag = fc[~np.eye(NPARCEL, dtype=bool)]
                print(
                    f"[{subject}/{run}/{variant_tag}] saved {out_npy.name}, {out_png.name}  "
                    f"TR={combined.shape[1]}  mean off-diag r={off_diag.mean():.3f}"
                )

    print("\ndone")


if __name__ == "__main__":
    main()
