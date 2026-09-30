# GPT adpted from Hongmi's script

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nb
import scipy as sp
import scipy.io
from sklearn.linear_model import LinearRegression


# -----------------------------
# Paths
# -----------------------------

ROOTDIR = Path("/home/darekar1/proj/mmm")
DATADIR = ROOTDIR / "data"


# -----------------------------
# Parameters
# -----------------------------

subjects = [
    "sub-03",
    "sub-04",
    "sub-05"
]

task = "NATencoding"

# Add a session here once its fmriprep giis + confounds tsv are staged into
# data/<subject>/func/ and freesurfer_surface_smoothing_1.sh has produced its
# desc-sm4 files. Sessions whose inputs are absent are reported and skipped,
# so this is safe to run while only part of the data is in place.
sessions = ["ses-19", "ses-20"]
runs = ["run-01", "run-02"]

hemis = ["L", "R"]

old_desc = "sm4"

# Confound-regression variants to compare.
MOTION_COLS = [
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
]

CONFOUND_VARIANTS = {
    "base": {
        "cols": MOTION_COLS,
        "drift": ["linear", "quadratic"],
    },
    "basecsfwm": {
        "cols": MOTION_COLS + ["csf", "white_matter"],
        "drift": ["linear", "quadratic"],
    },
    "baseacc6": {
        "cols": MOTION_COLS + [f"a_comp_cor_{i:02d}" for i in range(6)],
        "drift": ["linear", "quadratic"],
    },
}


def build_covariates(confounds, spec):
    covariates = confounds[spec["cols"]].copy().fillna(0)

    if "linear" in spec["drift"]:
        covariates["linear"] = np.linspace(0, 1, len(covariates))
    if "quadratic" in spec["drift"]:
        covariates["quadratic"] = covariates["linear"] ** 2

    return covariates


# -----------------------------
# Process subjects
# -----------------------------

parser = argparse.ArgumentParser(
    description="Confound-regress smoothed surface data, one output per variant."
)
parser.add_argument(
    "--force", action="store_true",
    help="recompute outputs that already exist instead of skipping them",
)
args = parser.parse_args()

n_written = n_skipped = n_missing = 0

for variant_tag, variant_spec in CONFOUND_VARIANTS.items():

    print(f"\n=== Variant: {variant_tag} ===")
    new_desc = f"sm4zscored{variant_tag}"

    for subject in subjects:

        funcdir = DATADIR / subject / "func"

        for session, run in itertools.product(sessions, runs):

            print(f"\nProcessing {subject} {session} {run}")


            # -----------------------------
            # Load confounds
            # -----------------------------

            confound_file = (
                funcdir
                / f"{subject}_{session}_task-{task}_{run}_desc-confounds_timeseries.tsv"
            )

            if not confound_file.exists():
                print(f"  [missing] {confound_file.name}, skipping")
                n_missing += 1
                continue

            confounds = pd.read_csv(
                confound_file,
                sep="\t"
            )

            covariates = build_covariates(confounds, variant_spec)


            # -----------------------------
            # Process hemispheres
            # -----------------------------

            for hemi in hemis:

                print(f"  Hemisphere {hemi}")


                infile = (
                    funcdir
                    / f"{subject}_{session}_task-{task}_{run}_hemi-{hemi}_space-fsaverage6_desc-{old_desc}_bold.func.gii"
                )

                outfile = (
                    funcdir
                    / f"{subject}_{session}_task-{task}_{run}_hemi-{hemi}_space-fsaverage6_desc-{new_desc}_bold"
                )

                if (not args.force
                        and outfile.with_suffix(".npy").exists()
                        and outfile.with_suffix(".mat").exists()):
                    print(f"    [exists] {outfile.name}, skipping")
                    n_skipped += 1
                    continue

                if not infile.exists():
                    print(f"    [missing] {infile.name}, skipping")
                    n_missing += 1
                    continue


                # Load GIFTI
                img = nb.load(infile)

                epi = np.array(
                    [darray.data for darray in img.darrays] # type: ignore
                )

                # Check TR alignment
                if epi.shape[0] != len(covariates):
                    raise ValueError(
                        f"{subject} {run} {hemi}: "
                        f"{epi.shape[0]} TRs in GIFTI but "
                        f"{len(covariates)} rows in confounds"
                    )

                print(f"    Input shape: {epi.shape}")


                # -----------------------------
                # Confound regression
                # -----------------------------

                reg = LinearRegression()

                reg.fit(
                    covariates,
                    epi
                )

                residual = epi - reg.predict(covariates)


                # Convert TR x vertices -> vertices x TR
                residual = residual.T


                # -----------------------------
                # Z-score each vertex over time
                # -----------------------------

                residual = sp.stats.zscore(
                    residual,
                    axis=1
                )


                print(f"    Output shape: {residual.shape}")


                # -----------------------------
                # Save
                # -----------------------------

                np.save(
                    outfile.with_suffix(".npy"),
                    residual
                )

                scipy.io.savemat(
                    outfile.with_suffix(".mat"),
                    {
                        "residual": residual
                    }
                )

                print(f"    Saved {outfile}")
                n_written += 1


print(f"\nDone. wrote {n_written}, already present {n_skipped}, "
      f"inputs missing {n_missing}")
