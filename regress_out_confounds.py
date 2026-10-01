# GPT adpted from Hongmi's script

import argparse

import numpy as np
import pandas as pd
import nibabel as nb
import scipy as sp
import scipy.io
from sklearn.linear_model import LinearRegression

from confounds import CONFOUND_VARIANTS, build_covariates
from layout import HEMIS, confounds_tsv, iter_runs, surf_bold, surf_clean


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

    for r in iter_runs():

        print(f"\nProcessing {r}")


        # -----------------------------
        # Load confounds
        # -----------------------------

        confound_file = confounds_tsv(r)

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

        for hemi in HEMIS:

            print(f"  Hemisphere {hemi}")


            infile = surf_bold(r, hemi, "sm4")
            out_npy = surf_clean(r, hemi, variant_tag, ".npy")
            out_mat = surf_clean(r, hemi, variant_tag, ".mat")

            if not args.force and out_npy.exists() and out_mat.exists():
                print(f"    [exists] {out_npy.stem}, skipping")
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
                    f"{r} {hemi}: "
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
                out_npy,
                residual
            )

            scipy.io.savemat(
                out_mat,
                {
                    "residual": residual
                }
            )

            print(f"    Saved {out_npy.with_suffix('')}")
            n_written += 1


print(f"\nDone. wrote {n_written}, already present {n_skipped}, "
      f"inputs missing {n_missing}")
