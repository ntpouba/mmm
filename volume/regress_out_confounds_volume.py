"""
Volume-space equivalent of regress_out_confounds.py: takes the masked+smoothed
volumes from mask_and_smooth.py, restricts to voxels covered by the Schaefer-400
volume atlas (same voxels extract_parcel_timecourses_volume.py will average
over -- restricting here, not before smoothing, keeps a consistent flat voxel
index/order across subjects and cuts the regression down to only the voxels
actually used downstream), regresses out confounds and z-scores each voxel's
timeseries via nilearn.signal.clean.

Uses the shared CONFOUND_VARIANTS/build_covariates from confounds.py, same as
regress_out_confounds.py (custom hand-picked columns, not nilearn's named
strategies), so the volume variants are directly comparable to the surface ones. nilearn.signal.clean's
own detrending/filtering are turned off (detrend=False, filter=False) since
our own covariates already carry the linear/quadratic drift terms per variant
-- letting nilearn's defaults run too would double up on detrending.
"""
import sys
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.signal import clean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from confounds import CONFOUND_VARIANTS, build_covariates  # noqa: E402

SUBJECTS = ["sub-03", "sub-04", "sub-05"]
SESSION = "ses-19"
RUNS = ["run-01", "run-02"]
TR = 1.5

DATADIR = ROOT / "data"
ATLAS_PATH = ROOT / "standard/Schaefer2018_400Parcels_MNI152NLin2009cAsym_2mm.nii.gz"


def smoothed_path(subject, run):
    return (
        DATADIR / subject / "func"
        / f"{subject}_{SESSION}_task-NATencoding_{run}_space-MNI152NLin2009cAsym_res-2_desc-sm4_bold.nii.gz"
    )


def confound_path(subject, run):
    return (
        DATADIR / subject / "func"
        / f"{subject}_{SESSION}_task-NATencoding_{run}_desc-confounds_timeseries.tsv"
    )


def cleaned_out_path(subject, run, variant_tag):
    return (
        DATADIR / subject / "func"
        / f"{subject}_{SESSION}_task-NATencoding_{run}_space-MNI152NLin2009cAsym_res-2_desc-sm4zscored{variant_tag}.npy"
    )


def main():
    atlas_mask = nb.load(ATLAS_PATH).get_fdata() > 0
    print(f"Atlas mask: {atlas_mask.sum()} voxels")

    for subject in SUBJECTS:
        for run in RUNS:
            print(f"\nProcessing {subject} {run}")

            smoothed = nb.load(smoothed_path(subject, run)).get_fdata(dtype=np.float32)
            # (X, Y, Z, T) -> (n_atlas_voxels, T)
            voxel_timeseries = smoothed[atlas_mask]
            print(f"  Voxel timeseries shape: {voxel_timeseries.shape}")

            confounds = pd.read_csv(confound_path(subject, run), sep="\t")

            if voxel_timeseries.shape[1] != len(confounds):
                raise ValueError(
                    f"{subject} {run}: "
                    f"{voxel_timeseries.shape[1]} TRs in volume but "
                    f"{len(confounds)} rows in confounds"
                )

            # The full 4D volume is no longer needed once the atlas voxels are
            # extracted; drop it before looping variants (~3GB per run).
            del smoothed

            for variant_tag, variant_spec in CONFOUND_VARIANTS.items():
                covariates = build_covariates(confounds, variant_spec)

                # nilearn.signal.clean wants (n_timepoints, n_features).
                signals = voxel_timeseries.T

                cleaned = clean(
                    signals,
                    confounds=covariates.values,
                    detrend=False,
                    filter=False,
                    standardize="zscore_sample",
                    t_r=TR,
                )

                cleaned = cleaned.T  # back to (n_atlas_voxels, T)

                out_path = cleaned_out_path(subject, run, variant_tag)
                np.save(out_path, cleaned)
                print(f"  [{variant_tag}] saved {out_path.name}, shape={cleaned.shape}")


if __name__ == "__main__":
    main()
