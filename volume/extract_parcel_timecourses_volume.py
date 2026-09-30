"""
Volume-space equivalent of extract_parcel_timecourses.py: averages the
confound-regressed, z-scored voxel timeseries (from
regress_out_confounds_volume.py) into the 400 Schaefer parcels, using
standard/Schaefer2018_400Parcels_MNI152NLin2009cAsym_2mm.nii.gz directly --
already verified to have identical shape/affine to the subject BOLD volumes,
and its integer labels 1-400 already match the LH(1-200)/RH(201-400) row
order in Schaefer2018_400Parcels_17Networks_order.txt (verified against
world-space X coordinates), so no relabeling is needed and the same
by-network plotting code applies unchanged.
"""
import sys
from pathlib import Path

import nibabel as nb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from confounds import VARIANT_TAGS  # noqa: E402

SUBJECTS = ["sub-03", "sub-04", "sub-05"]
SESSION = "ses-19"
RUNS = ["run-01", "run-02"]

NPARCEL = 400

DATADIR = ROOT / "data"
ATLAS_PATH = ROOT / "standard/Schaefer2018_400Parcels_MNI152NLin2009cAsym_2mm.nii.gz"


def cleaned_path(subject, run, variant_tag):
    return (
        DATADIR / subject / "func"
        / f"{subject}_{SESSION}_task-NATencoding_{run}_space-MNI152NLin2009cAsym_res-2_desc-sm4zscored{variant_tag}.npy"
    )


def parcel_out_path(subject, run, variant_tag):
    out_dir = DATADIR / subject / "func" / "parcellated"
    out_dir.mkdir(exist_ok=True)
    return (
        out_dir
        / f"{subject}_{SESSION}_task-NATencoding_{run}_space-MNI152NLin2009cAsym_res-2_desc-schaefer400{variant_tag}.npy"
    )


def main():
    atlas_data = nb.load(ATLAS_PATH).get_fdata()
    atlas_mask = atlas_data > 0
    # Labels for the voxels in atlas_mask, in the same flat order that
    # `smoothed[atlas_mask]` produced in regress_out_confounds_volume.py.
    atlas_labels_in_mask = atlas_data[atlas_mask].astype(int)

    for subject in SUBJECTS:
        for run in RUNS:
            for variant_tag in VARIANT_TAGS:
                cleaned = np.load(cleaned_path(subject, run, variant_tag))  # (n_atlas_voxels, TR)

                parcel_rows = []
                for label in range(1, NPARCEL + 1):
                    parcel_tc = cleaned[atlas_labels_in_mask == label, :]  # (n_voxels_in_parcel, TR)
                    parcel_rows.append(parcel_tc.mean(axis=0))

                combined = np.vstack(parcel_rows)  # (400, TR)

                out_path = parcel_out_path(subject, run, variant_tag)
                np.save(out_path, combined)
                print(f"[{subject}/{run}/{variant_tag}] saved {out_path} shape={combined.shape}")

    print("\ndone")


if __name__ == "__main__":
    main()
