"""
Volume-space equivalent of freesurfer_surface_smoothing_1.sh: skull-strip
(apply fMRIPrep's brain mask, since desc-preproc_bold.nii.gz ships with real
skull/scalp/eye signal still in it -- the mask is a separate file fMRIPrep
does not apply for you) and then Gaussian-smooth (FWHM=4mm, matching the
surface pipeline's smoothing extent).

Order matters here: masking happens BEFORE smoothing, so the Gaussian blur
never mixes real non-brain signal into brain-edge voxels. Smoothing runs on
the full 3D volume (not a flattened/atlas-restricted array), since the blur
needs real spatial neighbors -- restricting to atlas voxels happens later,
in regress_out_confounds_volume.py, once smoothing no longer needs the full
spatial grid.
"""
from pathlib import Path

import nibabel as nb
import numpy as np
from nilearn.image import smooth_img

SUBJECTS = ["sub-03", "sub-04", "sub-05"]
SESSION = "ses-19"
RUNS = ["run-01", "run-02"]

SMOOTH_FWHM = 4.0

ROOT = Path(__file__).resolve().parent.parent
DATADIR = ROOT / "data"


def bold_path(subject, run):
    return (
        DATADIR / subject / "func"
        / f"{subject}_{SESSION}_task-NATencoding_{run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    )


def mask_path(subject, run):
    return (
        DATADIR / subject / "func"
        / f"{subject}_{SESSION}_task-NATencoding_{run}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
    )


def smoothed_out_path(subject, run):
    return (
        DATADIR / subject / "func"
        / f"{subject}_{SESSION}_task-NATencoding_{run}_space-MNI152NLin2009cAsym_res-2_desc-sm4_bold.nii.gz"
    )


def main():
    for subject in SUBJECTS:
        for run in RUNS:
            print(f"\nProcessing {subject} {run}")

            bold_img = nb.load(bold_path(subject, run))
            mask = nb.load(mask_path(subject, run)).get_fdata().astype(bool)

            bold_data = bold_img.get_fdata(dtype=np.float32)
            print(f"  Input shape: {bold_data.shape}")

            # Skull-strip: zero every voxel outside the brain mask, in place
            # to avoid doubling peak memory on a ~6GB (float32) 4D array.
            bold_data *= mask[..., None]

            masked_img = nb.Nifti1Image(bold_data, bold_img.affine, bold_img.header)

            smoothed_img = smooth_img(masked_img, fwhm=SMOOTH_FWHM)

            out_path = smoothed_out_path(subject, run)
            nb.save(smoothed_img, out_path)
            print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
