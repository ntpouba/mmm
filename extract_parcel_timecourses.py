# Adapted from Yoonjung's script w/ Claude
# Extracts Schaefer 400-parcel (17 Networks) timecourses from fsaverage6 surface
# data that has already been smoothed + confound-regressed + z-scored.
# Parcellation step, called per run and variant by postprocess.py.

import numpy as np
import nibabel.freesurfer as fs

from atlas import STANDARD_DIR
from layout import HEMIS

# ------------------------------------------------------------------
# Parcel / atlas info
# ------------------------------------------------------------------
NPARCEL_PER_HEMI = 200

ANNOTS = {
    "lh": STANDARD_DIR / "standard" / "fsaverage6_Schaefer2018" / "lh.schaefer2018_400parcels_17networks_order.annot",
    "rh": STANDARD_DIR / "standard" / "fsaverage6_Schaefer2018" / "rh.schaefer2018_400parcels_17networks_order.annot",
}

# vlabel: vertex -> parcel id per hemi. ctable: colortable. struct_names: parcel names (index 0 = background)
vlabel, ctable, struct_names = {}, {}, {}
for h, path in ANNOTS.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing annot file: {path}")
    vlabel[h], ctable[h], names = fs.read_annot(str(path))
    struct_names[h] = [n.decode() if isinstance(n, bytes) else n for n in names]
    n_real_parcels = len(struct_names[h]) - 1  # minus background/medial wall
    if n_real_parcels != NPARCEL_PER_HEMI:
        raise ValueError(
            f"{h}: annot has {n_real_parcels} parcels, expected {NPARCEL_PER_HEMI}"
        )


def parcellate(residual_L, residual_R):
    """Per-hemisphere (vertices, TR) residuals -> (400, TR) parcel timecourses."""
    residuals = {"L": residual_L, "R": residual_R}

    parcel_rows = []
    for hemi_bids, hemi_fs in HEMIS.items():
        residual = residuals[hemi_bids]  # vertex x TR

        v = vlabel[hemi_fs]
        # annot label 0 is background/medial wall; real parcels are 1..NPARCEL_PER_HEMI
        for p_idx in range(1, NPARCEL_PER_HEMI + 1):
            parcel_tc = residual[v == p_idx, :]  # (n_vertices_in_parcel, TR)
            parcel_mean = parcel_tc.mean(axis=0)  # (TR,) — mean timecourse per parcel
            parcel_rows.append(parcel_mean)

    # LH parcels 1-200 then RH parcels 201-400, matching standard/ label order
    return np.vstack(parcel_rows)  # (400, TR)
