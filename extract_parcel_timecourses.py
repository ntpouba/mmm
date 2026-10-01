# Adapted from Yoonjung's script w/ Claude
# Extracts Schaefer 400-parcel (17 Networks) timecourses from fsaverage6 surface
# data that has already been smoothed + confound-regressed + z-scored.

from pathlib import Path

import numpy as np
import nibabel.freesurfer as fs

from confounds import VARIANT_TAGS
from layout import HEMIS, iter_runs, parcels, surf_clean

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
STANDARD_DIR = Path("standard")

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


def process_variant(variant_tag):
    print(f"\n=== Variant: {variant_tag} ===")

    for r in iter_runs():
        hemi_files = {hemi: surf_clean(r, hemi, variant_tag) for hemi in HEMIS}
        missing = [f.name for f in hemi_files.values() if not f.exists()]
        if missing:
            print(f"[{r}] missing {', '.join(missing)}, skipping")
            continue

        try:
            parcel_rows = []
            for hemi_bids, hemi_fs in HEMIS.items():
                residual = np.load(hemi_files[hemi_bids])  # vertex x TR

                v = vlabel[hemi_fs]
                # annot label 0 is background/medial wall; real parcels are 1..NPARCEL_PER_HEMI
                for p_idx in range(1, NPARCEL_PER_HEMI + 1):
                    parcel_tc = residual[v == p_idx, :]  # (n_vertices_in_parcel, TR)
                    parcel_mean = parcel_tc.mean(axis=0)  # (TR,) — mean timecourse per parcel
                    parcel_rows.append(parcel_mean)

            # LH parcels 1-200 then RH parcels 201-400, matching standard/ label order
            combined = np.vstack(parcel_rows)  # (400, TR)

            out_path = parcels(r, variant_tag)
            out_path.parent.mkdir(exist_ok=True)
            np.save(out_path, combined)
            print(f"[{r}] saved {out_path} shape={combined.shape}")

        except Exception as e:
            print(f"[{r}] ERROR: {e}")
            continue


def main():
    for tag in VARIANT_TAGS:
        process_variant(tag)

    print("\ndone")


if __name__ == "__main__":
    main()
