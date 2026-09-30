# Adapted from Yoonjung's script w/ Claude
# Extracts Schaefer 400-parcel (17 Networks) timecourses from fsaverage6 surface
# data that has already been smoothed + confound-regressed + z-scored.

import re
from pathlib import Path

import numpy as np
import nibabel.freesurfer as fs

from confounds import VARIANT_TAGS

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
DATADIR = Path("data")
STANDARD_DIR = Path("standard")

# subjects to skip (BIDS sub- labels, no zero-padding assumptions)
EXCLUDE_SUBS = {"003", "004", "005"}

# ------------------------------------------------------------------
# Parcel / atlas info
# ------------------------------------------------------------------
NPARCEL_PER_HEMI = 200
HEMI_BIDS_TO_FS = {"L": "lh", "R": "rh"}  # BIDS filename hemi -> annot hemi

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

# ------------------------------------------------------------------
# Filename parsing for the preprocessed per-hemisphere .npy files
# ------------------------------------------------------------------
BIDS_PATTERN = re.compile(
    r"sub-(?P<sub>[A-Za-z0-9]+)"
    r"(?:_ses-(?P<ses>[A-Za-z0-9]+))?"
    r"_task-(?P<task>[A-Za-z0-9]+)"
    r"(?:_run-(?P<run>[A-Za-z0-9]+))?"
    r"_hemi-(?P<hemi>[LR])"
    r"_space-(?P<space>[A-Za-z0-9]+)"
    r"_desc-(?P<desc>[A-Za-z0-9]+)"
    r"_bold\.npy$"
)

def find_subject_dirs(datadir: Path, exclude: set[str]) -> list[Path]:
    subs = sorted(
        p for p in datadir.glob("sub-*")
        if p.is_dir() and p.name.replace("sub-", "") not in exclude
    )
    return subs


def find_hemi_l_files(func_dir: Path, desc: str) -> list[Path]:
    """Find left-hemi processed files; right-hemi partner is derived by string swap."""
    return sorted(func_dir.glob(f"*_hemi-L_*_desc-{desc}_bold.npy"))


def process_variant(subject_dirs, target_desc: str, out_desc: str):
    print(f"\n=== Variant: desc-{target_desc} -> desc-{out_desc} ===")

    for sub_dir in subject_dirs:
        sub = sub_dir.name.replace("sub-", "")
        func_dir = sub_dir / "func"
        if not func_dir.is_dir():
            print(f"[sub-{sub}] no func/ dir, skipping")
            continue

        l_files = find_hemi_l_files(func_dir, target_desc)
        if not l_files:
            print(f"[sub-{sub}] no hemi-L desc-{target_desc} files found, skipping")
            continue

        out_dir = func_dir / "parcellated"
        out_dir.mkdir(exist_ok=True)

        for l_file in l_files:
            m = BIDS_PATTERN.match(l_file.name)
            if not m:
                print(f"[sub-{sub}] could not parse filename, skipping: {l_file.name}")
                continue
            ent = m.groupdict()

            r_file = Path(str(l_file).replace("_hemi-L_", "_hemi-R_"))
            if not r_file.exists():
                print(f"[sub-{sub}] missing R partner for {l_file.name}, skipping")
                continue

            try:
                parcel_rows = []
                for hemi_bids, hemi_fs in HEMI_BIDS_TO_FS.items():
                    npy_file = l_file if hemi_bids == "L" else r_file
                    residual = np.load(npy_file)  # vertex x TR

                    v = vlabel[hemi_fs]
                    # annot label 0 is background/medial wall; real parcels are 1..NPARCEL_PER_HEMI
                    for p_idx in range(1, NPARCEL_PER_HEMI + 1):
                        parcel_tc = residual[v == p_idx, :]  # (n_vertices_in_parcel, TR)
                        parcel_mean = parcel_tc.mean(axis=0)  # (TR,) — mean timecourse per parcel
                        parcel_rows.append(parcel_mean)

                # LH parcels 1-200 then RH parcels 201-400, matching standard/ label order
                combined = np.vstack(parcel_rows)  # (400, TR)

                bids_bits = [f"sub-{ent['sub']}"]
                if ent.get("ses"):
                    bids_bits.append(f"ses-{ent['ses']}")
                bids_bits.append(f"task-{ent['task']}")
                if ent.get("run"):
                    bids_bits.append(f"run-{ent['run']}")
                bids_bits.append(f"space-{ent['space']}")
                bids_bits.append(f"desc-{out_desc}")
                out_name = "_".join(bids_bits) + ".npy"

                out_path = out_dir / out_name
                np.save(out_path, combined)
                print(f"[sub-{sub}] saved {out_path} shape={combined.shape}")

            except FileNotFoundError as e:
                print(f"[sub-{sub}] FILE NOT FOUND: {e}")
                continue
            except Exception as e:
                print(f"[sub-{sub}] ERROR on {l_file.name}: {e}")
                continue


def main():
    subject_dirs = find_subject_dirs(DATADIR, EXCLUDE_SUBS)
    if not subject_dirs:
        print(f"No subject directories found under {DATADIR.resolve()}")
        return

    for tag in VARIANT_TAGS:
        process_variant(
            subject_dirs,
            target_desc=f"sm4zscored{tag}",
            out_desc=f"schaefer400{tag}",
        )

    print("\ndone")


if __name__ == "__main__":
    main()