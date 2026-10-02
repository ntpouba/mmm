"""
Post-fmriprep pipeline, one run at a time:

    raw fsaverage6 .func.gii (DATASET)
      -> freesurfer_surface_smoothing_1.sh   4mm FWHM, into a temp dir
      -> regress_out_confounds.py            per confound variant, z-scored
      -> extract_parcel_timecourses.py       Schaefer-400 parcel means
      -> derivatives/<sub>/<ses>/func/<prefix>_space-fsaverage6_desc-schaefer400<variant>.npy

The smoothed surfaces are the only intermediate on disk; they live in a temp
dir under scratch/tmp/ that is deleted as soon as the run is done (or fails).
Runs whose parcels already exist for every variant are skipped unless --force.

Usage:
    python postprocess.py                                   # every run of every task
    python postprocess.py --task NATencoding --sessions ses-19 ses-20 --jobs 4
"""
import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from confounds import CONFOUND_VARIANTS, build_covariates
from extract_parcel_timecourses import parcellate
from layout import HEMIS, ROOT, SCRATCH, TASKS, confounds_tsv, iter_runs, parcels, surf_bold
from regress_out_confounds import load_surface, regress_zscore

SMOOTH_SH = ROOT / "freesurfer_surface_smoothing_1.sh"
TMP_DIR = SCRATCH / "tmp"


def process_run(r, force):
    """Returns (status, message) with status one of "done", "skipped", "failed"."""
    outputs = {v: parcels(r, v) for v in CONFOUND_VARIANTS}
    if not force and all(p.exists() for p in outputs.values()):
        return "skipped", "parcels exist"

    inputs = [confounds_tsv(r)] + [surf_bold(r, hemi) for hemi in HEMIS]
    missing = [p.name for p in inputs if not p.exists()]
    if missing:
        return "failed", f"missing raw input {', '.join(missing)}"

    try:
        with tempfile.TemporaryDirectory(dir=TMP_DIR, prefix=f"{r.prefix}_") as tmp:
            epi = {}
            for hemi, fs_hemi in HEMIS.items():
                smoothed = os.path.join(tmp, f"hemi-{hemi}_desc-sm4_bold.func.gii")
                proc = subprocess.run(
                    ["bash", str(SMOOTH_SH), fs_hemi, str(surf_bold(r, hemi)), smoothed],
                    capture_output=True, text=True,
                )
                if proc.returncode != 0:
                    return "failed", f"smoothing hemi-{hemi}: {proc.stderr.strip()}"
                epi[hemi] = load_surface(smoothed)  # (TR, vertices)

            confounds = pd.read_csv(confounds_tsv(r), sep="\t")

            for variant, spec in CONFOUND_VARIANTS.items():
                covariates = build_covariates(confounds, spec)
                residual = {hemi: regress_zscore(epi[hemi], covariates) for hemi in HEMIS}
                combined = parcellate(residual["L"], residual["R"])

                # save inside the temp dir, then move into place, so a crash
                # mid-save never leaves a truncated file that a rerun would skip
                staged = os.path.join(tmp, f"{variant}.npy")
                np.save(staged, combined)
                outputs[variant].parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, outputs[variant])

        return "done", f"{len(outputs)} variants, shape {combined.shape}"

    except Exception as e:
        return "failed", f"{type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Smooth, confound-regress and parcellate every discovered run."
    )
    parser.add_argument("--task", choices=TASKS, help="default: every task in layout.TASKS")
    parser.add_argument("--subjects", nargs="+", metavar="sub-XX")
    parser.add_argument("--sessions", nargs="+", metavar="ses-XX")
    parser.add_argument("--force", action="store_true",
                        help="recompute runs whose parcels already exist")
    parser.add_argument("--jobs", type=int, default=1, help="runs processed in parallel")
    args = parser.parse_args()

    runs = list(iter_runs(task=args.task, subjects=args.subjects, sessions=args.sessions))
    print(f"{len(runs)} runs discovered")
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    counts = {"done": 0, "skipped": 0, "failed": 0}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(process_run, r, args.force): r for r in runs}
        # results arrive in completion order, so the bar never stalls behind a slow run
        with tqdm(total=len(runs), unit="run") as bar:
            for future in as_completed(futures):
                status, message = future.result()
                counts[status] += 1
                tqdm.write(f"[{status:7s}] {futures[future]}: {message}")
                bar.set_postfix(counts)
                bar.update()

    print(f"\ndone {counts['done']}, skipped {counts['skipped']}, failed {counts['failed']}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
