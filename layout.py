from __future__ import annotations

import itertools
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATADIR = ROOT / "data"
SCRATCH = ROOT / "scratch"

SUBJECTS: list[str] = ["sub-03", "sub-04", "sub-05"]
SESSIONS: list[str] = ["ses-19", "ses-20"]
RUNS: list[str] = ["run-01", "run-02"]
TASK: str = "NATencoding"

HEMIS: dict[str, str] = {"L": "lh", "R": "rh"}  # BIDS filename hemi -> FreeSurfer hemi


@dataclass(frozen=True)
class Run:
    sub: str
    ses: str
    run: str
    task: str = TASK

    @property
    def prefix(self) -> str:
        return f"{self.sub}_{self.ses}_task-{self.task}_{self.run}"

    def __str__(self) -> str:
        return f"{self.sub} {self.ses} {self.run}"


def iter_runs(
    subjects: Iterable[str] = SUBJECTS,
    sessions: Iterable[str] = SESSIONS,
    runs: Iterable[str] = RUNS,
    task: str = TASK,
) -> Iterator[Run]:
    for sub, ses, run in itertools.product(subjects, sessions, runs):
        yield Run(sub, ses, run, task)


# ------------------------------------------------------------------
# Inputs (flat working dir data/<subject>/func/)
# ------------------------------------------------------------------

def funcdir(r: Run) -> Path:
    return DATADIR / r.sub / "func"


def confounds_tsv(r: Run) -> Path:
    return funcdir(r) / f"{r.prefix}_desc-confounds_timeseries.tsv"


def surf_bold(r: Run, hemi: str, desc: str | None = None) -> Path:
    """fsaverage6 surface BOLD; desc=None is fmriprep's own file, "sm4" the smoothed one."""
    desc_part = f"_desc-{desc}" if desc else ""
    return funcdir(r) / f"{r.prefix}_hemi-{hemi}_space-fsaverage6{desc_part}_bold.func.gii"


def mni_bold(r: Run) -> Path:
    return funcdir(r) / f"{r.prefix}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"


def mni_brain_mask(r: Run) -> Path:
    return funcdir(r) / f"{r.prefix}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"


# ------------------------------------------------------------------
# Derived per-run outputs
# ------------------------------------------------------------------

def surf_clean(r: Run, hemi: str, variant: str, suffix: str = ".npy") -> Path:
    """Smoothed, confound-regressed, z-scored surface data (vertex x TR)."""
    return funcdir(r) / f"{r.prefix}_hemi-{hemi}_space-fsaverage6_desc-sm4zscored{variant}_bold{suffix}"


def parcels(r: Run, variant: str) -> Path:
    """Schaefer-400 parcel timecourses (400 x TR), LH parcels then RH."""
    return funcdir(r) / "parcellated" / f"{r.prefix}_space-fsaverage6_desc-schaefer400{variant}.npy"


def fc_matrix(r: Run, variant: str, suffix: str = ".npy") -> Path:
    return SCRATCH / "fc_matrices" / f"{r.prefix}_{variant}_fc400{suffix}"


def fc_bynetwork_png(r: Run, variant: str) -> Path:
    return SCRATCH / "fc_matrices_bynetwork" / r.sub / r.ses / r.run / f"{variant}_fc400_bynetwork.png"


if __name__ == "__main__":
    if sys.argv[1:] != ["smooth-jobs"]:
        sys.exit("usage: python3 layout.py smooth-jobs")
    # one tab-separated line per run and hemi: fs_hemi, input gii, output gii
    for r in iter_runs():
        for hemi, fs_hemi in HEMIS.items():
            print(fs_hemi, surf_bold(r, hemi), surf_bold(r, hemi, "sm4"), sep="\t")
