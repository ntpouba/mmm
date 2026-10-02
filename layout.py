from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = Path("/home/datasets/mmm")  # raw fmriprep output: read only, never written to
DERIV = ROOT / "derivatives"          # parcellated timecourses, the only pipeline files kept
SCRATCH = ROOT / "scratch"            # analysis outputs + postprocess temp files

TASKS: tuple[str, ...] = ("NATencoding", "NATretrieval")

HEMIS: dict[str, str] = {"L": "lh", "R": "rh"}  # BIDS filename hemi -> FreeSurfer hemi

# every fmriprep run has exactly one confounds tsv, whatever spaces it was output in
_CONFOUNDS_RE = re.compile(
    r"(?P<sub>sub-[A-Za-z0-9]+)_(?P<ses>ses-[A-Za-z0-9]+)_task-(?P<task>[A-Za-z0-9]+)"
    r"(?:_(?P<run>run-[0-9]+))?_desc-confounds_timeseries\.tsv"
)


@dataclass(frozen=True)
class Run:
    sub: str
    ses: str
    task: str
    run: str | None = None  # most NATretrieval sessions have no run- entity

    @property
    def prefix(self) -> str:
        run_part = f"_{self.run}" if self.run else ""
        return f"{self.sub}_{self.ses}_task-{self.task}{run_part}"

    def __str__(self) -> str:
        return " ".join(p for p in (self.sub, self.ses, self.task, self.run) if p)


def iter_runs(
    task: str | None = None,
    subjects: Iterable[str] | None = None,
    sessions: Iterable[str] | None = None,
) -> Iterator[Run]:
    """Every run in DATASET for `task` (default: all of TASKS), optionally
    restricted to some subjects/sessions, sorted by sub, ses, task, run."""
    tasks = {task} if task else set(TASKS)
    subjects = set(subjects) if subjects is not None else None
    sessions = set(sessions) if sessions is not None else None

    runs = []
    for tsv in DATASET.glob("sub-*/ses-*/func/*_desc-confounds_timeseries.tsv"):
        m = _CONFOUNDS_RE.fullmatch(tsv.name)
        if m is None or m["task"] not in tasks:
            continue
        if subjects is not None and m["sub"] not in subjects:
            continue
        if sessions is not None and m["ses"] not in sessions:
            continue
        runs.append(Run(m["sub"], m["ses"], m["task"], m["run"]))

    yield from sorted(runs, key=lambda r: (r.sub, r.ses, r.task, r.run or ""))


# ------------------------------------------------------------------
# Raw inputs (DATASET/<sub>/<ses>/func/)
# ------------------------------------------------------------------

def raw_funcdir(r: Run) -> Path:
    return DATASET / r.sub / r.ses / "func"


def confounds_tsv(r: Run) -> Path:
    return raw_funcdir(r) / f"{r.prefix}_desc-confounds_timeseries.tsv"


def surf_bold(r: Run, hemi: str) -> Path:
    return raw_funcdir(r) / f"{r.prefix}_hemi-{hemi}_space-fsaverage6_bold.func.gii"


def mni_bold(r: Run) -> Path:
    return raw_funcdir(r) / f"{r.prefix}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"


def mni_brain_mask(r: Run) -> Path:
    return raw_funcdir(r) / f"{r.prefix}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"


# ------------------------------------------------------------------
# Outputs (DERIV / SCRATCH only)
# ------------------------------------------------------------------

def parcels(r: Run, variant: str) -> Path:
    """Schaefer-400 parcel timecourses (400 x TR), LH parcels then RH."""
    return DERIV / r.sub / r.ses / "func" / f"{r.prefix}_space-fsaverage6_desc-schaefer400{variant}.npy"


def fc_matrix(r: Run, variant: str) -> Path:
    return SCRATCH / "fc_matrices" / f"{r.prefix}_{variant}_fc400.npy"


def fc_avg(sub: str, task: str, variant: str) -> Path:
    """Mean of one subject's per-run FC matrices for one task and variant."""
    return SCRATCH / "fc_avg" / f"{sub}_task-{task}_{variant}_fc400.npy"


def fc_avg_png(sub: str, task: str, variant: str) -> Path:
    return SCRATCH / "fc_avg_plots" / task / sub / f"{variant}.png"


def hippo_profile(r: Run, variant: str) -> Path:
    """One run's hippocampal FC profile: (400,) correlation with each cortical parcel."""
    return SCRATCH / "hippocampal_profiles" / f"{r.prefix}_{variant}_hippo400.npy"


def hippo_avg(sub: str, task: str, kind: str) -> Path:
    """kind "profiles": (n_variants, 400) mean profiles; kind "similarity": (n_variants, n_variants)."""
    return SCRATCH / "hippocampal_avg" / f"{sub}_task-{task}_hippo_{kind}.npy"


def hippo_png(sub: str, task: str, kind: str) -> Path:
    """kind "similarity" or "profile"."""
    return SCRATCH / "hippocampal_plots" / task / sub / f"{kind}.png"


def hippo_overview_png(task: str) -> Path:
    return SCRATCH / "hippocampal_plots" / task / f"{task}_overview.png"


def fc_avg_grid_png(task: str) -> Path:
    """All subjects x variants for one task in one figure, next to the per-subject folders."""
    return SCRATCH / "fc_avg_plots" / task / f"{task}_grid.png"
