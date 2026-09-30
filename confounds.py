"""
Single source of truth for the confound-regression variants, shared by the
surface (regress_out_confounds.py) and volume (volume/*) pipelines and by every
downstream script that loops over variants.

Every variant is the same base model -- 6 rigid-body motion parameters plus
linear and quadratic drift -- with extra fMRIPrep confound columns on top, so a
variant is defined just by what it adds. Adding or dropping a variant here
changes both the regression and every downstream VARIANT_TAGS loop at once.

Scripts under volume/ put the project root on sys.path before importing this.
"""
import numpy as np

MOTION_COLS = [
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
]

DRIFT = ["linear", "quadratic"]

# variant tag -> confound columns added on top of the base model
_EXTRA_COLS = {
    "base": [],
    "basecsfwm": ["csf", "white_matter"],
    "baseacc6": [f"a_comp_cor_{i:02d}" for i in range(6)],
}

CONFOUND_VARIANTS = {
    tag: {"cols": MOTION_COLS + extra, "drift": DRIFT}
    for tag, extra in _EXTRA_COLS.items()
}
VARIANT_TAGS = list(CONFOUND_VARIANTS)


def build_covariates(confounds, spec):
    """confounds: fMRIPrep confounds DataFrame; spec: a CONFOUND_VARIANTS entry.
    Returns the (T, n_covariates) design, NaNs (first-row derivatives etc.) -> 0."""
    covariates = confounds[spec["cols"]].copy().fillna(0)

    if "linear" in spec["drift"]:
        covariates["linear"] = np.linspace(0, 1, len(covariates))
    if "quadratic" in spec["drift"]:
        covariates["quadratic"] = covariates["linear"] ** 2

    return covariates
