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
    covariates = confounds[spec["cols"]].copy().fillna(0)

    if "linear" in spec["drift"]:
        covariates["linear"] = np.linspace(0, 1, len(covariates))
    if "quadratic" in spec["drift"]:
        covariates["quadratic"] = covariates["linear"] ** 2

    return covariates
