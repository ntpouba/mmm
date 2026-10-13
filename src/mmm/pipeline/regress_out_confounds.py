# GPT adpted from Hongmi's script
#
# Confound regression step, called per run and hemisphere by postprocess.py.
# Nothing is written to disk here: the residuals go straight to parcellation.

import numpy as np
import nibabel as nb
import scipy as sp
import scipy.stats
from sklearn.linear_model import LinearRegression


def load_surface(path):
    """GIFTI surface timeseries -> (TR, vertices) array."""
    img = nb.load(path)

    return np.array(
        [darray.data for darray in img.darrays] # type: ignore
    )


def regress_zscore(epi, covariates):
    """epi: (TR, vertices); covariates: (TR, n_regressors) from
    confounds.build_covariates. Returns the residuals as (vertices, TR),
    each vertex z-scored over time."""

    # Check TR alignment
    if epi.shape[0] != len(covariates):
        raise ValueError(
            f"{epi.shape[0]} TRs in GIFTI but "
            f"{len(covariates)} rows in confounds"
        )


    # -----------------------------
    # Confound regression
    # -----------------------------

    reg = LinearRegression()

    reg.fit(
        covariates,
        epi
    )

    residual = epi - reg.predict(covariates)


    # Convert TR x vertices -> vertices x TR
    residual = residual.T


    # -----------------------------
    # Z-score each vertex over time
    # -----------------------------

    residual = sp.stats.zscore(
        residual,
        axis=1
    )

    return residual
