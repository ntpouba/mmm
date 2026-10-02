#! /bin/bash
# use freesurfer mri_surf2surf to smooth functional images preprocessed with fmriprep
# hongmi lee 5/30/22
#
# Smooths ONE fsaverage6 surface file. postprocess.py calls this once per run
# and hemisphere, writing into a temp dir it deletes once the run is
# parcellated. Can also be run by hand:
#
#   freesurfer_surface_smoothing_1.sh <lh|rh> <input.func.gii> <output.func.gii>
#
# Exits non-zero on any failure, leaving no partial output behind.

if [ $# -ne 3 ]; then
    echo "usage: $0 <lh|rh> <input.func.gii> <output.func.gii>" >&2
    exit 2
fi
fs_hemi=$1
sval=$2
tval=$3

smoothfwhm=4

# mri_surf2surf resolves `--s fsaverage6` inside $SUBJECTS_DIR, not as a path.
# Set it explicitly: an inherited SUBJECTS_DIR pointing at a dataset collection
# (e.g. /home/datasets) has no fsaverage6 in it and every call dies reading
# fsaverage6/surf/?h.sphere.reg. $FREESURFER_HOME/subjects is FreeSurfer's own
# stock template directory and is the default when nothing overrides it.
export SUBJECTS_DIR=${SUBJECTS_DIR_OVERRIDE:-${FREESURFER_HOME}/subjects}

if [ ! -f "${SUBJECTS_DIR}/fsaverage6/surf/lh.sphere.reg" ]; then
    echo "ERROR: no fsaverage6 surfaces under SUBJECTS_DIR=${SUBJECTS_DIR}" >&2
    echo "       expected ${SUBJECTS_DIR}/fsaverage6/surf/lh.sphere.reg" >&2
    exit 1
fi

if [ "${fs_hemi}" != "lh" ] && [ "${fs_hemi}" != "rh" ]; then
    echo "ERROR: hemi must be lh or rh, got '${fs_hemi}'" >&2
    exit 2
fi

if [ ! -f "${sval}" ]; then
    echo "ERROR: missing input ${sval}" >&2
    exit 1
fi

if mri_surf2surf --s fsaverage6 --hemi ${fs_hemi} \
    --sval "${sval}" --fwhm ${smoothfwhm} --tval "${tval}" </dev/null; then
    # mri_surf2surf can exit 0 having written nothing usable
    if [ ! -s "${tval}" ]; then
        echo "ERROR: ${tval} was not written" >&2
        rm -f "${tval}"
        exit 1
    fi
else
    status=$?
    echo "ERROR: mri_surf2surf exited ${status} for ${sval}" >&2
    rm -f "${tval}"   # don't leave a truncated file behind
    exit 1
fi
