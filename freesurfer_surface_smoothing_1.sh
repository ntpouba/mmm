#! /bin/bash
# use freesurfer mri_surf2surf to smooth functional images preprocessed with fmriprep
# to be run on the lab server
# hongmi lee 5/30/22
#
# Inputs are read from the flat working dir data/sub-XX/func/ -- copy a session's
# fmriprep giis there from data/sub-0XX/ses-YY/func/ before running.
#
# Which runs to smooth, and every input/output path, come from layout.py
# (`python3 layout.py smooth-jobs`); add a session there as it is staged. Runs
# whose input is not present are reported and skipped, so this is safe to
# re-run while only part of the data is in place. Outputs that already exist
# are left alone unless --force is passed.

LAYOUT=$(dirname "$0")/layout.py

# mri_surf2surf resolves `--s fsaverage6` inside $SUBJECTS_DIR, not as a path.
# Set it explicitly: an inherited SUBJECTS_DIR pointing at a dataset collection
# (e.g. /home/datasets) has no fsaverage6 in it and every call dies reading
# fsaverage6/surf/?h.sphere.reg. $FREESURFER_HOME/subjects is FreeSurfer's own
# stock template directory and is the default when nothing overrides it.
export SUBJECTS_DIR=${SUBJECTS_DIR_OVERRIDE:-${FREESURFER_HOME}/subjects}

if [ ! -f "${SUBJECTS_DIR}/fsaverage6/surf/lh.sphere.reg" ]; then
    echo "ERROR: no fsaverage6 surfaces under SUBJECTS_DIR=${SUBJECTS_DIR}"
    echo "       expected ${SUBJECTS_DIR}/fsaverage6/surf/lh.sphere.reg"
    exit 1
fi
echo "SUBJECTS_DIR=${SUBJECTS_DIR}"

smoothfwhm=4  # must match the desc-sm4 output names in layout.py

force=0
if [ "$1" = "--force" ]; then
    force=1
    echo "--force: existing smoothed files will be overwritten"
fi

n_done=0
n_skipped=0
n_missing=0
n_failed=0

job_list=$(python3 "${LAYOUT}" smooth-jobs) || { echo "ERROR: ${LAYOUT} smooth-jobs failed"; exit 1; }

while IFS=$'\t' read -r fs_hemi sval tval; do

    [ -n "${fs_hemi}" ] || continue   # empty job list still yields one blank line

    if [ ! -f "${sval}" ]; then
        echo "[missing] ${sval}"
        n_missing=$((n_missing + 1))
        continue
    fi

    if [ -f "${tval}" ] && [ ${force} -eq 0 ]; then
        echo "[exists ] ${tval}"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    echo "[smooth ] $(basename "${sval}")"
    if mri_surf2surf --s fsaverage6 --hemi ${fs_hemi} \
        --sval "${sval}" --fwhm ${smoothfwhm} --tval "${tval}" </dev/null; then
        # mri_surf2surf can exit 0 having written nothing usable
        if [ -s "${tval}" ]; then
            n_done=$((n_done + 1))
        else
            echo "[FAILED ] ${tval} was not written"
            n_failed=$((n_failed + 1))
        fi
    else
        echo "[FAILED ] mri_surf2surf exited $? for $(basename "${sval}")"
        rm -f "${tval}"   # don't leave a truncated file to be skipped next run
        n_failed=$((n_failed + 1))
    fi

done <<< "${job_list}"

echo
echo "smoothed ${n_done}, already present ${n_skipped}, inputs missing ${n_missing}, FAILED ${n_failed}"
if [ ${n_failed} -gt 0 ]; then
    exit 1
fi
