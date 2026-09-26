#! /bin/bash
# use freesurfer mri_surf2surf to smooth functional images preprocessed with fmriprep
# to be run on the lab server
# hongmi lee 5/30/22
#
# Inputs are read from the flat working dir data/sub-XX/func/ -- copy a session's
# fmriprep giis there from data/sub-0XX/ses-YY/func/ before running.
#
# Add a session to `sessions` as it is staged. Sessions whose input is not
# present are reported and skipped, so this is safe to re-run while only part
# of the data is in place. Outputs that already exist are left alone unless
# --force is passed.

EXPDIR=./data

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

subjects=( 03 04 05 )
sessions=( 19 20 )
runs=( 01 02 )
smoothfwhm=4

force=0
if [ "$1" = "--force" ]; then
    force=1
    echo "--force: existing smoothed files will be overwritten"
fi

n_done=0
n_skipped=0
n_missing=0
n_failed=0

for SUBJ in "${subjects[@]}"; do

    SN=sub-${SUBJ}

    for SES in "${sessions[@]}"; do

        for run in "${runs[@]}"; do

            for hemi_pair in "lh L" "rh R"; do
                set -- $hemi_pair
                fs_hemi=$1
                bids_hemi=$2

                stem=${EXPDIR}/${SN}/func/${SN}_ses-${SES}_task-NATencoding_run-${run}_hemi-${bids_hemi}_space-fsaverage6
                sval=${stem}_bold.func.gii
                tval=${stem}_desc-sm${smoothfwhm}_bold.func.gii

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

                echo "[smooth ] ${SN} ses-${SES} run-${run} ${fs_hemi}"
                if mri_surf2surf --s fsaverage6 --hemi ${fs_hemi} \
                    --sval "${sval}" --fwhm ${smoothfwhm} --tval "${tval}"; then
                    # mri_surf2surf can exit 0 having written nothing usable
                    if [ -s "${tval}" ]; then
                        n_done=$((n_done + 1))
                    else
                        echo "[FAILED ] ${tval} was not written"
                        n_failed=$((n_failed + 1))
                    fi
                else
                    echo "[FAILED ] mri_surf2surf exited $? for ${SN} ses-${SES} run-${run} ${fs_hemi}"
                    rm -f "${tval}"   # don't leave a truncated file to be skipped next run
                    n_failed=$((n_failed + 1))
                fi
            done

        done

    done

done

echo
echo "smoothed ${n_done}, already present ${n_skipped}, inputs missing ${n_missing}, FAILED ${n_failed}"
if [ ${n_failed} -gt 0 ]; then
    exit 1
fi
