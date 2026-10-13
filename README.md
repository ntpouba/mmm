# mmm

Run everything from anywhere in the repo with `uv run python -m mmm.<module>`. All outputs resolve from the repo root, so the working directory doesn't matter.

## Setup

```bash
uv sync          # installs mmm (editable) + dependencies
```

Raw fMRIPrep data is read from `/home/datasets/mmm`. The pipeline's smoothing step needs FreeSurfer (`$FREESURFER_HOME`).

## Layout

```
src/mmm/
  utils/      layout (all paths), atlas, confounds
  pipeline/   raw fMRIPrep -> parcel timecourses
  analyses/
    connectivity/     FC matrices, LOO-ISFC, hippocampal FC profiles
    lag_correlation/  stimulus feature -> BOLD lag correlations
movie_order.csv       clip onsets/durations per subject, session and run
data/stimuli/         movies + per-clip feature caches (envelopes/, motion/)
derivatives/          parcel timecourses written by the pipeline
outputs/
  figures/            plots, nested like src/mmm/analyses
  cache/              intermediate arrays, safe to delete and regenerate
  tmp/                pipeline temp files
  logs/
```

## 1. Pipeline: raw data → parcel timecourses

```bash
uv run python -m mmm.pipeline.postprocess                                  # every run of every task
uv run python -m mmm.pipeline.postprocess --task NATencoding --sessions ses-29 ses-30 --jobs 4
```

Smooths (4mm FWHM), regresses confounds (variants `base`, `basecsfwm`, `baseacc6`) and parcellates to Schaefer-400, writing to `derivatives/`. Options: `--task`, `--subjects sub-XX …`, `--sessions ses-XX …`, `--jobs N`, `--force`. Runs that already have parcels are skipped, so rerunning after new data only processes the new runs; `--force` recomputes them.

Everything except `hippocampal_fc_profile` reads these parcels, so run this first when new sessions arrive.

## 2. Connectivity

```bash
uv run python -m mmm.analyses.connectivity.compute_fc_matrix
uv run python -m mmm.analyses.connectivity.compute_loo_isfc
uv run python -m mmm.analyses.connectivity.hippocampal_fc_profile --jobs 4
```

| Script | Options | What it does | Writes |
|---|---|---|---|
| `compute_fc_matrix` | none | FC per run × variant, then mean per subject × task × variant | `cache/connectivity/{fc_matrices,fc_avg}/`, `figures/connectivity/fc_avg/<task>/` |
| `compute_loo_isfc` | none | leave-one-out ISFC for Negative Space, ses-19 (hardcoded) | `cache/connectivity/isfc_matrices/`, `figures/connectivity/isfc_bynetwork/` |
| `hippocampal_fc_profile` | `--jobs N`, `--force` | per-run hippocampus → cortex FC profile, then averages and variant similarity | `cache/connectivity/hippocampal_*/`, `figures/connectivity/hippocampal/<task>/` |

`compute_fc_matrix` recomputes every run each time (~20 s). `hippocampal_fc_profile` works from the MNI-space BOLD directly, so it doesn't depend on the pipeline; it skips runs already done unless `--force` (a full forced run is ~16 min with `--jobs 4`).

## 3. Lag correlation (stimulus → BOLD)

```bash
# first showing of each clip
uv run python -m mmm.analyses.lag_correlation.audio_envelope_corr              # auditory cortex x audio envelope
uv run python -m mmm.analyses.lag_correlation.motion_energy_corr               # VisCent x motion energy
uv run python -m mmm.analyses.lag_correlation.audio_envelope_corr "Fool" "Big Take"
uv run python -m mmm.analyses.lag_correlation.motion_energy_corr --exclude "Snack Attack"

# every showing of The Bench and From Dad To Son
uv run python -m mmm.analyses.lag_correlation.audio_repeated_corr
uv run python -m mmm.analyses.lag_correlation.motion_repeated_corr
```

- With no clip names, every clip with both a movie in `data/stimuli/movies/` and a parcellated first session is run, except The Bench and From Dad To Son (handled by the `*_repeated_corr` scripts). Name clips to run only those; `--exclude` drops some.
- A clip's first run computes and caches its features in `data/stimuli/{envelopes,motion}/` (motion: ~25 s per clip); later runs take seconds.
- Figures go to `outputs/figures/lag_correlation/{audio_envelope_corr,motion_energy_corr}/{individual,overview,repeated}/`. Peak tables print to the terminal; redirect into `outputs/logs/` to keep them.
- Negative lag = stimulus leads BOLD; a peak is marked plausible (`*`) at -5 to -2 TR with r > 0.1.
- The visual ROI is `ROI_TAG` in `motion_energy_corr.py` (currently `VisCent`).
- Feature caches are keyed by clip name only. If a movie file is replaced with a different version, delete its `.npz` files in `data/stimuli/envelopes/` and `data/stimuli/motion/`.

## Typical order when new data arrives

```bash
uv run python -m mmm.pipeline.postprocess --jobs 4
uv run python -m mmm.analyses.connectivity.compute_fc_matrix
uv run python -m mmm.analyses.connectivity.hippocampal_fc_profile --jobs 4
uv run python -m mmm.analyses.lag_correlation.audio_envelope_corr
uv run python -m mmm.analyses.lag_correlation.motion_energy_corr
uv run python -m mmm.analyses.lag_correlation.audio_repeated_corr
uv run python -m mmm.analyses.lag_correlation.motion_repeated_corr
```
