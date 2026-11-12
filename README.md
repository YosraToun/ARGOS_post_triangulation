# ARGOS Post-Triangulation

This repository contains the post-processing pipeline used to compare marker-based and markerless upper-limb motion capture data. The code covers data validation, preprocessing, biomechanical model creation, inverse kinematics, error analysis, and visualization.

## Overview

The workflow is organized into a few steps:

1. Clean and synchronize raw marker-based and markerless C3D files.
2. Build personalized upper-limb biomechanical models from a static trial.
3. Run inverse kinematics on the task trials and export joint kinematics.
4. Compare markerless results against marker-based reference data.
5. Generate figures and dashboards for review.

## Repository Layout

- `main_preprocessing.py`: preprocesses raw marker-based and markerless data, corrects markerless sampling frequency, filters signals, and creates fused markerless datasets.
- `main_mbo.py`: builds personalized models and runs inverse kinematics for marker-based, SynthPose, and SynthRTMPose data.
- `create_model_upperlimb.py` : create the biomechanical models used by the kinematics pipeline (three model available : Marker-based, SynthPose, and SynthRTMPose).
- `kinematics_to_all_figure.py`: prepares kinematics error data and generates summary tables and boxplots from the `Kinematics/` results.
- `mbo_generate_figures.py`: generates static comparison plots from the kinematics outputs for all subjects and models.
- `mbo_visualization_rerun.py` : generates a 3D visualisation of the results of the inverse kinematics for a given subject and model by selecting the npy files.
- `kinematics.py`: solves inverse kinematics for a C3D trial and exports joint angles as CSV, with optional `.npy` optimization metadata.
- `analysis_3D.py`: compares absolute and anatomical-frame marker errors and exports per-point error tables.
- `comparaison_3D_generate_parquet.py`: generates the parquet file used by the Dash error explorer.
- `comparaison_3D_dash_visualisation.py`: interactive dashboard for marker position error analysis.
- `check_optimisation_results.py`: prints optimization metadata saved by the kinematics step.
- `smoke_test_data.py`: validates file presence, frame counts, and missing marker segments.

## Environment Setup

Create the conda environment from the provided file:

```bash
conda env create -f environment.yml
conda activate ARGOS_post_triangulation
```

The environment targets Python 3.11 and includes the main dependencies used by the pipeline, such as `biorbd`, `biobuddy`, `ezc3d`, `polars`, `pandas`, `plotly`, `dash`, `numpy`, and `scipy`.


## Typical Workflow

### 1. Preprocess the data

Run:

```bash
python main_preprocessing.py
```

This will:

- trim the marker-based trials,
- correct markerless frame rates,
- filter markerless trajectories,
- export preprocessed marker-based and markerless datasets,
- create fused markerless outputs such as `SynthRTMPose` and `SynthRTMPoseMB_Marker`.

### 2. Build models and run inverse kinematics

Run:

```bash
python main_mbo.py
```

This processes the preprocessed marker-based, SynthPose, and SynthRTMPose datasets. Output CSV files are written under `Kinematics/`, grouped by subject and model type. When enabled, the solver also saves `.npy` files containing optimization information.

### 3. Generate static figures

Run:

```bash
python mbo_generate_figures.py
```

This script compares marker-based and markerless kinematics and saves PNG figures under `figures/`.


### 4. Generate comparison data


Run:

```bash
python comparaison_3D_generate_parquet.py
```

This prepares the parquet dataset used for error analysis and dashboard visualization.

### 5. Explore errors interactively

Run:

```bash
python comparaison_3D_dash_visualisation.py
```

The app loads `data/pre_processed/absolute_position_errors.parquet` and provides filters for subject, task, point, and model.


### 6. Generate kinematics error figures

Run:

```bash
python kinematics_to_all_figure.py
```

The script reads kinematics from `Kinematics/`, using `Marker_based` as the reference and comparing `Synthpose`, `SynthRTMPose`, and `Rtmpose`. Its current configuration processes the strong side only, using the subject laterality mapping defined in the script, and excludes 10 frames at both ends of the `16-comb-hair` task. The input folder and these analysis settings can be changed near the top of `kinematics_to_all_figure.py`.

The script writes the strong-side error data to `figures/kinematic_errors_strong_side.parquet` and an Excel summary to `figures/kinematic_errors_summary.xlsx`. It then generates PNG boxplots under `figures/error_deg/boxplot/`: per-subject and per-task plots are grouped into subject subfolders, with additional all-subject plots grouped by task in `task/` and by degree of freedom in `dof/`.


## Outputs

The main generated artifacts are:

- preprocessed C3D files in `data/pre_processed/`
- kinematics CSV files in `Kinematics/`
- optional optimization dumps as `.npy`
- comparison parquet files for error analysis
- static figures in `figures/`


## Quick Sanity Check

If you want to validate the dataset before processing, run:

```bash
python smoke_test_data.py
```

This checks file presence, frame counts, and missing-marker logs across the subject folders.
