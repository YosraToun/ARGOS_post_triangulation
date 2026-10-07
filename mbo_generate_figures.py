import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
base_folder = Path("Kinematics")

model_to_analyze = "with_2dof_hand"

folder_MB = base_folder / "Marker_based" / model_to_analyze
folder_ML_SynthRTM = base_folder / "SynthRTMpose" / model_to_analyze
folder_ML_Synthpose = base_folder / "Synthpose" / model_to_analyze
# folder_fake_ML = base_folder / "test" / model_to_analyze / "fake_markerless"

# Figures output folder
figure_root = Path("./figures")
figure_root.mkdir(exist_ok=True)

NON_ANGLE_VARIABLES = ["TimeFrame", "TimeFrame(s)", "time", "frame", "timestamp"]


# ---------------------------------------------------------------------
# READ CSV AND CONVERT ANGLES
# ---------------------------------------------------------------------
def get_csv_dict(subject_folder):
    csv_files = {}
    for csv_path in subject_folder.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path)

            # DROP timeline columns
            for col in df.columns:
                if col in NON_ANGLE_VARIABLES:
                    df = df.drop(columns=[col])

            # # Convert only numeric angle variables
            # for col in df.columns:
            #     if np.issubdtype(df[col].dtype, np.number) and col not in NON_ANGLE_VARIABLES:
            #         df[col] = np.rad2deg(df[col])
            
            # Convert only numeric angle variables (skip first 3 columns which are translations)
            for i, col in enumerate(df.columns):
                if np.issubdtype(df[col].dtype, np.number) and col not in NON_ANGLE_VARIABLES:
                    if i >= 3:  # Only convert columns from index 3 onwards (angles)
                        df[col] = np.rad2deg(df[col])

            csv_files[csv_path.name] = df

        except Exception as e:
            print(f"⚠ Could not read {csv_path}: {e}")

    return csv_files


# ---------------------------------------------------------------------
# RESAMPLE MARKER-BASED → MARKERLESS
# ---------------------------------------------------------------------
def resample_markerbased_to_markerless(df_ml, df_mb):
    n_ml = len(df_ml)
    n_mb = len(df_mb)

    # normalized time
    t_ml = np.linspace(0, 1, n_ml)
    t_mb = np.linspace(0, 1, n_mb)

    df_mb_interp = pd.DataFrame()

    for col in df_mb.columns:
        if np.issubdtype(df_mb[col].dtype, np.number):
            df_mb_interp[col] = np.interp(t_ml, t_mb, df_mb[col])
        else:
            df_mb_interp[col] = df_mb[col].iloc[0]

    return df_mb_interp


def generate_task_figure(subject_name, task_name, aligned_data):
    """
    Parameters
    ----------
    subject_name : str
    task_name : str
    aligned_data : dict[str, pd.DataFrame]

        Example:
        {
            "Markerless": df_ml,
            "MarkerBased": df_mb,
            "FakeMarkerless":  ,
        }
    """

    reference_df = next(iter(aligned_data.values()))

    variables = [col for col in reference_df.columns if col not in NON_ANGLE_VARIABLES]

    n_vars = len(variables)

    cols = 3
    rows = math.ceil(n_vars / cols)

    subject_fig_folder = figure_root / subject_name
    subject_fig_folder.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))

    axes = np.array(axes).flatten()

    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "black",
        "cyan",
        "magenta",
    ]

    for idx, var in enumerate(variables):

        ax = axes[idx]

        for model_idx, (model_name, df_model) in enumerate(aligned_data.items()):

            if var not in df_model.columns:
                continue

            ax.plot(
                df_model[var].values,
                label=model_name,
                color=colors[model_idx % len(colors)],
                alpha=0.8,
                linewidth=1.5,
            )

        ax.set_title(var)
        ax.set_xlabel("Frame Index")
        # Assign Translation (mm) to the first 3 columns, Angle (deg) to the rest
        if idx < 3:
            ax.set_ylabel("Translation (mm)")
        else:
            ax.set_ylabel("Angle (deg)")
        ax.grid(True, alpha=0.3)

    # remove unused axes
    for idx in range(n_vars, len(axes)):
        fig.delaxes(axes[idx])

    # single legend for the figure
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(labels), 5),
        bbox_to_anchor=(0.5, 0.98),
    )

    fig.suptitle(
        f"{subject_name} - {task_name}",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    filepath = subject_fig_folder / f"{task_name}.png"

    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"   ✔ Saved figure: {filepath}")


# ---------------------------------------------------------------------
# MAIN LOOP: GENERATE CSV + FIGURES
# ---------------------------------------------------------------------
comparison_rows = []

for subject in folder_MB.iterdir():
    if not subject.is_dir():
        continue

    subject_name = subject.name
    print(f"\n=== Processing subject: {subject_name} ===")

    ml_subject_folder = folder_ML_Synthpose / subject_name
    if not ml_subject_folder.exists():
        print(f"❌ No markerless data for {subject_name}")
        continue

    ml_Synthpose_csv = get_csv_dict(ml_subject_folder)
    mb_csv = get_csv_dict(subject)
    ml_SynthRTM_csv = get_csv_dict(folder_ML_SynthRTM / subject_name)

    common_csv = set(ml_Synthpose_csv.keys()) & set(mb_csv.keys())
    common_csv = set(ml_Synthpose_csv.keys()) & set(mb_csv.keys()) & set(ml_SynthRTM_csv.keys())

    for csv_name in common_csv:
        df_ml_Synthpose = ml_Synthpose_csv[csv_name]
        df_mb = mb_csv[csv_name]
        df_ml_SynthRTM = ml_SynthRTM_csv[csv_name]

        # Report frames
        n_ml = len(df_ml_Synthpose)
        n_mb = len(df_mb)
        ratio = n_mb / n_ml if n_ml > 0 else np.nan

        print(f"\n➡ {subject_name} – {csv_name}")
        print(f"   Markerless frames     = {n_ml}")
        print(f"   Marker-based frames   = {n_mb}")
        print(f"   Detected freq ratio   = {ratio:.2f}")

        aligned_data = {
            "Markerless_Synthpose": df_ml_Synthpose,
            "MarkerBased": df_mb,
            "Markerless_SynthRTM": df_ml_SynthRTM,
        }

        # -------------------------
        # GENERATE FULL FIGURE
        # -------------------------
        task_name = csv_name.replace(".csv", "")

        generate_task_figure(subject_name, task_name, aligned_data)

# # ---------------------------------------------------------------------
# # FINAL CSV
# # ---------------------------------------------------------------------
# comparison_df = pd.DataFrame(comparison_rows)
# comparison_df.to_csv(f"comparison_dataset_{model_to_analyze}.csv", index=False)

# print(f"\n=== comparison_dataset_{model_to_analyze}.csv generated successfully ===")
# print(f"=== All figures generated in ./figures/{model_to_analyze}/<subject>/task.png ===")
