from pathlib import Path
from click import pause
import pandas as pd
import polars as pl
import numpy as np



def normalize_dof(name: str) -> str:
    """
    Supprime le côté pour fusionner gauche et droite.
    """

    if name.startswith("R_"):
        return name[2:]

    if name.startswith("L_"):
        return name[2:]

    return name



def generate_parquet(path_data_kinematics: Path, reference_name:str, dof_hand:str,
                     models_hpe:list[str], 
                     process_only_strong_side:bool,
                     subject_laterality:dict[str,str],
                     remove_before_after:dict[str,tuple[int,int]], filename_output:Path):
    
    reference_root = path_data_kinematics / reference_name / dof_hand
    rows = []

    for subject_dir in sorted(reference_root.glob("S*")):

        subject = subject_dir.name

        print(f"Processing {subject}")

        for marker_file in sorted(subject_dir.glob("*_RL.csv")):

            task = marker_file.stem.replace("_RL", "")

            print(f"   {task}")

            marker_df = pd.read_csv(marker_file)

            angle_cols = [
                c
                for c in marker_df.columns
                if "Rot" in c
            ]

            for method in models_hpe:

                model_file = (
                    path_data_kinematics
                    / method
                    / dof_hand
                    / subject
                    / marker_file.name
                )

                if not model_file.exists():

                    print(f"Missing: {model_file}")

                    continue

                model_df = pd.read_csv(model_file)

                n_frames = min(
                    len(marker_df),
                    len(model_df),
                )

                marker_trim = marker_df.iloc[:n_frames]
                model_trim = model_df.iloc[:n_frames]

                for kinematic in angle_cols:
                    if "Hand" not in kinematic:
                        if process_only_strong_side:

                            laterality = subject_laterality[subject]
                            if laterality == "R" and kinematic.startswith("L_"):
                                continue

                            if laterality == "L" and kinematic.startswith("R_"):
                                continue

                        diff_rad = (
                            model_trim[kinematic].values
                            - marker_trim[kinematic].values
                        )

                        diff_deg = np.rad2deg(diff_rad)

                        abs_diff_deg = np.abs(diff_deg)

                        dof = normalize_dof(kinematic)

                        for frame in range(n_frames):
                            if task in remove_before_after:
                                before, after = remove_before_after[task]

                                if frame < (before/100)*n_frames or frame >= n_frames - after/100*n_frames:
                                    continue
                            
                            rows.append(
                                {
                                    "subject": subject,
                                    "task": task,
                                    "model": method,
                                    "frame": frame,

                                    "kinematic": kinematic,
                                    "dof": dof,

                                    "error_rad": float(
                                        diff_rad[frame]
                                    ),

                                    "error_deg": float(
                                        diff_deg[frame]
                                    ),

                                    "abs_error_deg": float(
                                        abs_diff_deg[frame]
                                    ),
                                }
                            )

    df = pl.DataFrame(rows)

    print(df.shape)
    df.write_parquet(
        filename_output
    )

    print(f"Saved: {filename_output}")

if __name__ == "__main__":
    path_data_kinematics = Path("./Kinematics_SB2026_all")

    reference_name = "Marker_based"
    dof_hand = "with_2dof_hand"
    model_hpe = [
        "Synthpose",
        "SynthRTMPose",
        "Rtmpose",
    ]
    subject_laterality = {
        "S08": "R",
        "S09": "R",
        "S10": "R",
        "S11": "R",
        "S12": "R",
        "S13": "L",
        "S14": "R",
        "S15": "R",
        "S16": "R",
        "S17": "R",
        "S18": "R",
        "S19": "R",
        "S20": "R",
        "S21": "R",
        "S22": "R",
        "S23": "R",
        "S24": "R",}

    remove_before_after = {
        "16-comb-hair":(10, 10),
    }

    process_only_strong_side = True
    filename_output = "kinematic_errors_strong_side.parquet" if process_only_strong_side else "kinematic_errors.parquet"
    generate_parquet(
        path_data_kinematics=path_data_kinematics,
        reference_name=reference_name,
        dof_hand=dof_hand,
        models_hpe=model_hpe,
        process_only_strong_side=process_only_strong_side,
        subject_laterality=subject_laterality,
        remove_before_after=remove_before_after,
        filename_output=filename_output
    )