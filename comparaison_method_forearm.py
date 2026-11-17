import snip_ezc3d as snip
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import ezc3d 
# fonction 1 : Generate the homogeneoux matrix of the arm from the marker-based data
import polars as pl
import pandas as pd


dict_task = {
    "00-static-stand" : "static",
    "01-eat-yaourt" : "eat",
    "02-cut-food" : "cut",
    "03-open-bottle" : "open bottle",
    "03-Open_a_Bottle_and_Pour" : "open and pour",
    "04-Hand_to_Head" : "hand to head",
    "04-take-in-box" : "take in box",
    "05-use-computer" : "computer",
    "06-drawing": "drawing",
    "06-Open_Kit" : "open kit",
    "07-use-ruler" : "ruler",
    "08-cut-scissor": "cut scissor",
    "09-erase-eraser": "erase eraser",
    "10-reach_and_grasp_far" : "R&G far",
    "11-reach_and_grasp_mid" : "R&G mid",
    "12-reach_and_grasp_near" : "R&G near",
    "13-playdoe" : "playdoe",
    "14-put-toothpaste" : "put toothpaste",
    "15-brush-teeth" : "brush teeth",
    "16-comb-hair" : "comb hair",
    "17-hand-to-back" : "hand to back",
    "18-lift-tray" : "lift tray",
    "20-catch-throw-ball" : "catch throw ball",
    "21-abduction" : "GH abduction",
    "22-flexion" : "GH flexion",
    "23-rotation-plus-supination-full-arm-extended-front" : "GH rot + El sup extend",
    "24-Elbow_Flexion_Extension" : "Elbow flexion",
    "24-Elbow_Pronosupination_Bras_Bend" : "Elbow pronosupination",
    "26-Cymbals": "cymbals",
}

def define_frame_forearm(points_data,points_name,wrist_flexion_axis,EJC_name,WJC_name):
    # Get the indices of the markers
    EJC_index = points_name.index(EJC_name)
    WJC_index = points_name.index(WJC_name)

    # Get the positions of the markers
    EJC = points_data[:, EJC_index, :]
    WJC = points_data[:, WJC_index, :]

    # Calculate the origin of the arm frame (EJC)
    origin = EJC

    # Calculate the x-axis (from EJC to WJC)
    y_axis = EJC - WJC
    y_axis /= np.linalg.norm(y_axis, axis=0)

    # Calculate the z-axis (normal to the plane formed by GH, EJC, and WJC)
    z_axis = wrist_flexion_axis

    # Calculate the x-axis (cross product of z and y axes)
    x_axis = np.cross(z_axis.T, y_axis.T).T
    x_axis /= np.linalg.norm(x_axis, axis=0)

    # renormalize the y-axis to ensure orthogonality
    y_axis = np.cross(z_axis.T, x_axis.T).T
    y_axis /= np.linalg.norm(y_axis, axis=0)

    # Create the homogeneous transformation matrix for each frame
    n_frames = points_data.shape[2]
    H_arm = np.zeros((4, 4, n_frames))
    
    for i in range(n_frames):
        H_arm[0:3, 0, i] = x_axis[:, i]
        H_arm[0:3, 1, i] = y_axis[:, i]
        H_arm[0:3, 2, i] = z_axis[:, i]
        H_arm[0:3, 3, i] = origin[:, i]
        H_arm[3, :, i] = [0, 0, 0, 1]

    return H_arm

def compute_wrist_flexion_axis_rtmpose(c3d_path):
    acq = ezc3d.c3d(str(c3d_path))
    points_data, points_name, points_ind = snip.get_points_ezc3d(acq)
    # Perform the necessary calculations to generate the wrist flexion axis
    HM2 = points_data[:, points_name.index("R_MCP_index"), :]
    HM5 = points_data[:, points_name.index("R_MCP_little"), :]
    # Calculate the wrist flexion axis
    wrist_flexion_axis = HM2 - HM5
    wrist_flexion_axis /= np.linalg.norm(wrist_flexion_axis, axis=0)

    frame = define_frame_forearm(points_data,points_name,wrist_flexion_axis, EJC_name="R_EJC", WJC_name="R_WJC")

    return frame

def compute_wrist_flexion_axis_MB(c3d_path):
    acq = ezc3d.c3d(str(c3d_path))
    points_data, points_name, points_ind = snip.get_points_ezc3d(acq)
    # Perform the necessary calculations to generate the wrist flexion axis
    RS = points_data[:, points_name.index("R_RS"), :]
    US = points_data[:, points_name.index("R_US"), :]
    # Calculate the wrist flexion axis
    wrist_flexion_axis = RS - US
    wrist_flexion_axis /= np.linalg.norm(wrist_flexion_axis, axis=0)

    frame = define_frame_forearm(points_data,points_name,wrist_flexion_axis, EJC_name="R_EJC", WJC_name="R_WJC")

    return frame

def compute_wrist_flexion_axis_fakeRtmpose(c3d_path):
    acq = ezc3d.c3d(str(c3d_path))
    points_data, points_name, points_ind = snip.get_points_ezc3d(acq)
    # Perform the necessary calculations to generate the wrist flexion axis
    HM2 = points_data[:, points_name.index("R_HM2"), :]
    HM5 = points_data[:, points_name.index("R_HM5"), :]
    # Calculate the wrist flexion axis
    wrist_flexion_axis = HM2 - HM5
    wrist_flexion_axis /= np.linalg.norm(wrist_flexion_axis, axis=0)

    frame = define_frame_forearm(points_data,points_name,wrist_flexion_axis, EJC_name="R_EJC", WJC_name="R_WJC")

    return frame

def compute_wrist_flexion_axis_synthpose(c3d_path):
    acq = ezc3d.c3d(str(c3d_path))
    points_data, points_name, points_ind = snip.get_points_ezc3d(acq)
    # Perform the necessary calculations to generate the wrist flexion axis
    RS = points_data[:, points_name.index("r_lwrist"), :]
    US = points_data[:, points_name.index("r_mwrist"), :]
    # Calculate the wrist flexion axis
    wrist_flexion_axis = RS - US
    wrist_flexion_axis /= np.linalg.norm(wrist_flexion_axis, axis=0)

    frame = define_frame_forearm(points_data,points_name,wrist_flexion_axis, EJC_name="R_EJC", WJC_name="R_WJC")

    return frame

def compare_frames(H_plane, H_isb):

    n_frames = H_plane.shape[2]

    quaternions = np.zeros((n_frames, 4))
    angles = np.zeros(n_frames)

    for i in range(n_frames):

        R_plane = H_plane[:3, :3, i]
        R_isb   = H_isb[:3, :3, i]
        if np.isnan(R_plane).any() or np.isnan(R_isb).any():
            quaternions[i, :] = np.array([np.nan, np.nan, np.nan, np.nan])
            angles[i] = np.nan
            continue
        R_rel = R_plane @ R_isb.T

        rot = Rot.from_matrix(R_rel)

        quaternions[i, :] = rot.as_quat()  # [x,y,z,w]

        angles[i] = np.rad2deg(rot.magnitude())

    return quaternions, angles

def comp_method(c3d_path):
    frame_rtmpose= compute_wrist_flexion_axis_rtmpose(c3d_path)
    frame_MB = compute_wrist_flexion_axis_MB(c3d_path)
    frame_fakeRtmpose = compute_wrist_flexion_axis_fakeRtmpose(c3d_path)
    frame_synthpose = compute_wrist_flexion_axis_synthpose(c3d_path)
    # comparaison of the two methods
    
    # Error relative to the ISB frame
    quaternion_rtm,error_angles_rtm = compare_frames(frame_rtmpose, frame_MB)
    quaternion_synth,error_angles_synth = compare_frames(frame_synthpose, frame_MB)
    quaternion_fakeRtm,error_angles_Rtm_using_MB = compare_frames(frame_fakeRtmpose, frame_MB)
    # Error due to the 
    quaternion_fakeMBfakeML,error_angles_MBvsML_hand_as_wrist = compare_frames(frame_rtmpose, frame_fakeRtmpose)
    
    return error_angles_rtm,error_angles_Rtm_using_MB,error_angles_synth,error_angles_MBvsML_hand_as_wrist

def batch_process(reference_root,filename_output):
    """Process a batch of c3d files and compare the two methods for elbow flexion axis estimation and generate a parquet file for all subject and task with the angles between the two methods (in degrees)
    Parameters
    ----------
    reference_root : _type_
        _description_
    """
    rows = []
    for subject_dir in sorted(reference_root.glob("S*")):
        subject = subject_dir.name
        print(f"Processing {subject}")
        for c3d_file in sorted(subject_dir.glob("*.c3d")):
            task = c3d_file.stem
            print(f"   {task}")
            error_angles_rtm, error_angles_Rtm_using_MB, error_angles_synth, error_angles_MBvsML_hand_as_wrist = comp_method(c3d_file)
            for angle in error_angles_rtm:
                rows.append({"subject": subject,"model":"RTM-ML", "task": dict_task[task], "angle": angle})
            for angle in error_angles_Rtm_using_MB:
                rows.append({"subject": subject,"model":"RTM-MB", "task": dict_task[task], "angle": angle})
            for angle in error_angles_synth:
                rows.append({"subject": subject,"model":"Synth", "task": dict_task[task], "angle": angle})
            for angle in error_angles_MBvsML_hand_as_wrist:
                rows.append({"subject": subject,"model":"RTM-ML vs RTM-MB", "task": dict_task[task], "angle": angle})
    df = pl.DataFrame(rows)

    print(df.shape)
    # check if the folder exists, if not create it
    filename_output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(
        filename_output
    )   


def generate_stats_from_parquet(filename_input, filename_output):
    """
    Generate an Excel file with:

    Sheet 1: statistics for each subject and task
    Sheet 2: statistics for each task across all subjects

    Statistics:
    - mean
    - std
    - median
    - min
    - max
    - n_frames
    """
    df = pl.read_parquet(filename_input)

    # Remove NaNs if present
    df = df.filter(~pl.col("angle").is_nan())

    # -------------------------------------------------------------
    # Subject × Task
    # -------------------------------------------------------------
    stats_subject_task = (
        df.group_by(["subject", "task","model"])
        .agg(
            [
                pl.len().alias("n_frames"),
                pl.col("angle").mean().alias("mean_deg"),
                pl.col("angle").std().alias("std_deg"),
                pl.col("angle").median().alias("median_deg"),
                pl.col("angle").min().alias("min_deg"),
                pl.col("angle").max().alias("max_deg"),
            ]
        )
        .sort(["subject", "task"])
    )

    # -------------------------------------------------------------
    # Task across all subjects
    # -------------------------------------------------------------
    stats_task = (
        df.group_by(["task", "model"])
        .agg(
            [
                pl.len().alias("n_frames"),
                pl.col("angle").mean().alias("mean_deg"),
                pl.col("angle").std().alias("std_deg"),
                pl.col("angle").median().alias("median_deg"),
                pl.col("angle").min().alias("min_deg"),
                pl.col("angle").max().alias("max_deg"),
            ]
        )
        .sort("task")
    )

    # -------------------------------------------------------------
    # Export to Excel
    # -------------------------------------------------------------
    with pd.ExcelWriter(filename_output) as writer:

        stats_subject_task.to_pandas().to_excel(
            writer,
            sheet_name="subject_task",
            index=False,
        )

        stats_task.to_pandas().to_excel(
            writer,
            sheet_name="task_global",
            index=False,
        )

    print(f"Saved statistics to: {filename_output}")

    return stats_subject_task, stats_task

def generate_boxplots_from_parquet(filename_input, output_dir):
    """
    Generate:
    - one boxplot per subject (all tasks)
    - one global boxplot (all subjects)

    Parameters
    ----------
    filename_input : Path
        parquet file generated by batch_process
    output_dir : Path
        output directory for figures
    """

    import polars as pl
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(filename_input)

    df = (
        df.filter(~pl.col("angle").is_nan())
          .to_pandas()
    )

    sns.set_style("whitegrid")

    # --------------------------------------------------
    # One figure per subject
    # --------------------------------------------------

    subjects = sorted(df["subject"].unique())

    for subject in subjects:

        df_sub = df[df["subject"] == subject]

        plt.figure(figsize=(12, 6))

        sns.boxplot(
            data=df_sub,
            x="task",
            y="angle",
            hue="model",
        )

        plt.title(f"{subject} - Angle difference between methods")
        plt.xlabel("Task")
        plt.ylabel("Angle (deg)")
        plt.xticks(rotation=90)
        plt.legend(title="Method")
        plt.tight_layout()

        plt.savefig(
            output_dir / f"{subject}_boxplot.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # --------------------------------------------------
    # Global figure
    # --------------------------------------------------
    plt.figure(figsize=(14, 6))

    sns.boxplot(
        data=df,
        x="task",
        y="angle",
        hue="model",
    )

    plt.title("All subjects - Angle difference between methods")
    plt.xlabel("Task")
    plt.ylabel("Angle (deg)")
    plt.xticks(rotation=90)
    plt.legend(title="Method", loc="upper right")
    plt.tight_layout()

    plt.savefig(
        output_dir / "all_subjects_boxplot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    plt.figure(figsize=(8,6))

    sns.boxplot(
        data=df,
        x="model",
        y="angle"
    )

    plt.ylabel("Difference to ISB frame (deg)")
    plt.xlabel("")
    plt.title("Comparison of arm frame definition methods")

    plt.tight_layout()
    plt.savefig(
        output_dir / "model_comparison.png",
        dpi=300
    )
    plt.close()

    print(f"Figures saved in {output_dir}")


if __name__ == "__main__":
    # Example usage
    # c3d_file_path = "data/pre_processed/marker_based/S14/05-use-computer.c3d"
    # print("Comparing the two methods for elbow flexion axis estimation...")
    # angles = comp_method(c3d_file_path)
    # print("Angles between the two methods (in degrees):")
    # print(angles)
    reference_root = Path("data/pre_processed/marker_less/SynthRTMposeMB_Marker")
    filename_parquet = Path("figures") / "angles_between_methods_wrist" / "angles_between_methods_wrist.parquet"
    filename_excel = Path("figures") / "angles_between_methods_wrist" / "stats_angles_between_methods_wrist.xlsx"
    figure_folder = Path("figures") / "angles_between_methods_wrist"


    print(f"Processing batch of c3d files and saving to {filename_parquet}...")
    batch_process(reference_root, filename_parquet)
    generate_stats_from_parquet(filename_parquet, filename_excel)
    generate_boxplots_from_parquet(filename_parquet, figure_folder)

